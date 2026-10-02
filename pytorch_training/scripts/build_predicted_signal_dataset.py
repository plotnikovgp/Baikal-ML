#!/usr/bin/env python3
"""Apply one frozen hit classifier and write matched predicted-signal events.

Both MC and experimental inputs are decoded to physical units, normalized to the
classifier's training statistics for inference, and finally stored using one
common angle-reconstruction normalization.  Only hits above the classifier
threshold are written.  Events must contain at least ``min_hits`` selected hits
on at least ``min_strings`` detector strings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

SPLITS = ("train", "val", "test")
EVENT_KEYS = ("ev_ids", "prime_prty", "cluster_ids", "num_un_strings")
HIT_KEYS = ("channels", "labels", "t_res")


def data_path(group: h5py.Group, key: str) -> str | None:
    if f"{key}/data" in group:
        return f"{key}/data"
    if key in group and isinstance(group[key], h5py.Dataset):
        return key
    return None


def norm_values(h5: h5py.File, group: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(h5[f"{group}/mean"], dtype=np.float32),
        np.asarray(h5[f"{group}/std"], dtype=np.float32),
    )


def append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    old = len(dataset)
    dataset.resize(old + len(values), axis=0)
    dataset[old:] = values


def appendable(
    parent: h5py.Group,
    name: str,
    source: h5py.Dataset,
    compression: str | None = None,
    dtype=None,
) -> h5py.Dataset:
    shape = (0,) + source.shape[1:]
    options = {"chunks": True}
    if compression:
        options["compression"] = compression
    return parent.require_group(name).create_dataset(
        "data",
        shape=shape,
        maxshape=(None,) + source.shape[1:],
        dtype=dtype or source.dtype,
        **options,
    )


def particle_mask(ids: np.ndarray, particle: str) -> np.ndarray:
    if particle == "all":
        return np.ones(len(ids), dtype=bool)
    prefixes = ids.astype("S5")
    wanted = {"muatm": b"muatm", "nuatm": b"nuatm", "nue2": b"nue2_"}[particle]
    return prefixes == wanted


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.encoder import Encoder, EncoderTwoHead

    if args.model_type == "encoder_two_head":
        model = EncoderTwoHead(
            in_features=5,
            hidden_size=args.hidden,
            num_shared_layers=args.layers,
            num_cls_layers=args.cls_layers,
            num_tres_layers=args.tres_layers,
            dim_feedforward_size=args.ff_size,
            n_heads=args.heads,
            cls_out_size=2,
            tres_out_size=1,
            dropout_p=0.0,
        )
    else:
        model = Encoder(
            in_features=5,
            hidden_size=args.hidden,
            num_layers=args.layers,
            dim_feedforward_size=args.ff_size,
            n_heads=args.heads,
            out_size=2,
            dropout_p=0.0,
            use_cls_token=False,
        )
    state = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(
        state["model"] if "model" in state else state, strict=False
    )
    # Some historical two-head checkpoints used a deeper t_res-only head.  That
    # head is irrelevant here; require the shared trunk and classification head
    # to match exactly while allowing only t_res-head incompatibilities.
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith("tres_head")]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("tres_head")
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Classifier checkpoint mismatch: missing={bad_missing}, unexpected={bad_unexpected}"
        )
    return model


def max_kept_for_split(args: argparse.Namespace, split: str) -> int | None:
    value = getattr(args, f"max_kept_{split}")
    return None if value <= 0 else value


def process_split(
    source: h5py.File,
    output: h5py.File,
    split: str,
    model: torch.nn.Module,
    device: torch.device,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    model_mean: np.ndarray,
    model_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, int | float]:
    src = source[split]
    dst = output.create_group(split)
    starts_path = data_path(src, "ev_starts")
    data_key = data_path(src, "data")
    ids_path = data_path(src, "ev_ids")
    channels_path = data_path(src, "channels")
    assert starts_path and data_key and ids_path and channels_path
    starts_source = src[starts_path]
    n_events = len(starts_source) - 1

    event_sources: dict[str, h5py.Dataset] = {}
    event_outputs: dict[str, h5py.Dataset] = {}
    for key in EVENT_KEYS:
        path = data_path(src, key)
        if path:
            event_sources[key] = src[path]
            event_outputs[key] = appendable(dst, key, src[path])

    hit_sources: dict[str, h5py.Dataset] = {}
    hit_outputs: dict[str, h5py.Dataset] = {}
    for key in HIT_KEYS:
        path = data_path(src, key)
        if path:
            hit_sources[key] = src[path]
            hit_outputs[key] = appendable(dst, key, src[path], compression=args.compression)
    data_out = appendable(dst, "data", src[data_key], compression=args.compression)
    prob_out = dst.require_group("signal_probability").create_dataset(
        "data",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        chunks=True,
        compression=args.compression,
    )
    starts_out = dst.require_group("ev_starts").create_dataset(
        "data", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=True
    )
    starts_out[0] = 0

    scanned = eligible = kept = written_hits = 0
    truth_signal_selected = truth_noise_selected = truth_signal_total = 0
    max_kept = max_kept_for_split(args, split)
    q_cap = (args.q_cap - model_mean[0]) / model_std[0] if args.q_cap > 0 else None

    for event_start in range(0, n_events, args.batch_size):
        if max_kept is not None and kept >= max_kept:
            break
        event_end = min(event_start + args.batch_size, n_events)
        starts = np.asarray(starts_source[event_start : event_end + 1], dtype=np.int64)
        hit_start, hit_end = int(starts[0]), int(starts[-1])
        rel = starts - hit_start
        ids = np.asarray(src[ids_path][event_start:event_end])
        wanted = particle_mask(ids, args.particle)
        scanned += event_end - event_start
        eligible += int(wanted.sum())
        if not wanted.any():
            continue

        stored = np.asarray(src[data_key][hit_start:hit_end], dtype=np.float32)
        raw = stored * source_std + source_mean
        classifier_features = (raw - model_mean) / model_std
        channels = np.asarray(src[channels_path][hit_start:hit_end])
        event_batch = {
            key: np.asarray(values[event_start:event_end]) for key, values in event_sources.items()
        }
        hit_batch = {
            key: np.asarray(values[hit_start:hit_end]) for key, values in hit_sources.items()
        }
        wanted_indices = np.flatnonzero(wanted)
        lengths = np.diff(rel)[wanted_indices]
        seq_len = int(lengths.max())
        x = np.zeros((len(wanted_indices), seq_len, 5), dtype=np.float32)
        mask = np.zeros((len(wanted_indices), seq_len), dtype=bool)
        for row, event_idx in enumerate(wanted_indices):
            a, b = int(rel[event_idx]), int(rel[event_idx + 1])
            x[row, : b - a] = classifier_features[a:b]
            mask[row, : b - a] = True
        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        mask_t = torch.from_numpy(mask).to(device, non_blocking=True)
        if q_cap is not None:
            x_t[..., 0].clamp_(max=float(q_cap))
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"),
        ):
            model_output = model(x_t, mask_t)
            probabilities = torch.sigmoid(model_output[..., 1]).float().cpu().numpy()

        event_values: dict[str, list[np.ndarray]] = {key: [] for key in event_outputs}
        hit_values: dict[str, list[np.ndarray]] = {key: [] for key in hit_outputs}
        selected_data: list[np.ndarray] = []
        selected_probs: list[np.ndarray] = []
        new_lengths: list[int] = []
        for row, event_idx in enumerate(wanted_indices):
            if max_kept is not None and kept + len(new_lengths) >= max_kept:
                break
            a, b = int(rel[event_idx]), int(rel[event_idx + 1])
            signal = probabilities[row, : b - a] >= args.threshold
            count = int(signal.sum())
            if count < args.min_hits:
                continue
            selected_channels = channels[a:b][signal]
            if np.unique(selected_channels // args.channels_per_string).size < args.min_strings:
                continue
            for key, source_values in event_sources.items():
                if key == "num_un_strings":
                    value = np.asarray(
                        np.unique(selected_channels // args.channels_per_string).size,
                        dtype=source_values.dtype,
                    )
                else:
                    value = event_batch[key][event_idx]
                event_values[key].append(value)
            for key in hit_sources:
                values = hit_batch[key][a:b][signal]
                hit_values[key].append(values)
                if key == "labels":
                    truth_signal_selected += int(np.count_nonzero(values != 0))
                    truth_noise_selected += int(np.count_nonzero(values == 0))
            if "labels" in hit_sources:
                all_labels = hit_batch["labels"][a:b]
                truth_signal_total += int(np.count_nonzero(all_labels != 0))
            selected_raw = raw[a:b][signal]
            selected_data.append(((selected_raw - output_mean) / output_std).astype(np.float32))
            selected_probs.append(probabilities[row, : b - a][signal].astype(np.float32))
            new_lengths.append(count)

        if not new_lengths:
            continue
        for key, values in event_values.items():
            append(event_outputs[key], np.asarray(values, dtype=event_outputs[key].dtype))
        for key, values in hit_values.items():
            append(hit_outputs[key], np.concatenate(values))
        append(data_out, np.concatenate(selected_data))
        append(prob_out, np.concatenate(selected_probs))
        new_starts = written_hits + np.cumsum(new_lengths, dtype=np.int64)
        append(starts_out, new_starts)
        kept += len(new_lengths)
        written_hits += int(sum(new_lengths))
        if kept and (kept % 10_000 < len(new_lengths)):
            print(
                f"{split}: scanned={scanned:,} eligible={eligible:,} "
                f"kept={kept:,} hits={written_hits:,}",
                flush=True,
            )

    precision = truth_signal_selected / max(truth_signal_selected + truth_noise_selected, 1)
    recall_in_kept = truth_signal_selected / max(truth_signal_total, 1)
    return {
        "source_events_scanned": scanned,
        "particle_eligible_events": eligible,
        "kept_events": kept,
        "kept_hits": written_hits,
        "truth_signal_selected": truth_signal_selected,
        "truth_noise_selected": truth_noise_selected,
        "selected_hit_precision": precision,
        "truth_signal_recall_within_kept_events": recall_in_kept,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-norm-source", required=True)
    parser.add_argument("--output-norm-source", required=True)
    parser.add_argument("--output-norm-group", default="signal_norm_param")
    parser.add_argument(
        "--model-type", choices=("encoder", "encoder_two_head"), default="encoder_two_head"
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--cls-layers", type=int, default=0)
    parser.add_argument("--tres-layers", type=int, default=0)
    parser.add_argument("--ff-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--particle", choices=("all", "muatm", "nuatm", "nue2"), default="all")
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--min-strings", type=int, default=2)
    parser.add_argument("--channels-per-string", type=int, default=36)
    parser.add_argument("--q-cap", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-kept-train", type=int, default=0)
    parser.add_argument("--max-kept-val", type=int, default=0)
    parser.add_argument("--max-kept-test", type=int, default=0)
    parser.add_argument("--compression", choices=("lzf", "gzip"), default="lzf")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device).eval()
    summaries = {}
    with (
        h5py.File(args.source, "r") as source,
        h5py.File(args.model_norm_source, "r") as model_norm_h5,
        h5py.File(args.output_norm_source, "r") as output_norm_h5,
        h5py.File(output_path, "w") as output,
    ):
        source_mean, source_std = norm_values(source, "norm_param")
        model_mean, model_std = norm_values(model_norm_h5, "norm_param")
        output_mean, output_std = norm_values(output_norm_h5, args.output_norm_group)
        norm_group = output.create_group("norm_param")
        norm_group.create_dataset("mean", data=output_mean)
        norm_group.create_dataset("std", data=output_std)
        for split in SPLITS:
            if split not in source:
                continue
            summaries[split] = process_split(
                source,
                output,
                split,
                model,
                device,
                source_mean,
                source_std,
                model_mean,
                model_std,
                output_mean,
                output_std,
                args,
            )
        metadata = {"arguments": vars(args), "summaries": summaries}
        output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.3f} GB)", flush=True)


if __name__ == "__main__":
    main()
