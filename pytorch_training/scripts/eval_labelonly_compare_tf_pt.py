import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Encoder

TYPE_NAMES = {0: "muatm", 1: "nuatm", 2: "nue2"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt-checkpoint", required=True)
    p.add_argument("--tf-saved-model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument(
        "--tf-train-norm-data",
        default=None,
        help=(
            "HDF5 whose norm_param/mean,std were used for TF-model training. "
            "Eval features are de-normalized with --data norm_param and re-normalized "
            "with this file's norm_param. Defaults to --data."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-subset-cut", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap selected eval batches. 120 batches at batch_size=128 is ~15k events.",
    )
    p.add_argument(
        "--max-events-per-type",
        type=int,
        default=None,
        help="Stratified cap: keep up to this many muatm/nuatm/nue2 events.",
    )
    return p.parse_args()


def event_type(ev_id):
    if b"muatm" in ev_id:
        return 0
    if b"nuatm" in ev_id:
        return 1
    return 2


def load_pt_model(path):
    model = Encoder(
        in_features=5,
        hidden_size=128,
        num_layers=5,
        dim_feedforward_size=512,
        n_heads=1,
        out_size=2,
        dropout_p=0.0,
        use_cls_token=False,
        return_only_cls_token=False,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    if torch.cuda.is_available():
        print(f"PyTorch device: {torch.cuda.get_device_name(0)}")
        return model.cuda().eval()
    print("PyTorch device: CPU")
    return model.eval()


def setup_tf():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("TensorFlow visible GPUs:", ", ".join(gpu.name for gpu in gpus))
    else:
        print("TensorFlow visible GPUs: none (CPU)")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def selected_batches(n_events, batch_size, split, val_subset_cut):
    n_batches = (n_events + batch_size - 1) // batch_size
    step = val_subset_cut if split == "val" else 1
    return range(0, n_batches, step)


def collate_events(f, split, batch_idx, batch_size, max_seq_len):
    ev_starts = f[f"{split}/ev_starts/data"]
    data_ds = f[f"{split}/data/data"]
    labels_ds = f[f"{split}/labels/data"]
    ev_ids = f[f"{split}/ev_ids/data"]

    event_start_idx = batch_idx * batch_size
    event_end_idx = min((batch_idx + 1) * batch_size, len(ev_starts) - 1)
    starts = ev_starts[event_start_idx : event_end_idx + 1]
    global_start, global_end = int(starts[0]), int(starts[-1])

    raw_x = data_ds[global_start:global_end]
    raw_y = labels_ds[global_start:global_end]
    batch_n = len(starts) - 1
    x = np.zeros((batch_n, max_seq_len, raw_x.shape[-1]), dtype=np.float32)
    y = np.zeros((batch_n, max_seq_len), dtype=np.int32)
    mask = np.zeros((batch_n, max_seq_len), dtype=bool)
    event_sizes = np.zeros(batch_n, dtype=np.int64)

    for i in range(batch_n):
        s = int(starts[i] - global_start)
        e = int(starts[i + 1] - global_start)
        length = min(e - s, max_seq_len)
        x[i, :length] = raw_x[s : s + length]
        y[i, :length] = raw_y[s : s + length]
        mask[i, :length] = True
        event_sizes[i] = length

    types = np.array(
        [event_type(eid) for eid in ev_ids[event_start_idx:event_end_idx]], dtype=np.int32
    )
    event_indices = np.arange(event_start_idx, event_end_idx, dtype=np.int64)
    return x, y, mask, event_sizes, types, event_indices


def stratified_event_selection(f, split, batch_size, val_subset_cut, max_events_per_type):
    ev_ids = f[f"{split}/ev_ids/data"][:]
    n_events = len(ev_ids)
    counts = {k: 0 for k in TYPE_NAMES}
    keep = set()
    keep_batches = []
    for bi in selected_batches(n_events, batch_size, split, val_subset_cut):
        batch_keep = False
        start = bi * batch_size
        end = min((bi + 1) * batch_size, n_events)
        for event_idx in range(start, end):
            typ = event_type(ev_ids[event_idx])
            if counts[typ] >= max_events_per_type:
                continue
            keep.add(event_idx)
            counts[typ] += 1
            batch_keep = True
        if batch_keep:
            keep_batches.append(bi)
        if all(counts[t] >= max_events_per_type for t in TYPE_NAMES):
            break
    print("Stratified event counts:", {TYPE_NAMES[k]: v for k, v in counts.items()})
    return keep_batches, keep


def load_norm_params(path):
    with h5py.File(path, "r") as f:
        return (
            f["norm_param/mean"][:].astype(np.float32),
            f["norm_param/std"][:].astype(np.float32),
        )


def precision_at_recall(y_true, y_score, min_recall=0.9):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ok = recall >= min_recall
    if not np.any(ok):
        return np.nan, np.nan
    idx = np.argmax(precision[ok])
    ok_idx = np.flatnonzero(ok)[idx]
    th = thresholds[min(ok_idx, len(thresholds) - 1)] if len(thresholds) else np.nan
    return float(precision[ok_idx]), float(th)


def metrics_for(y_true, y_score, threshold):
    pred = y_score >= threshold
    tp = np.sum(pred & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))
    fn = np.sum((~pred) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    p90, th90 = precision_at_recall(y_true, y_score)
    return {
        "auc": float(auc),
        "ap": float(ap),
        "p_at_r90": p90,
        "threshold_at_r90": th90,
        "precision_at_0p5": float(precision),
        "recall_at_0p5": float(recall),
    }


def macro_pr(y_true, y_score, event_ids, event_types, target_type, threshold, min_signal_hits=1):
    n_events = len(event_types)
    pred = y_score >= threshold
    truth = y_true == 1

    tp = np.bincount(event_ids, weights=(pred & truth).astype(np.float32), minlength=n_events)
    fp = np.bincount(event_ids, weights=(pred & ~truth).astype(np.float32), minlength=n_events)
    fn = np.bincount(event_ids, weights=(~pred & truth).astype(np.float32), minlength=n_events)
    n_sig = np.bincount(event_ids, weights=truth.astype(np.float32), minlength=n_events)

    keep = (event_types == target_type) & (n_sig >= min_signal_hits)
    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = tp[keep] / (tp[keep] + fp[keep])
        recalls = tp[keep] / (tp[keep] + fn[keep])
    precisions[(tp[keep] + fp[keep]) == 0] = np.nan
    recalls[(tp[keep] + fn[keep]) == 0] = np.nan
    return float(np.nanmean(precisions)), float(np.nanmean(recalls)), int(np.sum(keep))


def summarize_model(name, y_true, y_score, hit_types, hit_event_ids, event_types, threshold):
    rows = []
    overall = metrics_for(y_true, y_score, threshold)
    rows.append(
        {
            "model": name,
            "type": "overall",
            "n_hits": len(y_true),
            "n_events": len(event_types),
            **overall,
        }
    )
    for typ, typ_name in TYPE_NAMES.items():
        m = hit_types == typ
        met = metrics_for(y_true[m], y_score[m], threshold)
        macro_p, macro_r, macro_n = macro_pr(
            y_true, y_score, hit_event_ids, event_types, typ, threshold
        )
        rows.append(
            {
                "model": name,
                "type": typ_name,
                "n_hits": int(m.sum()),
                "n_events": int(np.sum(event_types == typ)),
                **met,
                "macro_precision_at_0p5": macro_p,
                "macro_recall_at_0p5": macro_r,
                "macro_events": macro_n,
            }
        )
    return rows


def plot_pr_curves(out_dir, all_scores, y_true, hit_types):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, score in all_scores.items():
        p, r, _ = precision_recall_curve(y_true, score)
        ax.plot(r, p, label=f"{name} overall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Label-only truth: signal = label != 0")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves_overall.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for typ, ax in zip(TYPE_NAMES, axes):
        m = hit_types == typ
        if m.sum() == 0 or len(np.unique(y_true[m])) < 2:
            ax.text(0.5, 0.5, "no positive/negative mix", ha="center", va="center")
        else:
            for name, score in all_scores.items():
                p, r, _ = precision_recall_curve(y_true[m], score[m])
                ax.plot(r, p, label=name)
        ax.set_title(TYPE_NAMES[typ])
        ax.set_xlabel("Recall")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Precision")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves_by_event_type.png", dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_tf()
    pt_model = load_pt_model(args.pt_checkpoint)
    tf_model = tf.saved_model.load(args.tf_saved_model).signatures["serving_default"]
    tf_norm_path = args.tf_train_norm_data or args.data
    tf_mean, tf_std = load_norm_params(tf_norm_path)
    print(f"TF normalization params from: {tf_norm_path}")

    all_true, all_pt, all_tf = [], [], []
    all_hit_types, all_hit_event_ids, all_event_types = [], [], []
    event_offset = 0

    with h5py.File(args.data, "r") as f:
        eval_mean = f["norm_param/mean"][:].astype(np.float32)
        eval_std = f["norm_param/std"][:].astype(np.float32)
        q_up_norm = (100.0 - tf_mean[0]) / tf_std[0]
        n_events = len(f[f"{args.split}/ev_starts/data"]) - 1
        selected_event_indices = None
        if args.max_events_per_type is not None:
            batch_indices, selected_event_indices = stratified_event_selection(
                f, args.split, args.batch_size, args.val_subset_cut, args.max_events_per_type
            )
        else:
            batch_indices = list(
                selected_batches(n_events, args.batch_size, args.split, args.val_subset_cut)
            )
        if args.max_batches is not None and selected_event_indices is None:
            batch_indices = batch_indices[: args.max_batches]
        print(
            f"Evaluating {len(batch_indices)} batches "
            f"(~{len(batch_indices) * args.batch_size:,} events before final short batch)."
        )
        for bi in tqdm(batch_indices, desc="Evaluating PT + TF", unit="batch"):
            x, labels, mask, event_sizes, event_types, event_indices = collate_events(
                f, args.split, bi, args.batch_size, args.max_seq_len
            )
            if selected_event_indices is not None:
                keep_rows = np.array(
                    [idx in selected_event_indices for idx in event_indices], dtype=bool
                )
                if not np.any(keep_rows):
                    continue
                x = x[keep_rows]
                labels = labels[keep_rows]
                mask = mask[keep_rows]
                event_sizes = event_sizes[keep_rows]
                event_types = event_types[keep_rows]
            valid = mask.reshape(-1)
            truth = (labels.reshape(-1)[valid] != 0).astype(np.int32)

            with torch.no_grad():
                xt = torch.from_numpy(x)
                mt = torch.from_numpy(mask)
                if torch.cuda.is_available():
                    xt = xt.cuda(non_blocking=True)
                    mt = mt.cuda(non_blocking=True)
                logits = pt_model(xt, mt).detach().cpu().numpy()
                pt_score = 1.0 / (1.0 + np.exp(-logits.reshape(-1, logits.shape[-1])[valid, 1]))

            x_tf = (x * eval_std + eval_mean - tf_mean) / tf_std
            x_tf[:, :, 0] = np.minimum(x_tf[:, :, 0], q_up_norm)
            x_tf = np.concatenate([x_tf, mask[..., None].astype(np.float32)], axis=-1)
            tf_out = tf_model(input_1=tf.constant(x_tf, dtype=tf.float32))["output_1"].numpy()
            tf_score = tf_out.reshape(-1, tf_out.shape[-1])[valid, 0]

            hit_event_ids = np.repeat(
                np.arange(event_offset, event_offset + len(event_sizes)), event_sizes
            )
            hit_types = np.repeat(event_types, event_sizes)

            all_true.append(truth)
            all_pt.append(pt_score)
            all_tf.append(tf_score)
            all_hit_event_ids.append(hit_event_ids)
            all_hit_types.append(hit_types)
            all_event_types.append(event_types)
            event_offset += len(event_sizes)

    y_true = np.concatenate(all_true)
    pt_score = np.concatenate(all_pt)
    tf_score = np.concatenate(all_tf)
    hit_types = np.concatenate(all_hit_types)
    hit_event_ids = np.concatenate(all_hit_event_ids)
    event_types = np.concatenate(all_event_types)

    print(f"Computing metrics for {len(y_true):,} hits and {len(event_types):,} events...")
    rows = []
    rows.extend(
        summarize_model(
            "pytorch_labelonly",
            y_true,
            pt_score,
            hit_types,
            hit_event_ids,
            event_types,
            args.threshold,
        )
    )
    rows.extend(
        summarize_model(
            "tf_encoder_remask_63_re1-1",
            y_true,
            tf_score,
            hit_types,
            hit_event_ids,
            event_types,
            args.threshold,
        )
    )

    fieldnames = sorted({k for row in rows for k in row})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_pr_curves(
        out_dir,
        {"PyTorch label-only": pt_score, "TF encoder_remask_63_re1-1": tf_score},
        y_true,
        hit_types,
    )

    for row in rows:
        print(row)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
