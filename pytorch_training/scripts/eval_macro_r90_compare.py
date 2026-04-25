import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_labelonly_compare_tf_pt import (  # noqa: E402
    TYPE_NAMES,
    collate_events,
    load_norm_params,
    load_pt_model,
    selected_batches,
    setup_tf,
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


def macro_at_threshold(y_true, y_score, event_ids, event_types, threshold, target_type=None):
    pred = y_score >= threshold
    truth = y_true == 1
    n_events = len(event_types)
    tp = np.bincount(event_ids, weights=(pred & truth).astype(np.float32), minlength=n_events)
    fp = np.bincount(event_ids, weights=(pred & ~truth).astype(np.float32), minlength=n_events)
    fn = np.bincount(event_ids, weights=(~pred & truth).astype(np.float32), minlength=n_events)
    n_sig = np.bincount(event_ids, weights=truth.astype(np.float32), minlength=n_events)
    keep = n_sig >= 1
    if target_type is not None:
        keep &= event_types == target_type
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp[keep] / (tp[keep] + fp[keep])
        recall = tp[keep] / (tp[keep] + fn[keep])
    precision[(tp[keep] + fp[keep]) == 0] = np.nan
    return float(np.nanmean(precision)), float(np.nanmean(recall)), int(np.sum(keep))


def macro_precision_at_recall(
    y_true, y_score, event_ids, event_types, min_recall=0.9, target_type=None
):
    n_events = len(event_types)
    truth = y_true == 1
    n_sig = np.bincount(event_ids, weights=truth.astype(np.float32), minlength=n_events)
    keep = n_sig >= 1
    if target_type is not None:
        keep &= event_types == target_type
    keep_count = int(np.sum(keep))
    if keep_count == 0:
        return np.nan, np.nan, np.nan, 0

    tp = np.zeros(n_events, dtype=np.float32)
    fp = np.zeros(n_events, dtype=np.float32)
    event_precision = np.full(n_events, np.nan, dtype=np.float32)
    event_recall = np.zeros(n_events, dtype=np.float32)
    sum_precision = 0.0
    precision_count = 0
    sum_recall = 0.0

    order = np.argsort(y_score)[::-1]
    best_precision = np.nan
    best_recall = np.nan
    best_threshold = np.nan
    pos = 0
    while pos < len(order):
        threshold = y_score[order[pos]]
        end = pos + 1
        while end < len(order) and y_score[order[end]] == threshold:
            end += 1

        changed_events = np.unique(event_ids[order[pos:end]])
        for ev in changed_events:
            if keep[ev]:
                if not np.isnan(event_precision[ev]):
                    sum_precision -= float(event_precision[ev])
                    precision_count -= 1
                sum_recall -= float(event_recall[ev])

        group_events = event_ids[order[pos:end]]
        group_truth = truth[order[pos:end]]
        tp += np.bincount(group_events, weights=group_truth.astype(np.float32), minlength=n_events)
        fp += np.bincount(
            group_events, weights=(~group_truth).astype(np.float32), minlength=n_events
        )

        for ev in changed_events:
            if not keep[ev]:
                continue
            pred_count = tp[ev] + fp[ev]
            event_recall[ev] = tp[ev] / n_sig[ev]
            if pred_count > 0:
                event_precision[ev] = tp[ev] / pred_count
                sum_precision += float(event_precision[ev])
                precision_count += 1
            else:
                event_precision[ev] = np.nan
            sum_recall += float(event_recall[ev])

        macro_recall = sum_recall / keep_count
        if macro_recall >= min_recall and precision_count > 0:
            macro_precision = sum_precision / precision_count
            if np.isnan(best_precision) or macro_precision > best_precision:
                best_precision = macro_precision
                best_recall = macro_recall
                best_threshold = threshold
        pos = end

    return float(best_precision), float(best_recall), float(best_threshold), keep_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-checkpoint", required=True)
    parser.add_argument("--tf-saved-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tf-train-norm-data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    setup_tf()
    pt_model = load_pt_model(args.pt_checkpoint)
    tf_model = tf.saved_model.load(args.tf_saved_model).signatures["serving_default"]
    tf_mean, tf_std = load_norm_params(args.tf_train_norm_data)

    all_true, all_pt, all_tf = [], [], []
    all_event_ids, all_types, all_hit_types = [], [], []
    event_offset = 0

    with h5py.File(args.data, "r") as f:
        eval_mean = f["norm_param/mean"][:].astype(np.float32)
        eval_std = f["norm_param/std"][:].astype(np.float32)
        q_up_norm = (100.0 - tf_mean[0]) / tf_std[0]
        n_events = len(f[f"{args.split}/ev_starts/data"]) - 1
        batch_indices = list(selected_batches(n_events, args.batch_size, args.split, 1))
        for bi in tqdm(batch_indices, desc="Evaluating", unit="batch"):
            x, labels, mask, event_sizes, event_types, _ = collate_events(
                f, args.split, bi, args.batch_size, args.max_seq_len
            )
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

            all_true.append(truth)
            all_pt.append(pt_score)
            all_tf.append(tf_score)
            all_event_ids.append(
                np.repeat(np.arange(event_offset, event_offset + len(event_sizes)), event_sizes)
            )
            all_hit_types.append(np.repeat(event_types, event_sizes))
            all_types.append(event_types)
            event_offset += len(event_sizes)

    y_true = np.concatenate(all_true)
    scores = {
        "pytorch": np.concatenate(all_pt),
        "tf": np.concatenate(all_tf),
    }
    event_ids = np.concatenate(all_event_ids)
    event_types = np.concatenate(all_types)

    for name, score in scores.items():
        micro_p90, micro_th = precision_at_recall(y_true, score, min_recall=0.9)
        macro_p90, macro_recall, macro_th, macro_events = macro_precision_at_recall(
            y_true, score, event_ids, event_types, min_recall=0.9
        )
        print(f"\n{name}: micro P@R90={micro_p90:.6f}, threshold={micro_th:.6f}")
        print(
            f"{'overall':7s}: macro_P@R90_one_global_threshold={macro_p90:.6f}, "
            f"macro_recall={macro_recall:.6f}, threshold={macro_th:.6f}, events={macro_events}"
        )
        for typ in [None, 0, 1, 2]:
            label = "overall" if typ is None else TYPE_NAMES[typ]
            mp, mr, n = macro_at_threshold(y_true, score, event_ids, event_types, micro_th, typ)
            gp, gr, gn = macro_at_threshold(y_true, score, event_ids, event_types, macro_th, typ)
            print(
                f"{label:7s}: macro@microR90_th precision={mp:.6f}, recall={mr:.6f}, "
                f"events={n}; macro@global_macroR90_th precision={gp:.6f}, "
                f"recall={gr:.6f}, events={gn}"
            )


if __name__ == "__main__":
    main()
