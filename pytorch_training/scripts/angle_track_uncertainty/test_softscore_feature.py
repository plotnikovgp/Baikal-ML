#!/usr/bin/env python3
"""Smoke-check score alignment and zero-impact upgrade from five features."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import h5py
import numpy as np
import torch
from infer_rethresholded_for_audit import selected_batch
from softscore_model import upgrade_track_model
from train_mixed_universal import AdaptedTrackDataset, model_from_checkpoints


def make_dataset(
    data: Path,
    anchors: Path,
    expected_mean: np.ndarray,
    expected_std: np.ndarray,
    *,
    score_feature: bool,
) -> AdaptedTrackDataset:
    return AdaptedTrackDataset(
        str(data),
        "val",
        16,
        128,
        False,
        16,
        False,
        anchor_path=str(anchors),
        keep=None,
        expected_mean=expected_mean,
        expected_std=expected_std,
        score_feature=score_feature,
    )


def check_batch(plain: AdaptedTrackDataset, scored: AdaptedTrackDataset, *, has_p: bool):
    old = plain[0]
    new = scored[0]
    assert old[0].shape[-1] == 5 and new[0].shape[-1] == 6
    torch.testing.assert_close(old[0], new[0][..., :5], rtol=0, atol=0)
    for left, right in zip(old[1:], new[1:]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    probability = new[0][..., 5]
    valid = new[2]
    assert torch.all(probability[~valid] == 0)
    if has_p:
        assert torch.all((probability[valid] >= 0.7) & (probability[valid] <= 1))
    else:
        assert torch.all(probability[valid] == 1)
    return old, new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-data", type=Path, required=True)
    parser.add_argument("--pred-anchors", type=Path, required=True)
    parser.add_argument("--gt-data", type=Path, required=True)
    parser.add_argument("--gt-anchors", type=Path, required=True)
    parser.add_argument("--angle-checkpoint", type=Path, required=True)
    parser.add_argument("--track-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with h5py.File(args.gt_data, "r") as h5:
        mean = np.asarray(h5["norm_param/mean"], dtype=np.float32)
        std = np.asarray(h5["norm_param/std"], dtype=np.float32)
    pred_plain = make_dataset(args.pred_data, args.pred_anchors, mean, std, score_feature=False)
    pred_scored = make_dataset(args.pred_data, args.pred_anchors, mean, std, score_feature=True)
    plain_batch, scored_batch = check_batch(pred_plain, pred_scored, has_p=True)
    with h5py.File(args.pred_data, "r") as h5:
        prepared = {
            "input_norm_mean": mean.tolist(),
            "input_norm_std": std.tolist(),
            "max_seq_len": 128,
            "center_time": False,
        }
        kept, audit_x, audit_mask = selected_batch(
            h5["val"],
            0,
            16,
            0.7,
            prepared,
            np.asarray(h5["norm_param/mean"]),
            np.asarray(h5["norm_param/std"]),
            36,
            0,
            1,
            score_feature=True,
        )
        assert len(kept) == 16
        torch.testing.assert_close(scored_batch[0], torch.from_numpy(audit_x), rtol=0, atol=1e-6)
        torch.testing.assert_close(scored_batch[2], torch.from_numpy(audit_mask), rtol=0, atol=0)
    gt_plain = make_dataset(args.gt_data, args.gt_anchors, mean, std, score_feature=False)
    gt_scored = make_dataset(args.gt_data, args.gt_anchors, mean, std, score_feature=True)
    gt_plain_batch, gt_scored_batch = check_batch(gt_plain, gt_scored, has_p=False)

    original = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, "cpu")
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    original.load_state_dict(checkpoint["model"], strict=True)
    upgraded = model_from_checkpoints(args.angle_checkpoint, args.track_checkpoint, 50.0, "cpu")
    upgraded.load_state_dict(checkpoint["model"], strict=True)
    upgraded = upgrade_track_model(upgraded)
    original.eval()
    upgraded.eval()
    with torch.inference_mode():
        old_dir, old_point = original(plain_batch[0], plain_batch[2])
        new_dir, new_point = upgraded(scored_batch[0], scored_batch[2])
    torch.testing.assert_close(old_dir, new_dir, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(old_point, new_point, rtol=1e-5, atol=1e-5)
    centered = upgrade_track_model(copy.deepcopy(original), "centered").eval()
    with torch.no_grad():
        centered.backbone.embed[0].weight[:, 10].normal_(0, 0.1)
    with torch.inference_mode():
        gt_old_dir, gt_old_point = original(gt_plain_batch[0], gt_plain_batch[2])
        gt_centered_dir, gt_centered_point = centered(gt_scored_batch[0], gt_scored_batch[2])
    torch.testing.assert_close(gt_old_dir, gt_centered_dir, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gt_old_point, gt_centered_point, rtol=1e-5, atol=1e-5)
    print(
        {
            "events": len(old_dir),
            "max_direction_diff": (old_dir - new_dir).abs().max().item(),
            "max_point_diff_m": (old_point - new_point).abs().max().item(),
        }
    )


if __name__ == "__main__":
    main()
