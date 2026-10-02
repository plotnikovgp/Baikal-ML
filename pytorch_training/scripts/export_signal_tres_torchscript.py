#!/usr/bin/env python3
"""Export the best joint hit signal/t_res checkpoint as a raw-input TorchScript graph.

The exported module accepts raw [Q, time, x, y, z] values and a boolean validity
mask. Input normalization and t_res de-normalization are part of the graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
from torch import Tensor, nn

from models.encoder import EncoderTwoHead


class RawHitSignalTres(nn.Module):
    """Production wrapper with all preprocessing/postprocessing in TorchScript."""

    def __init__(
        self,
        model: nn.Module,
        input_mean: Tensor,
        input_std: Tensor,
        tres_mean_ns: float,
        tres_std_ns: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", input_mean.reshape(1, 1, 5))
        self.register_buffer("input_std", input_std.reshape(1, 1, 5))
        self.register_buffer("tres_mean_ns", torch.tensor(tres_mean_ns))
        self.register_buffer("tres_std_ns", torch.tensor(tres_std_ns))

    def forward(self, raw_hits: Tensor, valid_mask: Tensor) -> Tensor:
        normalized_hits = (raw_hits - self.input_mean) / self.input_std
        raw_output = self.model(normalized_hits, valid_mask)
        signal_probability = torch.sigmoid(raw_output[..., 1])
        abs_tres_ns = raw_output[..., 2] * self.tres_std_ns + self.tres_mean_ns
        result = torch.stack(
            (
                raw_output[..., 0],
                raw_output[..., 1],
                signal_probability,
                abs_tres_ns,
            ),
            dim=-1,
        )
        return result * valid_mask.unsqueeze(-1).to(dtype=result.dtype)


def load_model(checkpoint: Path) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)

    model = EncoderTwoHead(
        in_features=5,
        hidden_size=128,
        num_shared_layers=5,
        num_cls_layers=0,
        num_tres_layers=0,
        dim_feedforward_size=512,
        n_heads=1,
        cls_out_size=2,
        tres_out_size=1,
        dropout_p=0.0,
    )
    # This checkpoint predates the current config-driven construction of the
    # regression head and stores a 128 -> 256 -> 1 MLP with GELU.
    model.tres_head = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 1),
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.normalization_h5, "r") as h5:
        input_mean = torch.as_tensor(h5["norm_param/mean"][:], dtype=torch.float32)
        input_std = torch.as_tensor(h5["norm_param/std"][:], dtype=torch.float32)

    tres_mean_ns = 7.8
    tres_std_ns = 26.2
    eager = RawHitSignalTres(
        load_model(args.checkpoint),
        input_mean,
        input_std,
        tres_mean_ns,
        tres_std_ns,
    ).eval()

    torch.manual_seed(42)
    example_hits = input_mean.reshape(1, 1, 5) + torch.randn(2, 32, 5) * input_std.reshape(1, 1, 5)
    example_mask = torch.ones(2, 32, dtype=torch.bool)
    example_mask[0, 23:] = False

    with torch.inference_mode():
        traced = torch.jit.trace(
            eager,
            (example_hits, example_mask),
            check_trace=True,
            strict=True,
        )
        traced = torch.jit.freeze(traced.eval())

    model_path = args.output_dir / "noise_sig_tres_abs_merged_raw_input_cpu.pt"
    torch.jit.save(traced, model_path)

    # Verify serialization and dynamic sequence length against eager execution.
    loaded = torch.jit.load(model_path, map_location="cpu").eval()
    test_hits = input_mean.reshape(1, 1, 5) + torch.randn(3, 73, 5) * input_std.reshape(1, 1, 5)
    test_mask = torch.ones(3, 73, dtype=torch.bool)
    test_mask[0, 61:] = False
    test_mask[1, 47:] = False
    with torch.inference_mode():
        expected = eager(test_hits, test_mask)
        actual = loaded(test_hits, test_mask)
    max_abs_error = float((expected - actual).abs().max())
    # The frozen CPU Transformer may choose a fused attention kernel whose
    # float32 rounding differs slightly from eager mode after t_res is scaled
    # back to nanoseconds. This remains far below any meaningful physics scale.
    if max_abs_error > 5e-4:
        raise RuntimeError(f"TorchScript verification failed: max abs error={max_abs_error}")

    metadata = {
        "format": "TorchScript",
        "torch_version": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_file": model_path.name,
        "target_definition": {
            "signal": "label != 0 (track and cascade MC signal hits)",
            "t_res": "absolute residual time; regression loss was applied only to MC signal hits",
        },
        "inputs": {
            "raw_hits": {
                "dtype": "float32",
                "shape": "[batch, hits, 5]",
                "columns": ["Q", "time_ns", "x_m", "y_m", "z_m"],
                "normalization": "embedded: (raw_hits - input_mean) / input_std",
            },
            "valid_mask": {
                "dtype": "bool",
                "shape": "[batch, hits]",
                "semantics": "true for a real hit, false for padding",
            },
        },
        "outputs": {
            "dtype": "float32",
            "shape": "[batch, hits, 4]",
            "columns": [
                "noise_logit",
                "signal_logit",
                "signal_probability",
                "abs_t_res_ns",
            ],
            "padding": "all four output columns are zero where valid_mask is false",
        },
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "tres_denormalization": {
            "formula": "abs_t_res_ns = normalized_prediction * std + mean",
            "mean_ns": tres_mean_ns,
            "std_ns": tres_std_ns,
        },
        "validation_reference": {
            "checkpoint_validation": {
                "auc": 0.9990654854,
                "threshold_at_90pct_recall": 0.7342026830,
                "precision_at_90pct_recall": 0.9883934451,
                "recall": 0.8999998330,
                "t_res_mae_ns": 3.4064316750,
            },
            "saved_report_at_threshold_0p5": {
                "precision": 0.9605299188,
                "recall": 0.9652931178,
                "f1": 0.9629056278,
            },
            "t_res_mae_ns": 4.8331,
            "t_res_median_abs_error_ns": 1.8370,
            "t_res_q68_abs_error_ns": 3.0120,
        },
        "verification": {
            "dynamic_test_shape": [3, 73, 5],
            "max_abs_error_vs_eager": max_abs_error,
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(model_path)
    print(metadata_path)
    print(f"max_abs_error_vs_eager={max_abs_error:.9g}")


if __name__ == "__main__":
    main()
