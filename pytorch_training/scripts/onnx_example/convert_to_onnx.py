"""Convert a PyTorch Encoder checkpoint to ONNX format.

This script is fully self-contained — the Encoder architecture is defined
inline so there is no dependency on the project's `models` package.

Both batch size and sequence length are fixed in the exported ONNX graph
because PyTorch's nn.MultiheadAttention bakes internal reshape dimensions
during tracing.  Defaults: batch_size=1, seq_len=256.

Usage:
    python convert_to_onnx.py
    python convert_to_onnx.py --checkpoint path/to/best.ckpt
    python convert_to_onnx.py --seq-len 128 --batch-size 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = str(
    SCRIPT_DIR.parents[1]
    / "checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff512_hs512_bs128/best_2020.ckpt"
)
DEFAULT_OUTPUT = str(SCRIPT_DIR / "encoder_noise_sig.onnx")

MODEL_PARAMS = {
    "in_features": 5,
    "hidden_size": 512,
    "num_layers": 5,
    "dim_feedforward_size": 512,
    "n_heads": 1,
    "out_size": 2,
    "dropout_p": 0.0,
}


# ---------------------------------------------------------------------------
# Encoder architecture (self-contained copy)
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int,
        dim_feedforward_size: int,
        n_heads: int,
        out_size: int,
        dropout_p: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.first_layer = nn.Linear(in_features, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, n_heads, dim_feedforward_size, dropout_p, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        mask = (~mask).float()
        x = self.first_layer(x)
        x = self.enc(x, src_key_padding_mask=mask)
        return self.head(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Encoder to ONNX")
    parser.add_argument("-c", "--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to .ckpt file")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Fixed sequence length (default: 256)"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size (default: 1)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Export / verify
# ---------------------------------------------------------------------------
def load_pytorch_model(checkpoint_path: str) -> nn.Module:
    model = Encoder(**MODEL_PARAMS)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    opset_version: int,
    batch_size: int,
    seq_len: int,
) -> None:
    dummy_x = torch.randn(batch_size, seq_len, MODEL_PARAMS["in_features"])
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        output_path,
        input_names=["x", "mask"],
        output_names=["logits"],
        opset_version=opset_version,
    )
    print(f"ONNX model exported to {output_path}")
    print(f"  Fixed shape: batch_size={batch_size}, seq_len={seq_len}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX validation check: PASSED")


def verify_outputs(model: nn.Module, onnx_path: str, batch_size: int, seq_len: int) -> None:
    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, MODEL_PARAMS["in_features"]).astype(np.float32)
    mask_np = np.ones((batch_size, seq_len), dtype=bool)

    with torch.no_grad():
        pt_out = model(torch.tensor(x_np), torch.tensor(mask_np)).numpy()

    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {"x": x_np, "mask": mask_np})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    mean_diff = float(np.abs(pt_out - ort_out).mean())

    print("\nVerification (PyTorch vs ONNX Runtime):")
    print(f"  Input shape:     ({batch_size}, {seq_len}, {MODEL_PARAMS['in_features']})")
    print(f"  Output shape:    {pt_out.shape}")
    print(f"  Max difference:  {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")

    if max_diff < 1e-5:
        print("  Result: PASSED — outputs match closely")
    elif max_diff < 1e-3:
        print("  Result: OK — minor numerical differences")
    else:
        print("  Result: WARNING — outputs differ significantly")


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_pytorch_model(args.checkpoint)

    print(f"Exporting to ONNX (opset {args.opset})...")
    export_to_onnx(model, args.output, args.opset, args.batch_size, args.seq_len)

    verify_outputs(model, args.output, args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
