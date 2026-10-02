#!/usr/bin/env python3
"""Export and validate the raw-input joint signal/t_res model as ONNX."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.export_signal_tres_torchscript import RawHitSignalTres, load_model


class OnnxTransformerLayer(nn.Module):
    """Exact one-head, post-norm TransformerEncoderLayer without static reshapes."""

    def __init__(self, source: nn.TransformerEncoderLayer) -> None:
        super().__init__()
        attention = source.self_attn
        self.in_proj_weight = nn.Parameter(attention.in_proj_weight.detach().clone())
        self.in_proj_bias = nn.Parameter(attention.in_proj_bias.detach().clone())
        self.out_proj = attention.out_proj
        self.linear1 = source.linear1
        self.linear2 = source.linear2
        self.norm1 = source.norm1
        self.norm2 = source.norm2
        self.scale = 1.0 / math.sqrt(attention.embed_dim)

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))

        # Reproduce the checkpoint implementation exactly. EncoderTwoHead
        # converts (~valid_mask) to float before passing it as key padding mask;
        # PyTorch therefore adds 1.0 at padded key positions.
        additive_mask = (~valid_mask).to(dtype=scores.dtype).unsqueeze(1)
        scores = scores + additive_mask
        attention = torch.softmax(scores, dim=-1)
        attention_output = self.out_proj(torch.matmul(attention, v))
        x = self.norm1(x + attention_output)
        feed_forward = self.linear2(F.relu(self.linear1(x)))
        return self.norm2(x + feed_forward)


class OnnxTwoHead(nn.Module):
    """ONNX-friendly equivalent of this checkpoint's EncoderTwoHead."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.first_layer = source.first_layer
        self.layers = nn.ModuleList(OnnxTransformerLayer(layer) for layer in source.shared.layers)
        self.cls_head = source.cls_head
        self.tres_head = source.tres_head

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x, valid_mask)
        return torch.cat((self.cls_head(x), self.tres_head(x)), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.normalization_h5, "r") as h5:
        mean = torch.as_tensor(h5["norm_param/mean"][:], dtype=torch.float32)
        std = torch.as_tensor(h5["norm_param/std"][:], dtype=torch.float32)

    source_model = load_model(args.checkpoint)
    source_wrapper = RawHitSignalTres(source_model, mean, std, 7.8, 26.2).eval()
    model = RawHitSignalTres(OnnxTwoHead(source_model), mean, std, 7.8, 26.2).eval()
    torch.manual_seed(42)
    example_hits = mean.reshape(1, 1, 5) + torch.randn(2, 32, 5) * std.reshape(1, 1, 5)
    example_mask = torch.ones(2, 32, dtype=torch.bool)
    example_mask[0, 23:] = False

    with torch.inference_mode():
        conversion_error = float(
            (source_wrapper(example_hits, example_mask) - model(example_hits, example_mask))
            .abs()
            .max()
        )
    if conversion_error > 5e-4:
        raise RuntimeError(f"ONNX-friendly attention differs from checkpoint: {conversion_error}")

    onnx_path = args.output_dir / "noise_sig_tres_abs_merged_raw_input_cpu.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example_hits, example_mask),
            onnx_path,
            input_names=["raw_hits", "valid_mask"],
            output_names=["hit_predictions"],
            dynamic_axes={
                "raw_hits": {0: "batch", 1: "hits"},
                "valid_mask": {0: "batch", 1: "hits"},
                "hit_predictions": {0: "batch", 1: "hits"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph, full_check=True)
    node_types = {node.op_type for node in graph.graph.node}

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    shape_errors: dict[str, float] = {}
    rewrite_errors: dict[str, float] = {}
    source_errors: dict[str, float] = {}
    for batch, hits in ((1, 48), (3, 73), (2, 11)):
        raw_hits = mean.reshape(1, 1, 5) + torch.randn(batch, hits, 5) * std.reshape(1, 1, 5)
        valid_mask = torch.ones(batch, hits, dtype=torch.bool)
        if hits > 4:
            valid_mask[0, hits - 3 :] = False
        with torch.inference_mode():
            source_expected = source_wrapper(raw_hits, valid_mask).numpy()
            expected = model(raw_hits, valid_mask).numpy()
        actual = session.run(
            ["hit_predictions"],
            {"raw_hits": raw_hits.numpy(), "valid_mask": valid_mask.numpy()},
        )[0]
        shape_errors[f"{batch}x{hits}"] = float(np.max(np.abs(expected - actual)))
        rewrite_errors[f"{batch}x{hits}"] = float(np.max(np.abs(source_expected - expected)))
        source_errors[f"{batch}x{hits}"] = float(np.max(np.abs(source_expected - actual)))

    if max(source_errors.values()) > 2e-3:
        raise RuntimeError(f"ONNX validation failed vs checkpoint: {source_errors}")

    bench_hits = mean.reshape(1, 1, 5) + torch.randn(1, 48, 5) * std.reshape(1, 1, 5)
    bench_mask = torch.ones(1, 48, dtype=torch.bool)
    feeds = {"raw_hits": bench_hits.numpy(), "valid_mask": bench_mask.numpy()}
    for _ in range(10):
        session.run(["hit_predictions"], feeds)
    started = time.perf_counter()
    for _ in range(100):
        session.run(["hit_predictions"], feeds)
    latency_ms = (time.perf_counter() - started) * 10.0

    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["onnx"] = {
        "model_file": onnx_path.name,
        "opset": args.opset,
        "inputs": ["raw_hits", "valid_mask"],
        "output": "hit_predictions",
        "dynamic_axes": ["batch", "hits"],
        "normalization_nodes_present": "Sub" in node_types and "Div" in node_types,
        "onnx_checker": "passed",
        "onnxruntime_provider": "CPUExecutionProvider",
        "max_abs_error_vs_pytorch_by_shape": shape_errors,
        "max_abs_error_attention_rewrite_by_shape": rewrite_errors,
        "max_abs_error_vs_source_checkpoint_by_shape": source_errors,
        "cpu_latency_1thread_48hits_ms": latency_ms,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(onnx_path)
    print(f"max_abs_error_attention_rewrite={max(rewrite_errors.values()):.9g}")
    print(f"max_abs_error_vs_pytorch={max(shape_errors.values()):.9g}")
    print(f"max_abs_error_vs_source_checkpoint={max(source_errors.values()):.9g}")
    print(f"cpu_latency_1thread_48hits_ms={latency_ms:.3f}")
    print(f"normalization_nodes_present={metadata['onnx']['normalization_nodes_present']}")


if __name__ == "__main__":
    main()
