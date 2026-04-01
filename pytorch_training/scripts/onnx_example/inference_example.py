"""Run inference with an ONNX-exported Encoder model.

The ONNX model has fixed batch_size and sequence length (baked during export).
Inputs are automatically padded/truncated to match.  Events are processed one
at a time (default export batch_size=1).

Usage:
    python scripts/onnx_example/inference_example.py
    python scripts/onnx_example/inference_example.py --model encoder_noise_sig.onnx
    python scripts/onnx_example/inference_example.py --model encoder_noise_sig.onnx --data path/to/data.h5
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_MODEL = str(Path(__file__).resolve().parent / "encoder_noise_sig.onnx")
SIGNAL_THRESHOLD = 0.5
IN_FEATURES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Encoder inference example")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Path to .onnx model")
    parser.add_argument("-d", "--data", default=None, help="Optional HDF5 data file")
    parser.add_argument("--n-events", type=int, default=4, help="Number of events to process")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIGNAL_THRESHOLD,
        help="Signal classification threshold",
    )
    return parser.parse_args()


def get_model_shapes(session: ort.InferenceSession) -> tuple[int, int]:
    """Read the fixed (batch_size, seq_len) from the ONNX model input shape."""
    x_shape = session.get_inputs()[0].shape
    return int(x_shape[0]), int(x_shape[1])


def load_events_from_h5(data_path: str, max_events: int = 4):
    """Load individual events from an HDF5 dataset (variable-length hits)."""
    import h5py

    with h5py.File(data_path, "r") as f:
        split = "val" if "val" in f else list(f.keys())[0]
        ev_starts = f[f"{split}/ev_starts/data"][:]
        raw_data = f[f"{split}/data/data"][:]

    n_events = min(max_events, len(ev_starts) - 1)
    events = []
    for i in range(n_events):
        start, end = int(ev_starts[i]), int(ev_starts[i + 1])
        events.append(raw_data[start:end, :IN_FEATURES].astype(np.float32))
    return events


def generate_random_events(n_events: int, hits_per_event: int = 64):
    """Generate random events for demonstration."""
    np.random.seed(0)
    return [
        np.random.randn(hits_per_event, IN_FEATURES).astype(np.float32) for _ in range(n_events)
    ]


def pad_event(hits: np.ndarray, batch_size: int, seq_len: int):
    """Pad/truncate a single event's hits into the model's fixed input shape."""
    x = np.zeros((batch_size, seq_len, IN_FEATURES), dtype=np.float32)
    mask = np.zeros((batch_size, seq_len), dtype=bool)
    n_hits = min(len(hits), seq_len)
    x[0, :n_hits] = hits[:n_hits]
    mask[0, :n_hits] = True
    return x, mask, n_hits


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    args = parse_args()

    print(f"Loading ONNX model: {args.model}")
    session = ort.InferenceSession(args.model)

    batch_size, seq_len = get_model_shapes(session)
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    print(f"Inputs:  {[(m.name, m.shape, m.type) for m in input_meta]}")
    print(f"Outputs: {[(m.name, m.shape, m.type) for m in output_meta]}")
    print(f"Fixed shape: batch_size={batch_size}, seq_len={seq_len}\n")

    if args.data:
        print(f"Loading data from: {args.data}")
        events = load_events_from_h5(args.data, max_events=args.n_events)
    else:
        print(f"Using {args.n_events} random events")
        events = generate_random_events(args.n_events)

    print(f"Processing {len(events)} events...\n")
    print(f"Threshold: {args.threshold}\n")

    for i, hits in enumerate(events):
        x, mask, n_hits = pad_event(hits, batch_size, seq_len)
        logits = session.run(None, {"x": x, "mask": mask})[0]
        probs = sigmoid(logits[0, :n_hits, 1])
        n_signal = int((probs > args.threshold).sum())

        print(f"Event {i}:")
        print(f"  Hits: {n_hits}")
        print(f"  Signal hits (P > {args.threshold}): {n_signal} / {n_hits}")
        print(f"  Mean P(signal): {probs.mean():.4f}")
        print(f"  Min  P(signal): {probs.min():.4f}")
        print(f"  Max  P(signal): {probs.max():.4f}")
        if n_hits <= 10:
            print(f"  Per-hit P(signal): {np.array2string(probs, precision=4)}")
        print()


if __name__ == "__main__":
    main()
