# PyTorch Training for Baikal-GVD

Neural network training framework for Baikal-GVD MC data using PyTorch and PyTorch Geometric.

## Supported Training Types

- **noise_sig** - Noise-signal hits classification
- **track_cascade** - Track-cascade hits classification  
- **angle** - Angle reconstruction
- **energy** - Energy reconstruction
- **tres** - Time residual prediction
- **direction** - Direction reconstruction

All types support domain adaptation variants (`_da` suffix).

## Installation

```bash
pip install -r requirements.txt

# For graph neural networks (optional)
pip install torch-geometric
```

## Quick Start

### Using Hydra Configuration

```bash
# Run with an experiment config
python train.py +experiment=noise_sig_2020

# Run with domain adaptation
python train.py +experiment=noise_sig_da

# Override parameters from command line
python train.py +experiment=noise_sig_2020 training.lr=1e-4 training.batch_size=128

# Run validation only
python train.py +experiment=noise_sig_2020 val_mode=true from_checkpoint=/path/to/checkpoint.ckpt
```

### Configuration Structure

```
conf/
├── config.yaml          # Default configuration
├── train_type/          # Training type configs (noise_sig, angle, etc.)
├── model/               # Model architecture configs (encoder, lstm, etc.)
├── data/                # Data loading configs (single, multi dataset)
└── experiment/          # Full experiment configs
```

### Creating a New Experiment

Create a new file in `conf/experiment/`:

```yaml
# @package _global_
defaults:
  - override /train_type: noise_sig
  - override /model: encoder
  - override /data: single

exp_project: my_project
exp_name: my_experiment

data:
  path: /path/to/data.h5
  val_subset_cut: 3

model:
  params:
    hidden_size: 512
    num_layers: 5

training:
  lr: 1.0e-3
  batch_size: 256
```

## Multi-Dataset Training

Train on multiple datasets with different sampling probabilities:

```yaml
# @package _global_
defaults:
  - override /train_type: noise_sig_da
  - override /model: encoder
  - override /data: multi

data:
  datasets:
    - name: mc_data
      path_to_data: /path/to/mc.h5
      val_subset_cut: 3
    - name: exp_data
      path_to_data: /path/to/exp.h5
      val_subset_cut: 3
      preprocessor: no_labels
      DatasetType: no_labels
  weights: [0.5, 0.5]

train_type:
  domain_adaptation_loss_k: 0.1
  label_dataset_name: mc_data
```

### Features

- Train on multiple datasets with controlled sampling rates
- Individual validation metrics per dataset
- Save best models for each dataset (`save_best_per_dataset: true`)
- Custom prefilters and parameters per dataset

### Performance Optimizations

```yaml
data:
  num_workers: 4
  prefetch_factor: 3
  persistent_workers: true
  pin_memory: true
  cache_datasets: true
```

## Utility Scripts

### Generate Signal Predictions

Save noise-signal predictions to H5 file:

```bash
python scripts/noise_sig_nn_data_utils/predict_to_h5.py \
  --checkpoint /path/to/checkpoint.ckpt \
  --config /path/to/config.yaml \
  --data /path/to/data.h5 \
  --version 1 \
  --batch_size 512 \
  --output-path /path/to/output_predictions.h5
```

### Filter Events by Signal

Filter H5 file to keep only events with sufficient signal hits/strings:

```bash
python scripts/noise_sig_nn_data_utils/filter_by_signal.py \
  --original /path/to/original.h5 \
  --predictions /path/to/predictions.h5 \
  --threshold 0.5 \
  --min-strings 3 \
  --min-signal-hits 10 \
  --output /path/to/filtered.h5
```

Parameters:
- `--threshold` - Signal probability threshold (default: 0.5)
- `--min-strings` - Minimum unique signal strings required (default: 3)
- `--min-signal-hits` - Minimum signal hits required (default: 10)

## Project Structure

```
pytorch_training/
├── conf/                 # Hydra configuration files
├── training/
│   └── trainer.py        # Main training loop logic
├── train_types/          # Training type implementations
│   ├── base.py           # BaseTrainType abstract class
│   ├── noise_sig.py      # Noise-signal classification
│   ├── angle.py          # Angle reconstruction
│   └── ...
├── data_utils/
│   ├── dataloaders.py    # Dataset/DataLoader creation
│   ├── preprocessors.py  # Data preprocessing
│   └── readers.py        # H5 file readers
├── models/
│   ├── encoder.py        # Transformer encoder
│   ├── lstm.py           # LSTM model
│   ├── graphnet.py       # Graph neural networks
│   └── ...
├── metrics/              # Metric calculation classes
├── scripts/              # Utility scripts
│   └── noise_sig_nn_data_utils/
│       ├── predict_to_h5.py      # Generate signal predictions
│       └── filter_by_signal.py   # Filter events by signal
└── train.py              # Main entry point
```

## Available Models

| Model | Config | Description |
|-------|--------|-------------|
| Encoder | `encoder` | Transformer encoder |
| EncoderCLS | `encoder_cls` | Transformer with CLS token |
| LSTM | `lstm` | Bidirectional LSTM |
| CNN | `cnn` | 1D Convolutional network |
| GCN | `gcn` | Graph Convolutional Network |
| GAT | `gat` | Graph Attention Network |
| Graphnet | `graphnet` | Custom graph network |

## Logging

Training logs to:
- Console output
- TensorBoard (in checkpoint directory)
- Weights & Biases (if not disabled)

```bash
# Disable W&B logging
WANDB_MODE=disabled python train.py +experiment=noise_sig_2020
```

## Development

Format code:
```bash
black .
ruff check . --fix
```

Run linting:
```bash
ruff check .
```
