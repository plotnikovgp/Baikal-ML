## Project for training NN's on baikal MC data based on torch and torch_geometric

### Currently supported training types are
- noise-signal hits classification
- track-cascade hits classification
- angle-reconstruction
- t_res
- track-cascade & angle-reconstruction || track-cascade & t_res 

### How to run
```
python train.py -c=train_configs/<your_config> -dw
```
remove `-dw` if you want to track your run in wandb (login from terminal required before that)

### Multi-Dataset Training

The codebase now supports training on multiple datasets with different sampling probabilities. This allows you to:

- Train on multiple datasets with controlled sampling rates
- Validate on multiple datasets with individual metrics for each dataset
- Save best models for each individual dataset
- Apply custom prefilters and parameters for each dataset
- Use custom names for each dataset

To use multi-dataset training, configure your YAML file with a `dataset_configs` section as shown in the examples below.

#### Performance Optimizations

Several performance optimizations are available to improve data loading speed:

```yaml
# Performance optimization parameters
num_workers: 4            # Number of data loading worker processes
prefetch_factor: 3        # Number of batches to prefetch per worker
persistent_workers: true  # Keep worker processes alive between iterations
pin_memory: true          # Pin memory for faster CPU->GPU transfer
cache_datasets: true      # Cache datasets in memory for faster access
```

These settings are especially useful for multi-dataset training to reduce loading times.

#### Basic Multi-Dataset Setup

```yaml
dataset_configs:
  - # First dataset
    path_to_data: "/path/to/dataset1.h5"
    tres_cut: 10.0
    
  - # Second dataset
    path_to_data: "/path/to/dataset2.h5"
    tres_cut: 15.0

# Optional sampling weights
dataset_weights: [0.7, 0.3]  # 70% from first dataset, 30% from second

# Optional: save best model for each dataset separately
save_best_per_dataset: true
```

#### Custom Named Datasets

You can provide custom names for each dataset, which will be used in metric logging and saved model filenames:

```yaml
dataset_configs:
  - # First dataset with a custom name
    name: "clean_data"
    path_to_data: "/path/to/clean_dataset.h5"
    tres_cut: 10.0
    
  - # Second dataset with a custom name
    name: "noisy_data"
    path_to_data: "/path/to/noisy_dataset.h5"
    tres_cut: 15.0
```

This will result in metrics like `clean_data_loss`, `noisy_data_loss` and checkpoint files named `best_clean_data.ckpt` and `best_noisy_data.ckpt`.

#### Custom Prefilters for Each Dataset

You can specify individual prefilters and parameters for each dataset:

```yaml
dataset_configs:
  - # First dataset with strict prefiltering
    name: "clean_data"
    path_to_data: "/path/to/dataset1.h5"
    tres_cut: 10.0
    data_prefilter_params:
      min_hits: 15
      max_hits: 100
      min_signal_percentage: 0.6
    
  - # Second dataset with different prefiltering
    name: "noisy_data"
    path_to_data: "/path/to/dataset2.h5"
    tres_cut: 15.0
    data_prefilter_params:
      min_hits: 10
      max_hits: 150
      min_signal_percentage: 0.5
```

See the example config files in `train_configs/` for complete examples:
- `multi_dataset_prefilters_example.yaml` - example with custom prefilters
- `multi_dataset_named_example.yaml` - example with named datasets and performance optimizations

### Project structure

- trainining
    - train_utils.py - all high-level logic for model training & evaluation

- data_utils
    - dataloaders.py - create torch datasets and dataloaders, collator logic
    - preprocessors.py - preparing data for different tasks, add noise, filter etc
    - readers.py - read data from H5 file and run preprocessor
    - MultiDatasetSampler - combines multiple datasets with specified probabilities
    - CachingDatasetWrapper - caches dataset items in memory for faster access

- metrics - all logic for metrics calculation 

- models - each file contains some architecture 

- train_configs - yaml configs describing task type, model etc
    - multi_dataset_prefilters_example.yaml - example with custom prefilters for each dataset
    - multi_dataset_named_example.yaml - example with named datasets and performance optimizations

- examples
    - multi_dataset_example.py - code example showing how to use the multi-dataset functionality

