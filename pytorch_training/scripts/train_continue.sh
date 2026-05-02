#!/bin/bash

# Train on GPU 0 - continue from checkpoint without DA (gr0)
CUDA_VISIBLE_DEVICES=0 python train.py -c train_configs/noise_sig_da_k02_gr0_continue.yaml &

# Train on GPU 1 - continue from checkpoint with DA (gr005)
CUDA_VISIBLE_DEVICES=1 python train.py -c train_configs/noise_sig_da_k02_gr005_continue.yaml &

# Wait for both processes to complete
wait

echo "Both training runs completed!"

