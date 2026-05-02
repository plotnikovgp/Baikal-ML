#!/bin/bash

python plot_prediction_distributions.py \
  --checkpoint checkpoints/noise_sig_domain_adaptation/encoder_nl5_hs512_dff512_nh1_noise_sig_da_k01_gr01_bs128/best_sig_noise_2020.ckpt \
  --config train_configs/noise_sig_da_k01_gr01.yaml \
  --output plots/noise_sig_da_k01_gr01.png \
  --model_name "DA (k=0.1, alpha=0.1)"


