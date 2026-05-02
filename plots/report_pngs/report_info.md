# Report reproducibility — report

_Generated 2026-04-30T18:20:33_

## Git

- commit: `3c6b0596fff248910941ce5ef507465c0ef60306`
- branch: `new_dev`
- dirty: **True**

<details><summary>git status --porcelain</summary>

```
M pytorch_training/models/__init__.py
 M pytorch_training/models/encoder.py
 M pytorch_training/scripts/generate_report.py
 M pytorch_training/train_types/noise_sig.py
 M pytorch_training/training/trainer.py
?? plots/
?? pytorch_training/.cursorignore
?? pytorch_training/.pre-commit-config.yaml
?? pytorch_training/conf/experiment/noise_sig_and_tres.yaml
?? pytorch_training/conf/experiment/noise_sig_and_tres_two_head.yaml
?? pytorch_training/conf/experiment/noise_sig_and_tres_two_head_mae.yaml
?? pytorch_training/conf/experiment/noise_sig_or_labels_zmirror.yaml
?? pytorch_training/conf/model/encoder_two_head.yaml
?? pytorch_training/conf/train_type/noise_sig_and_tres.yaml
?? pytorch_training/data_utils/README.md
?? pytorch_training/external_models/
?? pytorch_training/logs/
?? pytorch_training/models/icecube_transformer.py
?? pytorch_training/scripts/calc_metrics.py
?? pytorch_training/scripts/calculate_metrics.py
?? pytorch_training/scripts/compare_exp_mc_distributions.py
?? pytorch_training/scripts/compare_signal_def.py
?? pytorch_training/scripts/compare_theta_phi.py
?? pytorch_training/scripts/create_new_set_optimized.py
?? pytorch_training/scripts/create_normalized_particle_subset.py
?? pytorch_training/scripts/distribution_plots.py
?? pytorch_training/scripts/eval_da.py
?? pytorch_training/scripts/extract_muatm_events.py
?? pytorch_training/scripts/lint.sh
?? pytorch_training/scripts/noise_sig_nn_data_utils/create_random_preds.py
?? pytorch_training/scripts/notebooks/
?? pytorch_training/scripts/plot.sh
?? pytorch_training/scripts/plot_da_k01_gr01.sh
?? pytorch_training/scripts/plot_da_k02_gr0.sh
?? pytorch_training/scripts/plot_hit_counts_by_dataset.py
?? pytorch_training/scripts/plot_prediction_distributions.py
?? pytorch_training/scripts/plot_prediction_distributions_backup.py
?? pytorch_training/scripts/plot_training_metrics.py
?? pytorch_training/scripts/plot_tres_distribution.py
?? pytorch_training/scripts/plot_two_experiments.py
?? pytorch_training/scripts/quick_normalize.py
?? pytorch_training/scripts/run_experiments.py
?? pytorch_training/scripts/setup-hooks.sh
?? pytorch_training/scripts/test_forward.py
?? pytorch_training/scripts/train_continue.sh
?? pytorch_training/test.py
?? pytorch_training/train_configs/angle_da_with_exp.yaml
?? pytorch_training/train_configs/enc_graphnet_stack_noise_sig.yaml
?? pytorch_training/train_configs/encoder_angle_10-3.yaml
?? pytorch_training/train_configs/encoder_angle_old.yaml
?? pytorch_training/train_configs/noise_sig.yaml
?? pytorch_training/train_configs/noise_sig_da_k01_gr01.yaml
?? pytorch_training/train_configs/noise_sig_da_k02_gr0.yaml
?? pytorch_training/train_configs/noise_sig_da_k02_gr005.yaml
?? pytorch_training/train_configs/noise_sig_da_k02_gr005_continue.yaml
?? pytorch_training/train_configs/noise_sig_da_k02_gr0_continue.yaml
?? pytorch_training/train_configs/noise_sig_da_k20_gr0.yaml
?? pytorch_training/train_configs/noise_sig_da_with_exp.yaml
```
</details>


## Command

```bash
cd /home/plotnikovgp/baikal/Baikal-ML/pytorch_training
CUDA_VISIBLE_DEVICES=0 \
python pytorch_training/scripts/generate_report.py --checkpoint pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt --no-zmirror-checkpoint pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt --zmirror-checkpoint pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128_zm/best.ckpt --da-checkpoint pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_da_hs128/best_mc_2020.ckpt --tres-checkpoint pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_tres_hs128/best.ckpt --mc-data-path /home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_5gb.h5 --mc-events-per-type 50000 --exp-max-events 50000 --embedding-max-events 10000 --html-output pytorch_training/plots/report_full.html --report-format html
```

## Settings

- signal definition: `label_nonzero` (|label| != 0)
- threshold: `0.5`
- mc-split: `val`, mc-events-per-type: `50000`
- exp-split: `train`, exp-max-events: `50000`
- model: hs=128, dff=512, type=encoder
- threshold-points: 80

## Counts

- MC events loaded: **75000**
- EXP events loaded: **50000**

| cut | MC muon kept | EXP kept |
|---|---|---|
| ≥0h ≥0s @ thr=0.5 | 25000 | 50000 |
| ≥8h ≥2s @ thr=0.5 | 5542 | 3187 |

## Files

- **checkpoint** — `/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt`  (3,987,728 B, mtime 2026-04-25T21:03:40, sha1 `55d04249f63f…`)
- **no_zmirror_checkpoint** — `/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt`  (3,987,728 B, mtime 2026-04-25T21:03:40, sha1 `55d04249f63f…`)
- **zmirror_checkpoint** — `/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128_zm/best.ckpt`  (3,987,728 B, mtime 2026-04-25T21:05:57, sha1 `a41900b7a2ad…`)
- **da_checkpoint** — `/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_da_hs128/best_mc_2020.ckpt`  (4,063,540 B, mtime 2026-04-25T19:23:47, sha1 `1375b54050e4…`)
- **tres_checkpoint** — `/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_experiments/k_nsol_labelneq0_tres_hs128/best.ckpt`  (3,988,240 B, mtime 2026-04-25T20:45:41, sha1 `8b10598704d2…`)
- **mc_data** — `/home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_5gb.h5`  (5,285,026,944 B, mtime 2026-04-25T20:07:07, sha1 `1bcbec9aec95…`)
- **exp_data** — `/home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_exp_filtered_norm_w_model_enc_v2_fix_q.h5`  (1,060,839,776 B, mtime 2025-11-14T20:10:06, sha1 `c4dd4f4152b7…`)

## Predictions

- saved to `plots/report_pngs/predictions.npz` (load with `np.load(path)`)
- arrays: `mc_ev_starts`, `mc_ev_type`, `mc_energy`, `mc_probs`, `mc_true_sig`, `mc_t_res`, `mc_channels`, `exp_ev_starts`, `exp_probs`, `exp_channels` (and optionally `tres_true`, `tres_pred`)

## Outputs

- PNG directory: `plots/report_pngs`
- PDF (if requested): `plots/report.pdf`

## Environment

```json
{
  "python": "3.11.4",
  "platform": "Linux-6.8.0-106-generic-x86_64-with-glibc2.39",
  "torch": "2.1.1+cu118",
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce RTX 3090",
  "cuda_visible_devices": "0"
}
```