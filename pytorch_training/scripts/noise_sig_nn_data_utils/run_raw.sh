python write_preds_for_raw_data.py \
  --checkpoint /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff512_hs512_bs128/best_2020.ckpt \
  --config /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/train_configs/noise_sig.yaml \
  --raw-data /home3/ivkhar/Baikal/data/h5s/baikal_2020_flat.h5 \
  --norm-data /home2/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5 \
  --output /home/plotnikovgp/baikal/data/baikal_2020_flat_nn_preds.h5 \
  --output-folder preds \
  --particles muatm nuatm nue2 \
  --batch-size 256 \
  --prefix raw \
  --events-limit 2000

# further use in filter_data_multiprocessing.py
