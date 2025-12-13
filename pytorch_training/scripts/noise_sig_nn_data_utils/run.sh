python predict_to_h5.py \
 --checkpoint /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/checkpoints/noise_sig_rerun_v2/encoder_nl5_nh1_dff512_hs512_bs128/best_2020.ckpt \
 --config /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/train_configs/noise_sig.yaml \
 --data /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5 \
 --version 1 \
 --batch_size 512 \
 --splits train val test \
 --output-path /home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_nn_v1_sig_probs.h5 \
 --events-limit 100000

threshold=0.5
num_strings=2
num_signal_hits=8
python filter_by_signal.py \
 --original /home/plotnikovgp/baikal/Baikal-ML/pytorch_training/data/baikal_2020_sig-noise_mid-eq_normed.h5 \
 --predictions /home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_nn_v1_sig_probs.h5 \
 --output /home/plotnikovgp/baikal/data/baikal_2020_sig-noise_mid-eq_normed_filtered_t${threshold}_s${num_strings}_h${num_signal_hits}.h5 \
 --threshold ${threshold} \
 --min-strings ${num_strings} \
 --min-signal-hits ${num_signal_hits} \
 --splits train val test
