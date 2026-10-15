# DA audit of report-era noise/signal checkpoints (2026-10-15)

## Which model was in `generate_report`?

The saved report at `pytorch_training/plots/report_noise_sig_tres_abs_merged_v2/report_info.md` used the **non-DA two-head model** `checkpoints/noise_sig_experiments/noise_sig_tres_abs_merged/best.ckpt` as its primary checkpoint. It supplied `k_nsol_labelneq0_da_hs128_k0p001/best_mc_2020.ckpt` separately via `--da-checkpoint` for comparative plots. The latter is not the checkpoint selected for this audit because an earlier held-out test found poor signal/noise performance.

The audited good-DA checkpoint is `checkpoints/noise_sig_experiments/k_nsol_labelneq0_da_hs128_k0p0001/best_mc_2020.ckpt`. Its no-DA counterpart is `checkpoints/noise_sig_experiments/k_nsol_labelneq0_hs128/best.ckpt`. Both are five-layer, 128-hidden, one-head encoders trained with `label != 0` as signal; the DA run uses a gradient reversal domain head with domain-loss weight `1e-4`.

## Protocol

- Script: `da_report_check.py`, run on cluster63 with `--n 5000`.
- MC: 10,000 disjoint `muatm` events from the 2020 5 GB file's `val` split; first 5,000 train the *independent frozen-feature probe*, next 5,000 test it. MC hit labels are `label != 0`.
- Experiment: 5,000 `val` events for probe training, 5,000 disjoint `test` events for testing. Experimental data have no hit truth. Both datasets use a raw 2-unique-string / 8-hit preselection and the experimental features are renormalized to the MC statistics.
- Event-level features: masked mean of final encoder hit embeddings. Probe: standardized logistic regression, fitted afresh per frozen checkpoint on balanced MC/experimental events. AUC is measured only on probe-test events.
- Hit metrics and MC–experiment score-distribution Wasserstein-1 distance (W1) use the probe-test events. The predicted 2-string / 8-hit cut is applied at score threshold 0.5. W1 is reported both on each model's own selected events and on the intersection of events accepted by both models.
- At most 256 hits are fed to each model per event. Only four events in the entire MC validation file have more than 256 hits, and none in experimental test.

The original report instead used 25,000 MC `muatm` validation events and 50,000 experimental **train** events, so these are a held-out audit, not a numerical reproduction of its plots.

## Results on the held-out probe-test cohort

The main numbers below use the mathematically correct two-class signal score `softmax(logits)[1]`.

| Metric | No DA | DA (`k=1e-4`) |
|---|---:|---:|
| Independent domain-probe AUC (event features) | 0.9171 | 0.9058 |
| DA head AUC, raw domain label direction | — | 0.4093 |
| MC hit ROC AUC | 0.99863 | 0.99833 |
| MC hit average precision | 0.98640 | 0.98397 |
| MC hit precision / recall at score ≥0.5 | 0.9384 / 0.9580 | 0.9395 / 0.9456 |
| MC–experiment hit-score W1, all events | 0.00919 | 0.00799 |
| W1 after each model's predicted 2s8h cut | 0.05745 | 0.05727 |
| W1 on the same accepted events (intersection) | 0.05976 | 0.05897 |
| MC / experimental events passing predicted 2s8h | 1,066 / 671 | 1,030 / 649 |

The raw DA-head AUC of 0.4093 is equivalent to 0.5907 separability if its score direction is reversed. It is not evidence of indistinguishable domains. The independent probe still separates domains strongly (AUC 0.906); the DA improvement over no DA is small. W1 improves by about 13% before the cut but only about 0.3% after each model's cut, and about 1.3% on the fixed shared cohort. No uncertainty intervals were computed for these differences, so they should not be called significant. MC hit recall at a fixed 0.5 score falls by about 1.25 percentage points.

## Legacy score convention

`generate_report.py` and training-time classification metrics use `sigmoid(logit_1)` even though the model is trained with two-class cross-entropy. This is internally consistent with their reported threshold, but `softmax(logits)[1]` is the actual two-class probability. Scores, event cuts and W1 differ between the conventions. Under the legacy convention:

| Metric | No DA | DA (`k=1e-4`) |
|---|---:|---:|
| MC hit precision / recall at score ≥0.5 | 0.9302 / 0.9631 | 0.9425 / 0.9422 |
| W1, all hits | 0.01203 | 0.00948 |
| W1 after model-specific predicted 2s8h | 0.05396 | 0.05262 |
| W1 on shared accepted events | 0.05722 | 0.05353 |

The absolute W1 values cannot be compared directly with the original report because checkpoint, cohort, split and score convention differ. The JSON from this run is at `/home/plotnikovgp/tmp/da_report_check_2026-10-15.json` on cluster63.

## Re-run

On cluster63, with the original data/checkpoint paths available:

```bash
cd /home/plotnikovgp/tmp/baikal-angle-uncertainty-commit
/home/plotnikovgp/tmp/ENTER/bin/python3.11 da_report_check.py \
  --n 5000 --output /home/plotnikovgp/tmp/da_report_check_2026-10-15.json
```

The script intentionally keeps all paths to the report-era data/checkpoints explicit near its top; adjust those constants for another host.
