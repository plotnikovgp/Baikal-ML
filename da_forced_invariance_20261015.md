# Artificially forcing the independent domain-probe AUC to 0.5

This is a **diagnostic ablation, not a recommended production checkpoint**.
It uses the previously selected MC/EXP noise-signal model from
`da_output_w1_20261015.md`, with its correct padding mask and masked domain
pooling. The supervised signal label is `hit_label != 0`. MC and EXP events
both have the same raw preselection: at least 8 hits on at least 2 strings.
No `impose_reco_cuts` is applied: this is a hit-classification experiment, not
an angle-reconstruction comparison.

The target is AUC of an **independent probe trained on mean-pooled encoder
features**, not AUC of the adversarial head. MC `muatm` and experimental
events are the domains. Two disjoint evaluation halves each contain 5,000 MC
validation events and 5,000 experimental test events. The scalar calibration
is fitted on the first 5,000 labeled MC validation events only; EXP labels
are never used.

## What was tried

1. **Iterative nullspace projection (INLP).** A logistic domain classifier is
   fitted to pooled features, its direction is removed from *every hit's*
   128D embedding, and an independent logistic probe is re-fitted after each
   step. Projection rank is selected without using the final evaluation half.
   The numerical basis is explicitly orthonormalized in float64. A direct
   control at full rank gives constant features and AUC=0.5, but hit AP falls
   to the signal prevalence, 0.0916. On a separate-validation selection,
   127 directions gave validation AUC 0.506; on the final half the raw AUC
   was 0.448 (symmetric separability 0.552) and hit AP was 0.295. The last
   one-dimensional signal can invert its domain association across splits.
2. **Keep only the task axis.** For a two-logit linear hit head, only the
   difference of its two weight vectors is needed for the signal probability.
   Projecting every hit to this 1D axis changes scores by at most 1.2e-7,
   so hit AP/recall/W1 are unchanged. The independent linear domain probe
   AUC falls from 0.881 to 0.589 on the same validation-to-test protocol.
3. **Add independent Gaussian logit jitter per event** to that remaining
   task axis. This can move AUC arbitrarily close to 0.5, but the hit task
   collapses: at sigma 8, holdout AUC is 0.517 and hit AP 0.521; at sigma
   128, AUC is 0.507 and AP 0.097. These are means over three random seeds.
4. **Deterministic event centering (best artificial construction).** For
   original hit-logit differences `d_ij`, use

       d'_ij = d_ij - mean_j(d_ij) + c

   and output `sigmoid(d'_ij)`. The single global offset `c = -8.1080713`
   maximizes hit F1 on the MC calibration split. Equivalently, all non-task
   dimensions are discarded, and the remaining hit embedding is centered
   within each event. Its mean-pooled event embedding is now the **same
   constant for every MC and EXP event**. Fresh linear and random-forest
   probes on this pooled embedding both return AUC exactly 0.5. A real
   `CenteredHitAxisAdapter` forward pass matches the analytic hit scores to
   3.6e-7 and has at most 7.2e-7 pooled-feature spread in a checked batch.

## Cost of the deterministic construction

Metrics below are against MC hit ground truth. Precision/recall use signal
threshold 0.5. W1 is the Wasserstein distance between MC and EXP **hit-score
distributions** after the model's predicted 2-string/8-signal-hit event cut;
it is not an angular metric. The centered model uses the same `c` in both
halves, with no refitting on either evaluation half.

| Half | Model | Hit AP | Precision | Recall | Post-cut W1 | MC/EXP events passing |
|---|---|---:|---:|---:|---:|---:|
| First | Original | 0.9822 | 0.9466 | 0.9299 | 0.04574 | 983 / 608 |
| First | Centered | 0.9659 | 0.9143 | 0.9090 | 0.02137 | 1075 / 613 |
| Second | Original | 0.9833 | 0.9480 | 0.9302 | 0.05866 | 987 / 581 |
| Second | Centered | 0.9694 | 0.9161 | 0.9146 | 0.02581 | 1111 / 568 |

On the **intersection of events accepted by both models**, post-cut W1 is
0.04321→0.01982 in the first half (962 MC / 576 EXP common events) and
0.05599→0.02294 in the second (974 MC / 544 EXP). Thus the improvement is
not explained solely by a changed event selection. The precision/recall and
AP losses are real: roughly 1.4–1.6 percentage points in AP, 3.2 points in
precision, and 1.6–2.1 points in recall.

## Why this is not full domain adaptation

The AUC=0.5 result is exact **only for the mean-pooled representation**.
After centering, an independent probe using quantiles and standard deviation
of the within-event hit logits still distinguishes MC from EXP: AUC 0.632
for logistic regression and 0.667 for a random forest on the second half.
These probes do not even use hit multiplicity. The domain information was
removed from one aggregation, not from the full set of hit features. The
smaller post-cut W1 must also be interpreted alongside hit-quality losses;
it does not prove that the model is physically calibrated.

The experiment is reproducible with `da_hit_axis_probe.py`,
`da_force_invariance.py`, `da_task_axis_noise_sweep.py`, and
`da_centered_hit_axis.py`. Result JSON files on cluster62 are in
`/home/plotnikovgp/tmp/baikal-da-strong-20261015/runs/` under the matching
names. A self-contained 3.9 MB adapter state dict is at
`cluster63:/home/plotnikovgp/tmp/baikal-angle-uncertainty-commit/pytorch_training/checkpoints/noise_sig_experiments/centered_hit_axis_20261015/adapter.ckpt`
(and at
`cluster62:/home/plotnikovgp/tmp/baikal-da-strong-20261015/runs/centered_hit_axis_adapter_20261015.ckpt`).
It includes the original network weights and calibrated offset. CPU loading:

```python
from da_centered_hit_axis import load_adapter_checkpoint
model = load_adapter_checkpoint("/home/plotnikovgp/tmp/baikal-angle-uncertainty-commit/pytorch_training/checkpoints/noise_sig_experiments/centered_hit_axis_20261015/adapter.ckpt", "cpu")
hit_logits, pooled_features = model(normalized_hits, valid_hit_mask)
```

Keep the original checkpoint as the default until downstream angle
reconstruction and experimental-data studies validate this postprocessor.
