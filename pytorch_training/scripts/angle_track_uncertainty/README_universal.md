# Universal direction + track-point + uncertainty model

The current single inference checkpoint is
`universal_angle_track_uncertainty_v4.pt`. It contains the all-particle
direction backbone, a compatible track-point head, the angular/transverse
uncertainty head, input normalization, and validation calibration factors. It
is a CPU-loadable PyTorch `state_dict` bundle without optimizer state. The
signal/noise hit classifier is **not** part of this file.

On cluster63 the bundle is at:

`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/universal_angle_track_uncertainty_v4.pt`

The model expects the prepared HDF5 format with signal/noise probability
`p >= 0.70`, at least eight retained hits on at least two strings. Direction
was also checked at `p >= 0.90`, but uncertainty radii were calibrated only
at `p >= 0.70`. The angle
backbone was trained on `nuatm`, `nue2`, and only **single-track** `muatm` MC.
The point target is a point on the truth line nearest the centroid of **all
retained hits**. Events with multiple simulated `muatm` tracks were removed
from both direction and point training. Point/radius outputs for multi-track
events and experimental data are not validated.

Run a ten-event CPU smoke test on cluster63:

```bash
CUDA_VISIBLE_DEVICES= /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
  /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/infer_unified_model.py \
  --checkpoint /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/universal_angle_track_uncertainty_v4.pt \
  --data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
  --split test --max-events 10 --batch-size 10 --threads 4 \
  --output /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/my_predictions.csv
```

Remove `--max-events 10` for a full split. `theta_deg`, `phi_deg`, direction
components, and `anchor_{x,y,z}_m` describe one reconstructed line in the
cluster-local coordinate frame. `angle_r68_deg` and `angle_r95_deg` are
validation-calibrated angular containment radii; `anchor_r68_m` and
`anchor_r95_m` are the corresponding transverse point-to-truth-line radii.
They are *marginal* intervals, not a jointly calibrated tube around the entire
track. The CSV's `track_point_scope` field flags muatm/experimental events for
which the point prediction has not been validated against a unique true line.

The clean predicted-hit MC test contains 27,038 events after cross-dataset
leakage exclusion and removal of multi-track `muatm`. Direction q50/q68 is
2.74°/4.73°; transverse point error q50/q68 is 3.55/5.49 m. Marginal
68%/95% test coverage is 66.2%/93.8% for angle and 65.2%/92.9% for the
point. On the independent 307,913-event GT-hit `nue2` test, coverage is
68.1%/95.1% for angle and 67.8%/94.5% for the point. Intervals are
validation-calibrated separately in the predicted-hit and GT-hit domains;
the CPU bundle uses the **predicted-hit** calibration. The high-uncertainty
tail is under-covered on predicted hits, so marginal coverage is not a
conditional guarantee. None of these MC numbers should be transferred to
experimental data without validation.

The earlier direct comparison over all 403,171 nue2 GT-signal test events was
**not independent**: 95,258 of those event IDs were already in the
predicted-hit training split of the universal checkpoint. Retain that result
only as a paired-input diagnostic, not as a test metric. The independent
comparison excludes those IDs (307,913 events). Its direction q50/q68 is
3.44°/5.73° for the universal model versus 2.37°/4.04° for the nue2
specialist; transverse point q50/q68 is 5.42/8.68 m versus 4.42/7.22 m.
`make_leak_free_masks.py` creates the cross-dataset masks used for all newer
experiments. The universal model's uncertainty calibration was fitted on
predicted-signal hits; GT-hit interval coverage remains only a domain-shift
diagnostic.

For the newer single-track-muatm experiments, `train_mixed_universal.py`
filters `muatm` events using the anchor file's `valid` flag. That flag requires
exactly one simulated muon track, a parsable source event, and a finite true
track point. Other particle types are retained. This is a deliberately
conservative applicability cut, not a signal/noise classifier. The stricter
predicted-hit validation/test masks also exclude GT train events already seen
by the nue2 specialist initializer. Full multi-muatm metrics are reported
separately as an out-of-scope diagnostic.

## Single-track-muatm universal model (v3)

The selected direction/point checkpoint is on cluster63 at
`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/continue_10k_same_lr/best.pt`.
It is one SetTransformer model, initialized from the nue2 specialist and
fine-tuned on all three particle types. The earlier v2 model was continued
for 10,000 extra steps at the original learning rate and at half that rate.
The original learning rate won by the predeclared validation score (6.875
versus 6.889); its best checkpoint was at extra step 9,500. No teacher or
ensemble is used at
inference. The hit input is converted to the **GT signal-file normalization**
even when the hits came from the signal/noise network. `muatm` training events
are retained only if the original MC has one simulated muon track and a valid
truth-line point. The p>=0.70, two-string/eight-hit cut is unchanged.

The checkpoint was selected solely on validation. The independent test uses
`masks_v2.npz`: predicted-hit test excludes events seen by the specialist's
GT training; GT nue2 test excludes events seen by the original universal
predicted-hit training. Both comparisons below evaluate exactly the same
events and hit inputs for all models.

| Model | Predicted-hit angle q50/q68 (27,038) | Single-muatm q68 (3,437) | Predicted-hit nue2 q68 (9,463) | GT nue2 angle q50/q68 (307,913) | GT nue2 line q68 |
| --- | --- | --- | --- | --- | --- |
| Universal v3, longer training | 2.77° / 4.79° | 10.09° | 5.83° | 2.41° / 4.09° | 6.81 m |
| Universal v2 | 2.80° / 4.85° | 11.06° | 5.87° | 2.43° / 4.14° | 6.99 m |
| Original universal | 3.21° / 5.49° | 11.54° | 7.81° | 3.44° / 5.73° | 8.68 m |
| Original nue2 specialist | 3.06° / 5.99° | 50.91° | 5.82° | 2.37° / 4.04° | 7.22 m |

The MC single-track requirement cannot be applied to experimental data
without a separate multiplicity/quality classifier. Multi-muatm and
experimental resolution/coverage remain unvalidated; the table must not be
read as a performance claim for all muatm events. The older v1 bundle remains
available at `runs/universal_predsignal_bundle_v1/` for comparison only.

`train_mixed_uncertainty.py` freezes the selected v3 direction/point model and
trains a Student-t uncertainty head for angular and transverse line errors.
Disjoint clean validation halves select the head and calibrate its 68%/95%
radii. The selected head is
`runs/mixed_universal/uncertainty_continue10k_low_lr/best.pt`; its metrics
and independent-test per-event predictions are in the same directory. The
v3 bundle embeds both heads and predicted-hit calibration.
`plot_mixed_uncertainty.py` produces angular/line reliability,
conditional-coverage, and error-distribution figures from the saved test
predictions. It shows useful uncertainty ranking, but a notable
high-difficulty undercoverage tail on predicted hits.

## Domain and hit-threshold audits

The v3 experimental theta distribution is close to the *full* muatm MC
distribution, but not to the rare single-track-muatm subset used for
supervised track-point training. On the common `p >= 0.90`, 2s8h cut,
13,349 experimental events have predicted theta q10/q50/q90 of
118.91°/142.13°/161.39°. The 10,835 unique muatm MC validation event IDs
have 123.46°/145.28°/164.80°. Their raw theta Wasserstein-1 distance is
3.48°; matching the MC hit-count/string-count distribution to experiment
reduces it to 1.66°. On the training `p >= 0.70` cut, the corresponding
numbers are 30,562 experimental and 25,054 unique muatm MC events, with
Wasserstein-1 distances 3.80° raw and 3.06° input-matched. Experiment
train/test differ by only 0.47° (`p >= 0.70`) or 0.51° (`p >= 0.90`).

This is a **prediction-distribution** check, not an experimental accuracy
measurement: experiment has no direction ground truth. At `p >= 0.90`, only
121/10,835 muatm MC validation events have a single usable truth track.
That subset's theta distribution is not representative of experiment.

The higher signal-hit threshold improves muatm MC direction q68 on the
**same 10,835 event IDs**, from 9.874° (`p >= 0.70`) to 9.362°
(`p >= 0.90`), paired bootstrap difference -0.512° with 95% interval
[-0.672°, -0.351°]. It hurts the 121 single-track events in this overlap,
so do not generalize the gain to every particle class. On the same 13,349
experimental events, switching `p >= 0.70` to `p >= 0.90` changes the
predicted direction by median 4.675° (q68 7.238°). `analyze_domain_predictions.py`,
`evaluate_mc_angle_by_threshold.py`, and `compare_hit_thresholds.py`
reproduce these checks from inference CSVs.

Both MC prepared-hit files contain repeated event IDs *within* splits (but
no IDs shared between train, validation, and test in the `p >= 0.70` file).
The domain/threshold comparisons deduplicate MC by first occurrence and
report the count. A paired v2-v3 evaluation on unique, leakage-free test IDs
still finds a v3 direction q68 gain: 4.904° to 4.839° on predicted hits
(26,229 IDs), with bootstrap difference 95% interval [-0.107°, -0.022°].
On GT nue2 hits it is 4.267° to 4.229° (281,804 IDs), interval
[-0.048°, -0.027°]. `compare_checkpoint_paired.py` reproduces this.

`train_mixed_universal.py` now supports an optional predicted-hit
threshold augmentation (`--pred-thinning-prob`, `--pred-threshold-min`,
`--pred-threshold-max`). It reselects hits using stored signal probabilities,
retains the 2s8h cut, and rebases the point target to the new hit centroid
so the physical truth line is unchanged. It is disabled by default and
is not part of the v3 bundle. `test_pred_thinning.py` checks the line
invariance and minimum-hit fallback.

## Selected stochastic-threshold model (v4)

Starting from v3, a 5,000-step run used `--pred-thinning-prob 0.5` and
uniformly varied the threshold from 0.70 to 0.90 on the augmented half of
predicted-hit training batches. It did **not** apply threshold thinning to
GT-hit nue2 batches or to validation/test data. The selected checkpoint was
at step 2,500:

`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/continue_5k_stochastic_p/best.pt`

On the same leakage-free test events, it improves the v3 direction and point
errors. After deduplicating event IDs, predicted-hit q68 angle is
4.839° -> 4.782° (paired bootstrap 95% interval for the change
[-0.092°, -0.027°]); GT nue2 q68 is 4.229° -> 4.194°
([-0.045°, -0.024°]). Transverse-line q68 is 5.260 -> 5.212 m on
predicted hits and 5.650 -> 5.518 m on GT nue2 hits. The raw row counts
are 27,038 predicted-hit and 307,913 GT-hit events; the unique-ID counts
are 26,229 and 281,804. Per-event residuals and paired-bootstrap results
are in `continue_5k_stochastic_p/test_best2500/test_residuals.npz` and
`continue_5k_stochastic_p/paired_v3_vs_thin.json`.

On the common experimental/MC `p >= 0.90` cut, muatm MC validation q68
improves from 9.362° to 9.221° over 10,835 unique shared IDs, with paired
bootstrap change interval [-0.224°, -0.101°]. On `p >= 0.70` it improves
10.720° -> 10.629° over 25,054 IDs, interval [-0.143°, -0.045°]. These
muatm truth angles are a diagnostic for multi-track events, not the
direction of a uniquely identified track. The rare single-track subset has
no statistically clear change.

Experimental-vs-muatm MC theta-distribution matching has not clearly
improved: after matching hit/string counts, Wasserstein-1 is 1.992° ->
2.022° at `p >= 0.90` (experiment test versus MC validation) and 2.954°
-> 3.007° at `p >= 0.70`. The small increases warn against claiming
better experimental reconstruction from MC resolution alone. There is no
experimental truth angle. `infer_direction_point_for_audit.py` evaluates
the unbundled direction/point checkpoint without emitting stale uncertainty
values; `compare_mc_quality_checkpoints.py` performs paired muatm checks.

For v4, `train_mixed_uncertainty.py` trained a fresh Student-t uncertainty
head and continued it at a lower learning rate, selecting the continuation
by clean validation NLL (7.145 -> 7.108). The selected head, metrics, and
independent-test predictions are in
`runs/mixed_universal/uncertainty_stochastic_p_low_lr/`. Calibration uses
a validation subset disjoint from head-selection validation. On the
predicted-hit test, marginal angle coverage is 66.2%/93.8% for its 68%/95%
radii and transverse-line coverage is 65.2%/92.9%. The hardest events remain
under-covered. The bundle at the top of this README embeds this new head;
the old v3 bundle is retained for comparison. Uncertainty radii at p>=0.90
or on experimental events have **not** been calibrated.

## Controlled continuation after v4

The stochastic-threshold augmentation did not make individual experimental
events invariant to the selected-hit threshold. On the same 2,126 experiment
test IDs, changing p>=0.70 to p>=0.90 changes v4 direction by median/q68
4.570°/7.323°, versus 4.498°/7.200° for v3. This is a useful limit on
what the v4 MC improvement implies.

Two further 3,000-step runs started from v4 with identical learning rates,
seeds, and data: one always tried p>=0.90 on predicted-hit training batches
(falling back to p>=0.70 when the 2s8h cut failed), the other used no new
threshold augmentation. Both selected step 1,000 on validation. They are
saved as `continue_v4_fixed_p090_3k` and `continue_v4_noaug_3k`, but neither
replaces v4:

| Unique-ID independent test q68 | v4 | Fixed-p090 continuation | No-new-augmentation continuation |
| --- | ---: | ---: | ---: |
| Predicted-hit direction [deg] | 4.782 | 4.773 | 4.794 |
| Predicted-hit transverse line [m] | 5.212 | 5.238 | 5.203 |
| GT nue2 direction [deg] | 4.194 | 4.178 | 4.172 |
| GT nue2 transverse line [m] | 5.518 | 5.629 | 5.542 |

For predicted-hit direction, the paired-bootstrap 95% intervals for the
change relative to v4 both cross zero (fixed-p090 [-0.035°, +0.018°],
no-new-augmentation [-0.010°, +0.043°]). The fixed-p090 run clearly
damages the point target: GT nue2 transverse-line q68 rises by 0.111 m,
95% interval [+0.099, +0.122] m. On the same 10,835 p>=0.90 muatm MC IDs,
direction q68 is 9.221° for v4, 9.186° for fixed-p090, and 9.167° for
no-new-augmentation; neither continuation's paired-bootstrap improvement
excludes zero. Experiment-vs-MC hit/string-matched theta Wasserstein-1 is
2.022°, 2.012°, and 2.009° respectively, also nearly unchanged. This is
why the v4 all-heads bundle remains the selected model.

The preparation/training pipeline is in `build_all_track_anchor_targets.py`,
`train_track_anchor_v3.py`, `make_leak_free_masks.py`,
`train_mixed_universal.py`, and `train_mixed_uncertainty.py`. The bundle is
produced by `bundle_universal_model.py`; `infer_unified_model.py` performs
standalone CPU inference. A 16-event CPU-only smoke test of the v4 bundle
succeeded. Training source checkpoints and full test metrics remain under
`/home/plotnikovgp/tmp/angle_reconstruction/` on cluster63.
