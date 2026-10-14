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

## Paired hit-threshold audit of v4

`infer_rethresholded_for_audit.py` reapplies a signal-probability threshold
to the same prepared p>=0.70 hits, recomputes the two-string/eight-hit cut
and hit centroid, and runs the selected v4 direction/point backbone. The
signal/noise checkpoint itself is unchanged. On the experimental test split,
the first 100 source rows at p>=0.90 reproduce all 46 retained rows and
every direction/point field of the separately prepared p>=0.90 file exactly.
The full muatm validation CSV reproduces its 11,267 rows in the same order;
the largest floating-point difference is 0.000031 degrees in phi and
0.000008 m in anchor coordinates. The p>=0.70 baseline also reproduces the
earlier 25,054-unique-ID muatm q68 of 10.629 degrees and matched theta W1
of 3.007 degrees.

The following comparison holds the **same** 10,835 muatm MC IDs and 2,126
experimental IDs at every threshold. MC truth direction is `prime_prty`,
which is ambiguous for multi-track muatm. `Line q68` uses only the 121
single-track muatm events with a valid true line. W1 compares predicted
theta distributions after matching MC to experimental hit/string counts;
it is a domain-shift diagnostic, **not** experimental accuracy.

| Minimum hit probability | Experiment test retained / 4,924 | muatm angle q68, common [deg] | Single-muatm angle q68 [deg] | Valid line q68 [m] | Experiment–MC theta W1, common [deg] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.70 | 4,924 | 9.621 | 12.451 | 6.401 | 2.937 |
| 0.75 | 4,470 | 9.557 | 15.963 | 5.777 | 2.667 |
| 0.80 | 3,921 | 9.425 | 16.424 | 6.546 | 2.336 |
| 0.85 | 3,215 | 9.247 | 18.530 | 6.758 | 1.913 |
| 0.90 | 2,126 | 9.221 | 23.273 | 7.747 | 2.022 |

For example, changing p>=0.70 to p>=0.80 improves all-muatm q68 by
0.196 degrees on the fixed IDs (paired-bootstrap 95% interval
[-0.364, -0.113] degrees), and reduces matched W1 by approximately
0.60 degrees (paired interval [-0.910, -0.295]). But p>=0.90 worsens
single-muatm angle q68 by 10.82 degrees (interval [+2.84, +16.99]) and
point-line q68 by 1.35 m (interval [+0.15, +2.92]). The `p>=0.85` W1 is
slightly smaller than `p>=0.90` on these fixed events, while retaining
more events. All threshold comparisons use first occurrence for repeated
MC event IDs; the experimental test IDs are unique.

To test the universal model rather than only muatm, the same sweep covered
all 4,185 nuatm validation rows and every fourth nue2 validation row
(5,436 rows distributed across its full source range). The next figures
hold the common p>=0.90 IDs fixed: 1,580 nuatm and 3,667 nue2.

| Minimum hit probability | nuatm angle q68 / line q68 | nue2 angle q68 / line q68 |
| --- | ---: | ---: |
| 0.70 | 2.526 deg / 3.642 m | 3.577 deg / 7.336 m |
| 0.75 | 2.585 deg / 3.636 m | 3.697 deg / 7.432 m |
| 0.80 | 2.672 deg / 3.738 m | 3.939 deg / 7.588 m |
| 0.90 | 2.882 deg / 3.889 m | 5.627 deg / 8.716 m |

At p>=0.80 the paired 95% intervals for the q68 angle change versus
p>=0.70 are [+0.044, +0.222] degrees for nuatm and [+0.252, +0.498]
degrees for nue2; the line q68 increases too. Even p>=0.75 worsens nue2
angle q68 with interval [+0.041, +0.206] degrees. This is validation
evidence, not a new independent test. It nevertheless rules out adopting a
higher hit threshold globally merely because multi-muatm `prime_prty` and
the experimental theta distribution improve. **Keep p>=0.70 as the
universal production input cut.** Higher thresholds remain optional
muatm/domain diagnostics; their uncertainty radii have not been calibrated.

On cluster63, `run_threshold_sweep_v4.sh` and
`run_other_particles_thresholds_v4.sh` reproduce the inference/evaluation;
`summarize_threshold_sweep_v4.py`,
`summarize_other_particles_thresholds_v4.py`, and
`compare_domain_threshold_common_v4.py` generate the paired tables,
bootstrap intervals, and static figures in
`runs/mixed_universal/threshold_sweep_v4/`. These scripts use only the
single selected v4 checkpoint, never an ensemble. Compact snapshots of the
tables and two trade-off figures are committed under
[`results/threshold_sweep_v4/`](results/threshold_sweep_v4/).

## Physical-fit, score-feature, graph, and loss ablations after v4

These experiments retain the same leakage-free masks, `p>=0.70` input cut,
single-muatm training filter, and one-model CPU inference requirement. The
selected production model remains v4 unless an alternative improves the
independent predicted-hit test **and** survives the experiment-versus-muatm-MC
distribution audit. Experimental events have no truth direction.

`evaluate_spacetime_direction.py` compared two simple physical direction
estimates using raw positions in metres and times in nanoseconds: the
position/time covariance and a ridge time-gradient fit. The water photon
speed in the graph experiment is 0.225 m/ns. On the selected MC subsets,
covariance-fit versus v4 NN angle q68 was 25.29° versus 10.58° for muatm,
6.36° versus 3.48° for nuatm, and 22.42° versus 4.62° for nue2. The
time-gradient fit was worse still. Linear blending did not beat the NN, so
neither physical fit was deployed. The per-event mean signal probability
was close, but not identical, between hit/string-matched experiment and
muatm MC (`q50` 0.8876 versus 0.8947; W1 0.00679); the fraction of hits
below 0.8 showed a larger shift (`q50` 0.150 versus 0.118; W1 0.0310).

An optional six-feature model added each hit's signal probability to the
five physical inputs. It is initialized exactly from v4 by zeroing the new
input column. Full fine-tuning and a within-event-centered score feature
both failed the v4 validation gate. Training *only* the absolute-score
column found a slightly lower validation score (6.8399 to 6.8309), but on
26,229 independent predicted-hit event IDs its angle q68 changed only
4.782° to 4.778° (paired 95% interval for change
[-0.014°, +0.012°]) and transverse-line q68 5.212 to 5.216 m.
Hit/string-matched experiment–MC theta W1 was essentially unchanged,
3.0075° to 3.0066° (paired interval for change
[-0.0032°, +0.0016°]). It is not a production improvement.

`physics_graph_upgrade.py` adds identity-initialized local graph blocks
with spatial separation, light-cone timing, and a signed Minkowski edge
feature computed in physical units. With the base network frozen, neither
two layers/12 neighbours nor one layer/8 neighbours beat the starting
validation score 6.83945 after 2,000 steps. The approximate CPU cost on a
16-event benchmark increased from 2.6–3.0 ms/event for v4 to 8.5 ms/event
for the larger graph or 4.8 ms/event for the compact graph. No graph
checkpoint replaced v4.

The predicted-hit MAE angular loss was compared against a half-scaled
chord distance and a half-scaled 8° trimmed-angular loss, starting from
the same v4 checkpoint. Chord did not beat the starting validation score.
Trimmed-angular selected step 2,000 with score 6.826 versus 6.839, but its
independent predicted-hit angle q68 was 4.795° versus v4's 4.782°
(paired interval for the change [-0.022°, +0.049°]). GT nue2 angle q68
did improve 4.194° to 4.157° ([-0.048°, -0.030°]), showing that an MC-only
gain does not necessarily transfer to the working predicted-hit input.
On experiment versus muatm MC, matched theta W1 changed 3.007° to
3.042° (interval for change [-0.005°, +0.077°]); the direction of the
change is not reassuring. The v4 checkpoint remains selected.

`track_augmentation.py` adds training-only hit-time jitter and event-time
offsets in ns, coordinate jitter in metres, and optional event-consistent
z rotations. Coordinate jitter rebases the truth-line point when the hit
centroid moves; rotations transform both the direction and point target.
`test_track_augmentation.py` verifies these invariants. With the original
v4 loss balance, a 2,000-step run using 1 ns per-hit time noise, 5 ns
event-time offset, and 0.1 m coordinate noise did not beat the starting
validation score (6.83945; final 6.84216). Its final predicted-hit q68
was slightly lower, 4.469° to 4.451°, but GT nue2 q68 increased from
4.089° to 4.099°. A separate 25%-probability random z-rotation run
ended at score 6.86648 and predicted-hit q68 4.499°. Neither passed
checkpoint selection; there is no independent-test or experimental-
distribution claim for these augs. A pred-hit-focused controlled run is
the follow-up to test whether the small noise gain is reproducible.

Small numerical audit snapshots and the signal-probability plot are saved
under [`results/ablations_v4/`](results/ablations_v4/); H5 inputs and
large checkpoint files remain on cluster63.

Finally, `run_predfocus_v4_3k.sh` compared 3,000-step continuations with
20% GT-hit training weight and double the predicted-hit validation-score
weight, with and without the same mild noise. Both used identical masks
and seed and enforced GT-nue2 validation q68 <=4.2°. Neither beat the
starting score 9.07388. At step 3,000 the unnoised run had score 9.14571,
predicted-hit angle/line q68 4.495°/4.722 m, and GT angle q68 4.136°;
the noisy run had score 9.13669, 4.507°/4.726 m, and 4.119° respectively.
The common starting values were 4.469°/4.691 m and 4.089° on these
validation subsets. This rules out selecting a candidate merely by
reweighting MC loss toward the predicted-hit sample in this short sweep;
no independent test was needed because no checkpoint passed selection.

## Longer, low-learning-rate continuation of v4

`run_low_lr_v4_5k.sh` continues the selected v4 checkpoint for 5,000
steps with batch size 256, the same leakage-free masks, stochastic
predicted-hit threshold in [0.70, 0.90], and the single-muatm training
filter. The two runs used backbone/point learning rates of
`1e-6`/`2e-6` and `2e-6`/`4e-6`, respectively. Both selected a slightly
lower validation score than the step-0 value 6.83898 (6.82060 at step
3,000 and 6.81925 at step 2,500). However, predicted-hit validation
angle q68 increased from 4.4695° to 4.4956° and 4.4854° at those
selected steps, respectively.

`run_low_lr_audit_v4.sh` evaluated the validation-selected checkpoints
on the independent, leakage-free test. Paired bootstrap samples unique
event IDs, not hits. Lower is better in each column:

| Checkpoint | Pred-hit angle q68 | Pred-hit line q68 | GT nue2 angle q68 | GT nue2 line q68 |
| --- | ---: | ---: | ---: | ---: |
| v4 baseline | 4.782° | 5.212 m | 4.194° | 5.518 m |
| 1e-6 / 2e-6 | 4.776° | 5.195 m | 4.168° | 5.549 m |
| 2e-6 / 4e-6 | 4.786° | 5.204 m | 4.173° | 5.552 m |

For predicted-hit angle, paired 95% intervals for the change from v4
were [-0.035°, +0.018°] and [-0.028°, +0.032°]; neither establishes an
improvement. GT-angle gains were statistically clear, but GT-line
resolution worsened by roughly 0.03 m in both runs. On experiment versus
muatm MC after the same p>=0.70 selection and hit/string matching,
theta W1 was 3.0075° for v4, 3.0010° for the 1e-6 run, and 3.0222° for
the 2e-6 run. Paired 95% intervals for W1 changes were [-0.0194°,
+0.0080°] and [+0.0011°, +0.0315°], respectively. The higher-LR run
therefore slightly but measurably worsened the domain match. Experimental
events have no direction truth, so W1 is a distributional diagnostic,
not an accuracy estimate. **Neither continuation replaces v4.** The
scripts and compact validation/test/domain audit JSON are in
[`results/low_lr_v4/`](results/low_lr_v4/); full checkpoints stay on
cluster63.

## Angular Huber, joint graph, and point-loss balance after v4

These controlled continuations start from the same selected v4 model,
retain stochastic threshold thinning, use the same cross-dataset masks,
and keep multi-track muatm out of supervised training. The independent
test is used only for variants that showed a validation gain; it is not
used to select a checkpoint.

`run_huber_v4_3k.sh` replaces only the predicted-hit component MAE with
half-scaled angular Huber loss. The 4° and 8° transition runs both selected
step 2,000. Their predicted-hit validation q68 was 4.445° and 4.463°,
versus 4.469° for v4; however, on 26,229 unique independent predicted-hit
test IDs, angle q68 was 4.789° and 4.791°, versus v4's 4.782°. Paired 95%
intervals for the changes were [-0.024°, +0.039°] and
[-0.024°, +0.039°]. The 4° run's GT nue2 angle q68 improved only slightly,
4.194° to 4.186°. After hit/string matching, experiment–muatm-MC theta W1
was 2.998° and 3.012°, versus 3.007° for v4; both changes had intervals
crossing zero. Neither Huber candidate replaces v4.

`run_joint_graph_v4_3k.sh` fine-tunes one identity-initialized, 8-neighbour
local physics graph block *together with* the Transformer and point head.
One variant uses spatial, timing, and light-propagation edges; the other
adds a signed Minkowski feature in physical metres/nanoseconds before
normalization. Both started at score 6.83898 and remained worse throughout
3,000 steps. Best post-start scores were 6.85592 and 6.85642,
respectively. Neither produced a selected checkpoint, so no test or
experimental-distribution claim is made for them.

`run_point_balance_v4_3k.sh` changes only the relative weight of the
transverse-line loss from 0.5 to 0.2 or 0.8. The selected validation
steps had predicted-hit angle/line q68 of 4.479°/4.711 m and
4.507°/4.722 m, both worse than the v4 starting 4.469°/4.691 m. Small
clean-GT gains are therefore insufficient to promote either candidate;
they were not evaluated on the independent test. The runnable scripts
and compact audit JSON are in [`results/huber_v4/`](results/huber_v4/),
[`results/joint_graph_v4/`](results/joint_graph_v4/), and
[`results/point_weight_v4/`](results/point_weight_v4/).

## Additional clean-hit particle supervision after v4

`run_clean_nuatm_v4_3k.sh` adds an angle-only stream of clean-GT nuatm
events from the all-particle 2-string/8-hit sample. The training stream
contains 380,648 rows (379,999 unique event IDs), with no overlap with
predicted-hit validation or test IDs. We tried auxiliary loss weights 0.1
and 0.3, using the same v4 initialization and stochastic predicted-hit
threshold. Both selected step 1,500 on validation, but neither improved
the independent predicted-hit test. On 26,229 unique test IDs, angle q68
was 4.788° and 4.812° versus 4.782° for v4; line q68 was 5.240 m and
5.242 m versus 5.212 m. The 0.3-weight run also worsened the matched
experiment–muatm-MC theta W1 from 3.007° to 3.038° (paired 95% change
interval [+0.011°, +0.051°]). Angle-only clean-particle supervision is
therefore not a production replacement. The 0.1-weight W1 was 3.016°
and its change interval crossed zero.

`build_paired_clean_anchor_targets.py` creates *paired* clean-hit track
anchors without rescanning the large raw MC archive: it matches the
already validated true-particle track point by event ID, then expresses
the same truth line relative to the clean-hit centroid. Only nuatm and
single-track muatm with a valid particle track are used. The generated
`all_signal_paired_track_anchors_v1.h5` stays on cluster63, is about
88 MB, and has 310,209 valid training rows, 4,893 validation rows, and
16,711 test rows. A geometry check found transverse disagreement below
0.00002 m in sampled events. `--clean-paired-data`,
`--clean-paired-anchors`, and `--clean-paired-weight` add this joint
angle-and-line stream to `train_mixed_universal.py` without changing the
default v4 training behavior.

`run_clean_paired_v4_3k.sh` tests weights 0.1 and 0.3. The 0.1 run
selected step 1,000: predicted-hit validation angle/line q68 changed
from 4.469°/4.691 m to 4.460°/4.667 m; clean-GT nue2 angle changed
from 4.088° to 4.076°. The 0.3 run's selected validation angle was
worse than v4 (4.477°), so only the 0.1 run was audited independently.
On one row per unique event ID:

| Metric | v4 | Clean-paired 0.1 | Paired 95% interval for change |
| --- | ---: | ---: | ---: |
| Pred-hit angle q68 | 4.782° | 4.788° | [-0.024°, +0.037°] |
| Pred-hit line q68 | 5.212 m | 5.171 m | [-0.065, -0.014] m |
| GT nue2 angle q68 | 4.194° | 4.177° | [-0.026°, -0.009°] |
| GT nue2 line q68 | 5.518 m | 5.507 m | [-0.022, -0.001] m |

With the same p>=0.70 signal-hit selection and hit/string matching,
experiment–muatm-MC theta W1 changed from 3.007° to 2.975°; the
paired 95% interval for the change was [-0.053°, -0.008°]. This is a
small distributional improvement, **not** experimental truth validation.
The candidate preserves the single-model architecture and CPU cost, but
its principal predicted-hit angular q68 has not improved. Furthermore,
many closely related candidates have now been evaluated, so small test
gains should be treated cautiously. **The production v4 checkpoint is
unchanged.** If this line-focused candidate is promoted later, its
uncertainty head must be retrained and bundled with it; the v4 head
should not be reused uncalibrated. Scripts and compact audit JSON are
in [`results/clean_nuatm_v4/`](results/clean_nuatm_v4/) and
[`results/clean_paired_v4/`](results/clean_paired_v4/).

## Explicit space-time track-frame input

`diagnose_physical_direction.py` first tests analytic geometry on the
same leakage-free predicted-hit validation used for model selection.
An oriented PCA axis has angle q68 7.846° versus 4.469° for v4 on
5,406 events. It is better than the network on 20.7% of individual
events, so it may contain complementary information, but cannot replace
the network directly. The reverse PCA axis has a median error near
175°, confirming the time-based orientation convention. This is a
validation-only diagnostic, not a test result.

`physics_feature_upgrade.py` adds six per-hit features to the *same*
Transformer: longitudinal and transverse position relative to an
event-wise PCA axis in physical metres, a time residual in nanoseconds
relative to propagation along that axis, and the three oriented axis
components. The old embedding columns are copied exactly and the new
columns initialized to zero, so step 0 gives the v4 prediction. The
physical input layer is learned at 1e-4 while the rest of the backbone
uses 5e-6. This remains a single model, with no ensemble. In a small
cluster63 CPU benchmark (140 events, up to 77 hits, four threads, ten
timed forwards), median batch latency was 0.992 s for v4 and 1.017 s
for the upgraded model, about 2.5% more. This is not a full production
throughput measurement.

`run_physics_input_v4_3k.sh` trained the upgraded model for 3,000 steps
with the same stochastic p=0.70–0.90 hit selection, data splits, and
single-muatm filter as v4. The selected step-3,000 validation score
was 6.817 versus 6.839 at initialization; predicted-hit angle q68
was only slightly lower, 4.462° versus 4.469°, while clean-GT nue2
angle q68 was 4.068° versus 4.088°. The independent unique-ID test
does **not** support promotion:

| Metric | v4 | Physical-input model | Paired 95% interval for change |
| --- | ---: | ---: | ---: |
| Pred-hit angle q68 | 4.782° | 4.807° | [-0.003°, +0.055°] |
| Pred-hit line q68 | 5.212 m | 5.204 m | [-0.040, +0.017] m |
| GT nue2 angle q68 | 4.194° | 4.177° | [-0.025°, -0.009°] |
| GT nue2 line q68 | 5.518 m | 5.569 m | [+0.041, +0.061] m |

Experiment–muatm-MC theta W1 after the same p>=0.70 hit selection and
hit/string matching was 3.007° for v4 and 3.022° for the new model;
the paired 95% interval for the change was [-0.009°, +0.032°]. The
experimental sample has no truth direction, so this is only a
distributional diagnostic. **The production v4 model remains unchanged.**
Scripts and compact audit JSON are in
[`results/physics_input_v4/`](results/physics_input_v4/); full
checkpoints stay on cluster63.

## Raw-hit joint signal/angle/line pilot

The `joint_raw_*` scripts test a larger input/architecture change: one
Transformer reads **all raw hits, including noise**, predicts signal
probability per hit, and softly pools the contextual hit embeddings into
direction and line-point heads. The signal/noise trunk starts from the
existing classifier checkpoint; the geometric heads start from scratch.
At inference this is one network, not an ensemble or a separate hard
signal-selection pass. Two padding-mask variants were trained because
the classifier checkpoint used a float additive mask: `legacy` exactly
reproduces that behavior, and `hard` uses a corrected boolean mask.
Both use BF16 for the trunk and FP32 for probabilities/geometric heads.

`build_joint_raw_index.py` matches the existing p>=0.70, 2-string/
8-hit selected events back to the 50 GB raw MC file by unique event ID.
The matched training cohort contains 1,187,878 unique events, of which
60,581 are single-track muatm, 379,999 nuatm, and 747,298 nue2. All
26,044 overlaps with the established evaluation IDs were removed from
training; validation (5,396 IDs) and test (26,229 IDs) are disjoint.
This pilot changes the *input hits* but still uses the pre-existing
event acceptance cohort; it is not an experiment on all raw events.
Only unambiguous single-track muatm enter supervised training. The
domain audit uses the same fixed 4,924 experimental and 25,054 unique
muatm-MC events as the v4 p>=0.70 comparison, with matching by selected
hit/string counts. `build_joint_raw_domain_index.py` maps those events
to their raw-hit rows.

The two 5,000-step runs selected validation angle/line q68 of
6.931°/6.195 m (`legacy`, step 4,500) and 6.875°/6.093 m (`hard`, step
5,000). The corrected-mask model was continued to 10,000 total steps,
selecting step 9,000: angle q50/q68 4.045°/6.126°, line q50/q68
4.127/5.813 m. On the closely matched v4 validation cohort, v4 has
angle q50/q68 2.611°/4.469° and line q50/q68 3.228/4.691 m. The
joint model remains substantially worse for every particle type, so
it was **not** promoted and no independent test result is claimed.

At 5,000 steps the raw model reduced hit/string-matched experimental
versus muatm-MC theta W1 from v4's 3.007° to 2.197° (paired 95%
interval for the change [-1.057°, -0.563°]). That advantage did not
survive longer training: the selected 9,000-step checkpoint has W1
3.012°, with change interval [-0.195°, +0.207°]. Thus the apparently
better domain alignment of the undertrained model traded away MC truth
accuracy and was not robust. A later, stricter audit also matches
observed vertical hit spread: on the *same* paired events, v4's
geometry-conditional W1 is 0.364°, whereas the 5,000-step raw model's
is 1.613° (paired 95% change interval [+0.745°, +1.379°]) and the
selected 9,000-step model's is 0.954° ([+0.219°, +0.734°]). The
apparent early marginal improvement therefore hid a deterioration
within geometry-matched events. Experimental data have no truth angles;
neither W1 is an accuracy measurement. The current v4 production
checkpoint and its calibrated uncertainty heads remain unchanged.

Commands are in `run_joint_raw_5k.sh`,
`run_joint_raw_hard_continue_10k.sh`, and `run_joint_raw_domain.sh`.
The large raw files, ID indices, and checkpoints remain on cluster63.
Compact audit JSON is in [`results/joint_raw_v1/`](results/joint_raw_v1/).

## Input-composition audit and equivariant pair-vote pilot

`audit_input_domain_shift.py` compares event-level physical hit summaries
on the fixed p>=0.70, 2s8h experimental (4,924 events) and unique
muatm-MC validation (25,054 events) cohorts. It checks each CSV event ID
against its HDF5 source row. After the established hit/string-count
matching, the experiment has shorter vertical hit spread (median
40.81 versus 47.53 m) and time spread (178.46 versus 193.36 ns), and
slightly lower mean signal probability (0.8876 versus 0.8947).
These are **observed input differences**, not proof of detector or model
error; the true particle composition of experimental events is unknown.

`match_domain_covariates.py` checks whether the theta-distribution
comparison is sensitive to that composition. With only hit/string
matching, v4 experiment–muatm-MC theta W1 is 3.007°. Also matching by
three coarse bins of *observed vertical hit spread* reduces it to
0.364°. All experimental events have matching MC support and effective
MC sample size remains 21,071 of 25,054. The paired-bootstrap 95%
interval for the W1 change is [-2.997°, -2.072°]. Within the three
vertical-spread bins, hit/string-matched theta W1 is 0.553°, 0.709°,
and 1.081°: the marginal 0.364° value should not be mistaken for zero
conditional shift. The result strongly indicates that the original
3° **marginal** theta difference is confounded by event geometry.
It does not establish experimental angular accuracy or justify forcing
the network's output distribution to match unweighted MC.
`compare_domain_models_paired.py` now accepts `--exp-input-features`
and `--mc-input-features` to report both the original hit/string-matched
W1 and the vertical-spread-conditioned W1 for any candidate on the
same event IDs, with paired bootstrap intervals. The raw-hit pilot's
re-audit JSON is in [`results/joint_raw_v1/`](results/joint_raw_v1/).

`--center-time-alpha` is an optional *diagnostic* input transform in
`infer_rethresholded_for_audit.py`; the default is zero and preserves
v4 behavior. On every fourth event of the fixed cohort (1,231
experimental and 6,493 unique muatm MC), full time centering changed
experimental directions by only 0.23° median. Overall muatm-MC angle
q68 changed from 10.958° to 11.017° and matched theta W1 from 2.703°
to 2.729°. A small single-track subset improved, but this did not
outweigh the all-muatm and domain results. No preprocessing change was
promoted.

`pair_vote_residual.py` tests a distinct architecture from the earlier
hidden-state graph blocks: at most 32 time-quantile hits form directed
pairs; a small network uses physical distances, time residuals, charge,
and alignment with v4's direction to cast azimuth-equivariant vector
votes. Its final layer is zero-initialized, so step 0 exactly equals
v4. The v4 backbone and point head are frozen, and only one residual
head is trained; this is a single model, not an ensemble. Unit tests
check identity initialization, gradients, time-translation invariance,
and z-rotation equivariance of the vote. In a 3,000-step pilot with
the same leakage-free masks and single-muatm training filter, the
best predeclared validation score remained at **step 0**. The
post-start step-2,500 predicted-hit angle q68 was 4.4677° versus
4.4689° at start, but GT nue2 q68 and the combined score worsened;
step 3,000 was 4.4876°. This sub-0.002° validation-only fluctuation
does not warrant independent-test or experimental-domain evaluation.
v4 and its uncertainty calibration remain the selected production model.
Compact results and the theta overlay are in
[`results/domain_input_shift_v2/`](results/domain_input_shift_v2/) and
[`results/pair_vote_v4/`](results/pair_vote_v4/).

## Vertical-span crop on predicted hits

`span_crop_augmentation.py` implements a training-only, event-wise trim of
the lowest or highest selected hits in physical z. It preserves at least
eight hits on at least two strings and recomputes the line-point target
relative to the new hit centroid; the truth line is unchanged. This tests
whether the shorter vertical extent seen in experimental data can be
simulated without changing the production inference interface. It is
applied only to the predicted-hit training stream, after stochastic
signal-probability thresholding, never to validation/test or the clean
GT-hit stream. The `--span-crop-*` defaults are disabled, so older runs
and the v4 bundle are unchanged.

On 25,972 sampled muatm-MC validation rows (including repeated event IDs),
probability 0.5 and a retained-hit fraction of 0.65--0.85 applied a crop
to 33.0% of events. Median z spread decreased from 47.07 to 42.38 m;
median selected hits decreased from 12 to 11. This is only an input
composition check, not a domain or accuracy improvement.

`run_span_crop_v4_3k.sh` ran a same-seed no-crop control and the crop
variant for 3,000 steps from exactly the same v4 checkpoint, with the
same leakage-free split, single-track-muatm filter, and validation rule.
Both selected **step 0** (score 6.8395). The best post-start crop score
was 6.8518 at step 2,000: predicted-hit angle/line q68 4.4815°/4.7024 m
and GT nue2 angle q68 4.0874°, versus v4's 4.4689°/4.6906 m and
4.0886°. The best post-start control score was 6.8645 at step 2,500:
4.5008°/4.7319 m and 4.1003°. These differences are validation-only;
neither model warrants independent-test or experimental-domain inference.
The unchanged v4 bundle remains selected. The compact validation table
and input-composition audit are in
[`results/span_crop_v4/`](results/span_crop_v4/).

## What the large-error events look like

`audit_large_angle_errors.py` runs the unchanged v4 checkpoint on the
leakage-free predicted-hit **test** mask, filters multi-track `muatm` as in
training, and joins each output to MC-only hit labels and `t_res` for
diagnosis. It keeps the normal `p>=0.70`, 2-string/8-hit input. There are
27,038 selected rows but only **26,229 unique event IDs**; all analyses
below count each ID once. Direction q50/q68/q95 is
2.762°/4.783°/22.902°. Large error means angle >=20°, and occurs in
1,576 events (6.0%): 533/3,435 single-track `muatm`, 288/14,121
`nuatm`, and 755/8,673 `nue2`.

The strongest observable correlate is the maximum 3D separation of the
selected hits. Of 2,524 events with span <100 m, 749 (29.7%) have >=20°
error. Of 10,490 with span >=200 m, only 32 (0.31%) do. The trend holds
within each particle type: for `muatm` 36.5% versus 0.26%, `nuatm` 21.3%
versus 0.17%, and `nue2` 28.3% versus 0.86%. This is an association, not
proof that artificially increasing span would fix the same events.
Nearly half (785/1,576) of the large-error cases have exactly eight hits.

Hit composition matters even at similar span. MC origin labels below zero
are common in `nuatm` and rare in `nue2`, consistent with track-like versus
cascade-like hits; the exact sign ontology comes from the external MC mask.
Within single-track `muatm` at 100--150 m span, events with <25% negative
labels have 201/1,015 (19.8%) large errors, versus 16/364 (4.4%) when
25--75% of labels are negative. For all `muatm`, 490/2,091 (23.4%) of the
low-negative-label group are large errors, versus 0/273 in the >=75%
negative-label group. The analogous pattern is present in `nuatm` and
`nue2`; the [risk plot](results/large_error_v4/risk_by_span_and_hit_label.png)
shows the per-particle rates and sample sizes are in `summary.json`.

Some extreme failures are mostly azimuthal ambiguity. Among the 106
events with angle error >=90°, median absolute theta error is only 8.4°,
100/106 have span <120 m, and 61/106 have exactly eight selected hits.
For 105 with sufficiently nonvertical directions, median absolute phi
error is 135°. These events often have nearly horizontal truth directions;
the sparse/compact hit pattern can support a badly wrong azimuth even when
theta is close.

False-positive signal/noise hits contribute, but are not the whole story.
At least one selected hit has MC label zero in 5,394/26,229 events. Their
>=20° error rate is 710/5,394 (13.2%), versus 866/20,835 (4.2%) when
none have label zero. Thus 55% of the large-error events have **no**
selected label-zero hit. `oracle_remove_noise.py` removes label-zero hits
using forbidden-at-inference MC truth, only if the resulting event still
passes 2 strings/8 hits. Just 3,352 events remain eligible. This oracle
changes unique-test q68 4.783° to 4.674° (paired event-bootstrap 95%
interval for the change [-0.135°, -0.075°]) and the >=20° count 1,576 to
1,492 (112 rescued, 28 newly above threshold). It is an upper-bound
diagnostic for the signal/noise stage, **not** a deployable reconstruction
metric. Most severe failures therefore require better handling of short,
cascade-heavy topology, not only a tighter `p` threshold.

`audit_photon_time_model.py` verifies that a Cherenkov arrival-time formula
in metres/nanoseconds is informative at the true MC line: median event-wise
absolute residual is 4.2 ns for single-track `muatm`, 1.4 ns for `nuatm`,
and 2.8 ns for `nue2` at effective refractive index 1.36, versus 132,
345, and 137 ns after reversing the true direction (the latter numbers
use index 1.33). Yet the validation-only `fit_photon_track.py` nonlinear
fit from v4 worsens overall angular q68 from 4.469° to 4.711° with a
5° direction prior; stronger 2° and 1° priors give 4.530° and 4.470°.
The 2° prior improves line q68 4.691 to 4.546 m and `nuatm` angle q68
3.477° to 3.322°, but worsens `muatm` angle 10.584° to 11.700°.
No direct physics fit is promoted. `photon_residual_upgrade.py` and
`train_photon_residual.py` implement a zero-initialized, single-model learned
correction from per-hit physical time residuals. A 3,000-step run selected
step 3,000 on the predeclared validation score. On the independent
predicted-hit test, after deduplicating 27,038 rows to 26,229 event IDs,
angle q50/q68 changes from **2.762°/4.783° to 2.625°/4.707°**, and
transverse line q68 from **5.213 to 4.764 m**. The paired-bootstrap 95%
interval for the angle q68 change is [-0.104°, -0.048°]. By particle,
angle q68 changes from 10.041° to 9.989° (`muatm`), 3.487° to 3.425°
(`nuatm`), and 6.065° to 5.991° (`nue2`).

This correction **does not fix the severe tail**: error >=20° occurs in
1,576 events for v4 and 1,585 for the correction (13 rescued, 22 newly
severe). At hit span <100 m, angle q68 is 18.832° versus 18.852°.
An observed physical diagnostic is still useful: among events with span
<150 m, >=20° errors occur in 3.8% when the median absolute photon-time
residual to the predicted line is <2 ns, versus 28.4% at 16--32 ns and
34.3% above 32 ns. This is a test-set association, not a fitted or
externally validated quality cut.

`infer_rethresholded_for_audit.py --photon-residual-checkpoint` evaluates
the same correction on experimental and muatm-MC events after the same
signal/noise threshold and 2-string/8-hit cut. For p>=0.70, the
hit/string-matched theta Wasserstein distance is 3.007° before and
2.975° after. Adding observed z-spread matching gives 0.364° before
and 0.368° after; the paired-bootstrap interval for this change includes
zero. Thus no evidence of a material theta-domain shift was found.
Compact test, validation, and domain results, including a before/after
theta plot, are in [`results/photon_residual_v4/`](results/photon_residual_v4/).
The experimental checkpoint remains at
`track_anchor/runs/physical_time_audit/photon_residual_3k/best.pt` on
cluster63. It has **not** replaced the production v4 bundle, whose
uncertainty heads have not been recalibrated for this direction/point
correction. Experimental events have no truth hit labels or directions,
and may include multiple `muatm` tracks; MC error rates above must not
be presented as experimental resolution.

## Short-topology follow-up and exact reco-test check (2026-10-13)

`audit_direction_hypotheses.py` and `audit_refit_hypotheses.py` show on
the leak-free validation split that choosing among fixed azimuth rotations
by photon-time residuals is worse than the original direction. Even an
oracle choosing the best of four azimuths only changes short-span (<120 m)
q68 from 17.024° to 15.044°. We do not use a multi-hypothesis selector.

`train_short_topology.py` fine-tunes v4 with 4x smoothly weighted loss for
short real-hit-span events (still using predicted-hit thinning, a clean GT
stream, and single-track `muatm`). Its validation improvement is small.
`interpolate_short_topology.py` finds no weight interpolation meeting a
predeclared overall-q68 and short-span gate. `replace_bundle_base.py`
creates a **training-only** bundle containing the short-tuned base; its
uncertainty heads are stale and must not be deployed. `train_photon_residual.py`
then trains a 3k-step physical-time correction on that base. On the
deduplicated 26,229-event internal test, v4+photon and short+photon have
angular q68 4.707° and 4.705° respectively; the paired-bootstrap interval
for the latter-minus-former q68 difference is [-0.034°, +0.032°]. The
short+photon >=20° count is 1,559 versus 1,585 for v4+photon, but the
short-span q68 improvement remains statistically uncertain. Thus this
fine-tune is not established as a meaningful overall advance.

For an exact external check, use the historical reco-normalized input
`/home/plotnikovgp/baikal/data/baikal_reco_reco_h8s2_norm.h5` on cluster63,
**not** the similarly named `baikal_reco_h8s2_norm.h5` on cluster62.
`build_predicted_signal_dataset.py --splits test` runs the frozen
`noise_sig_tres_abs_merged` classifier, keeps p>=0.70 hits, then enforces
>=8 selected hits on >=2 unique strings. It yields 58,100 events, of
which exactly 57,316 overlap the corrected fair all-GT-signal reco-test
file. `infer_rethresholded_for_audit.py --device cuda` evaluates the v4,
v4+photon and short+photon direction models. `evaluate_reco_paired_current.py`
joins on exact event IDs, computes vector/angular errors in degrees, and
applies the official cut to all methods identically:
`fCovMatrixStatus == 3`, `fInitRecDelta < 20`, and
`fFuncValue / (fNHits - 5) < 30`. Its standard-reco no-cut q68 exactly
reproduces the historical 9.620°, confirming source and angle conventions.

| Reco-test selection | N | Standard reco q50/q68 | v4 q50/q68 | v4+photon q50/q68 | short+photon q50/q68 |
|---|---:|---:|---:|---:|---:|
| No official cut | 57,316 | 2.833°/9.620° | 3.068°/6.014° | **2.877°/5.943°** | 2.873°/5.953° |
| Official cut | 54,424 | 2.568°/8.388° | 2.935°/5.671° | **2.756°/5.549°** | 2.738°/5.567° |

The paired-event 95% bootstrap interval for v4+photon minus standard
reco q68 is [-3.90°, -3.47°] without the official cut and
[-3.06°, -2.66°] with it. At q50 the standard reco is slightly better:
v4+photon minus reco is +0.044° (interval crosses zero) without the cut
and +0.189° with it. The short fine-tune's q68 change against v4+photon
is not resolved on this external set (CI [-0.022°, +0.045°] no cut;
[-0.006°, +0.050°] official cut). At GT-signal span >=450 m, with the
official cut, standard reco q50/q68 is 1.90°/6.97° versus v4+photon
2.30°/4.92° (N=5,023). This is a *minimum hit-span* selection, not
the reconstructed path length and not a disjoint length bin.

The experimental-versus-all-`muatm` theta check does not establish
a material domain improvement for short+photon over v4+photon:
hit/string-matched W1 2.975° -> 2.951° (paired-bootstrap delta CI includes
zero); also matching observed z-spread gives 0.368° -> 0.408° (CI includes
zero). Experimental data have no direction truth.

Reproducible metric tables, paired-bootstrap output and plots are in
[`results/reco_paired_current/`](results/reco_paired_current/). The two
photon-correction checkpoints are experimental; the production v4
uncertainty heads are **not** recalibrated for either correction.

## Paired-threshold photon-head follow-up (2026-10-13)

`train_photon_residual.py` can now resume a checkpoint, optionally unfreeze
the two base output heads (the encoder stays frozen), and train on paired
predicted-hit views at p>=0.70 and p>=0.90. The first resumed runs revealed
an optimizer bug: the hit-head parameter iterator was consumed while making
the gradient-clipping list, so the optimizer received no hit-head parameters.
`optimizer_groups` now materializes the parameters; an update test verifies
that an optimizer step changes the intended weights. Results from the runs
before this fix are invalid as tests of joint/head training.

A fixed 3,000-step head-only continuation with both thresholds and a 0.2
direction-consistency loss reaches validation predicted-hit q68 4.343° and
clean-GT q68 3.945°. On 26,229 unique independent internal test IDs, its
direction q50/q68 is 2.612°/4.701° versus 2.625°/4.707° for the original
v4+photon model. The paired-bootstrap 95% interval for the q68 change is
[-0.022°, +0.007°]; hence the gain is **not established**. Severe >=20°
errors change from 1,585 to 1,580. On the exact 57,316-event external reco
test, no official cut gives q50/q68 2.862°/5.923° versus the original
2.877°/5.943°; with the official cut, 2.738°/5.542° versus
2.756°/5.549°. The latter q68 change also has a bootstrap interval
crossing zero.

At p>=0.70, 4,924 experimental predictions change direction by q50/q68
0.109°/0.146° between the original and continued models. After matching
25,054 unique `muatm` MC predictions to experiment by hit/string counts,
theta W1 is 2.975° before and 2.956° after; additionally matching observed
vertical hit spread gives 0.368° and 0.369° (change interval includes zero).
These are distributional checks, not experimental resolution measurements.
The 3,000-step candidate is not promoted to the production bundle, and its
uncertainty heads remain stale.

A validation-only sweep of the original physical correction scale over
0.00, 0.15, 0.30, 0.45, 0.60 found the existing 0.30 best by the
predeclared score and predicted-hit q68 (4.350°); 0.45 slightly lowers
q50 but worsens q68 to 4.385°. No scale change is adopted.
Configurations and summary metrics are in
[`results/photon_head_followup_v1/`](results/photon_head_followup_v1/).
The candidate checkpoint is
`track_anchor/runs/physical_time_audit/photon_head_pair_fixed_3k/best.pt`
on cluster63.

## Reco preliminary cuts from fillTrackAnalysisTree_v1.0 (2026-10-13)

The supplied `fillTrackAnalysisTree_v1.0.cc` defines 13 active preliminary
conditions in `fPrelCutsPassed`; its commented-out `fThetaRec < 80` is **not**
an active condition. The external nue2 reco HDF5 stores only a few of these
fields, so `extract_reco_preselection_fields.py` joins the exact 57,316
paired test event IDs back to 50 source ROOT files. ROOT/HDF5 hit counts,
time fit, and angles match after accounting for the HDF5 integer truncation;
ROOT angles exactly match the independent reco-quality sidecar.

Ten conditions can be evaluated directly from ROOT scalars. The remaining
three—`fLogPHit_30_min > -9`, `fQacc_20/fQtot_20 > 0.05`, and
`fTRes_30_med < 60`—are generated by `BRecoMuonBDTEval` and are absent from
these source files. Therefore the measured selection is a **10-of-13
superset**, not the full `fPrelCutsPassed` cut. `evaluate_reco_preselection_partial.py`
applies the exact available inequalities to the paired event set:

| Selection | N | Standard reco direction q50/q68 | v4+photon | Paired-threshold candidate |
|---|---:|---:|---:|---:|
| ROOT-available 10/13 | 44,189 | 2.031°/5.730° | 2.342°/4.468° | 2.327°/4.450° |
| 10/13 plus official reco cut | 43,736 | 1.991°/5.524° | 2.314°/4.395° | 2.301°/4.374° |

On the 10/13 selection, candidate-minus-standard q68 is -1.280° with a
paired-bootstrap 95% interval [-1.418°, -1.126°]. Standard reco remains
better at q50; the NN is better at q68. The missing three conditions could
change both sample size and ranking. The supplied program's example file
lists are for 2023, whereas this external comparison uses 2020 reco ROOT
files; its preselection should not be called the same as the older official
three-condition reco cut. The raw ROOT contains the required event, geometry,
mask, and reconstructed-track objects, so the missing conditions are
calculable in principle. However, the BARS source/build found on cluster62
has an older `BRecoMuonBDTEval` API without the three new getters; the local
ROOT interpreter also fails to start with that installation. A compatible
evaluator build or a validated reimplementation is required for exact 13/13
cuts. Full per-condition counts and
metrics are in
[`results/reco_preliminary_partial/`](results/reco_preliminary_partial/).

## Corrected joint photon/base-head training (2026-10-13)

After fixing the exhausted-optimizer-iterator bug noted above, two actual
joint continuations were run from the original v4+photon checkpoint. Both
updated the photon residual head and the direction/point output heads while
keeping the Transformer encoder frozen. The first used hit/base learning
rates 2e-5/2e-6, 40% clean-GT loss weight, and 4,000 steps. The second
changed only the clean-GT weight to 10% and ran 3,000 steps. The same
leak-free masks, predicted-hit thinning, and single-track-muatm training
filter were used.

| Validation | Start | 40%-GT joint, step 4,000 | 10%-GT joint, step 3,000 |
|---|---:|---:|---:|
| Predicted-hit angle q68 | 4.351° | 4.394° | 4.383° |
| Clean-GT nue2 angle q68 | 3.954° | 3.924° | 3.947° |
| Predeclared validation score | 6.629 | 6.616 | 6.640 |

Every post-start checkpoint in both runs failed the predicted-hit q68
non-regression gate. The clean-GT gain in the 40%-GT run does not outweigh
the working-input degradation. Neither run produced an admissible
checkpoint; no independent-test or experimental-data comparison is claimed,
and the original production model remains unchanged. Configurations and
per-step validation metrics are in
[`results/photon_joint_fixed_v1/`](results/photon_joint_fixed_v1/).

## Primary-overlap audit and independent 2019 holdout (2026-10-13)

`audit_primary_overlap.py` compares the exact float32
`(primary theta, primary phi, primary energy)` triple. A match establishes a
shared **generated primary**, not necessarily an identical detector event:
ten sampled cross-dataset pairs showed almost no identical
`(channel, charge)` hit signatures. Nevertheless, file/ID-disjoint splits
are not primary-disjoint:

| Comparison | Same-primary fraction |
|---|---:|
| Reco val / test versus reco train | 94.48% / 94.77% |
| Reco test versus internal 2020 predicted-hit + GT train | 97.74% |
| Internal clean-GT val / test versus clean-GT train | 91.15% / 91.19% |

Only **329 of 57,316** paired 2020 reco-test events have a primary absent
from the internal 2020 train union. On those 329 events, paired-threshold NN
direction q50/q68 is 3.479°/6.916° versus standard reco 3.672°/9.941°;
the paired event-bootstrap 95% interval for NN-minus-reco q68 is
[-7.30°, -1.21°]. This tiny subset cannot establish a precise independent
reco-MC ranking. All earlier full reco-test numbers remain valid descriptive
measurements but should **not** be described as primary-independent tests.
The complete count audit is in
[`results/primary_overlap_v1/`](results/primary_overlap_v1/).

An exploratory reco-train auxiliary adaptation used the same frozen hit
classifier, p>=0.70 and 2-string/8-hit cut, retaining 469,625 train and
58,663 val events. Only the final photon-head direction rows were trainable.
At step 750, reco-val q68 changed 6.002° -> 5.982°, while independent
internal predicted-hit validation q68 worsened 4.343° -> 4.415°; the
original run saved no checkpoint. After the primary audit,
`train_reco_aux_direction.py` was tightened to log primary-disjoint controls
and require at least 1,000 primary-unseen reco-val events before promotion;
this dataset has only **631** after hit selection, so no future candidate
from this split can pass that gate. Neither the exploratory run nor its
reco-val score is a basis for deployment. This 631-event slice has baseline
q68 24.7° and is highly selection-biased, not a representative replacement
for the full reco-val. The stricter primary-unseen internal GT-val control
contains 3,385 events and baseline q68 4.567° versus 3.945° on the original
47,792-event ID-masked control; these also have different event mixes.
The exploratory run settings and key validation values are archived in
[`results/reco_aux_v1/`](results/reco_aux_v1/).

For a stronger out-of-season check, `extract_independent_2019_raw.py`
selected the first 5,000 raw nue2 events from each of 20 evenly spaced
2019 source parts, without inspecting labels. The resulting 100,000-event
sample has **zero** matching generated-primary triples with the 2020 train
union, and zero exact event-ID matches with the hit-classifier's train/val/test
files. The same frozen hit classifier at p>=0.70 retained 51,473 events,
with selected-hit precision 99.52% and signal recall 97.13% within those
events. `audit_2019_holdout.py` measures the unchanged v4 and paired-photon
direction models:

| 2019 predicted-hit selection | N | v4 q50/q68 | v4 + paired photon q50/q68 |
|---|---:|---:|---:|
| p>=0.70, >=8 hits/2 strings | 51,473 | 4.285°/7.877° | 4.178°/7.844° |
| Plus GT-signal >=8 hits/2 strings | 51,114 | 4.252°/7.790° | 4.148°/7.761° |

The photon-minus-v4 q68 change on all 2019 events is -0.033°; its paired
event-bootstrap interval is [-0.062°, +0.002°], so this is not a firm
out-of-season gain. The apparent seasonal gap versus the 2020 paired reco
set (q68 5.923°) is strongly confounded by topology: median GT-signal hit
span is **139 m** in 2019 versus **174 m** in 2020, and 19.0% versus 2.6%
of events have span <100 m. Reweighting the 2019 errors to the 2020 span
distribution gives q68 **5.189°**; matching both span and true-theta bins
gives **5.244°** (98.1% target-mass support). These are descriptive
standardizations, not proof of causal equivalence or a replacement for a
new primary-disjoint 2020 test. They do not support claiming severe
2020-specific overfit from the raw seasonal q68 gap alone. Full metrics and
span-bin counts are in
[`results/2019_holdout_v1/`](results/2019_holdout_v1/).

## Conditional lower hit threshold (2026-10-13)

The frozen signal/noise classifier at p>=0.50 supplies useful hits that a
p>=0.70 cut misses for clean upward-going events. Merely lowering the global
threshold is **not** safe: on 109,897 deduplicated, ID-masked 2020 MC-test
events it changes direction q68 from 9.516° to 9.822°, driven by multi-track
`muatm`, and worsens the hit/string-matched experimental-vs-`muatm` theta W1
from 3.062° to 3.895° on the same accepted events.

`infer_rethresholded_for_audit.py` therefore supports an optional two-pass
*single-checkpoint* rule: require the original p>=0.70 two-string/eight-hit
cut, predict once, and rerun at p>=0.50 only if the first-pass theta is below
90° **and** the lower threshold adds at least one hit. No event that fails
the original cut is newly accepted. The script validates that the prepared
HDF5 actually contains hits down to the requested threshold and records
`selected_threshold` per output row. The optional second pass uses the
same model weights, not an ensemble; on 4,924 experimental events it was
needed only 26 times. A 512-raw-event CPU smoke on upward `nue2` took 10.6 s
versus 8.4 s for a single pass on the same host; this is not a production
throughput benchmark.

The first 2019 holdout showed a gated q68 reduction from 7.844° to 6.986°.
To avoid selecting on that already inspected sample,
`extract_independent_2019_raw.py --part-shift 2 --exclude-manifest ...` made
a *second* 100,000-event sample from 20 different source parts. It has zero
exact event-ID or generated-primary overlap with the first 2019 sample and
zero generated-primary overlap with the 2020 train union. On its 51,360
common p>=0.70-accepted events, the unchanged direction model changes
q50/q68 from **4.138°/7.871° to 3.725°/6.999°**. The paired q68 change is
-0.872°; 95% source-part bootstrap interval **[-0.940°, -0.793°]**.

| Paired evaluation | N | p>=0.70 q50/q68 | Conditional q50/q68 |
|---|---:|---:|---:|
| 2020 internal all, ID-masked/deduplicated | 109,897 | 6.112°/9.516° | 6.082°/9.477° |
| 2020 internal `nue2` | 8,673 | 3.177°/5.984° | 2.895°/5.417° |
| 2020 internal `nuatm` | 14,121 | 1.994°/3.419° | 1.921°/3.310° |
| 2020 internal `muatm` | 87,103 | 7.202°/10.614° | 7.198°/10.604° |
| External reco-2020 `nue2`, no official reco cut | 57,316 | 2.862°/5.923° | 2.867°/5.950° |
| External reco-2020 `nue2`, official reco cut | 54,424 | 2.738°/5.543° | 2.743°/5.560° |

The external reco deterioration is small but statistically resolved: its
q68 change has paired event-bootstrap 95% intervals [+0.007°, +0.043°]
without and [+0.005°, +0.034°] with the official cut. Both NN variants
still beat standard reco q68 on those same events (9.620°/8.388°). The
likely input reason is measurable: among events common to p>=0.70, hits
with 0.50<=p<0.70 occur in 42.6% of first-holdout-2019 `nue2` events
and 58.5% of internal-2020 `nue2` events, with truth-signal purities
79.5% and 72.7%; on the external reco set they occur in only 3.0% of
events and their purity is 32.5%. This is a dataset-composition
association, not an experimental truth measurement.

For the same 4,924 experimental events and 87,103 unique `muatm` MC events,
hit/string-matched theta W1 changes from 3.062° to 3.034°; also matching
observed vertical hit spread gives 0.375° to 0.374°. Experimental truth is
unknown, so this only checks that the new rule does not visibly distort
the measured theta distribution. On 26,229 MC events with a valid truth
line, transverse line q68 improves from 4.755 to 4.720 m, but q95 worsens
from 17.62 to 18.59 m. The direction/point correction and conditional
input rule have **not** been recalibrated with the uncertainty heads.
This mode is available for further evaluation, **not promoted as the
default production pipeline**. Detailed counts are in
[`results/conditional_threshold_v1/`](results/conditional_threshold_v1/).

CPU example using a p>=0.50-prepared HDF5 and the same v4/photon weights:

```bash
python infer_rethresholded_for_audit.py \
  --bundle runs/mixed_universal/universal_angle_track_uncertainty_v4.pt \
  --photon-residual-checkpoint runs/physical_time_audit/photon_head_pair_fixed_3k/best.pt \
  --data /path/to/predsignal_p050.h5 --split test \
  --threshold 0.70 --conditional-low-threshold 0.50 \
  --gate-theta-max-deg 90 --device cpu --output /path/to/predictions.csv
```

## Added-hit photon-time gate (2026-10-13)

The preceding theta-only conditional rule improves independent `nue2` but
slightly hurts the external reco sample. An optional, more conservative
`--max-added-photon-residual-ns 50` gate now checks **every** newly admitted
hit against the first-pass track. It computes the idealized Cherenkov arrival
time using positions in metres, times in ns, refractive index 1.36, and
`c=0.299792458 m/ns`. The unknown event clock is estimated by the median
residual of the original p>=0.70 hits. The same checkpoint is rerun at
p>=0.50 only when first-pass theta is <90°, at least one hit is added, and
all added hits are within 50 ns of that clock. The p>=0.70 two-string/eight-hit
acceptance remains fixed; this is **not** an ensemble. This option does not
recalibrate the uncertainty outputs.

The 50 ns criterion was explored on the first two disjoint 2019 samples and
the external reco comparison. A third, previously unused 100,000-event 2019
`nue2` sample from 20 different source parts (`part-shift 4`) was then
prepared with the unchanged frozen hit classifier. It has zero exact event-ID
or generated-primary overlap with the first two samples and zero
generated-primary overlap with the 2020 internal train union. On the same
51,386 p>=0.70-accepted events, actual inference gives direction
q50/q68/q95 **4.084°/7.743°/39.610°** at baseline and
**3.855°/7.447°/39.062°** with the 50 ns gate. The paired q68 reduction is
0.296°; its 95% source-part bootstrap interval is **[0.244°, 0.349°]**.
This is independent evidence for the `nue2` gain, not a new model weight.

| Paired cohort | N | Baseline q50/q68 | 50 ns gate q50/q68 |
|---|---:|---:|---:|
| 2020 internal, ID-masked/deduplicated all | 109,897 | 6.112°/9.516° | 6.090°/9.497° |
| 2020 internal `nue2` | 8,673 | 3.177°/5.984° | 2.970°/5.628° |
| 2020 internal `nuatm` | 14,121 | 1.994°/3.419° | 1.900°/3.284° |
| 2020 internal `muatm` | 87,103 | 7.202°/10.613° | 7.202°/10.612° |
| External reco `nue2`, paired fair sample | 57,316 | 2.862°/5.923° | 2.861°/5.916° |
| External reco `nue2`, official reco cut | 54,424 | 2.738°/5.543° | 2.738°/5.537° |
| External reco, ROOT-available 10/13 preliminary cuts | 44,189 | 2.327°/4.450° | 2.327°/4.452° |

The external reco q68 changes are **not** statistically resolved: paired
event-bootstrap 95% intervals are [-0.020°, +0.0001°] without and
[-0.014°, +0.005°] with the older official reco cut. The 10/13 preliminary
selection is only a superset of the supplied program's full 13-condition
cut. These 2020 comparisons are descriptive because most generated
primaries overlap training. On 26,229 MC events with a valid truth line,
transverse anchor q68 improves 4.755 -> 4.656 m, while q95 rises
17.625 -> 17.849 m; the tail needs further attention.

On the same 4,924 experimental events and 87,103 unique `muatm` MC events,
only two experimental and 55 `muatm` predictions rerun. With matching
weights fixed from the baseline observed hits/strings, theta W1 changes
3.062° -> 3.065°; additionally matching vertical hit spread gives
0.375° -> 0.376°. These are effectively unchanged distributional checks,
**not** experimental angular-resolution measurements. The saved
[`theta_distribution.png`](results/conditional_photon_gate_v1/theta_distribution.png)
shows both methods. The mode remains **experimental and off by default**:
uncertainty heads are calibrated for p>=0.70 only, the external reco q68 gain
is unproven, and the transverse-error q95 is slightly worse. It adds a
second CPU pass for selected events; a 512-raw-event `nue2` smoke took
10.4 s versus 8.4 s for a single pass on the same host (not a production
throughput benchmark).

Reproducible inputs, cohort counts, bootstrap intervals, and the plot are in
[`results/conditional_photon_gate_v1/`](results/conditional_photon_gate_v1/).
The third raw HDF5 has SHA256
`191491f16c3fbf75ce3d62a9808585a7bf1ba3b10b1425055654d777d93441f5`.
CPU example: add `--max-added-photon-residual-ns 50` to the command above,
using an HDF5 prepared at p>=0.50.

## Independent 2019 truth-line check and fourth holdout (2026-10-13)

`build_all_track_anchor_targets.py` now resolves 2019 IDs with a two-part
source-part suffix such as `nue2_1000_4_0`. The companion
`extract_2019_truth_points.py` recovers the cluster-local true track point
from the source MC. The original third holdout C has a valid single truth
line for all 100,000 raw events. A **fourth** holdout D selected 20 previously
unused source parts (`part-shift 6`), yielding 95,144 raw events. It has zero
exact event-ID or generated-primary overlap with A/B/C, and zero
generated-primary overlap with the 2020 training union. The frozen hit
classifier at p>=0.50 kept 54,207 events, with selected-hit precision 98.90%
and recall 99.01% within kept events. Nineteen D raw rows have IDs repeated
across detector clusters and are excluded from the truth-line comparison;
this prevents a wrong cluster-local point from being joined to an event.

The paired p>=0.70 acceptance and the 50 ns added-hit gate were unchanged.
Each row is the same fixed cohort for all methods; q values are q50/q68/q95:

| Independent `nue2` set | N | Direction baseline | Direction gate | Line baseline | Line gate |
|---|---:|---:|---:|---:|---:|
| C | 51,386 | 4.084°/7.743°/39.610° | 3.855°/7.447°/39.062° | 3.869/6.236/19.260 m | 3.754/6.104/19.353 m |
| D | 49,803 | 4.159°/7.817°/39.677° | 3.915°/7.472°/39.002° | 3.894/6.284/19.428 m | 3.782/6.146/19.389 m |

On D, the gate-minus-baseline direction-q68 source-part-bootstrap 95%
interval is [-0.388°, -0.299°]; line-q68 improves by 0.139 m. The line-q95
change from the second pass is not consistent across C and D. A *diagnostic*
midpoint of the two predicted absolute points gives line q68/q95
6.111/19.201 m on C and 6.168/19.340 m on D. Its q68 improvement over
the first-pass point is supported by source-part bootstraps in both samples;
on D the q95 change interval is [-0.239, -0.026] m. This midpoint averages
two predictions, so it is **not a production candidate** under the
single-model/no-ensemble requirement. It identifies point-head instability
under hit-threshold changes and motivates a learned single-output
point-consistency loss instead. The shipped inference path remains unchanged.

The D raw HDF5 SHA256 is
`c5b2c84e1aa80c029a61687c0e9aaa1428b6c8676d27e44148ba7a10cbf7ed87`.
Its manifest, classifier summary, primary-overlap audit, and the paired
truth-line metrics with bootstrap intervals are in
[`results/conditional_photon_gate_v1/`](results/conditional_photon_gate_v1/).

## Point-only photon-head continuation (2026-10-13)

The conditional gate changes the selected hits and can move the predicted
track point. `train_photon_residual.py` now supports an absolute-point
consistency term across p>=0.70 and p>=0.90 hit views. A controlled
`--point-only-head` mode freezes the base model, shared photon layers, and
the two direction-output rows; it updates only the final point-vote rows.
`--point-selection` selects on predicted-hit line q68, with guards against
direction or clean-GT line degradation. The earlier angle-oriented selector
did not evaluate the working predicted-hit line and selected no point-only
checkpoint. These flags are opt-in and leave the previous default unchanged.

Two 2,000-step joint direction+point trials illustrate the trade-off: with
point-consistency weights 1.0 and 0.3, predicted-hit line q68 reached
4.294 and 4.301 m at step 2,000 (from 4.315 m), but direction q68 rose
from 4.343 to 4.350 degrees in both cases, violating the existing angle
guard. A point-only trial under the inherited angle-oriented selector
preserved direction but saved no improved checkpoint. The point-specific
selector below uses the same training trajectory and saves step 1,500.

The selected step-1500 checkpoint is
`runs/physical_time_audit/photon_point_only_selected_2k/best.pt` under the
angle-reconstruction workspace (SHA256
`fd4d674392eabef558a5ddee2dbfa019cda6018b7da35329413b9cc632488fc3`).
A state-dict audit confirms that every shared and direction parameter is
bitwise unchanged from `photon_head_pair_fixed_3k/best.pt`; only the last
point rows changed. On the predicted-hit validation set (5,406 events),
line q68 is 4.315 -> 4.299 m and q95 is 13.784 -> 13.540 m, while
direction q68 remains 4.343 degrees. Clean-GT line q68 moves 5.969 ->
5.983 m; the selection guard allows up to 0.05 m.

For an untouched **fifth** 2019 `nue2` sample E, 20 new source parts
(`part-shift 8`) yield 100,000 raw events. There is zero exact event-ID or
generated-primary overlap with A/B/C/D or the 2020 training union. The
frozen signal classifier at p>=0.50 keeps 56,681 events; its selected-hit
precision/recall are 98.901%/99.008% within that kept set. All 52,129
p>=0.70-accepted events have a unique valid truth line. On this *same*
cohort, q values below are q50/q68/q95:

| E method | Direction [degrees] | Transverse line [m] |
|---|---:|---:|
| Original point, p>=0.70 | 4.138/7.857/40.801 | 3.891/6.267/20.144 |
| Point-only, p>=0.70 | 4.138/7.857/40.801 | 3.815/6.186/19.990 |
| Original point, 50 ns gate | 3.933/7.537/40.514 | 3.779/6.151/20.165 |
| Point-only, 50 ns gate | 3.933/7.540/40.520 | 3.688/6.041/20.008 |

For the point-only gate versus original gate, the source-part-bootstrap
95% interval of the line-q68 change is [-0.133, -0.077] m, and line-q95
is [-0.262, -0.051] m. The direction-q68 interval is [-0.003, +0.010]
degrees, consistent with no change. Only 161 of 52,129 gate decisions
changed. These are genuine independent-sample point improvements, not
prediction averaging.

The shift is **not uniform across datasets**: on the ID-masked,
deduplicated internal 2020 MC test, 26,229 events with a valid truth line
give line q68 4.656 -> 4.673 m for old versus point-only gate (paired
bootstrap interval [-0.009, +0.038] m). Direction q68 on all 109,897
events is 9.497 -> 9.498 degrees (interval [-0.001, +0.002] degrees).
The internal 2020 test shares generated primaries with the training union,
so it is descriptive, not an independent promotion criterion. On the
external reco comparison, direction q68 is 5.916 degrees for both gates
on the 57,316-event paired cohort; with the existing official reco cut it
is 5.537 -> 5.538 degrees, and on the available 10/13 preliminary-cut
superset it is 4.452 -> 4.454 degrees. The full 13-condition preliminary
selection is **not yet reproduced**: three derived quantities require the
ROOT/BARS evaluator from the supplied `fillTrackAnalysisTree_v1.0.cc`.
The cluster62 source ROOT has the underlying `BRecoMuon`, `BEvent`,
`ScanfitMask`, geometry, and channel-mask branches; the limitation is the
runtime/evaluator, not missing raw data.

For the same fixed 4,924 experimental and 87,103 unique `muatm` MC
events, only 2 and 56 predictions respectively rerun. Baseline-matched
theta Wasserstein-1 distance changes 3.062 -> 3.064 degrees when matching
hits/strings, and 0.375 -> 0.376 degrees when also matching observed
vertical hit spread. This detects no substantial new theta-distribution
shift, but it is **not** a truth-quality measure on experiment.

Results and reproducibility manifests are in
[`results/point_consistency_v1/`](results/point_consistency_v1/). The E raw
HDF5 SHA256 is
`b197aacc50b448e0735e6eb27743c4efcf385a5f1550e8e37f2a503b36f47733`.
This checkpoint is **not promoted to production** pending full exact reco
preselection, cross-period investigation of the line discrepancy, and
uncertainty-head recalibration. It remains a single model, not an ensemble;
the conditional gate itself can require a second CPU pass.

### Exact preliminary-cut runtime audit

Cluster62 does have ROOT 6.34.04 and a BARS checkout at
`/home/ivkhar/BARS/bars_source/bars` (local commit `b9accaef`,
2025-02-27). The supplied `fillTrackAnalysisTree_v1.0.cc` does **not**
compile against that checkout: its `BRecoMuonBDTEval` lacks
`InitBDT_v3_LE/HE`, the seven-argument `EvalVars`, and getters for
`fLogPHit_30_min`, `fQacc_20`, `fQtot_20`, and `fTRes_30_med`; the energy
helpers are also absent. The source ROOT does contain the `BRecoMuon`,
unfiltered `BEvent`, `ScanfitMask`, geometry, and channel-mask branches.
Thus the 10/13 ROOT-field selection above remains a superset, **not** the
full 13/13 cut. A newer compatible BARS checkout/container is needed for
the supplied program's exact three derived conditions; no numerical
substitute has been presented as an exact result.

## Sixth independent `nue2` holdout and primary-disjoint audit (2026-10-13)

A sixth raw 2019 holdout F was reserved **after** choosing the point-only
checkpoint and 50 ns gate. It uses 20 new source parts (`part-shift 10`),
100,000 raw events, and the same frozen signal classifier. The extractor
now accepts repeated `--exclude-manifest` arguments and rejected overlap
with all five earlier holdouts. A separate audit found zero common source
parts, exact event IDs, or exact generated primaries between F and each of
A/B/C/D/E. F also has zero generated-primary overlap with the 2020
training union. All 100,000 raw events have a unique valid truth line.
At p>=0.50 the classifier keeps 56,274 events; selected-hit precision is
98.882% and recall within kept events is 99.020%. Fixed p>=0.70 and
2-string/8-hit acceptance leaves **51,580** paired events. The raw HDF5
SHA256 is `e88afc94dfe3ee9ba13fd600f6d21d77ff3c408d7f559d559f8db247e12931a0`.

| F method | Direction q50/q68/q95 [degrees] | Transverse line q50/q68/q95 [m] |
|---|---:|---:|
| Original point, p>=0.70 | 4.109/7.737/40.340 | 3.929/6.308/19.839 |
| Point-only, p>=0.70 | 4.109/7.737/40.340 | 3.847/6.216/19.665 |
| Original point, 50 ns gate | 3.908/7.422/39.720 | 3.809/6.171/19.983 |
| Point-only, 50 ns gate | 3.911/7.424/39.720 | 3.722/6.074/19.737 |

Point-only versus original at the **same** gate changes line q68 by
-0.097 m, with a 95% source-part-bootstrap interval [-0.123, -0.071] m;
line q95 changes by -0.246 m, interval [-0.406, -0.132] m. The angle-q68
interval [0.000, +0.015] degrees does not establish an improvement; the
direction rows themselves remain frozen, while 163/51,580 gate decisions
change through the first-pass point. Relative to the original p>=0.70
pipeline, the point-only gate improves direction q68 by 0.314 degrees
(95% interval [0.253, 0.358]) and line q68 by 0.234 m (interval
[0.201, 0.267]). This replicates the independent E direction/line result
without an ensemble or model selection on F. The manifests, classifier
summary, overlap checks, and paired bootstrap result are in
[`results/point_consistency_v1/`](results/point_consistency_v1/).

`audit_timegate_mc_quality.py` now optionally excludes every exact
generated primary seen in either 2020 training stream, in addition to the
existing test-ID mask and deduplication. This leaves 99,429/109,897
internal test events overall, but only **841/8,673 `nue2`** events. On the
full 8,673, the old 50 ns gate direction q68 is 5.628 degrees and line
q68 is 6.213 m; on the 841 primary-disjoint `nue2`, they are 16.011
degrees and 10.132 m. The point-only gate changes the latter line q68 to
9.889 m, with a wide 95% event-bootstrap change interval [-0.507,
+0.066] m. Thus the reassuring full-2020 `nue2` numbers are heavily
affected by reused generated primaries, while the strictly novel subset
is small and compositionally different. Neither is a substitute for the
large independent 2019 holdouts.

`audit_primary_composition.py` checks whether this gap is only a
low-statistics/low-energy sample effect. The 841 primary-disjoint `nue2`
events have a median of 8 retained hits and median log10(primary energy)
of 3.757, versus 13 hits and 4.689 for the 7,832 reused-primary
events. Reweighting the latter to the novel group's hit-count, energy,
and true-theta cells gives direction q68 **9.535 degrees**, still below
the novel group's 16.011 degrees (all 841 novel events have support).
On the independent 2019 F set, the 8-9-hit stratum has 8,048 events and
direction q50/q68 of **6.605/13.383 degrees**, compared with
2.277/3.790 degrees for the 14,243 events with at least 24 hits.
Difficulty composition therefore explains part, but not all, of the
2020 gap. This is not a causal test of memorization: unobserved source
and detector covariates may differ too. In particular,
`train_mixed_universal.py` initializes from specialist checkpoints that
already saw some 2020 primaries, so a primary-disjoint mask applied only
to the final fine-tune would not create a genuinely independent model.
The machine-readable 2020 audit is in
[`results/primary_composition_v1/`](results/primary_composition_v1/).

`compare_2019_2020_nue2_novel.py` joins the same 841 events to the sixth
independent 2019 F holdout and reproduces the recorded per-event hit
thresholds from their p>=0.50 files. F has 51,580 events and q68 7.422
degrees overall; after reweighting F to the novel 2020 hit-count, primary
energy and true-theta cells, its q68 is 13.138 degrees (all 841 novel
events supported; effective comparison sample size 11,177). Matching
observed z-span as well gives 12.399 degrees on 839/841 novel events.
The remaining gap is therefore not resolved by those coarse covariates.
MC-only label-zero hits differ sharply: at fixed p>=0.70, 589/841
novel-2020 events contain at least one, versus 5,633/51,580 F events.
With the current 50 ns gate, the counts are 606/841 versus 7,344/51,580,
and the median label-zero fraction is 0.125 in novel 2020 versus zero in
F. Reweighting on hit count and this forbidden-at-
experiment label-zero fraction reverses the comparison (F q68 23.153
degrees versus 15.971 degrees for the supported 837 novel events), but
the effective F sample drops to 995 and the result is not causal.
On the same 841 novel-2020 IDs, fixed p>=0.70 q68 is 16.277 degrees
versus 16.011 degrees with the gate, so the gate is not the source of the
large discrepancy. An oracle removal of all selected label-zero hits
would leave **532/841** novel events with fewer than eight hits, even
before checking strings. A stricter signal/noise threshold alone cannot
fix this group while preserving the 2-string/8-hit input cut; sparse-hit
robustness is the next model target. None of these MC-label diagnostics
transfers directly to experiment, where labels and true angles are absent.

A 3,000-step direction-only continuation was also tested, freezing the
shared hit layers and point-vote rows. The final point coordinates can
still move slightly because the model projects them perpendicular to the
updated direction. No step passed the predicted-hit direction-q68 guard
and improved the existing validation score; no checkpoint was selected.
The complete validation snapshots are in
[`results/point_consistency_v1/direction_only_pilot/`](results/point_consistency_v1/direction_only_pilot/).
The point-only candidate remains experimental for universal deployment:
the reco gain is unresolved, the exact 13-condition reco cut needs a newer
BARS evaluator, and uncertainty heads are calibrated for the older input
rule.

## Random sparse-hit training augmentation (2026-10-14)

`sparse_hit_augmentation.py` adds a training-only eventwise crop to 8-12
random hits with at least two strings. It rebases the point target after
the hit centroid changes and keeps the physical truth line unchanged.
`test_sparse_hit_augmentation.py` checks this invariant; on a real
128-event training batch at crop probability 0.5, 56 events changed and
the maximum valid-line transverse discrepancy was about 1e-5 m.

`run_sparse_crop_v4.sh` compared 2,000-step continuations from the same
v4 checkpoint, seed 211 and losses: control versus crop probability 0.5.
Only the predicted-hit training stream was cropped; GT-hit batches and
validation were unchanged. Both used stochastic p=0.70-0.90 threshold
thinning, the single-track-muatm filter, and a validation score that also
weighs 8-12-hit predicted events. The common step-0 score was **9.1731**;
no later checkpoint in either run improved it. At step 2,000 the control
score/predicted-hit q68/8-12-hit q68 were 9.1777/4.4829°/4.6887°,
and the crop run's were 9.1920/4.5029°/4.6872°, versus
9.1731/4.4695°/4.6682° initially. The 8-12-hit `nue2` validation q68
was 17.298° initially and 17.972° at the crop run's final step, albeit
on only 242 events. Neither run produced a selected `best.pt`; there is
no independent-test or experimental-domain result for this augmentation.
Random removal from otherwise accepted events did not replicate the
false-positive-hit composition of the difficult 2020 subset. The next
experiment should use the classifier's lower-probability hits or a
score-aware robustness objective, with independent-sample and
experiment-versus-muatm distribution checks before any promotion.

## Lower classifier-threshold training ablation (2026-10-14)

`build_p050_anchor_from_p070.py` transfers audited single-track truth lines
from the p>=0.70 **training** split to a 250,000-row p>=0.50 training set,
then rebases each line's point target to the new hit centroid. It rejects
all evaluation IDs. There are 96,591 eligible rows (91,177 unique IDs),
including 69,203 `nue2`, 23,250 `nuatm`, and 4,138 `muatm` rows; the stream
is therefore strongly `nue2`-weighted. No eligible row overlaps evaluation.
`test_build_p050_anchor_from_p070.py` checks the line-rebasing invariant.

`train_mixed_universal.py` now accepts an optional weighted lower-threshold
predicted-hit stream; without the new arguments its old behavior is unchanged.
`run_p050_mix_v4.sh` continued the same v4 checkpoint for 1,500 steps with
stream weights 0.10 and 0.20 (seed 211). The common step-0 predicted-hit
q68 was 4.4695 degrees and validation score 9.1731. At the
0.20 run's best step 1,250, q50/q68 was 2.6142/4.4667 degrees, 8-12-hit
q68 4.6600 versus 4.6682 initially, and score 9.1629. The 0.10 run's
best step had q50/q68 2.6203/4.4819 degrees and score 9.1726. These
changes are too small to promote without independent and experimental-domain
checks; neither candidate replaces v4. The checkpoints remain in
`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/`.

## Paired reco median-gap audit (2026-10-14)

`analyze_reco_central_gap.py` joins eventwise v4, v4+photon, the standard
reconstruction, selected classifier hits, and a **separate** fair-input
GT-signal dataset. Its machine-readable output is in
[`results/reco_central_gap_v3/`](results/reco_central_gap_v3/). The cut is
the 10 preliminary conditions available in ROOT out of the requested 13:
three BRecoMuonBDTEval-derived quantities require a newer BARS installation.
It is a superset of the full 13/13 selection, not a substitute for it.

On the same 44,189 `nue2` events, standard reco has direction q50/q68
2.031/5.730 degrees; v4 on p>=0.70 classifier hits has 2.518/4.611,
and v4+photon has 2.342/4.468. The paired-bootstrap 95% interval for
v4+photon minus standard reco q50 is +0.276 to +0.346 degrees. Thus its
tail is better but its central resolution is worse. At the standard-reco
median error, 5,694 events have a good reco and worse NN result, versus
3,758 in the reverse direction. The net 1,936-event shortfall is mostly
events with at least four selected strings: 4 strings contribute 931,
5-6 strings 833, and 7-9 strings 102; two-string events favor NN by 441.
For 17-24 selected hits, NN/reco q50 is 1.585/1.216 degrees. Even among
36,577 events with *zero* selected label-zero hits, it is 2.080/1.730.
This is a central, clean-event deficit, not just a few outliers.

There is also an important input asymmetry. The reco source file for the
classifier was already filtered with `take_reco_hits=True`,
`take_signal_hits=False` in `filter_data_reco.py`: its input hit count equals
reco `fNHits` for 99.985% of test events. The frozen classifier cannot
recover raw pulses omitted before inference. On a distinct fair dataset
retaining all GT-signal hits (an oracle unavailable on experiment), the
*same* v4 weights reach q50/q68 1.968/3.309 degrees on the paired 44,189
events, versus standard reco 2.031/5.730. The q50 difference is -0.064
degrees (paired-bootstrap 95% interval -0.104 to -0.023). Full GT input
shrinks or reverses the median gap, but does not prove a deployable gain;
the clean zero-noise subset still has GT-input NN/reco q50 1.874/1.730.

The next **diagnostic**, before further training or architecture search, is
to start from the unfiltered 2020 raw `baikal_reco.h5`, apply the same frozen
signal/noise classifier and 2-string/8-hit cut, and compare on exact matched
IDs. That tests whether the missing raw pulses are the main bottleneck.

## Exact-cluster full-raw reco input check (2026-10-14)

That diagnostic is now complete. `extract_reco_unmasked_exact.py` matched
all 57,316 paired IDs to the *specific* source cluster by exact equality
between its reco-masked raw hits and the filtered reco event. Matching only
the event ID would be wrong: raw files contain multiple clusters for some
IDs. The extracted file retains **all 5,170,357 raw hits** from those
clusters; `reco_mask` is used to establish identity, not to select input
hits. The frozen signal/noise checkpoint, p>=0.70 threshold, and 2-string/
8-hit cut retained 54,492 events, with median 16 selected hits versus 12
in the old reco-masked branch. The same v4 angle bundle then ran on CPU.

`analyze_reco_unmasked_raw.py` joins exact IDs and computes the paired
results in [`results/reco_unmasked_raw_v1/`](results/reco_unmasked_raw_v1/).
Within the ROOT-observable preliminary 10/13 cut, 42,003/44,189 events
pass the raw-input classifier selection. On **these same 42,003 events**:

| Input/method | Direction q50 | Direction q68 |
| --- | ---: | ---: |
| Standard reco | 1.983° | 5.576° |
| v4 on old reco-masked classifier input | 2.472° | 4.517° |
| v4+photon on old reco-masked input | 2.297° | 4.373° |
| **Same v4 on full-raw classifier input** | **2.121°** | **3.530°** |

Full-raw v4 improves q50 by 0.351° over the *same* v4 weights on the old
masked input (paired-bootstrap 95% interval: 0.321-0.379° improvement).
It still trails standard reco q50 by 0.138° (95% interval 0.097-0.172°),
while improving q68 by 2.046°. Its q95 is 13.23° versus 32.97° for old
v4 and 61.32° for standard reco on the same events. The raw-input selection
rejects 2,186 of the 44,189 cut events; their standard-reco q50 is 3.195°,
so all-method comparisons must remain paired rather than comparing each
branch on a different event set.

The residual median deficit is again in well-constrained events. In the
old-input four-string stratum, raw v4/reco q50 is 2.290/1.915°; with five
to six strings it is 1.372/0.946°. Among raw-selected events with 17-24
hits, the gap is almost closed (2.051/2.031°), but for >=40 hits it is
1.031/0.739°. The full raw input explains much of the prior q50 mismatch
and improves the tail substantially; further broad hyperparameter sweeps
are not justified by the latest ablations. A remaining targeted option is
to study timing/geometry residuals or a physics-guided refit on the
clean, high-multiplicity events, then verify on a separate holdout and
experimental data. This is still not the full 13/13 selection because the
three unavailable BRecoMuonBDTEval-derived cuts were not applied.

## Frozen photon correction on full-raw reco input (2026-10-14)

Without fitting or reselecting on the external reco sample, the already
trained `photon_head_pair_fixed_3k/best.pt` direction/point correction was
run on the full-raw p>=0.70 selected-hit file above. A 256-event smoke
found at most 0.033° direction difference between the CPU bundle inference
and the uncorrected GPU audit path before applying the correction.
`analyze_reco_unmasked_raw.py --raw-photon-predictions` checks exact ID and
hit-count alignment and saves paired bootstrap intervals and strata in
[`results/reco_unmasked_raw_photon_v2/`](results/reco_unmasked_raw_photon_v2/).

On the same 42,003 events passing the ROOT-available preliminary 10/13
selection and both classifier branches:

| Method | Direction q50 | Direction q68 | Direction q95 |
| --- | ---: | ---: | ---: |
| Standard reco | 1.983° | 5.576° | 61.315° |
| v4+photon, old reco-masked input | 2.297° | 4.373° | 33.054° |
| v4, full-raw input | 2.121° | 3.530° | 13.228° |
| **v4+photon, full-raw input** | **1.954°** | **3.360°** | **13.334°** |

The full-raw photon correction improves q50 by 0.168° and q68 by 0.170°
relative to full-raw v4; paired-bootstrap 95% intervals for its change are
[-0.179°, -0.152°] and [-0.193°, -0.151°]. Against standard reco, its
q50 difference is -0.029° with interval **[-0.070°, +0.011°]**, so a
median advantage is *not* established; its q68 difference is -2.215°
with interval [-2.400°, -2.067°]. With the older official reco cut as
well (41,566 common events), q50/q68 is 1.939°/3.330° for full-raw
v4+photon versus 1.946°/5.380° for standard reco. The slight q95 increase
from full-raw v4 to v4+photon should not be hidden.

The old clean/high-string precision limit is reduced but not eliminated:
in the old-input 5-6-string stratum, full-raw v4+photon/reco q50 is
1.225°/0.946°. The same frozen photon checkpoint was previously checked
on the fixed p>=0.70 experimental-versus-all-`muatm` cohort:
hit/string-matched predicted-theta W1 changed 2.975° -> 2.956° and
z-spread-conditioned W1 0.368° -> 0.369°. This is a distributional
check only; experiment has no true directions. At that stage the photon
correction was not yet bundled because its uncertainty heads needed
recalibration. The next section describes the subsequent bundle; full exact
13/13 reco cuts remain unavailable.

## Single-file photon-corrected direction/point/uncertainty bundle (2026-10-14)

`photon_uncertainty_bundle.py` wraps the frozen v4 direction/point/uncertainty
network and the previously trained frozen photon residual head in **one**
backbone (no ensemble). It validates that the two source checkpoints have
identical backbone tensors and preprocessing. `recalibrate_photon_uncertainty.py`
then fits multiplicative uncertainty radii on the 2,747-row 2020 validation
slice and evaluates them on the disjoint ID-masked 27,038-row 2020 test slice
(26,229 unique IDs). This is **not generated-primary-disjoint** from 2020
training, as the overlap audit above explains. No network weights are trained
or selected on this test.

On that 2020 mixed-particle test, the frozen corrected model gives direction
q50/q68 **2.582°/4.661°** and reference-point transverse-distance q50/q68
**3.077/5.045 m**. The 2020-only radii cover 66.9%/93.8% of true angles
at nominal 68%/95%, and 66.9%/92.9% of true track lines. Ranking is useful:
the half with smallest predicted angle uncertainty has q68 **1.999°**, and
the analogous point half has q68 **2.795 m**. These marginal numbers hide
conditional undercoverage: the hardest uncertainty decile has angle/point
68%-coverage 58.2%/59.0%. See
[`results/photon_uncertainty_recalibration_v1/quality_calibration.png`](results/photon_uncertainty_recalibration_v1/quality_calibration.png).

The **2019 nue2 out-of-season check exposed a material calibration failure**:
with the 2020-only angle factors, the 51,473-event 2019-A sample has
68%/95% angle coverage just **51.4%/85.9%**, despite direction q50/q68
4.179°/7.844° matching the prior independent angle audit. A second
2019-B sample was rethresholded from the frozen-classifier p>=0.50 file to
p>=0.70. The procedure was first verified on 2019-A: all twelve HDF5
arrays, including hit features, scores, event IDs and normalization,
matched the existing p>=0.70 file exactly. The 51,360 accepted 2019-B
events and their generated primaries are disjoint from 2019-A.

`calibrate_cross_season_angle.py` therefore takes, separately for 68% and
95%, the larger of the **2020-validation** factor and the factor fitted on
**2019-A only**. It does not use 2019-B or the 2020 test to set factors.

| Angle-radius calibration | 2019-B coverage 68% / 95% | 2020 MC test coverage 68% / 95% |
| --- | ---: | ---: |
| 2020 only | 51.8% / 86.1% | 66.9% / 93.8% |
| **Cross-season conservative** | **68.1% / 94.8%** | **81.4% / 97.9%** |

This is a deliberate coverage/precision trade-off: the single-file
cross-season bundle
`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/photon_uncertainty_recalibration_v1/universal_angle_track_uncertainty_photon_crossseason_v2.pt`
is safer for an unknown season but overconservative on 2020 MC. Its 68%/95%
angle multipliers are 2.60175/6.45668 times `sigma_angle_deg`; point
multipliers remain 1.59626/3.45839 times `sigma_anchor_m`, **validated only
on 2020 MC**. In CSV inference, angular radii are capped at the physical
maximum of 180° (coverage is unchanged). The full audit is in
[`results/photon_uncertainty_recalibration_v1/crossseason_calibration.json`](results/photon_uncertainty_recalibration_v1/crossseason_calibration.json),
and the 2020 selective-quality/coverage plot for these final factors is in
[`results/photon_uncertainty_crossseason_v2/quality_calibration.png`](results/photon_uncertainty_crossseason_v2/quality_calibration.png).

The v1 and cross-season v2 bundle `state_dict` tensors are exactly equal.
A 256-event CPU check produced identical direction and point values; only
angle radii changed. On all 4,924 p>=0.70 experimental events, the v1 bundle
matched the previously domain-checked standalone photon correction to a
median direction difference of 0.0016° (CPU/GPU numerical variation).
Thus the fixed-cohort experimental/`muatm` theta-distribution check above
still applies to v2. Experimental events have no angle/point truth, so this
is **not** an experimental coverage claim. Multi-particle `muatm` scenes
still do not have a uniquely validated single reference point.

CPU inference on an HDF5 whose hits have already passed the frozen
signal/noise classifier at p>=0.70 and the 2-string/8-hit cut:

```bash
cd pytorch_training/scripts/angle_track_uncertainty
python infer_unified_model.py \
  --checkpoint /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/photon_uncertainty_recalibration_v1/universal_angle_track_uncertainty_photon_crossseason_v3.pt \
  --data /path/to/p070_selected_hits.h5 --split test \
  --output /path/to/predictions.csv --threads 8
```

The bundle does **not** contain the hit signal/noise classifier. The CSV
contains direction, point, uncalibrated `sigma_*`, and the calibrated 68%/95%
radii. `--device cuda` is optional; CPU is the default. Do not interpret
the point head or its radii as a unique track estimate on multi-track events.

## Out-of-season truth-line check and cross-season point radii (2026-10-14)

The source 2019 MC on cluster62 supplies a unique cluster-local truth line
for every raw event in independent nue2 holdouts A and B. The existing
`extract_2019_truth_points.py` recovered 100,000 valid points per sample;
`audit_2019_point_uncertainty.py` joined them to the same p>=0.70 selected
events used above. The point error is the transverse distance from the
predicted reference point to the **true infinite track line**, not a distance
at an arbitrary +200 m lever arm. There is no event-ID or generated-primary
overlap between A and B, or generated-primary overlap with 2020 training.

The frozen v2 network yields point-to-line q50/q68 **3.900/6.265 m** on
51,473 A events and **3.885/6.206 m** on 51,360 B events. Its 2020-only
point radii under-cover B: nominal 68%/95% intervals contain only
**60.8%/91.0%** of true lines. `calibrate_cross_season_track.py` fits only
on A, taking the maximum of its factor and the pre-existing 2020-validation
factor. B and the 2020 test influence neither the model nor these factors.

| Point-radius calibration | 2019-B coverage 68% / 95% | 2020 MC test coverage 68% / 95% |
| --- | ---: | ---: |
| 2020 only | 60.8% / 91.0% | 66.9% / 92.9% |
| **Cross-season conservative** | **68.2% / 95.0%** | **73.9% / 95.3%** |

The resulting single-file
`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/photon_uncertainty_recalibration_v1/universal_angle_track_uncertainty_photon_crossseason_v3.pt`
is the current recommended bundle. Relative to v2, **every network tensor,
direction, and reference point is bitwise unchanged**; only the point
radius factors change to 1.85289/4.22721 times `sigma_anchor_m`. The
cross-season angle factors from v2 remain 2.60175/6.45668. The 2019-B
output CSV was rerun with v3 and matched v2's direction and point exactly.
The v3 checkpoint SHA256 is
`7cb8f77cfdb2a7cf021d090be5246b83f8a94c7513cc24c793ab019c67686f41`.
On B, retaining the half of events with smallest predicted uncertainty
reduces direction q68 7.871° -> 3.138° and point-distance q68
6.206 -> 3.582 m. See the independent
[`2019-B quality and calibration plot`](results/photon_uncertainty_crossseason_v3/2019_b/quality_calibration.png)
and the corresponding
[`2020 MC test plot`](results/photon_uncertainty_crossseason_v3/quality_calibration.png).

The v3 point intervals are marginally calibrated on B, not perfectly
conditional: the easiest predicted-sigma deciles cover roughly 65-68% at
nominal 68%, while the hardest decile covers 82%. On 2020 MC, both angle
and point intervals are conservative **marginally**, but 2020 nue2 point
95%-coverage is only 90.6%, versus 95.3% for the particle mixture. This
species/topology dependence prevents claiming universal conditional
calibration. A one-slope monotone power-law recalibration fitted on A did
not resolve it: the 2019-B hardest point decile still over-covered (~80%
at nominal 68%), while the 2020 hardest decile under-covered (~62%). It
was not promoted. Experimental data still have no true
line, and multi-track `muatm` remains outside the validated single-point
scope. The checkpoint does not include signal/noise hit classification;
the CPU command above works unchanged with the v3 filename.

## Conditional photon-time hit recovery, q50 diagnosis (2026-10-14)

The strongest remaining *input* hypothesis was tested without retraining or
ensembling. `infer_rethresholded_for_audit.py` first runs the frozen v3 model
on classifier-selected p>=0.70 hits. Only for predicted upward directions
(`theta < 90°`), it may rerun **the same model** with p>=0.50 hits, provided
every newly added hit is within 50 ns of the first-pass photon propagation
time. The accepted event cohort always satisfies the original p>=0.70,
2-string/8-hit cut. This is an optional inference mode, not a new model.

The p>=0.50 full-raw 2020 reco file reproduces the previous p>=0.70 file
**exactly in all ten HDF5 datasets** after rethresholding (54,492 events,
1,148,062 hits); thus input preparation is not confounded. The external
reco-MC file contains **nue2 only**. On the 42,003 common events passing the
ROOT-observable preliminary 10/13 reco cuts:

| Direction error | q50 | q68 | q95 |
| --- | ---: | ---: | ---: |
| Standard reco | 1.983° | 5.576° | 61.315° |
| Frozen v3, p>=0.70 | 1.954° | 3.360° | 13.334° |
| Same v3, conditional photon gate | **1.845°** | **3.147°** | **13.146°** |

The gate reruns 11,778/42,003 events. Paired event bootstrap gives a
gate-minus-standard-reco q50 difference of -0.138° (95% interval
**[-0.178°, -0.103°]**), whereas the nominal p>=0.70 q50 is statistically
tied with reco (interval [-0.070°, +0.012°]). Gate-minus-nominal q50 is
-0.108° ([-0.123°, -0.097°]); q68 is -0.214° ([-0.237°, -0.188°]).
The available older official reco cut in addition to 10/13 leaves 41,566
events and likewise gives q50 1.832° gate versus 1.946° standard reco.
**This is still descriptive**: the unavailable 3/13 original cuts cannot
be reproduced with installed BARS, and some 2020 generated primaries
overlap training. See
[`photon_gate_reco/summary.json`](results/photon_uncertainty_crossseason_v3/photon_gate_reco/summary.json)
and its [`strata.csv`](results/photon_uncertainty_crossseason_v3/photon_gate_reco/strata.csv).

The nominal model's central weakness is not simply very long tracks. For
8-11 selected hits (10,565 events), q50 is 3.151° nominal versus 2.860°
standard reco; the gate lowers it to 2.806°. In the 3-string stratum
(9,974 events), q50 is 2.441° versus 2.283° reco, falling to 2.237° with
the gate. Very long 300-450 m GT spans remain a difficult subpopulation
(nominal/gate/reco q50 1.015°/0.959°/0.644°, 3,563 events), but they do
not explain the aggregate median. With reco's 1.983° error as a fixed
pivot, nominal NN loses 6,456 reco-good events but rescues 6,661 reco-bad
ones; the gate increases the net rescued count from 205 to 1,103. Those
counts, unlike separate subgroup medians, decompose the global median
comparison.

On independent out-of-season **2019-B nue2** (51,360 events), direction
q50/q68 improves from 4.138°/7.871° to 3.921°/7.498°; point-to-true-line
q50/q68 improves from 3.885/6.206 m to 3.765/6.075 m. The source-part
bootstrap q68 difference is [-0.403°, -0.315°] for direction and
[-0.158, -0.101] m for the line. See
[`2019_b_photongate_audit.json`](results/photon_uncertainty_crossseason_v3/2019_b_photongate_audit.json).
The 2020 ID-masked MC test shows analogous direction improvements for
nue2 (3.177°/5.984° -> 2.970°/5.628°) and nuatm
(1.994°/3.419° -> 1.900°/3.284°), but negligible effect on multi-particle
muatm (only 55/87,103 reruns). The small primary-disjoint nue2 subset in
that 2020 test (841 events) is inconclusive; 2019-B is the stronger
independent nue2 evidence. Full MC details are in
[`photon_gate_mc2020.json`](results/photon_uncertainty_crossseason_v3/photon_gate_mc2020.json).

The fixed-cohort experimental check has 4,924 accepted events and only
**two** gated reruns; the corresponding muatm MC has 55/87,103. Matched
theta-distribution Wasserstein-1 distance, conditioning on hit/string count
and z-spread, changes only 0.375° -> 0.376° (no meaningful domain
adaptation gain or distribution shift). See
[`photon_gate_domain/`](results/photon_uncertainty_crossseason_v3/photon_gate_domain/).
Experimental events have no angular truth. For gated p>=0.50 events the
frozen uncertainty heads still output raw `sigma_angle_deg` and
`sigma_anchor_m`, but the **p>=0.70-calibrated radii are intentionally
blank**: coverage for this changed input distribution has not been
validated. This mode should not replace the default v3 bundle until its
uncertainty calibration is resolved. It needs no second network, but may
invoke the same network twice on a subset; CPU latency is measured below.

### Gated-input uncertainty and CPU audit (2026-10-14)

`audit_gate_uncertainty.py` fits prospective gate-specific angle and
true-line radius factors **only on the 11,131 changed 2019-A nue2 events**.
It verifies no shared event IDs or exact generated primaries with 2019-B,
then reports untouched 2019-B and ID-masked 2020 test coverage. It does
not overwrite the v3 bundle or promote a new calibration. The A-fitted
68%/95% factors are 2.300/5.503 for angle and 1.692/3.982 for line,
smaller than existing v3 p>=0.70 factors 2.602/6.457 and 1.853/4.227.

| Gated events | Angle coverage with existing v3 factors, 68% / 95% | Line coverage, 68% / 95% |
| --- | ---: | ---: |
| Independent 2019-B nue2 (10,816) | 72.6% / 96.4% | 71.3% / 95.5% |
| 2020 masked nue2 (2,223) | 79.1% / 96.7% | 75.8% / 94.6% |
| 2020 masked nuatm (4,053) | 83.8% / 98.9% | 77.5% / 97.6% |
| 2020 masked muatm (55; only 10 valid truth lines) | **12.7% / 54.5%** | 30.0% / 90.0% |

The overall single-track coverage is somewhat conservative, but not
uniformly so. In the 2020 gated nue2 hardest predicted point-sigma decile,
nominal 95% line coverage is only **88.3%** (222 events); in 2019-B's
hardest angle-sigma decile it is 93.6% (1,081 events). The small 2020
primary-disjoint gated nue2 subset (121 events) also under-covers, though
it is too small for a reliable standalone correction. On the rare
upward-predicted muatm events, even angle coverage fails badly; their
single-track point target is usually not well-defined. See the full
[`photon_gate_uncertainty_audit_v2.json`](results/photon_uncertainty_crossseason_v3/photon_gate_uncertainty_audit_v2.json),
including uncertainty-decile rows. Thus the **gate's blank calibrated
radii remain intentional**; an A-only scalar factor is not a universal
fix. Raw predicted sigmas remain available as ranking scores. Experimental
events have no GT coverage measurement.

CPU timing on cluster63, 2,048 consecutive 2019-B p>=0.50 source events
(1,904 satisfy the high-threshold 2s8h cut), eight PyTorch threads,
batch 256: two p>=0.70 passes took 15.60 and 15.65 seconds; the gate run
with 388 reruns took 18.09 seconds. That is roughly **16% more wall time**
or 8.2 -> 9.5 ms per accepted event, with peak RSS about 1.03-1.08 GB.
This times network inference on already hit-classified HDF5 files, not the
upstream signal/noise network or I/O from raw detector events. The gate
is a CPU-feasible single-network quality option, not a free inference step.

## Wider photon-time gate selected on 2019-A, validated on B (2026-10-14)

The 50 ns photon-residual requirement above was conservative. A **predefined**
5, 10, 20, 30, 50, 100, 200 ns sweep was computed from frozen v3 high- and
low-threshold predictions on 2019-A; the per-event photon selection was
independently verified against the actual 50 ns inference on B (zero gate
mismatches). A showed monotonic direction q50/q68 gains up to 200 ns, so
**200 ns was selected before looking at B**. The added-hit GT-signal
fraction fell from 93.7% at 5 ns to 80.8% at 200 ns; this purity trade-off
is visible in the
[`A/B threshold plot`](results/photon_uncertainty_crossseason_v3/photon_gate_threshold_sweep_ab.png).

| Independent 2019-B nue2, 51,360 events | Direction q50 / q68 | Line distance q50 / q68 |
| --- | ---: | ---: |
| p>=0.70 only | 4.138° / 7.871° | 3.885 / 6.206 m |
| conditional p>=0.50, 50 ns | 3.921° / 7.498° | 3.765 / 6.075 m |
| conditional p>=0.50, **200 ns** | **3.740° / 7.041°** | **3.683 / 5.919 m** |

At 200 ns, 19,585 B events rerun the **same** model. Source-part bootstrap
95% intervals for 200 ns minus p>=0.70 are [-0.904°, -0.746°] at angle
q68, [-0.325, -0.251] m at line q68, and [-0.333, -0.021] m at line q95.
See the [`A sweep`](results/photon_uncertainty_crossseason_v3/photon_gate_threshold_sweep_2019_a.json),
[`B sweep`](results/photon_uncertainty_crossseason_v3/photon_gate_threshold_sweep_2019_b.json),
and [`B angle/line audit`](results/photon_uncertainty_crossseason_v3/2019_b_photongate200_audit.json).

The exact full-raw reco 2020 cohort remains 42,003 nue2 events under the
ROOT-observable 10/13 cut. Standard reco direction q50/q68 is
1.983°/5.576°; frozen v3 p>=0.70 is 1.954°/3.360°; gate 50 ns is
1.845°/3.147°; **gate 200 ns is 1.823°/3.072°** with q95 12.847°
versus 61.315° standard reco. The 200 ns gate reruns 23,590 events.
Its gate-minus-standard-reco q50 difference is -0.160° with paired event
bootstrap interval [-0.201°, -0.124°]; gate-minus-p>=0.70 q68 is
-0.288° with interval [-0.320°, -0.254°]. The older official cut together
with 10/13 gives gate 200 ns q50/q68 1.809°/3.046° versus
1.946°/5.380° standard reco. These reco results remain descriptive:
the unavailable 3/13 cuts and generated-primary overlap prevent claiming
a fully independent head-to-head test. See
[`photon_gate200_reco/`](results/photon_uncertainty_crossseason_v3/photon_gate200_reco/).

On the 2020 ID-masked MC test, nue2 direction q50/q68 improves
3.177°/5.984° -> 2.893°/5.430° and true-line q50/q68
3.659/6.317 -> 3.474/6.116 m. `nuatm` improves more modestly; the
effect on dominant `muatm` is negligible (285 reruns among 87,103 unique
events). The **841** primary-disjoint 2020 nue2 events are harder and do
not establish a gain: their angle-q68 and line-q68 change intervals both
span zero. This is a separate out-of-distribution warning, not grounds to
tune the threshold on those same test events. See
[`photon_gate200_mc2020.json`](results/photon_uncertainty_crossseason_v3/photon_gate200_mc2020.json).

On the fixed experimental 2s8h cohort, 18/4,924 events rerun. With muatm
MC matched using unchanged p>=0.70 hit/string counts and z spread, the
predicted-theta Wasserstein-1 distance changes 0.375° -> 0.369°; without
z-spread matching it changes 3.062° -> 3.064°. These are negligible
distribution changes, not evidence of improved experimental truth accuracy.
See [`photon_gate200_domain/`](results/photon_uncertainty_crossseason_v3/photon_gate200_domain/).

The changed-input uncertainty is **still not reliably calibrated**. With
the old conservative radii, independent 2019-B gated events cover
68.5%/95.1% of true angles and 69.5%/94.5% of true lines, but the hardest
angle-sigma decile covers only 64.9%/92.8%. The small primary-disjoint
2020 nue2 gated subset under-covers much more (57.3%/87.1% angle and
56.4%/85.8% line); rare muatm gated events also remain invalid for a
single-track calibration. `audit_gate_uncertainty.py` now records the
threshold used; its full [200 ns report](results/photon_uncertainty_crossseason_v3/photon_gate200_uncertainty_audit.json)
does **not** promote new factors. The inference CSV therefore keeps the
calibrated radius fields empty for reruns, retaining raw predicted sigmas.

CPU timing on the same 2,048 2019-B source events as the 50 ns benchmark:
722 gated reruns take 20.47 s versus 15.60/15.65 s for p>=0.70 only,
about **31%** extra wall time or 10.8 instead of 8.2 ms per accepted
event. The frozen hit classifier's runtime is excluded. The 200 ns gate
is the better *quality* option among tested thresholds; the unchanged v3
p>=0.70 bundle remains the calibrated default for downstream coverage.

Optional CPU inference on an already classifier-prepared p>=0.50 HDF5
(including per-hit probabilities and channel IDs) uses the **same** v3
single-file network checkpoint; no ensemble or retraining is required:

```bash
cd pytorch_training/scripts/angle_track_uncertainty
python infer_rethresholded_for_audit.py \
  --bundle /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/photon_uncertainty_recalibration_v1/universal_angle_track_uncertainty_photon_crossseason_v3.pt \
  --data /path/to/p050_selected_hits.h5 --split test \
  --threshold 0.70 --conditional-low-threshold 0.50 \
  --gate-theta-max-deg 90 --max-added-photon-residual-ns 200 \
  --output /path/to/predictions.csv --device cpu --threads 8
```

Rows with `selected_threshold=0.50` expose raw sigmas but deliberately
have blank calibrated radius columns. The frozen signal/noise classifier
that prepares the HDF5 is separate from this checkpoint and command.

## What still limits the median on the external reco cohort (2026-10-14)

The latest [paired input audit](results/reco_input_gap_gate200_v2/summary.json)
joins the exact full-raw MC clusters, the p>=0.70 classifier-selected hits,
the frozen v3 and 200 ns gate predictions, and ROOT reco **by event ID**.
The raw HDF5 and paired reco arrays have different row orders; direct
row-wise comparison is invalid. All results below use the same 42,003 nue2
events passing the ROOT-observable preliminary 10/13 cut. This is
descriptive: three original BRecoMuonBDTEval-derived cuts are unavailable,
and some 2020 generated primaries overlap training.

| Cohort | N | Standard reco direction q50/q68 | v3 gate200 q50/q68 |
| --- | ---: | ---: | ---: |
| All paired 10/13 events | 42,003 | 1.983° / 5.576° | **1.823° / 3.072°** |
| Reco >=20 hits and >=12 triplets | 5,679 | **0.622° / 0.989°** | 0.832° / 1.223° |
| Same, with >=90% raw true-signal-hit recall at p>=0.70 | 5,192 | **0.612° / 0.961°** | 0.817° / 1.201° |
| NN 8-11 selected hits, <70% signal recall | 426 | **2.816°** / 6.734° | 2.964° / **5.361°** |

Thus the clean, high-reco-quality precision gap remains when nearly all
true signal hits survive the classifier. Missing hits are **not** a
complete explanation. Both components contribute in the high-quality
cohort: theta q50 is 0.420° NN versus 0.318° reco, and phi q50 is
0.814° versus 0.593°. The same high-quality direction-q50 pattern holds
in each source part A-E (NN 0.794-0.930° versus reco 0.575-0.659°), not
just one source file. That does not remove the training-overlap caveat.

Hit selection still matters at low multiplicity. With 8-11 selected hits
and <70% raw signal recall, the *nominal* p>=0.70 NN q50 is 6.233°;
the fixed gate200 lowers it to 2.964°. Conversely, when standard reco uses
1-4 more fitted hits than the NN retains, its q50 is 1.215° versus
1.839° for gate200 (4,336 events); when the NN has >=10 more selected
hits, gate200 is 1.354° versus reco 2.095° (6,668 events). ROOT `fNHits`
counts fitted reco hits, not all available raw hits, so this difference
is an observational diagnostic, not proof that one algorithm discarded
the other's exact hit set.

At the common reco-median pivot of 1.983°, gate200 loses 5,828
reco-good events and rescues 7,174 reco-bad events, a **net 1,346-event
gain**. The two crossover groups have similar median signal recall
(0.933 vs 0.947) and GT span (179 vs 178 m); losses have fewer selected
hits (14 vs 17) and slightly smaller ROOT `fThetaErr` (0.467 vs 0.548).
In the high-quality subset the NN still wins event-wise 42.5% of the
time, but reco has the better central precision. These observations
favor a precision-floor/timing-geometry hypothesis over a simple
"long-track" or classifier-only diagnosis.

The **unchanged p>=0.70** uncertainty radius is conservative on this
external cohort: nominal `angle_r68_deg` covers 78.7% overall and 86.7%
in the high-quality subset. This does *not* validate gate200 radii;
they remain blank for changed-input rows. No model weights or
experimental theta predictions changed in this audit. A targeted
single-model timing/geometry correction should only be considered after
independent-season validation and a repeated MC-vs-experiment theta check;
another broad architecture sweep is not justified by these results.
The executable audit is [`audit_reco_input_gap.py`](audit_reco_input_gap.py),
with all strata in [`strata.csv`](results/reco_input_gap_gate200_v2/strata.csv).

### A simple geometric axis is not the missing precision (2026-10-14)

Before attempting another fit, [`audit_hit_pca_axis.py`](audit_hit_pca_axis.py)
tested a parameter-free direction from the largest principal component of
the p>=0.70 selected-hit XYZ cloud. The axis sign came from its correlation
with hit time; the MC-truth best sign was also measured only as a diagnostic.
No parameter was chosen on the holdout. On independent 2019-B nue2
(51,360 events), direction q50/q68 was **12.829°/24.934°** for this axis
versus **3.740°/7.041°** for frozen v3 gate200. The time sign was correct
96.4% of the time, and even the truth-oracle sign gave 12.824°/24.875°:
the main problem is geometric bias, not the orientation sign. With >=20
hits and PCA linearity >=0.9 (3,363 events), the axis still gave
4.828°/6.765° versus NN 1.125°/1.809°. The independent 2019-A result is
nearly identical. See the [A report](results/pca_axis_2019_a.json) and
[B report](results/pca_axis_2019_b.json). This axis should **not** replace
or be averaged with the NN. The earlier Cherenkov timing fit also worsened
overall q68 (reported above); a more elaborate refit would require a
new, predeclared independent test rather than tuning on the reco cohort.

### Gate200 uncertainty: five-bin transfer check (2026-10-14)

[`audit_gate_conditional_calibration.py`](audit_gate_conditional_calibration.py)
fits exactly five equal-count bins of the predicted angle and line sigmas
on the **19,861 gated 2019-A nue2 events only**, then evaluates untouched
2019-B and ID-masked 2020 MC. It reads the actual GT training source
`baikal_mc_merged2020_signal_8-2_angl_norm.h5` and asserts both the
3,022,664-primary training union and the established 225 gated
primary-disjoint 2020 nue2 events. An initially attempted different GT
file changed that disjoint cohort drastically and was rejected before
interpretation. Neither the fit nor its output modifies the bundle.

| Holdout | N | Old angle coverage 68% / 95% | Five-bin coverage 68% / 95% | Old line coverage 68% / 95% | Five-bin line coverage 68% / 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019-B gated nue2 | 19,585 | 68.5% / 94.3% | 67.2% / 95.0% | 69.5% / 94.2% | 67.6% / 94.6% |
| 2020 gated nue2, primary-disjoint | 225 | 57.3% / 84.9% | **56.9% / 88.0%** | 56.4% / 84.9% | **52.4% / 84.9%** |
| 2020 gated muatm | 285 | 18.2% / 62.1% | **18.6% / 67.4%** | 41.9% / 96.8% (31 valid lines) | 41.9% / 93.5% (31 valid lines) |

On the hardest predicted-angle-sigma decile of independent B, 95%
coverage rises from 92.8% to 94.7%, but 68% only from 64.9% to 65.5%.
The primary-disjoint 2020 nue2 angle-68% Wilson interval after the fit
is 50.4%-63.2%; this is not a harmless small-sample fluctuation around
68%. The rare gated `muatm` events violate the single-track uncertainty
assumption and experimental data have no truth with which to validate
coverage. Even taking the larger of old and fitted radii leaves the
2020 primary-disjoint angle coverage at 57.8% / 88.0%. Therefore this
conditional map is **not promoted**: changed-input rows still have blank
calibrated radii and only raw sigmas. The [full report](results/photon_uncertainty_crossseason_v3/photon_gate200_conditional5_audit.json)
includes factors, Wilson intervals, and sigma-decile coverage. Direction
predictions and the MC-versus-experiment theta comparison are unchanged.

### Reco events with unseen angle-model primaries (2026-10-14)

[`audit_reco_novel_primary.py`](audit_reco_novel_primary.py) checks the frozen
v3 bundle on the 2020 reco **train and validation** portions that were not used
in the earlier reco-test comparison. Inputs are the p>=0.70 signal/noise
selected hits, followed by the same >=8 hits / >=2 strings cut. It excludes
every exact float32 `(theta, phi, energy)` primary occurring in *any* split of
either internal angle-training GT or predicted-hit HDF5. Event IDs, prediction
order, and MC truth are matched against the standard-reco HDF5 before scoring.
This is primary-disjoint from the **angle model**, not a proof that the
signal/noise classifier never saw these events; its training overlap has not
been audited. It is also a covariate-shifted subset, not a random new-primary
version of the earlier high-quality reco cohort.

| Selection | N | NN q50 / q68 | Standard reco q50 / q68 |
| --- | ---: | ---: | ---: |
| All angle-model-primary-disjoint train + validation | 7,994 | 13.06° / 23.84° | 15.81° / 26.31° |
| Validation, three available official-quality sidecar cuts | 222 | 3.91° / 7.34° | 3.08° / 8.32° |

The 7,994 events represent 6,320 unique primaries. Keeping one event per
primary, a paired bootstrap gives 95% intervals of **[-3.42, -2.42]°** for
NN-minus-reco q50 and **[-3.01, -2.01]°** for q68: NN wins on this broad,
sparse subset. But only **70** of these events have reco `fNHits>=20`, and the
222-event official-quality comparison is too small for a firm precision
ranking (paired unique-primary intervals include zero for both quantiles).
The sidecar provides only three available quality cuts; it is **not** the
10/13 ROOT-cut selection used in the earlier reco-test audit, and cannot be
treated as the full 13/13 official selection.

The cohort is dominated by 8–11 selected-hit events and strongly differs
from the reused 2020 and independent 2019-B samples in track-hit fraction,
selected-hit purity, span, zenith, and energy. More importantly, the bundled
angle radii nominally targeting 68% / 95% contain the true direction only
**36.7% / 63.4%** of the time here. This is a real calibration warning;
do not report those radii as calibrated for this domain.

A frozen-model sensitivity check increased the signal/noise threshold from
`p>=0.70` to `p>=0.90` before the 2s8h cut. It retained only 65.2% / 65.9%
of novel train / validation events. On the **5,216 common novel events**,
q50 / q68 improved from 9.41° / 19.32° to 9.21° / 18.82° (paired
unique-primary bootstrap intervals for p090-minus-p070 are
[-0.44, -0.03]° / [-0.85, -0.24]°). On the far larger reused-MC common
cohort, both quantiles worsened. Therefore p090 is **not** promoted as a
global preprocessing change; no model parameters or production bundle were
changed. The [full reproducible report](results/reco_novel_primary_v3_p070_p090_audit_v1.json)
contains split-wise metrics, selection sizes, and coverage.

### Global zenith-bias correction: rejected cross-season control (2026-10-14)

[`audit_theta_bias_crossseason.py`](audit_theta_bias_crossseason.py) tested
whether a cheap post-hoc correction of predicted zenith could explain the
high-quality q50 gap. It fitted a regularized quadratic in
`cos(predicted theta)` on **2019-A only**, clipped shifts to +/-2°, asserted
that A and B have no exact shared generated `(theta, phi, energy)` primaries,
then evaluated untouched 2019-B. Both files use the frozen v3 gate200
directions; no network retraining or experimental-data tuning was involved.

| 2019-B selection | N | Frozen q50 / q68 | Zenith-corrected q50 / q68 |
| --- | ---: | ---: | ---: |
| All | 51,360 | **3.740° / 7.041°** | 3.918° / 7.143° |
| >=20 selected hits | 18,645 | **2.498° / 4.242°** | 2.780° / 4.492° |
| >=35 selected hits | 7,973 | **1.979° / 3.262°** | 2.342° / 3.500° |

The same correction also worsens A. Typical signed theta residual is close
to zero (B median +0.059° overall, -0.007° with >=20 hits), while the
conditional *mean* is pulled by large mistakes near the horizon and beyond.
This particular global bias correction is therefore **rejected**; it is not
evidence that all conditional geometry corrections are impossible. The
[machine-readable result](results/theta_bias_crossseason_v3_gate200_negative.json)
records the fitted coefficients and theta-bin diagnostics. The production
checkpoint, preprocessing, and experimental theta distribution are unchanged.

### Independent standard-reco validation inventory (2026-10-14)

A read-only inventory of the accessible cluster62 Baikal data tree found the
ready standard-reco HDF5s under `data/h5s/`, `data/filtered/`, and
`data/normed/`, plus the
ROOT files under `data/initial_data/reco/nu2.100PeV/recMay23/all/`; these
reco sources are the 2020 campaign. The sampled raw 2019 nue2 ROOT file
`data/initial_data/MC_2019/nue2/1000.root` has MC truth and hit branches but
no branch name containing `reco`, `recMuon`, `thetaFit`, `phiFit`, or `BDT`.
Thus 2019-A/B provide an independent *NN* direction check, not an independent
NN-versus-standard-reco comparison. This inventory is scoped to the inspected
cluster62 directories and ROOT sample; it does not prove that no suitable
reconstruction exists elsewhere. A reliable assessment of the subdegree
high-quality reco gap still needs standard-reco outputs on enough new
generated primaries, with the same cuts applied to both methods.

### Frozen uncertainty as an optional quality flag (2026-10-14)

[`audit_selective_quality_crossseason.py`](audit_selective_quality_crossseason.py)
sets two `sigma_angle_deg` thresholds from the **frozen v3 p>=0.70 model's
2019-A score distribution only**, at its 50th and 75th percentiles. The
full 2019-A prediction was rerun with the same v3 bundle used for B; event
IDs, hit counts, and disjoint generated primaries are checked. No B or
experimental truth is used to choose a threshold. The audit applies the
same fixed thresholds to untouched 2019-B and to the 7,994 2020 reco
events primary-disjoint from the angle model. Signal/noise classifier
training overlap on the latter remains unaudited.

| A-defined sigma limit | B retention | B q50 / q68 | B hit-count-only q50 / q68 at equal retention | Novel-2020 retention and q50 / q68 |
| --- | ---: | ---: | ---: | ---: |
| 2.323° | 49.7% | **2.016° / 3.123°** | 3.012° / 5.337° | 43.9%; 6.124° / 15.826° |
| 5.058° | 75.1% | **2.832° / 4.642°** | 3.611° / 6.673° | 84.3%; 11.225° / 21.636° |

Without selection, B q50/q68 is 4.138°/7.871° and novel-2020 is
13.060°/23.838°. At the 50% cut, a B bootstrap puts the sigma-minus-
equal-retention-hit-count q68 difference in **[-2.314, -2.113]°**.
The quality ranking persists *within* the 8–11, 12–19, and >=20-hit B
bins, so it is not just a multiplicity proxy. Nevertheless, the novel
2020 retained subset is still difficult and the previously observed
coverage failure of calibrated radii remains; this score is not an
absolute error guarantee.

Using the same p>=0.70 experimental and all-muatm MC event IDs,
the 2.323° score cut retains **27.8%** of experiment and **29.6%** of MC
after matching by baseline selected hit/string counts and vertical spread.
The matched predicted-theta W1 distance is **0.949°** after selection,
versus **0.378°** before selection; at 5.058°, retention is 83.7% / 84.4%
and W1 is 0.495°. These are distribution checks, not experimental-truth
accuracy measurements. The 50% A cut markedly reduces experimental
coverage and *worsens* the theta-domain match, so it is offered only as
an **optional quality flag**, not a default event cut or a replacement
checkpoint. The [full report](results/selective_quality_crossseason_v3_p070.json)
contains thresholds, hit-bin results, bootstraps, and matched-domain
support. The single-model direction output and full-cohort experimental
theta distribution are unchanged.

### Muatm composition of the experimental domain comparator (2026-10-14)

[`audit_muatm_domain_composition.py`](audit_muatm_domain_composition.py)
joins the fixed 4,924-event experimental and 25,054-unique-event muatm-MC
comparison to the existing track-anchor `valid` flag by event ID. It checks
that the anchor and selected-hit validation rows align, repeated event IDs
agree on source index and validity, and all source IDs were parsed. For muatm
in this anchor file, `valid` agrees with the metadata count for **exactly
one generated muon track**. Thus the previous description of this domain
comparator as "single-track muatm" was wrong: only **965 / 25,054 (3.85%)**
MC events satisfy that condition; the other **24,089** do not. The entire
2020 predicted-hit test similarly has 3,437 one-track muatm rows out of
90,526. Supervised *training* did filter to the unambiguous one-track
subset, but the experimental domain audit included **all** muatm MC.

| Muatm comparator | Unique MC events | Predicted-θ W1 vs experiment after hit/string/z-spread matching | Experimental cells without MC support | Effective MC N |
| --- | ---: | ---: | ---: | ---: |
| All | 25,054 | 0.378° | 0% | 21,071 |
| Exactly one generated muon | 965 | 3.122° on supported experiment | 22.1% | 144 |
| Not exactly one generated muon | 24,089 | 0.349° | 0% | 20,204 |

The earlier full-muatm W1 comparisons remain measurements of the intended
full MC cohort; only their occasional **single-track wording** was wrong.
The isolated one-track sample is too small and has poor covariate support
for a reliable experimental comparison. These distributions are consistent
with the importance of non-single muatm topologies, but experimental truth
composition is unknown and W1 is not an angular-accuracy metric. The
[machine-readable report](results/muatm_domain_composition_v3_p070.json)
contains both matching schemes and all support diagnostics. No checkpoint
or experimental prediction changed.

### Raw muatm direction and track-point coherence (2026-10-14)

[`audit_muatm_bundle_coherence.py`](audit_muatm_bundle_coherence.py) joins
the p>=0.70 2s8h validation/test events to the original merged HDF5 by
source part, event ID, and raw row index. It verifies primary angles and
single-track flags, and reads each generated muon's individual direction,
energy, and starting point. There are **112,157 unique muatm events** in
26 source parts: 4,400 have one generated muon; 107,757 have two or more
(17,518 with 2–4, 34,104 with 5–9, and 56,135 with at least 10).

In this particular MC file, **every individual generated muon direction
agrees with the event's primary direction to <0.01°**; the largest
observed difference is 0.000032°. This is a property of the stored
simulation labels, not proof that experimental bundles are exactly
collinear or that the network reconstructs each physical muon. It does
remove the feared *direction-label* conflict for training on this MC. The
separate track points remain ambiguous: the median per-event q68 of
transverse start-position spread is **19.05 m** across multi-muon events
(q68 over events 24.35 m), increasing to 20.53 m for >=10 muons. Thus
multi-muon events could be tested as **angle-only supervision**; they
must stay masked out of the single-point/line and uncertainty losses.
The [raw-data report](results/muatm_bundle_coherence_v2.json) records the
full multiplicity and angular/spatial quantiles. This audit itself did
not change any checkpoint or experimental prediction.

### Conservative multi-muatm angle-only ablation (2026-10-14)

`train_mixed_universal.py --extra-pred-multi-muatm-direction-only` can now
add an extra predicted-hit stream selected by `muatm_` event ID and invalid
single-track anchor. The primary training/validation streams still use
`--single-muatm-only`. The extra stream retains the original event IDs,
removes held-out IDs and exact primary-triple overlaps, and contributes
**direction loss only** because `point_loss` masks its invalid point target.
This experiment does not change inference architecture or require an
ensemble. Tests cover the selection and zero point gradient.

A controlled 3,000-step continuation of the v4 model used the same seed
(106), learning rates (3e-6 backbone, 6e-6 point), GT weight (0.4),
validation score, and single-track main stream as the existing
`continue_v4_noaug_3k` control. The only new stream had
`--extra-pred-weight 0.10`, giving it a 0.06 coefficient in the total
loss. After leakage checks it contained 1,676,396 train rows / 1,608,833
unique IDs; three exact primary overlaps were removed. The run is saved at
`runs/mixed_universal/angle_only_multi_muatm_w010_3k_seed106/`.

| Validation metric | Start (v4) | Control best, step 1,000 | Multi angle-only, step 1,000 |
| --- | ---: | ---: | ---: |
| Predeclared score, lower is better | 6.839 | **6.817** | 6.867 |
| Predicted-hit single-track direction q50/q68 | 2.611° / **4.469°** | **2.571°** / 4.491° | 2.602° / 4.542° |
| GT nue2 direction q68 | 4.088° | **4.055°** | 4.075° |
| Single-muatm direction q68 | 10.588° | **10.582°** | 10.754° |

All six multi-angle checkpoints at steps 500–3,000 scored worse than the
start (best 6.867, final 6.878), whereas the control improved. The trainer
therefore did **not** select or save a new best checkpoint. This ablation is
rejected; the deployed/frozen bundle and experimental theta predictions
are unchanged. It does not establish whether multi-muatm accuracy itself
improved, because no candidate passed validation for independent testing.
The result narrows this particular low-weight joint-training idea, not all
possible topology-conditioned designs.

### High-quality reco central-gap bias audit (2026-10-14)

[`audit_reco_precision_bias.py`](audit_reco_precision_bias.py) joins the
existing 2020 standard-reco fields, frozen v3 gate200 directions, and MC
truth by exact event ID, and checks the same ROOT-observable 10/13 cut.
The high-quality subset again contains **5,679** events with reco
`fNHits>=20` and `fNTriplets>=12`. Its direction q50/q68 is
**0.832°/1.223° NN** versus **0.622°/0.989° standard reco**. The NN wins
42.5% event-wise; reco has the sharper centre but much worse q90
(5.27° versus 2.55° for NN).

The NN's *signed* median residuals are only **+0.001° in theta** and
**-0.048° in phi**. The central gap persists in each populated true-theta
range (0–30°, 30–60°, 60–90°) and all eight 45° true-phi sectors; it is
not explained by one directional region or a global offset. A constant
theta/phi correction fitted on source parts A–C was just -0.0027°/-0.0420°.
On the held-out D–E source parts (2,332 high-quality events), it changed
NN q50/q68 from 0.85995°/1.30297° to 0.86145°/1.30635°; the paired
bootstrap 95% interval for the q50 change is [-0.01097°, +0.00804°].
Thus this *predeclared simple correction* is rejected. It reinforces a
resolution/precision-floor hypothesis, not a proof of the physical cause.

The [machine-readable report](results/reco_precision_bias_v2.json) includes
signed residual widths, per-angle-bin counts, and the source-part test.
These 2020 reco primaries may overlap angle-model training, and source-part
separation is not a new independent season. This is a diagnostic, not a
claim of independently validated subdegree superiority or a new model.
No checkpoint, inference path, or experimental theta distribution changed.

### Frozen NN-seeded photon-time refit on the reco precision subset (2026-10-14)

[`audit_frozen_photon_refit.py`](audit_frozen_photon_refit.py) applies the
existing `fit_photon_track.fit_one` algorithm to the frozen v3 **p>=0.70
input and prediction from the same event**. Its refractive index 1.36,
10 ns residual scale, 1° direction prior, 10 m point prior, and 50-function
evaluation cap were fixed from earlier MC-validation work, not selected on
the reco cohort. The comparison again uses exactly the 5,679 ID-matched
2020 events passing the observable 10/13 and high-reco-quality cuts. The
script additionally verifies that every selected-input `(theta, phi,
energy)` truth triple exactly equals its standard-reco source triple.

| Direction resolution | Frozen NN | NN-seeded time refit | Standard reco |
| --- | ---: | ---: | ---: |
| q50 | 0.833° | 0.967° | **0.622°** |
| q68 | 1.242° | 1.575° | **0.989°** |

The refit reduces its own median absolute photon-time residual from
6.40 to 5.30 ns, yet worsens the angle; its median direction shift is
0.718° and median CPU fit time is 5.4 ms/event, excluding NN inference.
Paired event bootstrap gives **[+0.111°, +0.161°]** for refit-minus-NN
q50. Thus a better fit to this idealized direct-photon model is not a
better direction estimator on the high-quality cohort.

[`audit_refit_label_composition.py`](audit_refit_label_composition.py)
shows that these inputs are already track-dominated: median **43**
truth-labelled track hits, **97.0%** track-hit fraction, and 56.6% of
events with no selected label-zero noise hit. The optional *oracle*,
forbidden-at-inference `--fit-hit-label track-only` mode keeps 5,667 events
with at least eight GT track hits. On those exact same events, direction
q50 is **0.833° NN**, **0.967° all-hit refit**, and **0.969° oracle
track-only refit**. Removing cascade/noise hits therefore does not rescue
this physical refit. This does not prove the cause of the NN–reco precision
gap; it rejects one concrete algorithmic remedy and suggests the timing
model/objective or fitted-line geometry is insufficient.

The [all-hit](results/refit_hq_prior1_all_v2.json),
[oracle-track](results/refit_hq_prior1_track_oracle_v2.json), and
[label-composition](results/refit_hq_prior1_label_composition_v1.json)
reports retain counts and diagnostics. Because the predeclared refit
worsened both q50 and q68 here, no independent 2019-B or experimental
distribution comparison was used to select it. The selected checkpoint,
hit threshold, uncertainty calibration, and experimental predictions
remain unchanged. This external 2020 reco cohort still may overlap
angle-model training primaries.

### Extended photon-residual gate: no production change (2026-10-14)

The earlier 200 ns gate was the upper edge of the first sweep. To check
whether it was artificially capped, the same frozen p>=0.70/p>=0.50
predictions were rescored on **2019-A only** at 200, 250, 300, 400, 500,
750, and 1000 ns. Before reading the result, we required at least a
0.05° q68 improvement relative to 200 ns, no q50 regression, and at
least 75% truth-signal fraction among reintroduced hits.

On 51,473 accepted events, 200 ns gave q50/q68 **3.764°/7.019°**;
750–1000 ns gave **3.754°/6.984°**. The 0.035° q68 gain misses the
predeclared minimum. The extra admitted hits remain mostly true signal
(79.6%), but the gain saturates by 750 ns. We therefore did **not**
select a new threshold, look at 2019-B for this choice, or alter the
checkpoint and MC/experimental predictions. The [full A sweep](results/photon_uncertainty_crossseason_v3/photon_gate_extended_2019_a.json)
is reproducible with [`sweep_added_hit_photon_gate.py`](sweep_added_hit_photon_gate.py)
and its `--photon-thresholds-ns` option.
