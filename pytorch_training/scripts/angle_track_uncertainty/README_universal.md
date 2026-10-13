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

The experimental-versus-single-track-`muatm` theta check does not establish
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
