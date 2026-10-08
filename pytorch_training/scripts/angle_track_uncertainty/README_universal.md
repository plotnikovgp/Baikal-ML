# Universal direction + track-point + uncertainty model

The current single inference checkpoint is
`universal_angle_track_uncertainty_v2.pt`. It contains the all-particle
direction backbone, a compatible track-point head, the angular/transverse
uncertainty head, input normalization, and validation calibration factors. It
is a CPU-loadable PyTorch `state_dict` bundle without optimizer state. The
signal/noise hit classifier is **not** part of this file.

On cluster63 the bundle is at:

`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/universal_angle_track_uncertainty_v2.pt`

The model expects the prepared HDF5 format with signal/noise probability
`p >= 0.70`, at least eight retained hits on at least two strings. The angle
backbone was trained on `nuatm`, `nue2`, and only **single-track** `muatm` MC.
The point target is a point on the truth line nearest the centroid of **all
retained hits**. Events with multiple simulated `muatm` tracks were removed
from both direction and point training. Point/radius outputs for multi-track
events and experimental data are not validated.

Run a ten-event CPU smoke test on cluster63:

```bash
CUDA_VISIBLE_DEVICES= /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
  /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/infer_unified_model.py \
  --checkpoint /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/universal_angle_track_uncertainty_v2.pt \
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
2.80°/4.85°; transverse point error q50/q68 is 3.72/5.67 m. Marginal
68%/95% test coverage is 66.8%/93.7% for angle and 65.5%/92.7% for the
point. On the independent 307,913-event GT-hit `nue2` test, coverage is
68.2%/95.1% for angle and 67.9%/94.5% for the point. Intervals are
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

## Single-track-muatm universal model (v2)

The selected direction/point checkpoint is on cluster63 at
`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/mixed_universal/single_mu_supervised_strict_3k/best.pt`.
It is one SetTransformer model, initialized from the nue2 specialist and
fine-tuned on all three particle types. No teacher or ensemble is used at
inference. The hit input is converted to the **GT signal-file normalization**
even when the hits came from the signal/noise network. `muatm` training events
are retained only if the original MC has one simulated muon track and a valid
truth-line point. The p>=0.70, two-string/eight-hit cut is unchanged.

The checkpoint was selected solely on validation. The independent test uses
`masks_v2.npz`: predicted-hit test excludes events seen by the specialist's
GT training; GT nue2 test excludes events seen by the original universal
predicted-hit training. Both comparisons below evaluate exactly the same
events and hit inputs for all three models.

| Model | Predicted-hit angle q50/q68 (27,038) | Single-muatm q68 (3,437) | Predicted-hit nue2 q68 (9,463) | GT nue2 angle q50/q68 (307,913) | GT nue2 line q68 |
| --- | --- | --- | --- | --- | --- |
| Universal v2 | 2.80° / 4.85° | 11.06° | 5.87° | 2.43° / 4.14° | 6.99 m |
| Original universal | 3.21° / 5.49° | 11.54° | 7.81° | 3.44° / 5.73° | 8.68 m |
| Original nue2 specialist | 3.06° / 5.99° | 50.91° | 5.82° | 2.37° / 4.04° | 7.22 m |

The MC single-track requirement cannot be applied to experimental data
without a separate multiplicity/quality classifier. Multi-muatm and
experimental resolution/coverage remain unvalidated; the table must not be
read as a performance claim for all muatm events. The older v1 bundle remains
available at `runs/universal_predsignal_bundle_v1/` for comparison only.

`train_mixed_uncertainty.py` freezes the selected v2 direction/point model and
trains a Student-t uncertainty head for angular and transverse line errors.
Disjoint clean validation halves select the head and calibrate its 68%/95%
radii. The selected head is
`runs/mixed_universal/uncertainty_student_continued_v3/best.pt`; its metrics
and independent-test per-event predictions are in the same directory. The
single bundled checkpoint above embeds both heads and predicted-hit
calibration. `plot_mixed_uncertainty.py` produces angular/line reliability,
conditional-coverage, and error-distribution figures from the saved test
predictions. It shows useful uncertainty ranking, but a notable
high-difficulty undercoverage tail on predicted hits.

The current experimental selected-hit file was made with `p >= 0.90`, not
the training threshold `p >= 0.70`. A five-event CPU schema smoke test passed,
but that is not a calibration or domain-adaptation result.

The preparation/training pipeline is in `build_all_track_anchor_targets.py`,
`train_track_anchor_v3.py`, `make_leak_free_masks.py`,
`train_mixed_universal.py`, and `train_mixed_uncertainty.py`. The bundle is
produced by `bundle_universal_model.py`; `infer_unified_model.py` performs
standalone CPU inference. A 16-event CPU-only smoke test of the v2 bundle
succeeded. Training source checkpoints and full test metrics remain under
`/home/plotnikovgp/tmp/angle_reconstruction/` on cluster63.
