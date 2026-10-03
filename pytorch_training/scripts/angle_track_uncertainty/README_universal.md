# Universal direction + track-point + uncertainty model

The single inference checkpoint is `universal_angle_track_uncertainty.pt`. It
contains the all-particle direction backbone, a compatible track-point head,
the angular/transverse uncertainty head, input normalization, and validation
calibration factors. It is a CPU-loadable PyTorch `state_dict` bundle without
optimizer state. The signal/noise hit classifier is **not** part of this file.

On cluster63 the bundle is at:

`/home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/universal_predsignal_bundle_v1/universal_angle_track_uncertainty.pt`

The model expects the prepared HDF5 format with signal/noise probability
`p >= 0.70`, at least eight retained hits on at least two strings. The angle
backbone was trained on `muatm`, `nuatm`, and `nue2`. The point target is a point
on the truth line nearest the centroid of **all retained hits**. Because most
`muatm` events contain multiple simulated tracks, their direction labels remain
in training, but only single-track MC events are used to train and calibrate
the point and transverse uncertainty. Point/radius outputs for multi-track
events and experimental data are not validated.

Run a ten-event CPU smoke test on cluster63:

```bash
CUDA_VISIBLE_DEVICES= /home/plotnikovgp/tmp/ENTER/bin/python3.11 \
  /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/infer_unified_model.py \
  --checkpoint /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/universal_predsignal_bundle_v1/universal_angle_track_uncertainty.pt \
  --data /home/plotnikovgp/tmp/angle_reconstruction/data/baikal_2020_all_predsignal_p070_2s8h.h5 \
  --split test --max-events 10 --batch-size 10 --threads 4 \
  --output /home/plotnikovgp/tmp/angle_reconstruction/track_anchor/runs/universal_predsignal_bundle_v1/my_predictions.csv
```

Remove `--max-events 10` for a full split. `theta_deg`, `phi_deg`, direction
components, and `anchor_{x,y,z}_m` describe one reconstructed line in the
cluster-local coordinate frame. `angle_r68_deg` and `angle_r95_deg` are
validation-calibrated angular containment radii; `anchor_r68_m` and
`anchor_r95_m` are the corresponding transverse point-to-truth-line radii.
They are *marginal* intervals, not a jointly calibrated tube around the entire
track. The CSV's `track_point_scope` field flags muatm/experimental events for
which the point prediction has not been validated against a unique true line.

The independent all-particle test contains 192,458 events. Direction q50/q68
is 3.98°/6.32°; across the 105,369 events with a unique truth line, transverse
point error q50/q68 is 5.41/8.66 m. Marginal 68%/95% test coverage is
67.6%/94.5% for angle and 68.2%/96.1% for the point. Coverage differs by
particle type (e.g. angle 95% coverage on nue2 is 92.5%), and these MC
numbers must not be transferred to experimental data without validation.

The current experimental selected-hit file was made with `p >= 0.90`, not
the training threshold `p >= 0.70`. A five-event CPU schema smoke test passed,
but that is not a calibration or domain-adaptation result.

The preparation/training pipeline is in `build_all_track_anchor_targets.py`,
`train_track_anchor_v3.py`, and `train_track_uncertainty_universal.py`. The
bundle is produced by `bundle_universal_model.py`; `infer_unified_model.py`
performs standalone CPU inference. Training source checkpoints and full test
metrics remain under `/home/plotnikovgp/tmp/angle_reconstruction/` on cluster63.
