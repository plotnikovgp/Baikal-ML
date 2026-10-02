# Angle, track and uncertainty reconstruction

This directory contains the current single-model Baikal-GVD pipeline:

- `train_angle_signal.py`: masked set-transformer direction backbone and HDF5 input reader;
- `train_track_anchor_v3.py`: direction plus a point on the reconstructed track;
- `train_track_uncertainty_v2.py`: frozen backbone and track model with a small angular/transverse uncertainty head;
- `plot_uncertainty_calibration.py`: test-set error-density, conditional-coverage and raw-reliability plots.
- `plot_theta_phi_calibration.py`: signed theta/phi residual plots and component-wise coverage diagnostics.
- `plot_track_distance_distribution.py`: distribution and uncertainty-quartile CDF of the transverse track-anchor error; the optional `--include-plus200` flag adds a lever-arm diagnostic.
- `plot_quality_calibration.py`: test-set q68 versus retained fraction (events ranked by predicted uncertainty, with an oracle lower bound) and 68%/95% conditional coverage by predicted-uncertainty decile. It consumes saved `test_predictions.npz` and the validation-calibration metrics JSON; no model rerun is needed.

The uncertainty head predicts an angular scale in degrees and a transverse track scale in metres. It is trained with a two-dimensional Student-t negative log likelihood (3 degrees of freedom). The direction and track-point weights remain frozen. A separate validation subset provides empirical containment-radius factors at 68% and 95%; the test set is not used for fitting or calibration.

For component plots, the same angular scale is used for theta and locally propagated to phi by dividing by `sin(theta_pred)`. The script calibrates theta and phi half-widths separately on validation events beyond the checkpoint-selection subset. Phi events with `sin(theta_pred) < 0.15` are excluded because azimuth is ill-defined near the poles; the exclusion is reported in the metrics.

The completed run used the `nue2_2020` MC sample with GT signal hits and a 2-string/8-hit selection. It did **not** include `muatm` events or experimental hits. The track reference point is the point on the truth line closest to the centroid of retained signal hits. The angular/anchor errors on the independent 403,171-event test set were q50/q68 = 2.329/3.983 degrees and 4.406/7.239 m. Marginal 68%/95% coverage after validation calibration was 68.22%/95.11% for angle and 68.04%/94.90% for transverse anchor error. Conditional coverage across difficulty bins is imperfect, especially for the track intervals; do not interpret these as guaranteed coverage on experimental data.

With the most confident 50% of events retained, direction q50/q68 is 1.303/1.897 degrees and transverse anchor q50/q68 is 2.757/3.800 m. The 95% anchor interval covers only 85.6% of the ninth predicted-uncertainty decile even though global coverage is 94.9%; the two marginal 68% regions contain both errors in 50.5% of test events, not 68%. The track metric is transverse anchor-to-true-line distance, not whole-track distance over a defined physical segment.

Training data, prepared track-anchor targets, base checkpoints and the fitted uncertainty-head checkpoint are not committed. They remain on cluster63 under `/home/plotnikovgp/tmp/angle_reconstruction/`. The selected head checkpoint and full metrics are in `track_anchor/runs/uncertainty_student/`; diagnostic figures and per-event test predictions are in its `diagnostics/` directory.

Run scripts from this directory so that the sibling imports resolve. Use `--help` for arguments. A typical uncertainty run supplies `--data`, `--anchors`, `--angle-checkpoint`, `--track-checkpoint`, and `--output-dir`; use `--loss student --student-df 3` to reproduce the selected variant. The plotting script additionally takes `--uncertainty-checkpoint` and the generated `metrics.json`.
