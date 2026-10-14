# Experimental per-hit photon gate (not the release default)

The released `release_v3/run_release_v3.sh` uses an eventwise 200 ns photon
gate: if any added p=0.50–0.70 hit is incompatible, the entire second pass is
rejected. `infer_rethresholded_for_audit.py --per-hit-photon-gate` instead
retains only the individually compatible added hits. The first-pass direction
and track anchor, checkpoint and 200 ns cut are unchanged. This option was
evaluated **after** the release setting was frozen; no model was retrained.

| MC set, NN p>=0.70 + 2s8h | N | Eventwise gate q50/q68 angle | Per-hit gate q50/q68 angle | Eventwise/per-hit line q68 |
| --- | ---: | ---: | ---: | ---: |
| Independent 2019-A nue2 | 51,473 | 3.764 / 7.019 deg | 3.759 / 6.998 deg | 5.940 / 5.936 m |
| Independent 2019-B nue2 | 51,360 | 3.740 / 7.041 deg | 3.731 / 7.014 deg | 5.919 / 5.911 m |
| Untouched 2019-F nue2 | 51,580 | 3.730 / 6.931 deg | 3.727 / 6.917 deg | 6.024 / 6.025 m |
| 2020 mixed-MC test | 109,897 | 6.083 / 9.480 deg | 6.081 / 9.477 deg | 4.716 / 4.707 m (line GT subset) |

On F the source-part bootstrap 95% CI for the q68 angle difference was
[-0.029,-0.009] degrees (per-hit minus eventwise). The line q68 CI included
zero. For 2020 mixed MC the angle q68 difference CI was [-0.009,-0.001]
degrees. The 2020 `nuatm` and generated-primary-disjoint subcohorts did not
show a clear benefit, and the 2020 reco-MC comparison has not been rerun with
this candidate. Therefore this remains an optional audit variant, not the
release default.

The experimental sample has no truth. After matching to muatm MC in selected
hit/string counts and z spread, theta Wasserstein-1 distance was 0.3691 deg
for the eventwise gate and 0.3681 deg for the per-hit gate: no meaningful
domain-shift change at this precision. See `domain/summary.json` and
`domain/theta_distribution.png`. Full 2020 MC quality splits are in
`mc2020_quality.json`. Events changed by this variant have no validated
conditional angle or line uncertainty calibration; their calibrated radii
remain blank in prediction CSVs.
