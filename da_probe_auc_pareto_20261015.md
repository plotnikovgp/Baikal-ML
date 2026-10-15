# Price of lowering the independent MC/EXP domain-probe AUC

The target here is **AUC of a freshly fitted frozen-encoder logistic-regression
probe**, not the adversarial head's own AUC. The domain comparison is MC
`muatm` versus experiment. The supervised hit loss still uses all MC particle
types. MC and EXP use the same raw event preselection: at least 8 hits on at
least 2 distinct strings. No ground-truth EXP hit labels are assumed.

All rows below use the **same untouched holdout**: 5,000 MC validation
`muatm` events (offset 10,000) and 5,000 experimental test events (offset
5,000). Each probe is independently fitted on the first 5,000 MC validation
and first 5,000 experimental validation events, using `StandardScaler` and
`LogisticRegression(C=1)`. The hit metrics are computed against MC hit labels;
the signal class is `label != 0`. Precision and recall use hit-score threshold
0.5. W1 is the 1D Wasserstein distance between **MC and EXP hit-score
distributions**, after each model's own predicted 2-string/8-hit event cut;
smaller is better. It is not an angular metric.

| Candidate | Probe AUC | MC hit AP | Precision @0.5 | Recall @0.5 | W1 after predicted cut | MC/EXP pass |
|---|---:|---:|---:|---:|---:|---:|
| Selected DA + output-W1 checkpoint | 0.8813 | 0.9833 | 0.9480 | 0.9302 | 0.0587 | 987 / 581 |
| DANN λ=2, step 200 | 0.7757 | 0.9646 | 0.9228 | 0.8923 | 0.0690 | 993 / 639 |
| DANN λ=5, step 200 | 0.7730 | 0.9594 | 0.9277 | 0.8883 | 0.0613 | 923 / 568 |
| DANN λ=10, step 250 | 0.7711 | 0.9556 | 0.9267 | 0.8630 | 0.0661 | 920 / 619 |
| Direct feature-mean matching, weight 10 | 0.8934 | 0.9837 | 0.9393 | 0.9452 | 0.0689 | 1040 / 633 |
| 5 fresh discriminator batches per step, λ=0.5 | 0.8796 | 0.9817 | 0.9364 | 0.9382 | 0.0680 | 996 / 624 |

The DANN rows start from the selected DA + output-W1 checkpoint, with encoder
learning rate 1e-4, discriminator learning rate 1e-3, 5 discriminator updates
per encoder update, 16 events per domain batch, and output-score W1 weight
0.2. The direct feature-mean row uses weight 10, learning rate 5e-5, 32
events per domain batch, no DANN loss, and the same output-score W1 weight.
The fresh-discriminator row uses 5 *different* domain batches per encoder
step, λ=0.5 and encoder learning rate 5e-5. Training diagnostics additionally
used a 1,000-event validation cohort; the table reports only the common
5,000-event holdout.

Because the cut membership changes, `da_stronger_bootstrap.py` also compares
the selected checkpoint with λ=5, step 200 on the **intersection** of events
accepted by both models. Two disjoint 5,000-event MC/EXP test halves are used;
the probe is trained on the same separate validation cohort for both halves.
Paired event bootstrap uses 150 replicates and reports the 95% interval for
`λ=5 minus selected`.

| Holdout half | Probe AUC selected → λ=5 | Δ AUC 95% CI | Fixed-cut W1 selected → λ=5 | Δ W1 95% CI | Shared MC/EXP pass |
|---|---:|---:|---:|---:|---:|
| First | 0.8832 → 0.7749 | −0.1161 to −0.1006 | 0.04585 → 0.05055 | +0.00308 to +0.00612 | 884 / 561 |
| Second | 0.8813 → 0.7730 | −0.1147 to −0.1007 | 0.05790 → 0.06265 | +0.00304 to +0.00646 | 892 / 543 |

**Conclusion:** An independent probe AUC of 0.5 was **not reached**. In this
tested family, increasing λ from 5 to 10 barely improves AUC (0.773→0.771)
while further reducing hit AP/recall. Compared with the selected checkpoint,
the best near-0.77 row costs about 2.4 percentage points of hit AP and 4.2
points of hit recall, and worsens fixed-cohort output-score W1 by about 0.0047.
This is a measured point on the Pareto frontier, **not** an estimate of the
unmeasured cost of 0.5. A constant encoder would force probe AUC=0.5 but also
reduce hit average precision to the positive-hit prevalence (about 0.0916 on
this holdout); that degenerate control is not useful domain adaptation.

The discriminator's own AUC (~0.54 for the λ=5 candidate) is not evidence of
invariant features: the independent probe still obtains ~0.77. MC and EXP may
also differ in physical event composition, so event-level invariance should
not be maximized without retaining the signal/noise task and output-level
checks.

Checkpoints and evaluation JSON files are on cluster62 under
`/home/plotnikovgp/tmp/baikal-da-strong-20261015/runs/pareto_*`. The selected
production candidate remains the checkpoint documented in
`da_output_w1_20261015.md`; **do not replace it with these stronger DANN
checkpoints**.
