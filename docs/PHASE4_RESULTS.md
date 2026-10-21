# Phase 4 GPU pilot: reviewed findings

Reviewed October 21, 2026. **All registered training/evaluation jobs completed;
full saved-artifact verification remains open.** The descriptive recommendation
is **analyse failures and overfitting before scaling**. Do not launch full
evaluation or Option B from this result, repeat the finished run, or tune against
its outer-test scores. The method has not established a consistent advantage.

## Evidence and execution

Run `phase4-checkpointed-20261020T145755Z-dd78ef55` used clean source
`6e09955514caffa52050459377e5b4736dedca75` and the registered checkpointed config.
All **36 fitting stages and 24 outer-test evaluations**, including six TAR
stages/evaluations, completed on Tesla T4. Activation checkpointing preserved
FP32, batch 256 and the registered training budgets. The original OOM run is
separate and preserved.

The saved execution ran from October 20, 14:58:15 UTC to 17:34:04 UTC:
**9,349.610 seconds (2 h 35 min 50 s)**, excluding 3.922 seconds of initial setup.
Sampled peak process RSS was **1,876,746,240 bytes**; this is host process memory,
not GPU memory. Provider service charge is unmeasured. The earlier synthetic
capacity check's 6.510 GiB allocation is not the whole run's peak GPU usage.

The [reviewed result](../results/phase4_gpu_pilot_review.json) records source,
config/lock hashes, dataset/split fingerprints, hardware/environment, paired
fold scores, grouped diagnostics, runtime and limitations. Raw JSON, row IDs,
model files, arrays and checkpoints remain private in Drive and ignored locally.

## Final downstream outer-test results

The complete [score CSV](../results/phase4_all_scores.csv) contains 108 records:
36 training, 36 internal validation, 12 grouped diagnostic and 24 outer-test
score sets. Classification includes AUROC/log loss; regression includes
R2/MSE/RMSE. Counts refer to score sets, not data rows. Exporting the saved
scores performed no new fitting, prediction or metric recomputation.

## Datasets and sample sizes

These are two version-1 MulTaBench pilot datasets under the pinned historical
source, both with **no original image column**. The remaining 18 eligible
datasets have not been evaluated by this pilot.

| Dataset | Full rows | Task / target | Text fields | Structured fields |
| --- | ---: | --- | --- | --- |
| [Product Sentiment v1](https://www.kaggle.com/datasets/chico89/multabench-product-sentiment/versions/1) | 5,091 | Four-class classification / `Sentiment` | `Product_Description` | `Product_Type`, numeric as in the pinned source |
| [Video Games Sales v1](https://www.kaggle.com/datasets/chico89/multabench-video-games-sales/versions/1) | 16,598 | Regression / `Global_Sales` | `Name`, `Publisher` | `Year`, `Platform`, `Genre` |

Each of three repeated outer holdouts used **900 fit + 100 internal validation**
rows. Main test sets contained 509 sentiment rows and 1,659 game-sales rows per
holdout. Remaining rows were excluded by the pilot cap: 3,582 and 13,939,
respectively. This is not full-data training or three disjoint K-fold test sets.
TAR additionally reserves its own 810/90 fit/validation partition inside the
900 permitted fitting rows. Diagnostic pipelines use smaller grouped splits
inside outer fold-0's 1,000-row training/validation pool; their exact score-row
counts appear in the CSV.

Sentiment classes: `Cannot Say`, `Negative`, `No Sentiment`, `Positive`.
Game names may repeat across main historical row holdouts; diagnostics group
by `Name`. Sentiment diagnostics group by `Product_Description`.
Dataset/feature provenance is in [the registered manifest](../configs/phase4_datasets.json).

## Main results by holdout

All four methods use LightGBM as the final learner. Frozen uses fixed E5/PCA
features; direct trains the gate on targets; residual is ReFuTab with OOF
structured-teacher offsets; TAR adapts the text encoder before downstream fit.

| Dataset / metric | Outer fold | Joint Frozen | Direct gated | Residual guided | Joint TAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product Sentiment / AUROC | 0 | 0.827275 | 0.832586 | 0.834818 | 0.895623 |
| Product Sentiment / AUROC | 1 | 0.861943 | 0.878527 | 0.876078 | 0.874298 |
| Product Sentiment / AUROC | 2 | 0.841015 | 0.833178 | 0.831509 | 0.855607 |
| Video Games Sales / R2 | 0 | 0.228539 | 0.152167 | 0.208082 | 0.256311 |
| Video Games Sales / R2 | 1 | 0.226265 | 0.170408 | 0.163027 | 0.220466 |
| Video Games Sales / R2 | 2 | 0.153422 | 0.081322 | 0.076849 | 0.206029 |

## Main means and variability

Means and sample standard deviations across three registered overlapping
holdouts; higher is better. These are descriptive, not independent-sample
significance estimates. AUROC and R2 are different metrics and are not averaged.

| Method | Product Sentiment macro OVR AUROC | Video Games Sales R2 |
| --- | --- | --- |
| Joint Frozen | 0.843411 +/- 0.017457 | 0.202742 +/- 0.042727 |
| Direct gated | 0.848097 +/- 0.026355 | 0.134632 +/- 0.047060 |
| Residual guided (ReFuTab) | 0.847469 +/- 0.024832 | 0.149319 +/- 0.066682 |
| Joint TAR | **0.875176 +/- 0.020022** | **0.227602 +/- 0.025889** |

Residual-minus-TAR mean differences were **-0.027707 AUROC** and **-0.078283 R2**.
Residual-minus-direct differences averaged **-0.000628 AUROC** and **+0.014687
R2**, with a positive difference in only **one of three holdouts on each task**.
The regression average benefit over direct was driven by fold 0; it was not
consistent. Residual beat frozen on average for classification, but trailed
frozen on every regression holdout. TAR had the highest mean on both tasks.

## Grouped complete-pipeline diagnostics

These evaluate unseen groups within the outer training pool, using independently
fitted complete pipelines; they are distinct from the main outer-test scores.

| Dataset / metric | Diagnostic fold | Direct held-out D | Residual held-out D |
| --- | ---: | ---: | ---: |
| Product Sentiment / AUROC | 0 | 0.839610 | 0.822893 |
| Product Sentiment / AUROC | 1 | 0.918782 | 0.914596 |
| Product Sentiment / AUROC | 2 | 0.858585 | 0.825613 |
| Video Games Sales / R2 | 0 | -0.023665 | -0.251296 |
| Video Games Sales / R2 | 1 | 0.007881 | 0.048835 |
| Video Games Sales / R2 | 2 | -0.009693 | -0.015919 |

| Dataset / metric | Direct held-out D mean | Residual held-out D mean | Residual minus direct | Positive paired folds |
| --- | --- | --- | --- | --- |
| Product Sentiment / AUROC | 0.872326 | 0.854367 | -0.017958 | 0/3 |
| Video Games Sales / R2 | -0.008492 | -0.072793 | -0.064301 | 1/3 |

Both sentiment methods achieved training AUROC 1.0 on all diagnostic folds.
Mean train-minus-D gaps were 0.127674 for direct and 0.145633 for residual.
Regression mean gaps were 0.728773 and 0.806989 respectively. These indicate
substantial training optimism and weak grouped generalization, particularly
for regression. They do not establish leakage or isolate its cause. Grouping
and smaller fitting samples also change task difficulty relative to main
historical row holdouts.

## Decision and verification limits

The preregistered `phase4_descriptive_continue_v1` rule returns
`analyze_failures_and_overfit_before_scaling`. Classification fails all four
criteria; regression passes only positive mean residual-minus-direct outer
score. No automatic redesign or new fitting followed this review.

The read-only review passed **209 checks**: independently recomputed aggregation;
global manifest/summary agreement; immutable source config and dependency-lock
hashes; all 36 stage-manifest hashes, scores, provenance and split bindings;
24 outer result/request bindings; and chronology. The last fit completed at
17:29:35.726372 UTC, preceding the earliest Drive evaluation-request creation
at 17:29:36.735 UTC. The eleven existing aggregation tests passed in 0.019 s.
The documentation/publication validator and staged whitespace checks passed.

This is a metadata/summary review. All weight files, teacher fitting-row scopes,
checkpoint tensors and prediction-array metrics have **not** been independently
read back in this review. The training runner records its own file inventories,
saved-model prediction round trips and prerequisite-stage verification; those
runtime checks do not replace the remaining independent artifact audit.
Phase 4 remains open for that audit and targeted failure/overfit analysis.

Next: audit existing saved predictions and fitting scopes without new training
or test-driven selection, investigate the regression/grouped-generalization
weakness, and document the final decision. Preserve the completed run. Phase 5
and Option B remain unstarted; no additional GPU fitting is needed for review.
