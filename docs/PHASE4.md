# Phase 4: matched pilot and decision gate

Status: **In progress**. The registered pilot configuration, source runner,
grouped diagnostics, result aggregation, separate GPU environment and thin
Colab notebook are implemented. A real GPU/TAR execution and the full matched
pilot remain pending. Infrastructure or subset completion does not complete
Phase 4. Read [HANDOFF.md](HANDOFF.md) and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
for the latest actual execution evidence.

## Registered scope

[phase4_pilot.json](../configs/phase4_pilot.json) fixes datasets, methods,
budgets, split rules and reporting before new pilot method scores. This is a
project registration in committed source, not an external preregistration.
The earlier Product Sentiment fold-0 Joint Frozen test score and Phase 2
teacher diagnostics were already seen. The pilot is exploratory; it cannot be
described as a newly blind confirmatory test. Dataset selection uses schema
and compute considerations rather than new method outcomes.

| Dataset | Task and primary metric | Text fields | Structured fields |
| --- | --- | --- | --- |
| `MUL_TEXT_PRODUCT_SENTIMENT` | Four-class, macro OVR AUROC | `Product_Description` | Numerical `Product_Type` |
| `REG_TEXT_VIDEO_GAMES_SALES` | Regression, R² | `Name`, `Publisher` | Numerical `Year`; categorical `Platform`, `Genre` |

Both original version-1 schemas have no image column. Every original curated
feature is retained, while labels, row IDs and target-derived metadata are
excluded from model features. In particular, Video Games has no regional-sales
or rank features. Original schemas, immutable archive/member hashes, content
audits and source limitations are recorded in
[phase4_datasets.json](../configs/phase4_datasets.json) and
[PHASE4_DATA.md](PHASE4_DATA.md). Raw data stays private and ignored.

Outer fold indices **0, 1, 2** invoke unchanged historical benchmark sampling
and splitting with a requested training cap of 1,000. Their outer seeds are
42, 43, 44; the source's separate internal validation seed remains 41. These
are **overlapping repeated holdouts**, rather than three disjoint evaluation
folds. Actual fit/validation/test counts and IDs are saved before fitting.
The Product Sentiment fold-0 assignments must match the archived Phase 1
reference; its prior score does not select a new model.

## Four matched conditions

All conditions use the same configured 100-tree LightGBM downstream adapter,
seed 42, original structured columns, training-only numerical/categorical
handling, task class order and held-out assignments. The source detector audits
the declared schema on fitting rows before execution. This is a controlled
matched-adapter comparison; the unchanged complete Phase 1 baseline execution
remains separately identified.

| CLI method | Representation and selection |
| --- | --- |
| `frozen` | Verified fixed E5-small-v2, then independent fitting-only float32 sklearn PCA-30 per text field. |
| `tar` | Unchanged pinned historical shared E5/LoRA training and its own internal selection, followed by fitting-only PCA-30 per field. Joint TAR is the primary baseline. |
| `direct` | Phase 3 frozen E5/PCA and the same gate architecture as residual, trained on the direct target without teacher offsets. |
| `residual` | Phase 3 frozen E5/PCA and gate, trained with 3-fold OOF structured-only CatBoost offsets; the teacher and temporary head are absent from final inference. |

Frozen/TAR preserve the historical `astype(str).fillna('')` text conversion.
Because conversion happens first, missing values can become literal `nan`,
`None` or `<NA>` text. Frozen PCA uses the original float32 sklearn math and
per-field matrix layout. Direct/residual share Phase 3's empty missing-text
policy and float64 PCA fitting followed by float32 features. Those two methods
therefore isolate the residual objective with matched preprocessing and
architecture; the documented source-baseline differences must accompany any
broader interpretation.

Direct/residual use the same fixed ten-epoch maximum, patience three, learning
rate 0.001, batch size 32, hidden width 64 and dropout 0.1. Their correction loss
on independent validation selects the gate; downstream fitting uses original
fitting targets. The teacher keeps the Phase 2 fixed 40 iterations/depth 4/rate
0.1 budget. No outer-validation or test label trains a teacher.

TAR explicitly supplies the historical CLI settings: LoRA rank 16, last three
layers, alpha 32, dropout 0.1, learning rate `1e-4`, weight decay 0.001, batch
256, at most 50 epochs and patience three. The source's FP32 training and
512-token maximum are retained, without automatic batch shrinking or mixed
precision. Regression TAR binning and its additional internal validation split
are owned by the pinned implementation and audited inside permitted fitting
rows. These settings are execution choices, not new empirical conclusions.

## Grouped complete-pipeline diagnostic

OOF teachers do not make the gate's training embeddings cross-fitted. For
each dataset's outer fold 0, the diagnostic takes its benchmark-training pool
`B_train = A + V` and makes three held-out diagnostic folds `D`. Exact text
groups use `Product_Description` for Product Sentiment and `Name` for Video
Games. Classification uses stratified grouped folds; regression uses grouped
folds. A fixed grouped split reserves 10% of each diagnostic complement's
groups for further internal selection. Classification chooses the first of at
most 100 deterministic candidates covering all classes in both A and V; this
is a class-coverage rule and never a score search. Infeasible splits fail.

Each diagnostic independently fits **direct and residual complete pipelines**:
new A-only PCA/structured preprocessing, new gate and downstream learner, and
new grouped three-fold OOF teachers for residual. V selects the gate. D fits
and selects nothing. The benchmark outer test is explicitly excluded. The
three D partitions cover that outer training pool once; their feature groups
do not overlap their corresponding A or V. Teacher folds inside diagnostic A
also preserve those groups.

Record train/validation/D scores, train-minus-D optimism, correction-loss
curves and effective sample sizes. Each pipeline retains its own feature
coordinates. Aggregate matched task predictions and metrics; do not concatenate
separately fitted PCA/gate vectors into one downstream table. Historical main
outer splits remain row-based, with exact group overlap audited and reported.
Grouping the diagnostic does not retroactively change the main benchmark
protocol or establish absence of all possible duplicate-content effects.

## Stage counts and test barrier

| Work | Count |
| --- | --- |
| Main fits: 2 datasets × 3 outer holdouts × 4 methods | 24 stages |
| Nested fits: 2 datasets × 3 grouped diagnostic folds × 2 gated methods | 12 stages |
| Total prerequisite fitted stages | **36** |
| Locked main outer-test prediction/evaluation calls | **24** |

These counts describe complete method stages, rather than individual CatBoost
models, epochs, prediction rows or separate CUDA jobs. All 36 stages must be
completed and verified before the first main outer-test prediction. Nested D
predictions are allowed earlier because they are inside benchmark training.
Raw-table hashing and stratification are non-fitting uses; no test features or
labels enter model fitting or selection.

A bounded CPU preflight can run only Product Sentiment outer fold 0 for
`frozen direct residual`: three main stages plus six nested stages, **9/36**.
It can report internal validation and grouped-D diagnostics while making
**zero outer-test predictions**. Its incomplete comparison supports no
improvement recommendation. CPU/GPU builds, settings, device and numerical
states have different provenance bindings. A subsequent GPU comparison needs
a **new complete GPU run**, rather than continuing or mixing CPU stages.

## Commands and artifacts

Prepare inputs without training:

```powershell
py -3.11 -m uv sync --locked
py -3.11 -m uv run --frozen python scripts/prepare_phase1.py
py -3.11 -m uv run --frozen python scripts/prepare_phase4.py
py -3.11 -m uv run --frozen python -m unittest discover -s tests -v
```

Optional bounded local CPU preflight, from clean committed source:

```powershell
py -3.11 -m uv run --frozen python scripts/run_pilot.py --config configs/phase4_pilot.json --run-id my-new-phase4-cpu-preflight --methods frozen direct residual --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --device cpu
```

For the complete GPU run, open the
[Colab notebook](https://colab.research.google.com/github/diyarh26/residual-guided-text-tabular/blob/main/notebooks/phase4_pilot.ipynb),
set the full reviewed project SHA and follow [PHASE4_GPU.md](PHASE4_GPU.md).
The default placeholder and explicit launch guard prevent implicit training.
The source command uses the separate locked GPU environment and durable Drive
output:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project environments/gpu --frozen python scripts/run_pilot.py \
  --config configs/phase4_pilot.json --run-id <unique-gpu-run-id> \
  --output-root /content/drive/MyDrive/ReFuTab/runs
```

Each new run keeps a global manifest, aggregate `summary.json`, full split
files, immutable method attempts, predictions, checksums, failure traces and
frozen-text cache under its private run directory. Completed stages are checked
before reuse. A failed/interrupted stage gets a new attempt directory. Fusion
epoch checkpoints and audited TAR Trainer checkpoints can resume only when
their source/data/config/split/environment bindings match. A partial LightGBM
fit restarts. Frozen text rows can be reused across requested blocks within
the exact frozen-model/runtime binding; fitted PCA, structured state, teachers,
gates and target-adapted encoders are never shared between evaluation folds.
See the GPU guide for the `--resume-run` contract and its unverified runtime
limits. Publish only reviewed aggregate summaries, never full row IDs, model
weights, embeddings, data or unrestricted failure logs.

## Descriptive recommendation

`pilot_report.aggregate_pilot` validates coverage, statuses, finite task metrics,
paired row alignment and the test barrier. It reports AUROC and R² separately,
matched residual-minus-TAR/direct/frozen differences, and grouped-D comparisons.
Means and sample standard deviations describe these overlapping holdouts;
there is no independent-sample significance test, p-value or cross-task average.

The fixed rule `phase4_descriptive_continue_v1` recommends `continue_unchanged`
only after all 36 valid stages and 24 outer evaluations are complete and,
**on each dataset**, all of the following hold:

- Mean residual-minus-TAR outer score is positive.
- Mean residual-minus-direct outer score is positive, with positive differences
  on at least two of the three outer holdouts.
- Mean residual-minus-direct score on the three grouped diagnostic D folds is
  positive.

Complete comparisons failing these criteria produce
`analyze_failures_and_overfit_before_scaling`. Missing or invalid required
comparisons produce `await_required_comparisons`, with missing TAR explicitly
marked pending GPU for CPU work. The rule is a descriptive pilot decision,
not evidence of statistical significance or all-benchmark improvement. Report
the actual results and recommendation to the user before full evaluation;
the runner never redesigns the method or starts Phase 5 automatically.
