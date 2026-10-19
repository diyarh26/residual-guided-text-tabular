# Experiment log

## October 19, 2026 - Phase 0 infrastructure

At the end of Phase 0, **no model training, baseline reproduction, benchmark
evaluation, or numerical performance experiment had been run**.

Activities: reviewed the complete eight-page course PDF, inspected repository
state/GitHub authentication, investigated historical upstream source, and
created documentation, source pin, and infrastructure checks. These are
research setup activities, not experimental evidence for the hypothesis.

Tooling note: `python` resolves to a Windows Store alias in this session; use
`py -3.11`. PDF extraction/rendering used installed `pypdf` and PyMuPDF. An
initial terminal print hit cp1252 Unicode encoding; the complete UTF-8 extraction
was read successfully and all eight rendered pages inspected. This did not
affect source content. PDF intermediates remain ignored under `tmp/pdfs/`.

Infrastructure validation passed locally and in a fresh public clone; staged
whitespace and publication-boundary checks passed, and the initial GitHub push
was verified. These checks do not evaluate a model. Details and source commit
provenance are in [HANDOFF.md](HANDOFF.md). No ML environment lock existed at
that milestone; Phase 1 subsequently added it.

## October 19, 2026 - Phase 1 setup and preregistered smoke baseline

User authorized Phase 1 after resolving dataset scope and separate per-dataset
training. Audited all 40 historical metadata schemas: 20 without image columns
are eligible. Prepared Kaggle Product Sentiment version 1 and pinned E5 revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52` with content hashes.

Environment setup initially selected Joblib 1.6.0, which removed
`joblib.externals.cloudpickle` required by Skrub 0.6.2. Early test-discovery
attempts failed during setup (NumPy was not installed yet, then the Joblib
incompatibility prevented imports), before any model experiment. Pinning original Joblib
1.5.2 resolved the incompatibility. Nine tests then passed, including original
split/metric behavior, train-only PCA, unknown-category handling, frozen encoder
inference, run safety, and fresh-clone bootstrap. A source review found and fixed
a fresh `--no-checkout` bootstrap guard bug before a network-clean setup claim.

Before any test score, fixed one CPU Joint Frozen LightGBM smoke configuration:
Product Sentiment, fold 0, original 1,000-row outer training cap, PCA 30, original
E5 prefix/pooling/tokenization and learner settings. Expected rows are 900 fit,
100 validation, and 509 test. This is a reduced baseline reproduction, not a
paper-table score or ReFuTab experiment. The executable code/config was committed
and pushed as `3188a243064f6c0eb8624dd5eed4744ec4d3c037` before the real run.

## October 19, 2026 - Completed original Joint Frozen smoke baseline

**Status: Completed.** Run ID `phase1-product-sentiment-frozen-fold0`.
Source `3188a243064f6c0eb8624dd5eed4744ec4d3c037`, clean at start; upstream
`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`, unchanged. The full machine-readable
[result](../results/phase1_product_sentiment_frozen.json) records exact revisions,
hashes, class order, packages, hardware, parameters, and artifact fingerprints.

| Item | Observed value |
| --- | --- |
| Dataset | Historical Product Sentiment, Kaggle version 1, 4 classes |
| Condition / learner | Original Joint Frozen / LightGBM |
| Original training cap / fold | 1,000 / 0 |
| Fit / validation / test rows | 900 / 100 / 509 |
| Test macro one-vs-rest AUROC | **0.8393533369478667** |
| Test log loss (secondary) | 0.8197310907083184 |
| Training / test inference seconds | 431.471 / 117.262 |
| Total wall seconds | 559.912 (9.33 minutes) |
| Sampled peak process RSS | 1,060,134,912 bytes (0.987 GiB) |
| Hardware / Python | Windows CPU, 8 logical CPUs, 4 Torch threads / 3.11.3 |
| Seeds | Base/outer/PCA/learner 42; internal validation 41 |

Command from that clean source commit:

```powershell
.venv\Scripts\python.exe scripts/run_baseline.py --config configs/phase1_frozen.json --run-id phase1-product-sentiment-frozen-fold0 --summary results/phase1_product_sentiment_frozen.json
```

PCA fitted on exactly 900 rows (384 to 30 dimensions), E5 remained in evaluation
mode with no gradients and identical before/after weight hashes, and exactly
one held-out prediction call occurred. Original AUROC/log loss independently
matched the saved predictions. Original source treats `Product_Type` as numeric;
the final input has that one structured column and 30 text PCA columns.

The independent [content audit](../results/phase1_content_audit.json) reproduced
the exact ordered splits and found no exact shared descriptions, feature pairs,
or full rows across these selected partitions. The full table still has four
excess duplicate full rows and seven duplicate descriptions; other folds require
their own audit. No deduplication or alternative split replaced the benchmark.

Raw IDs, predictions, PCA states, and logs persist locally under ignored
`results/raw/phase1-product-sentiment-frozen-fold0/`. Only reviewed aggregate
summaries are published. No external paid compute service was used; the run's
zero service charge excludes electricity/hardware. Later runner metadata leaves
provider charges unmeasured instead of assuming zero on other hosts.

Fresh public-clone source/versioned-data preparation passed; its first attempt
failed on temporary GitHub DNS resolution and succeeded on retry. The `.venv`
was installed fresh from the lock. Nine unit tests and notebook static checks
passed. Colab/GPU execution has not been performed. No model experiment failed.

Limitations: one multiclass task and one fold, reduced fitting sample, original
CPU defaults, and a minimal compatible dependency closure rather than the full
historical lock. No statistical improvement, ReFuTab, TAR, regression run, or
full-benchmark coverage is established. No test-driven tuning or second run was
performed. Phase 2 may now implement the general data interface and OOF teacher.

## October 19, 2026 - Phase 2 implementation checks and fixed smoke configuration

Implemented explicit dataset/schema and persisted split contracts plus a
structured-only CatBoost 1.2.8 teacher. The final full suite passed **43 tests
in 9.451 seconds**, including malformed-artifact guards. Infrastructure/link
and whitespace checks also passed.
Real tiny synthetic fits cover regression, binary and multiclass tasks, actual
model-fit row IDs, fixed-fold own-label exclusion, held-out label/feature
independence, category/missing handling, score conversion and checked save/load.
These are software checks, not benchmark improvement results. Expected-error
tests intentionally preserve failed manifests and reject overwritten artifacts.

Test setup initially attempted teacher imports before CatBoost installation;
locked environment installation resolved that dependency error. No Phase 2
benchmark model run has failed or completed at this entry's creation.

The configured pilot will use the exact Phase 1 Product Sentiment version-1
partitions and upstream source, 900 training/100 validation/509 reserved test
rows, with 3,582 excluded by the original reduced-cap sampling. The teacher
receives only numeric `Product_Type`. Each of three OOF models trains on 600
rows and predicts its excluded 300; a separate teacher trains on all 900 to
predict 100 validation rows. Seed 42, 40 iterations, depth 4, learning rate 0.1,
two CPU threads and native probabilities with clipping epsilon 1e-7 are fixed
before inspecting diagnostics. No fitted calibration, early stopping,
hyperparameter search, test prediction, or test scoring is included.

## Required entry for each later experiment

Record run ID/time/status (planned/running/completed/failed/interrupted), exact
project and upstream SHAs, config hash, dataset/model revisions and permissions,
split IDs/hashes, seed(s), command, dependency lock, hardware, runtime, peak
memory/cost, selected checkpoint criterion, metrics with definitions, artifacts,
and failure details. Include negative results. Changes to a run's configuration
create a new run ID. Never substitute correction-head scores for downstream
learner results or present partial benchmark coverage as complete.
