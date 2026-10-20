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
locked environment installation resolved that dependency error. This setup
preceded the real teacher run recorded below; no Phase 2 model experiment failed.

The configured pilot will use the exact Phase 1 Product Sentiment version-1
partitions and upstream source, 900 training/100 validation/509 reserved test
rows, with 3,582 excluded by the original reduced-cap sampling. The teacher
receives only numeric `Product_Type`. Each of three OOF models trains on 600
rows and predicts its excluded 300; a separate teacher trains on all 900 to
predict 100 validation rows. Seed 42, 40 iterations, depth 4, learning rate 0.1,
two CPU threads and native probabilities with clipping epsilon 1e-7 are fixed
before inspecting diagnostics. No fitted calibration, early stopping,
hyperparameter search, test prediction, or test scoring is included.

## October 19, 2026 - Phase 2 structured-only teacher smoke completed

Run **`phase2-product-sentiment-teacher-fold0`** completed from clean, pushed
source **`f805d81c0a5badc228f423b7b26befd5f137bafd`**. The immutable
[aggregate summary](../results/phase2_product_sentiment_teacher.json) contains
the exact configuration, upstream/data hashes, typed schema/content/split
fingerprints, dependency locks, hardware and raw artifact hashes.

```powershell
.venv/Scripts/python.exe scripts/run_teacher.py --config configs/phase2_teacher.json --run-id phase2-product-sentiment-teacher-fold0 --summary results/phase2_product_sentiment_teacher.json
```

| Item | Recorded outcome |
| --- | --- |
| Data | Product Sentiment version 1; original no-image schema; 5,091 rows; four classes |
| Fitting/prediction scope | Three OOF teachers, each 600 fit / 300 excluded-row predictions; one 900-fit / 100-validation teacher |
| Reserved rows | 509 outer test rows and 3,582 reduced-cap exclusions; no teacher fit/prediction |
| Teacher input | Numeric `Product_Type` only; text/target/IDs excluded |
| OOF diagnostic macro OVR AUROC / log loss | 0.806459815788321 / 0.5007548549097071 |
| Validation diagnostic macro OVR AUROC / log loss | 0.8202633422209448 / 0.4668987050618735 |
| Test predictions / scores | **0 / 0** |
| Runtime | **0.8031464 s**, excluding interpreter/import startup |
| Sampled peak process RSS during fit/save/load | **156,069,888 bytes**, approximately 0.145 GiB |
| Hardware | Existing local CPU, Intel Family 6 Model 140; 8 logical CPUs, 16,859,500,544 bytes RAM; CatBoost uses two threads |
| Cost | Script records unmeasured service charge; no paid external compute service used |

All runner checks passed: exactly one aligned OOF probability/offset vector per
training row, fitting/prediction row exclusion, separate A-only validation
teacher, explicit structured schema, no held-out labels in fitting, and native
model/array save/load. Class order: `Cannot Say`, `Negative`, `No Sentiment`,
`Positive`. No fitted calibration, checkpoint selection or hyperparameter
search occurred. The fixed configuration was committed before the diagnostics.

An independent read-only evidence audit matched every public/raw artifact
checksum and provenance hash; reconstructed Phase 1 assignments; checked the
three 600/300 fit/prediction audits, 900/100 validation audit and reserved-row
exclusion; verified `(900, 4)` OOF and `(100, 4)` validation arrays, class order,
finite offsets and metrics; and reproduced saved predictions from loaded
models. The audit performed no training or outer test prediction/scoring.
The public summary contains no raw IDs, prediction arrays, data or models.

Private bundles persist under ignored
`results/raw/phase2-product-sentiment-teacher-fold0/`; preserve this completed
attempt and its original summary. The old Phase 1 summary was not edited or
rerun. The software suite passed **43 tests in 9.451 s** before the recorded
run. Infrastructure/link, dependency-lock and publication checks passed.

Limitations: the real smoke covers one small multiclass structured-only teacher;
regression and binary paths were exercised with synthetic real-CatBoost tests.
These OOF/validation diagnostics are on different partitions from Phase 1's
test score and cannot establish a gain over that baseline. No fusion, TAR,
full-benchmark or final-test result is claimed. OOF teachers do not make future
supervised embeddings cross-fitted. Phase 2 meets its acceptance criteria;
Phase 3 fusion is next; wait for the user's instruction before starting it.

## October 19, 2026 - Phase 3 implementation and fixed synthetic smoke

User authorized Phase 3 and requested notice if GPU resources become necessary.
Implemented frozen pinned E5, fitting-only PCA-30 and structured preprocessing,
identity gate, three task losses, direct control, complete-pipeline fitting,
head/teacher-free inference, native downstream LightGBM and checked checkpoints.
**86 tests passed in 18.730 seconds**, with lock/infrastructure/link checks also
passing. Regression/binary/multiclass pipelines use real CatBoost, neural gates
and LightGBM with explicit deterministic text substitutes in unit tests.
Exact CPU dropout/shuffle continuation, forbidden/held-out data boundaries,
artifact semantic checks and CSV prediction are exercised.

An initial downstream helper test failed because Windows checkout bytes were
CRLF while the historical Git blob uses LF. Hashing normalized source bytes
against the inspected pinned blob fixed that provenance check; no historical
source was modified. No model experiment failed during implementation.

The fixed `phase3_smoke.json` generates a synthetic binary task: 60 fitting,
12 validation and 12 reserved test rows, two text fields, two numeric columns
and one categorical field. Actual E5-small-v2 is frozen at the same immutable
revision as Phase 1; PCA and vocabularies fit only those 60 rows. Gate hidden
64/dropout .1, category embeddings 8, context width 16, linear temporary head,
five AdamW epochs/batch 20/lr .001/weight decay .0001/patience 3/seed 42/two
CPU threads are fixed before diagnostics. Downstream uses 100 LightGBM trees.
This is an architecture smoke, not a benchmark pilot or improvement test.
The clean-source run and real inference readback are pending at this entry.

## October 19, 2026 - Phase 3 real-E5 synthetic smoke completed

Run **`phase3-synthetic-binary-real-e5`** completed from clean pushed source
**`23d4ce34a6e3b23f458dd9bf1d06d1c1be4035d3`**. The configuration above was
committed before diagnostics; no tuning or second training run followed.
The [original summary](../results/phase3_synthetic_binary_fusion.json) contains
source/config/lock hashes, generated-data/schema/split fingerprints, pinned
E5 hashes, hardware, learning curves and all raw artifact hashes.

```powershell
.venv/Scripts/python.exe scripts/run_fusion.py --config configs/phase3_smoke.json --run-id phase3-synthetic-binary-real-e5 --summary results/phase3_synthetic_binary_fusion.json
.venv/Scripts/python.exe scripts/predict.py --bundle results/raw/phase3-synthetic-binary-real-e5/bundle --input results/raw/phase3-synthetic-binary-real-e5/validation_features.csv --output results/raw/phase3-inference-readback/predictions.csv --threads 2
```

| Item | Actual outcome |
| --- | --- |
| Data | Original synthetic binary task; two text/two numeric/one categorical field |
| Partitions | 60 fitting, 12 validation, 12 reserved test rows |
| Teacher | Three 40-fit/20-excluded-row OOF models; separate 60-fit/12-validation teacher |
| Text | Actual pinned frozen E5-small-v2; fitting-only PCA `(2,30,384)` components |
| Gate selection | Epoch 1 selected; stopped after epoch 4 of maximum 5, patience 3 |
| Selected validation correction loss | 0.6027021408081055 |
| Downstream | Original structured block plus 60 fused coordinates; 100-tree LightGBM |
| Validation diagnostic AUROC / log loss | 0.7142857142857143 / 0.8421002030754804 |
| Reserved-test predictions / scores | **0 / 0** |
| Runtime | **47.227143 s**, excluding interpreter/import startup |
| Sampled peak process RSS | **987,308,032 bytes**, approximately **0.920 GiB** |
| Hardware | Existing Intel CPU; two Torch threads; no GPU |
| Cost | Unmeasured in script; no paid external compute service used |

All runtime checks passed. E5 state hash before/after remained
`880ab04eba37fac79b75f1697064b281566657e68b937eab5057c5c031cfb627` and all six
verified model/tokenizer files remained unchanged. PCA and structured fit IDs
equaled the 60 permitted fitting IDs. OOF teachers excluded their prediction
rows and validation was separate. The inference bundle has no teacher or
temporary-head weights; it keeps the trained structured encoder/gate and native
downstream learner, with verified pretrained E5 referenced externally.

Cold standalone CSV inference loaded the saved bundle and E5 in a new process.
Its saved class order (`negative`, `positive`) and all 12 validation row IDs
matched the reference. Maximum probability difference was
**1.1102230246251565e-16**. The separate
[readback audit](../results/phase3_inference_audit.json) checked all **20 raw file
hashes**, content/split binding, OOF row exclusions/coverage, preprocessing
scope, selected versus last checkpoint, class/order/shape, recomputed metrics
and unchanged E5 assets. That audit performed no training or test scoring.

Private artifacts persist under ignored
`results/raw/phase3-synthetic-binary-real-e5/`; separate CSV readback is under
`results/raw/phase3-inference-readback/`. Preserve them and their public
summaries. The original Phase 1/2 results were neither edited nor rerun.
The final source suite passed **86 tests in 18.730 s**; after adding the private
validation CSV export and CLI thread option, the three targeted runner/CSV
checks passed again in **3.116 s**. Dependency-lock, infrastructure/link,
publication and whitespace checks passed.

Limitations: synthetic validation only, no matched benchmark baseline or
statistical comparison. Training correction loss decreased slightly, while
validation correction loss was lowest at epoch 1; this does not establish useful
complementarity. Regression/multiclass pipelines and the direct control were
exercised with real neural/teacher/downstream models and explicit text fixtures
in tests; only this binary smoke used actual E5. Exact CPU continuation was
tested; GPU and Colab execution remain unvalidated. The nested benchmark
diagnostic is specified but not executed. No Phase 3 model experiment failed.
Phase 3 meets its implementation acceptance checks. Phase 4 pilots and the
Joint TAR GPU environment remain the next phase, pending user instruction.

## Phase 4 registration and software checks — October 19, 2026

Status: implementation/preparation complete; real pilot execution pending.
Registered `configs/phase4_pilot.json` and `configs/phase4_datasets.json`
before new method fits/scores. Product Sentiment v1 and Video Games Sales v1
both have original `image_col: null`. All six main and six grouped nested
case preflights pass, including historical A-only feature classification.
No outer-test prediction was made by these preflights.

The CPU full suite passed **152 tests in 32.508 seconds**. Coverage includes
grouped complete-pipeline splits and teacher exclusion, frozen cache/PCA,
original TAR integration with explicit CPU fixtures/native PEFT restoration,
stage and outer-prediction recovery, corruption rejection, aggregation gates,
and static notebook/source/launch guards. During development, a new test first
failed to import because of an extra parenthesis, then exposed that corruption
in a completed unselected stage was not checked on continuation. Both were
fixed; continuation now verifies every completed stage's exact file inventory.
These were software-test failures, not failed benchmark experiments.

CPU dependency lock validation and infrastructure/local-link/publication checks
passed. The CUDA Linux lock resolves separately and retains common CPU package
versions. No GPU wheel was installed locally and no CUDA/Colab run is claimed.
The recorded CPU preflight will follow a clean-source commit; it will not
include TAR or unlock the outer tests. See [PHASE4.md](PHASE4.md).

## Phase 4 CPU preflight and recovery — October 19–20, 2026

Run `phase4-product-sentiment-cpu-preflight`, source
`c1e7ec1cdbb79fbc2f000a060616a5e73e11c0d1` (clean and pushed), pinned upstream
`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`. Configuration SHA-256
`df6294fb90f8cb9dd72a2db52fb690eaa3752fb6b87a3d7a50ec7a39166d26cf`.
The registered config, dataset/model revisions, environment and split fingerprints
are in the reviewed [summary](../results/phase4_cpu_preflight.json); its
[readback audit](../results/phase4_cpu_preflight_audit.json) records provenance
and checks. This completes the CPU subset, **not Phase 4**.

Original command:

```powershell
.venv/Scripts/python.exe scripts/run_pilot.py --config configs/phase4_pilot.json --run-id phase4-product-sentiment-cpu-preflight --methods frozen direct residual --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --device cpu
```

The October 19 attempt encoded only the 1,000 benchmark-training rows with real
pinned frozen E5, completed frozen/direct stages, then failed during residual
checkpoint replacement with Windows `PermissionError: WinError 5`. Exact locking
cause is unconfirmed. All original 48 files and failed-attempt records remain
untouched under ignored `results/raw/phase4-product-sentiment-cpu-preflight/`.
On October 20, a byte-verified copy was made outside OneDrive and continued:

```powershell
$pilotResume = Join-Path $env:LOCALAPPDATA 'ReFuTab\runs\phase4-product-sentiment-cpu-preflight'
.venv/Scripts/python.exe scripts/run_pilot.py --config configs/phase4_pilot.json --resume-run $pilotResume --methods frozen direct residual --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --device cpu
```

Completed frozen/direct artifacts were verified and reused. Residual attempt 2
continued the saved epoch-3 optimizer/RNG state through epoch 10; old attempt 1
was preserved. All six grouped nested stages then completed. Each of eight
neural stages selected epoch 10 of the unchanged 10-epoch budget. There was no
model tuning or algorithm change after results.

| Condition | Main validation macro OVR AUROC | Main validation log loss |
| --- | ---: | ---: |
| Matched frozen adapter | 0.8166533713344859 | 0.95678180680636 |
| Direct gated | 0.8057462756561723 | 0.9669574899101643 |
| Residual gated | 0.800067021857754 | 0.9725217634341133 |

The three grouped D residual-minus-direct AUROC differences were
**-0.0244157, +0.0082152, -0.0106366**: mean **-0.0089457**, sample SD
**0.0163810**, one positive fold out of three. All downstream training AUROCs
were 1.0. These internal diagnostics show training optimism and no consistent
residual gain; they are not test scores or independent-sample evidence.
The fixed gate remains `await_required_comparisons`: TAR, regression and other
outer conditions are missing. No redesign or broader evaluation followed.

Completed **9/36 registered stages**, **zero new outer-test predictions/scores**.
The 509 main outer-test rows remained excluded from every fit and diagnostic.
The audit verified 142 completed-stage files, 16 teacher fits, 24 metric sets
recomputed from saved predictions, all 12 split fingerprints, grouped A/V/D and
nested OOF isolation, train-only PCA/structured/downstream fitting, and unchanged
six E5 assets. The audit performed no new model fitting or prediction.

Measured execution: **286.691709 s** initially plus **39.081796 s** on continuation
(sum **325.773505 s**); setup adds 4.235228 s and 3.364589 s, respectively.
These exclude interpreter/import startup and the overnight interruption.
Sampled process peak RSS: **999,112,704 bytes** initially; **658,583,552 bytes**
on continuation. Existing Intel CPU, two Torch threads, no GPU. Script records
service cost as unmeasured; no paid external compute was used.

The GPU notebook now prefills the reviewed implementation SHA while retaining
`LAUNCH_PILOT = False`. Actual CUDA/Colab/TAR execution remains unvalidated.
Start a fresh full GPU run; never merge CPU models into its comparison.
Final October 20 verification passed **152 tests in 25.723 seconds**, including
the prefilled-SHA and malformed-SHA guards. CPU lock, infrastructure/local-link,
publication-boundary and whitespace checks also passed.

## Phase 4 Colab preparation failure and backend repair — October 20, 2026

Status: setup/test failure reported by the user; local fix verified; Colab retry
and pilot training pending. The two supplied attachments are byte-identical
copies of one log, not two independent failures. Evidence and normalized details
are in [phase4_colab_setup_failure.json](../results/phase4_colab_setup_failure.json).
Bootstrap ID: `phase4-colab-20261020T140148Z-67a87c90`.
The notebook requests project source
`c1e7ec1cdbb79fbc2f000a060616a5e73e11c0d1`; the supplied excerpt omits actual
project-HEAD and hardware output, so neither is independently verified here.

The log confirms successful preparation of historical source
`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`, both versioned no-image dataset
CSVs/metadata with matching hashes, and all six frozen E5 files at revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52`. The isolated Python executable
was 3.11.15 on Linux. The GPU environment's test command reported **126 tests
in 8.803 s, failed with five import errors**. These whole-module import failures
prevented discovery of the full expected suite; do not count 126 as a full pass.
No registered pilot fitting or outer-test evaluation appears in this log.

Every failed module imported LightGBM, whose Matplotlib import inherited
`MPLBACKEND=module://matplotlib_inline.backend_inline` from Colab. That backend
is unavailable in the isolated environment. A local subprocess reproduced the
same `ValueError`; setting `MPLBACKEND=Agg` made Matplotlib/LightGBM import
successfully. The notebook helper now uses a copied child environment with
`MPLBACKEND=Agg` and logs that setting. It leaves the parent kernel and other
environment variables intact. Source pin, model/data/configuration, dependencies
and scientific training budgets are unchanged.

All **six notebook checks passed in 2.133 s**, including a real subprocess
regression test with the inherited Colab setting, successful dependency imports,
and preserved parent environment. No GPU or benchmark model was trained locally.
Existing Colab sessions can add `import os` and `os.environ["MPLBACKEND"] = "Agg"`
above the failed preparation/tests cell and rerun that cell, reusing verified
inputs and the current source checkout. Training remains behind the test/launch
guards. Actual Colab retry success, hardware, training runtime/memory and service
cost remain unreported; keep the failed bootstrap log in private Drive.

## Required entry for each later experiment

Record run ID/time/status (planned/running/completed/failed/interrupted), exact
project and upstream SHAs, config hash, dataset/model revisions and permissions,
split IDs/hashes, seed(s), command, dependency lock, hardware, runtime, peak
memory/cost, selected checkpoint criterion, metrics with definitions, artifacts,
and failure details. Include negative results. Changes to a run's configuration
create a new run ID. Never substitute correction-head scores for downstream
learner results or present partial benchmark coverage as complete.
