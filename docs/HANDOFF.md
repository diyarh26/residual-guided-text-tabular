# Project handoff

Updated: October 22, 2026 (Asia/Jerusalem).

## Current phase and status

**Next-direction discussion, October 22 (proposal only):** user asks what to do
after the weak validation result. Recommended next practical probe is I1
conditional innovation subspaces from the existing seven-idea note: investigate
which predictive text directions survive compression, then compare equal-size
ordinary supervised projection with conditional residual-guided projection.
Start with full frozen 384-dimensional embeddings, keep the main output budget
30 per field and downstream learner fixed, and use training-side grouped V.
PCA-30 and full-384 Frozen are diagnostic controls; full-384 is not an
equal-feature-budget comparison. Supervised PCA/PLS is an essential control,
since generic supervised compression is established work. This is a proposal
for a small CPU probe, not an accepted globally novel method, an implemented
redesign, permission to change the interrupted experiment, or a new GPU run.
The old run/checkpoints remain preserved and incomplete. Finish/assessment
decisions and any revised evaluation matrix must be declared explicitly.

**Preliminary validation interpretation, October 22:** user asked for an opinion
from current scores. The [scalar review](../results/scaling_partial_validation_review_20261022.json)
compares the fixed GitHub snapshot without new fitting/inference. On sentiment,
three-fold mean V AUROC at 900/2700/all is Frozen .84543/.85718/.87321,
Direct .85377/.86660/.87783, Residual .85079/.85248/.87314. Residual improves
absolutely with data but does not show a mean advantage over Direct at any
size, and is effectively tied with Frozen at largest A. Largest residual beats
Direct in 2/3 folds, with smaller gains outweighed by one negative gap. On the
two matched largest-A TAR folds, residual .87537 trails TAR .89965 (both gaps
negative); never compare TAR's two-fold mean as though it had three folds.
All 18 gates selected epoch 10 at the budget limit, motivating a later budget/
teacher diagnostic without proving nonconvergence or that more epochs help.
These are selection-validation observations, not D/generalization conclusions.
Keep the current recipe/run unchanged until the locked remaining comparisons
are available; no redesign or new GPU work was authorized/executed by this review.

**Actual standalone GPU run interrupted, GitHub progress recovered October 22.**
The user reports a usage-limit stop. GitHub branch
`results/scaling-20261021T182849Z-aa0994f6`, commit
`6baa18bdbc7041d8749b25f5420bbd3db7794518`, contains 35/72 completed fits and
one in-progress TAR fit, all on Product Sentiment (9 Frozen/9 Direct/9 Residual/
8 TAR completed). The sales dataset has no saved fitted stage yet. The CSV has
210 scalar metric rows / 70 score sets (35 train, 35 validation). D assessments
are zero because all 72 fits must finish first; zero old outer-test predictions.
The last upload was October 21 at 22:50:38 UTC (October 22, 01:50:38 Jerusalem).
The saved `running` status is a last snapshot, not proof of a live process.
This confirms real execution and actual GitHub publication, superseding the
pre-execution statements below. No algorithm-success decision is made from
these partial validation scores. Source is the unchanged `4971335...` pin.

Private checkpoints/logs are configured under the Drive mounted in that runtime:
`MyDrive/ReFuTab/standalone_scaling/scaling-20261021T182849Z-aa0994f6/`, with
bootstrap console logs in its sibling `_bootstrap` directory. That Drive was
not accessed in this readback; do not claim independent checkpoint validation.
The snapshot is cached privately in `results/raw/scaling-github-review-20261022/`.
Next: resume the same pinned notebook using that same Drive/current.json when
GPU resources are available; verified completed stages should be reused and
the interrupted TAR checkpoint recovered if compatible. No training launched
during this read-only progress check.

**Latest user instruction supersedes the transfer workflow below:** provide one
fully runnable notebook for another computer/Drive, with results saved to GitHub.
Current handoff is `notebooks/ReFuTab_Run_All.ipynb`: select GPU and Run all;
authorize Drive and enter a GitHub repository-write token through hidden input.
No old run, transfer or launch-variable edit is required. The standalone config
explicitly records the previous pilot audit as deferred to separate review;
it never claims that audit passed. All new data/split/grouped teacher/train-only
checks, fixed recipes and all-fit D barrier remain in place. Private checkpoints
and logs stay in the new Drive, with automatic compatible resumption using a
persisted session pointer. The supervisor pushes scalar JSON/CSV exports to
`results/<run-id>` branch in the existing repository during/following execution,
verifies write access before fitting, retries transient uploads and reports
final failures honestly. No new real GPU run or live result upload has occurred
from local preparation. All **181 local tests passed in 59.809 s**, including
fresh fitting without old artifacts, safe scalar exports, GitHub API fixture
updates/deduplication and supervised subprocess failure reporting. The notebook
has no saved outputs and all cells compile; actual GPU/live GitHub publication
await the user's runtime and credentials. The run-all notebook pins verified source
`4971335afe78c4ddfbb6d10a3170d23c350d58b2`.

**Drive change troubleshooting:** user switched runtime/account and mounted
another Drive; scaling cell 5 could not find the completed pilot's manifest.
The original Drive remains accessible. A standalone stdlib audit-transfer
helper and two-cell CPU notebook now package only the read-only audit inputs,
preserving original paths/bytes and omitting large neural checkpoints/caches.
See [transfer and restore steps](DRIVE_AUDIT_TRANSFER.md). No real transfer or
scaling launch has been performed by this session. Keep the training pin and
current prepared runtime; after restoring inputs rerun only the audit, since
the earlier capacity evidence file already exists. The actual prior artifact
audit remains pending until it runs against these files. All 174 local tests
passed in 55.308 s, including exact audit equivalence before/after the small
transfer, corruption/path checks, overwrite protection and notebook syntax.

**Latest user direction, October 21:** prioritize successful algorithm development
over the October 26 deadline and report preparation. Controlled scaling is the
next proposed Phase 4 investigation before deciding whether to redesign.
[SCALING_EXPERIMENT.md](SCALING_EXPERIMENT.md) specifies nested 900/2,700/largest
fitting sets, fixed grouped V/D, four methods and three paired replicates (72
pipelines). A non-fitting six-case preview found largest A sizes 3,297-3,298 for
sentiment and 10,736-10,756 for sales; no models or predictions ran. The old cap
cannot simply be increased because historical sampling can change test IDs.
The dedicated scaling runner, nested subsets, persisted attempts and all-fit
assessment barrier are implemented, with 170 passing tests (155.304 s final
run after adding optimizer-step accounting). The
real-data preflight passes all 18 size-specific cases without fitting or
predictions. Six original Drive split JSONs match every local ordered row
assignment; only a source-byte metadata hash differs with line endings, handled
by explicit portable anchor checks. The new `notebooks/phase4_scaling.ipynb`
keeps the old run intact and pins verified source
`f75cac6d8caa934d2f68d900420b438f890741c3`.
**GPU scaling has not started.** The older guarded notebook's setup must pass
the targeted saved-prediction/teacher/component-scope audit against the mounted
completed pilot plus TAR memory capacity before enabling `LAUNCH_SCALING`.
That prior-run dependency does not apply to the latest run-all notebook.
The audit is implemented and fixture-tested, not yet passed on all real saved
prediction/teacher files; broader neural-weight readback remains open.
This follow-up does not adopt a new architecture or start full evaluation.

**Phase 1 - Completed: historical source analysis and one real baseline reproduction.**
**Phase 2 - Completed: data contracts and leakage-safe teacher.**
The Phase 2 full suite passed 43 tests in 9.451 seconds, and infrastructure/link
checks passed. One clean-source CPU teacher smoke completed with all audit and
serialization checks passing; no outer test predictions or scores were made.
**Phase 3 - Completed: Option A fusion and saved-bundle inference.** The user
asked to be told if GPU resources are needed. CPU was sufficient: the bounded
real-E5 synthetic smoke and standalone CSV inference passed. Joint TAR,
benchmark pilots and full evaluation remain later-phase work.
**Phase 4 - In progress: GPU execution complete; final verification open.**
The checkpointed Tesla T4 run completed all **36 fits and 24 outer evaluations**
on October 20. Drive access was restored on October 21; direct readback and
209 metadata/aggregation checks confirm stage-manifest hashes, provenance,
coverage, paired metrics and the evaluation barrier's chronology. All six TAR
stages completed. See [reviewed findings](PHASE4_RESULTS.md) and the
[result](../results/phase4_gpu_pilot_review.json).
The [complete score CSV](../results/phase4_all_scores.csv) exports all 108 saved
train/validation/grouped-D/outer-test score sets, without row IDs or predictions.
The results report now includes dataset schemas, sample sizes and all holdouts.
TAR had the highest mean on both tasks. Residual fusion showed no consistent
gain over direct fusion and trailed TAR on both means; grouped diagnostics
were weaker for residual. The registered recommendation is
`analyze_failures_and_overfit_before_scaling`. Do not restart the completed run,
launch Phase 5/Option B, or tune on its test scores. Full saved-model/teacher/
prediction artifact readback and targeted overfit analysis remain before final
Phase 4 verification. No additional GPU fitting is required for that review.

**Research discussion prepared for October 22.** The user asked for at least
five creative new directions to discuss after sleeping, then pointed to the
course PDF. [Seven researched proposals](RESEARCH_IDEAS_2026-10-21.md) now cover
conditional innovation subspaces, matched-context differences, metadata-query
token pooling, multi-teacher residual agreement, tree-readable codes,
held-out-solver feature learning, and conditional negative-control interactions.
The PDF hash was rechecked and all eight existing renders inspected. The note
distinguishes measured pilot evidence, proposed mechanisms and prior-work
collisions, including recent TILE. It ranks I1/I5/I3 first for discussion.
No candidate has verified global novelty or measured gains. No architecture was
selected or implemented and no training ran. Next conversation: discuss the
current method/results and shortlist; keep the Phase 4 artifact audit open and
do not treat this research note as authorization to launch all seven ideas.

Phase 3 source is now implemented in `features.py`, `fusion.py`,
`downstream.py`, `pipeline.py`, `synthetic.py` and `fusion_runner.py`, with
thin training and CSV prediction scripts. **86 tests passed in 18.730 s**,
including original checks, all three synthetic task pipelines, direct control,
held-out perturbation, exact CPU continuation, CSV prediction and semantic
artifact validation. Lock/infrastructure/link checks pass. The real-E5 smoke
completed in 47.227 seconds with all checks passing and no reserved-test
predictions. Exact APIs and
the nested diagnostic design are in [PHASE3.md](PHASE3.md).

Dataset scope and training protocol are settled: use only datasets whose
original benchmark schema has no image column; train separate pipelines per
dataset/outer fold using internal validation and an untouched outer test split.
Do not ask the user to reconfirm these answers.

## Completed and verified work

- Phase 0 established this existing workspace, the public repository, persistent
  instructions, and the course requirements after reviewing all eight PDF
  pages. The PDF stays local and ignored. Its SHA-256 is
  `406ca54fc55e77d54058b0b14fa32ffddf3bb24d5a41ddc69ac08880399a8811`.
- Audited the exact E5/PCA/TAR, data, split, validation, and metric flows at
  historical MulTaBench `master`
  `599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`. The ignored detached checkout
  remains unmodified; no upstream code is vendored or relicensed.
- Validated all 40 original version-1 metadata schemas: **20 eligible datasets
  without image columns; 20 image-column datasets excluded**. Product Sentiment
  and Video Games Sales CSVs are now verified; the other 18 eligible CSVs need content
  hashes and feature audits before execution. See [DATASETS.md](DATASETS.md)
  and [configs/datasets.json](../configs/datasets.json).
- Locked the CPU Python 3.11 environment in `pyproject.toml` and `uv.lock`.
  Fresh `.venv` installation and **nine tests passed**. The tests cover original
  split semantics, binary holdout-size behavior, PCA-30 train-only fitting,
  unknown categories, E5 evaluation/no-gradient behavior, original multiclass
  metrics and fallback, failure manifests, and bootstrap overwrite protection.
- Added immutable source/data/model preparation, the original-baseline harness,
  a reduced smoke configuration, and a thin Colab notebook with persistent raw
  outputs. The notebook was statically checked; **Colab was not executed**.
- Fresh public clone preparation of pinned source and pilot data passed after
  retrying a temporary DNS failure. This did not repeat training or establish a
  complete fresh-clone model-download/training run.
- Phase 2 added explicit schema/table and partition APIs plus three-fold OOF
  CatBoost with regression/binary/multiclass support, native missing/unseen
  handling, training-only class order and validation teacher, finite additive
  offsets, row audits, checked persistence and a thin configured runner.
  All 43 tests passed, including the original nine, 16 data/split checks,
  14 teacher checks with real CatBoost and four runner checks. Full suite,
  locked dependency check and infrastructure/link checks passed locally.

See [BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md),
[REPRODUCIBILITY.md](REPRODUCIBILITY.md), [COLAB.md](COLAB.md), and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for details, exact setup, and failures.

## Recorded Phase 1 result

Run: `phase1-product-sentiment-frozen-fold0`.
Reviewed summary: [phase1_product_sentiment_frozen.json](../results/phase1_product_sentiment_frozen.json).

| Item | Actual result |
| --- | --- |
| Condition | Original LightGBM Joint Frozen; CPU; outer fold 0 |
| Dataset | `MUL_TEXT_PRODUCT_SENTIMENT`, Kaggle version 1, 5,091 rows, 4 classes |
| Reduced cap and actual partitions | 1,000 outer training rows: **900 fit / 100 validation**; **509 test** |
| Seeds | Base/outer/PCA/LightGBM: **42**; original internal validation: **41** |
| Macro one-vs-rest AUROC | **0.8393533369478667** |
| Original supplementary log loss | **0.8197310907083184** |
| Runtime | **559.912 s** total; **431.471 s** training; **117.262 s** test inference |
| Sampled peak process RSS | **1,060,134,912 bytes**, approximately **0.987 GiB** |
| Service charge | **USD 0** on existing local hardware; electricity/equipment excluded |

The clean, pushed source commit used by the run was
`3188a243064f6c0eb8624dd5eed4744ec4d3c037`. Real E5-small-v2 revision:
`ffb93f3bd4047442299a41ebb6fa998a38507c52`. Dataset CSV SHA-256:
`8a75a9880a5a1a619b16622c7f8f257ff8d19fde9b02864bf0059d5b0a8c2bb7`.
The summary records full environment, hardware, configuration, source/asset
hashes, class order, partition fingerprints, and raw artifact checksums.

The harness invoked unchanged original sampling, splitting, `fit`, and
`score_all_metrics`. It observed the actual preprocessing fit IDs, verified
PCA-30 fitted only on the 900 fit rows, and independently checked AUROC. E5
weights stayed unchanged, all gradients remained absent, and evaluation mode
was active; the original source's `requires_grad` flags were preserved.
**The Phase 1 held-out test partition was predicted and evaluated once.**
No test-score tuning or subsequent baseline evaluation was performed. The
Phase 2 teacher below makes no outer test predictions.

Raw IDs, predictions, PCA state, and upstream logs remain ignored under
`results/raw/phase1-product-sentiment-frozen-fold0/`; datasets and E5 weights
remain ignored under `data/` and `artifacts/`. Lightweight content-audit evidence
is recorded in `results/phase1_content_audit.json`. Preserve all original run
artifacts; run IDs and reviewed summaries must never be overwritten.

This result establishes that the small original baseline path works. It is one
dataset, one fold, and a reduced training cap, with no improvement claim or
complete reproduction of published benchmark tables. Original row splitting
retains possible duplicate-content overlap; see the content audit and dataset
limitations before interpreting broader experiments.

## Git and publication state

- Repository: https://github.com/diyarh26/residual-guided-text-tabular
- Branch: `main`.
- Latest verified pushed commit at edit time:
  `b667230cc8cf28c80d2c21446a18f7065ac6922c`
  (`analysis: review partial scaling validation gains and paired baseline gaps`).
- The checkpointed notebook/update instructions were pushed earlier.
  Compare actual HEAD with the remote to identify the current session tip.
- This field identifies the previously verified commit, avoiding a self-hash
  loop. Use `git log -1 --format="%H %s"` for the commit containing this handoff.

```powershell
uv sync --locked
uv run --frozen python -m unittest discover -s tests -v
py -3.11 scripts/verify_phase0.py
git status --short --branch
git log -1 --format="%H %s"
git ls-remote origin refs/heads/main
```

Do not rerun the recorded baseline merely to refresh documentation. Exact run
and preparation commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Recorded Phase 2 result and APIs

Implemented APIs: `src/refutab/data.py` (explicit schema/owned tables),
`splits.py` (saved ID partitions/fingerprints), `teacher.py` (three OOF CatBoost
teachers plus an A-only validation teacher), and `teacher_runner.py` with thin
`scripts/run_teacher.py`. General tasks support regression/binary/multiclass,
multiple text fields, numerical/categorical structured fields, missing/unseen
values, class mapping, finite correction offsets, and checked save/load.
No E5 or fusion training occurs. API details are in [PHASE2.md](PHASE2.md).

The locked environment adds CatBoost 1.2.8. `configs/phase2_teacher.json`
fixes 40 iterations/depth 4/learning rate 0.1/two CPU threads/seed 42 before
diagnostics. The runner reconstructs the exact Phase 1 assignments and uses
only `Product_Type` as the teacher feature. It reserves all 509 test rows,
makes no test predictions, and never scores the test. Four model fitting
audits and full row IDs remain private; reviewed summaries contain hashes and
OOF/validation diagnostics only. No calibration/early stopping/search is used.

Actual run: `phase2-product-sentiment-teacher-fold0`, clean pushed source
`f805d81c0a5badc228f423b7b26befd5f137bafd`. Reviewed result:
[phase2_product_sentiment_teacher.json](../results/phase2_product_sentiment_teacher.json).
Same pinned dataset/source and partitions as Phase 1, with typed content hash
`8561148d66c40b7fc4c2f721ea1c50b8cf40193cb635289777b1061b77308926`.
Three fold models each fit 600 rows and predict their excluded 300 rows;
every training row receives exactly one four-class OOF probability/offset.
The fourth teacher fits the 900 training rows to predict the 100 validation
rows. Class order is `Cannot Say`, `Negative`, `No Sentiment`, `Positive`.

OOF macro OVR AUROC **0.806459815788321**, log loss **0.5007548549097071**;
validation AUROC **0.8202633422209448**, log loss **0.4668987050618735**.
These are teacher diagnostics on different partitions from the Phase 1 test
score; they are not an improvement comparison. **Test predictions: 0; test
scores: 0.** Runtime **0.8031464 s** excluding interpreter/import startup;
sampled process peak RSS during fit/save/load **156,069,888 bytes** (~0.145 GiB).
Existing local CPU hardware was used; the script records service cost as
unmeasured. No E5 model loaded and no gate or correction head was trained.

Private artifacts persist under ignored
`results/raw/phase2-product-sentiment-teacher-fold0/`: full split IDs, schema,
four native models, OOF/validation predictions and offsets, fold assignments,
all fit/prediction IDs and audits, and checksums. Models and arrays loaded back
successfully with matching predictions. Preserve this run and its public
summary without overwriting them. No Phase 2 model experiment failed.

## Recorded Phase 3 result

Run `phase3-synthetic-binary-real-e5`, clean pushed source
`23d4ce34a6e3b23f458dd9bf1d06d1c1be4035d3`. Reviewed
[summary](../results/phase3_synthetic_binary_fusion.json) and
[inference audit](../results/phase3_inference_audit.json).
Original synthetic binary data: **60 fitting / 12 validation / 12 reserved
test rows**, two text fields, two numeric fields and one categorical field.
No benchmark dataset was evaluated during this phase.

Actual pinned E5-small-v2 encoded both fields frozen on CPU, with fitting-only
PCA-30 per field and structured preprocessing. Three OOF teachers fit 40 rows
and predict their excluded 20; the separate 60-fit teacher predicts validation.
The shared gate trained with fixed residual offsets; a head-free frozen gate
then supplied fused features to the 100-tree downstream LightGBM.

The fixed five-epoch budget stopped after **epoch 4** by patience three;
**epoch 1** had the minimum validation correction loss (0.6027021408081055).
Downstream validation AUROC **0.7142857142857143**, log loss
**0.8421002030754804**, are synthetic diagnostics only. Training loss decreased
slightly while validation correction loss did not improve after epoch 1;
this execution does not establish complementarity or benchmark improvement.
No tuning or second training attempt followed those diagnostics.

Runtime **47.227143 s**, excluding interpreter/import startup; sampled process
peak RSS **987,308,032 bytes** (~0.920 GiB). Existing Intel CPU, two Torch
threads, no GPU; service charge recorded as unmeasured. No paid external
compute was used. Encoder state hash remained
`880ab04eba37fac79b75f1697064b281566657e68b937eab5057c5c031cfb627`;
all six pinned model/tokenizer file hashes were unchanged.

The standalone `scripts/predict.py` cold-loaded the bundle and E5 files, with
no teacher or head weights and no upstream source needed for inference.
Its 12 saved-order probability rows matched saved validation predictions to
**1.1102230246251565e-16** maximum absolute difference. A readback audit checked
all **20 raw file hashes**, dataset/split binding, OOF exclusion/coverage,
preprocessing fit IDs, PCA shape `(2,30,384)`, selected gate versus checkpoint,
class/row ordering and metrics. **Reserved-test predictions/scores: 0/0.**

Private run bundle/checkpoint/teacher/raw IDs remain under ignored
`results/raw/phase3-synthetic-binary-real-e5/`; cold-inference CSV is under
`results/raw/phase3-inference-readback/`. Preserve both and original summaries.
Checkpoint tests establish exact same-environment CPU epoch continuation;
GPU, partial E5 batches and partial LightGBM resumption are untested/unsupported.
The nested benchmark diagnostic is designed in PHASE3 but has not run.
No Phase 3 model experiment failed.

## Phase 4 preparation and current next steps

The user authorized the needed work and requested notice when a GPU is needed.
`configs/phase4_pilot.json` registers both audited no-image datasets, three
original outer holdouts each, four matched conditions and three grouped nested
direct/residual diagnostics per dataset inside outer fold-0 training. This is
36 fitted stages and 24 outer-test prediction calls. All stages must complete
before any new outer-test predictions. Original Phase 1 visibility is disclosed;
this remains exploratory. The fixed descriptive decision rule does not permit
a gain recommendation with missing TAR or incomplete diagnostic coverage.

`pilot.py`, `pilot_protocol.py`, `pilot_features.py`, `tar.py` and
`pilot_report.py` implement the runner, source-compatible representations,
grouped split/OOF audits, safe caches/continuation, and aggregate reporting.
**152 tests passed in 25.723 seconds on October 20.** Both real tables pass all 12 main/nested
feature-role and split preflights. Separate Linux CUDA locks and the thin
`notebooks/phase4_pilot.ipynb` launcher are prepared; the full GPU/TAR pilot
has not executed in reviewed evidence. Colab setup is recorded below.
Full execution instructions are in [PHASE4.md](PHASE4.md)
and [PHASE4_GPU.md](PHASE4_GPU.md).

The CPU preflight is now recorded in
[phase4_cpu_preflight.json](../results/phase4_cpu_preflight.json) and its
[audit](../results/phase4_cpu_preflight_audit.json). Clean source was the pushed
commit `c1e7ec1cdbb79fbc2f000a060616a5e73e11c0d1`. Main Product Sentiment fold-0 frozen/direct/residual and all six
grouped direct/residual diagnostics completed: **9/36 stages; zero outer-test
predictions**. All 509 outer-test rows remained excluded from fitting.

The first execution on October 19 failed with Windows `WinError 5` during an
atomic checkpoint replacement inside OneDrive; the locking cause is unconfirmed.
On October 20, all 48 original files were copied and byte-verified outside
OneDrive. The original failure remains untouched. Completed frozen/direct stages
were reused; residual attempt 2 resumed epoch 3 and completed epoch 10, followed
by the six diagnostics. No model settings changed. Raw recovery artifacts are at
`%LOCALAPPDATA%/ReFuTab/runs/phase4-product-sentiment-cpu-preflight/`; the original
failed run stays at `results/raw/phase4-product-sentiment-cpu-preflight/`.

The audit verified 142 completed-stage files, 16 teacher fitting scopes, 24
saved-prediction metric sets, all 12 split fingerprints and unchanged six E5
assets. It performed no new fitting or inference. Total measured execution was
286.692 s initially plus 39.082 s on continuation, excluding setup/import and
the overnight interruption; maximum sampled process RSS was 999,112,704 bytes.

Validation AUROC: frozen 0.816653, direct 0.805746, residual 0.800067. On grouped
diagnostic D, residual-minus-direct differences were -0.024416, +0.008215,
-0.010637 (mean -0.008946, sample SD 0.016381); all downstream training AUROCs
were 1.0. These are internal exploratory diagnostics, with no consistent gain
and visible training optimism. No redesign followed. Missing GPU/TAR, regression
fits and remaining outer coverage block a complete pilot recommendation.

The user has now reached Colab preparation. Two pasted attachments contain the
same log for bootstrap `phase4-colab-20261020T140148Z-67a87c90`. Pinned source,
both datasets and all six E5 assets prepared successfully. The suite reported
126 discovered checks and five import errors sharing one Matplotlib backend
cause; the full suite was not successfully discovered/run. Hardware details and
the project `rev-parse` output were not in the supplied excerpt. The failure
record is [phase4_colab_setup_failure.json](../results/phase4_colab_setup_failure.json).

The next supplied log verifies clean project HEAD
`c1e7ec1cdbb79fbc2f000a060616a5e73e11c0d1` and launch of that same run ID.
Control flow reached `Fitting ...__direct`, which follows frozen-stage completion,
then failed at the first linear forward operation with the deterministic cuBLAS
workspace error. Raw Drive artifacts/hardware have not been independently read.
The follow-up record is [phase4_colab_cuda_failure.json](../results/phase4_colab_cuda_failure.json).

The subsequent GPU log reached direct, residual, then TAR. Direct/residual
therefore completed in this source's control flow, but their private artifacts
have not been independently audited. TAR failed in the first forward pass:
768 MiB requested, 551.81 MiB free on a GPU reporting 14.56 GiB. See
[phase4_colab_tar_oom.json](../results/phase4_colab_tar_oom.json).
The user confirmed only the current GPU is available. No outer test was unlocked.

The latest supplied log verifies source `6e09955514caffa52050459377e5b4736dedca75`
and fresh run `phase4-checkpointed-20261020T145755Z-dd78ef55`. The full-size
synthetic capacity check passed on Tesla T4 (6,989,661,184 bytes peak allocated).
The first TAR stage reached epoch 9 / optimizer step 36; temporary-head internal
validation loss fell from 1.276482 to 0.951114. These are progress diagnostics,
not downstream results. The worker-count warning did not stop training. See
[progress evidence](../results/phase4_colab_progress_epoch9.json). No complete
TAR stage or outer-test evaluation is reported yet.

**October 21 update: the run finished; do not resume or repeat it.** The
user's diagnostic and subsequent direct Drive readback confirm completion.
The [results review](PHASE4_RESULTS.md) supersedes the epoch-9 progress report.
Remaining work is independent saved-artifact verification and failure/overfit
analysis. The setup instructions below are historical/future recovery guidance,
not instructions to launch another job now.

1. Use the explicit memory-saving variant described in
   [PHASE4_MEMORY.md](PHASE4_MEMORY.md). It keeps FP32/batch 256 and all budgets,
   recomputing trainable BERT blocks with non-reentrant activation checkpointing
   and preserved dropout RNG. The original config and run remain unchanged.
2. Update the clean Colab checkout to the reviewed memory-fix source and run
   `scripts/run_memory_pilot.py --output-root /content/drive/MyDrive/ReFuTab/runs`.
   It first runs a full-size synthetic capacity check with real E5 weights,
   the frozen encoder also resident, and zero benchmark rows/optimizer updates.
   If this fails it stops, preserving diagnostics; do not promise that it fits.
3. On capacity success, its fresh run repeats the short first three conditions
   under the new source/config binding and runs the complete registered plan.
   Do not merge old fitted models or resume the old OOM ledger under new source.
   Existing data/model assets and the original private run remain intact.
4. Audit actual GPU capacity evidence, all artifacts/coverage and pilot results;
   report the descriptive recommendation before full evaluation. Phase 4 stays
   in progress. No Phase 5/Option B implementation is authorized by preparation.

## Remaining technical boundaries

- Joint TAR remains the primary comparison for supervised ReFuTab; original
  source calls, local-model redirection, save/load and continuation have CPU
  fixture coverage, and six actual GPU TAR stages now completed. Independent
  full saved-artifact readback remains open. No full
  evaluation or Option B is authorized by completion of this smoke run.
- The historical snapshot has no license file; preserve the ignored checkout
  and provenance policy in [UPSTREAM.md](UPSTREAM.md). Do not publish its source,
  datasets, weights, course PDF, credentials, or raw experiment data.
- OOF teacher outputs do not make a target-trained fusion representation OOF.
  Keep the planned nested diagnostic and coordinate-alignment safeguards for
  later method evaluation; do not silently redesign Option A.
- The course PDF's MulTaBench reference identifier is erroneous; use the
  verified MulTaBench paper identifier `2605.10616`. Requirements and attribution
  distinctions remain in [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md).
