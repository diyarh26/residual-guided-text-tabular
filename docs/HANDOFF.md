# Project handoff

Updated: October 19, 2026 (Asia/Jerusalem).

## Current phase and status

**Phase 1 - Completed: historical source analysis and one real baseline reproduction.**
**Phase 2 - Completed: data contracts and leakage-safe teacher.**
The Phase 2 full suite passed 43 tests in 9.451 seconds, and infrastructure/link
checks passed. One clean-source CPU teacher smoke completed with all audit and
serialization checks passing; no outer test predictions or scores were made.
**Phase 3 - Completed: Option A fusion and saved-bundle inference.** The user
asked to be told if GPU resources are needed. CPU was sufficient: the bounded
real-E5 synthetic smoke and standalone CSV inference passed. Joint TAR,
benchmark pilots and full evaluation remain later-phase work.
**Phase 4 - In progress.** The user authorized the needed work and asked to be
told when GPU resources are needed. Prepare the locked comparisons and grouped
nested diagnostics, run bounded CPU checks, and provide a concrete GPU launcher.
Do not claim Phase 4 completed without real TAR/comparison evidence.

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
  `2f9947eb1bd0a6ddad6548b47edc5eca60f5c5f8`
  (`experiments: record Phase 3 real-E5 smoke and inference verification`).
- Phase 4 preparation accompanies this handoff in
  a follow-up commit. The field above records the last verified commit at edit
  time; compare actual HEAD with the remote to identify the current session tip.
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
**152 tests passed in 32.508 seconds.** Both real tables pass all 12 main/nested
feature-role and split preflights. Separate Linux CUDA locks and the thin
`notebooks/phase4_pilot.ipynb` launcher are prepared; actual GPU/Colab execution
has not occurred. Full execution instructions are in [PHASE4.md](PHASE4.md)
and [PHASE4_GPU.md](PHASE4_GPU.md).

1. Commit/push this registered implementation before recording the bounded
   Product Sentiment fold-0 CPU preflight (nine stages; zero outer tests).
   Save reviewed evidence; do not mix these CPU models into the CUDA comparison.
2. Tell the user the full comparison now needs one CUDA GPU, and provide the
   Colab link and immutable source SHA. Run the fresh registered GPU plan when
   that resource is available; preserve actual OOM/failures without auto-tuning.
3. Audit full artifacts and coverage, report the pilot recommendation, then
   decide whether full evaluation is warranted. Phase 4 remains in progress;
   do not proceed silently to Phase 5 or Option B.

## Remaining technical boundaries

- Joint TAR remains the primary comparison for supervised ReFuTab; original
  source calls, local-model redirection, save/load and continuation have CPU
  fixture coverage, while real GPU training remains unvalidated. No full
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
