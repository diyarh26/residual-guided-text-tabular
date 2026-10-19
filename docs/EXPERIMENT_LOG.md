# Experiment log

## October 19, 2026 - Phase 0 infrastructure

**No model training, baseline reproduction, benchmark evaluation, or numerical
performance experiment has been run.** There are no scores or checkpoints.

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
provenance are in [HANDOFF.md](HANDOFF.md). There is no ML environment lock yet.

## October 19, 2026 - Phase 1 setup and preregistered smoke baseline

User authorized Phase 1 after resolving dataset scope and separate per-dataset
training. Audited all 40 historical metadata schemas: 20 without image columns
are eligible. Prepared Kaggle Product Sentiment version 1 and pinned E5 revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52` with content hashes.

Environment setup initially selected Joblib 1.6.0, which removed
`joblib.externals.cloudpickle` required by Skrub 0.6.2. Two test-discovery attempts
failed during imports before any model experiment. Pinning original Joblib
1.5.2 resolved the incompatibility. Nine tests then passed, including original
split/metric behavior, train-only PCA, unknown-category handling, frozen encoder
inference, run safety, and fresh-clone bootstrap. A source review found and fixed
a fresh `--no-checkout` bootstrap guard bug before a network-clean setup claim.

Before any test score, fixed one CPU Joint Frozen LightGBM smoke configuration:
Product Sentiment, fold 0, original 1,000-row outer training cap, PCA 30, original
E5 prefix/pooling/tokenization and learner settings. Expected rows are 900 fit,
100 validation, and 509 test. This is a reduced baseline reproduction, not a
paper-table score or ReFuTab experiment. The executable code/config will be
committed before its real run. No baseline score is available at this entry.

## Required entry for each later experiment

Record run ID/time/status (planned/running/completed/failed/interrupted), exact
project and upstream SHAs, config hash, dataset/model revisions and permissions,
split IDs/hashes, seed(s), command, dependency lock, hardware, runtime, peak
memory/cost, selected checkpoint criterion, metrics with definitions, artifacts,
and failure details. Include negative results. Changes to a run's configuration
create a new run ID. Never substitute correction-head scores for downstream
learner results or present partial benchmark coverage as complete.
