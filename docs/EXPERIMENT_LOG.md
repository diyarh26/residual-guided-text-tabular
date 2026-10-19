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

## Required entry for each later experiment

Record run ID/time/status (planned/running/completed/failed/interrupted), exact
project and upstream SHAs, config hash, dataset/model revisions and permissions,
split IDs/hashes, seed(s), command, dependency lock, hardware, runtime, peak
memory/cost, selected checkpoint criterion, metrics with definitions, artifacts,
and failure details. Include negative results. Changes to a run's configuration
create a new run ID. Never substitute correction-head scores for downstream
learner results or present partial benchmark coverage as complete.
