# Project plan

Last updated: October 20, 2026 (Asia/Jerusalem). Deadline: October 26, 2026.
Phases follow the user's order; Phase 7 report notes may be prepared alongside
evaluation, but no untested result may be written as a finding. Option B is
optional and lowest priority. Status values: Not started / In progress /
Completed / Blocked. [HANDOFF.md](HANDOFF.md) records the current session state.

## Phase 0 - Repository and research infrastructure

**Status: Completed.** Public repository and initial push verified; infrastructure
checks also passed from a fresh public clone. See HANDOFF for commit provenance.

Objective: establish an understandable, attributable, reproducible research
workspace without implementing or training the method.

- [x] Inspect the existing folder; retain it without nesting.
- [x] Read all eight official PDF pages and document authoritative requirements.
- [x] Verify GitHub account `diyarh26` and absence of the requested repository.
- [x] Investigate historical master, source compatibility, and license history.
- [x] Establish documentation, persistent agent instructions, and directory roles.
- [x] Validate infrastructure, local links, pin, and private-file exclusions.
- [x] Create public repository; review, commit, push, and verify remote SHA.
- [x] Prepare Phase 0 report and leave Phase 1 unstarted.

Deliverables: README, AGENTS, all requested docs, source lock, ignore/license
policy, directory scaffold, GitHub repository, infrastructure validator.
Acceptance: required documents and local links pass validation; PDF and outputs
are excluded; immutable historical SHA is documented; remote is accessible and
synchronized; handoff exposes all unresolved decisions. Course clarifications
may remain open if no dependent implementation is performed.

## Phase 1 - MulTaBench analysis and baseline reproduction

**Status: Completed.** Fresh locked CPU setup, nine tests, original baseline,
source/schema audits, and thin Colab runner are delivered. The notebook is
statically validated; actual Colab/GPU execution remains untested.

Objective: reproduce the historical benchmark before introducing ReFuTab.

Confirmed dataset scope: only benchmark datasets whose original schema has no
image column. Enumerate and validate the eligible roster in Phase 1.
Confirmed training protocol: train separately per dataset/outer fold on training
rows, use internal validation for selection, and keep test data untouched.

- [x] Validate leakage-safe split boundaries and historical multiclass metrics.
- [x] Confirm historical-source redistribution policy; retain original attribution.
- [x] Map E5/PCA/TAR, data schema, split source, validation, metrics, and integration
  points in the pinned source; Phase 0 inspection is preliminary analysis only.
- [x] Choose and pin compatible Python/ML dependencies and model/data revisions.
- [x] Add a small runnable baseline/smoke configuration and thin Colab runner.
- [x] Reproduce one original baseline on a small text-tabular task; save exact
  command, environment, hardware, seed, split IDs, metrics, and failure logs.
- [x] Test, document, commit, push, and verify.

Deliverables: source-flow analysis, dependency lock, smoke test, real baseline
artifact/metrics manifest, exact local/Colab commands. Acceptance: clean setup
can run the baseline with verified split/metric semantics and traceable output.
No new algorithm until the baseline works correctly.

Completed run: `phase1-product-sentiment-frozen-fold0`, source `3188a24`, macro
OVR AUROC 0.8393533369478667, log loss 0.8197310907083184, 559.912 seconds CPU,
sampled peak RSS 1,060,134,912 bytes. Original reduced cap gives 900 fit / 100
validation / 509 test rows. This meets the minimal baseline gate; no ReFuTab,
TAR, regression baseline, or full-benchmark result is claimed.

## Phase 2 - Data handling and leakage-safe teacher

**Status: Completed.** Dataset/split contracts, structured-only OOF teacher,
43 passing tests, and one recorded clean-source CPU teacher smoke are delivered.
The 509 outer test rows were neither predicted nor scored in Phase 2.

Objective: build a general data interface and trustworthy teacher signals.

- [x] Support arbitrary stable text/numerical/categorical schema, missing values,
  regression, binary classification, and multiclass classification.
- [x] Persist outer/inner/OOF split IDs, seeds, class ordering, and feature schema.
- [x] Fit 3-fold OOF structured-only CatBoost with isolated fold preprocessing.
- [x] Separate internal validation and final evaluation; handle binary clipping,
  multiclass class coverage, and probability/calibration decisions train-only.
- [x] Add disjointness, fit-provenance, ID/target exclusion, alignment, and
  deterministic failure tests; test, document, commit, push, and verify.

Deliverables: dataset/teacher APIs, configurations, provenance artifacts, tests.
Acceptance: each OOF row was excluded from its teacher fit, transformations
never fit on evaluation data, and supported tasks pass meaningful tests.

Recorded run: `phase2-product-sentiment-teacher-fold0`, source `f805d81`.
Three models each fit 600 rows/predict 300 excluded rows; a fourth fits all
900 training rows to predict 100 validation rows. Native score/offset bundles
pass save/load checks. Runtime 0.803 seconds, sampled peak RSS 156,069,888 bytes.
OOF/validation scores are diagnostics, not a final test or fusion result. See
[PHASE2.md](PHASE2.md) and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Phase 3 - Option A fusion

**Status: Completed.** 86 tests passed. The real pinned-E5 synthetic binary
smoke and standalone saved-bundle CSV inference passed on CPU; no GPU was
needed. No Phase 4 pilot or TAR run is implied.

Objective: implement the documented frozen-encoder residual-gated method.

- [x] Integrate pinned frozen E5, per-field train-fitted PCA-30, and structured
  encoding with missing/unknown handling and stable field ordering.
- [x] Implement near-identity context gating and temporary correction losses.
- [x] Implement training, selection, checkpoint/resume, serialization, and a
  clear prediction API with downstream learner and original structured data.
- [x] Add direct-target gated control; document supervised-representation overfit risk
  and design a valid cross-fitting/nested diagnostic with aligned feature spaces.
- [x] Test gradient freezing, dimensions, initialization, class ordering,
  round-trip predictions, and end-to-end synthetic tasks.
- [x] Test, document, commit, push, and verify.

Deliverables: modular source, training/inference entry points, artifact schema,
unit/integration/leakage tests. Acceptance: small synthetic runs complete for
all supported task types, inference works without teacher/head, invariants pass.

Recorded run: `phase3-synthetic-binary-real-e5`, source `23d4ce3`, 60 fitting /
12 validation / 12 reserved test rows and two text fields. Runtime 47.227 s,
sampled peak RSS 987,308,032 bytes. Selected epoch 1; patience stopped training
after epoch 4 of five maximum. Readback verified all 20 raw artifact hashes and
cold CSV inference (maximum difference 1.11e-16). These are synthetic execution
checks; nested benchmark evaluation and improvement claims remain later work.

## Phase 4 - Pilot experiments and decision gate

**Status: In progress.** User authorized the needed work and asked to be told
when GPU access is required. Two version-pinned datasets and matched pilot
settings are registered; 152 tests and real split/feature preflights pass.
Separate CUDA lock and Colab launcher are ready; real GPU/TAR execution remains pending.
The nine-stage Product Sentiment CPU preflight and saved-artifact audit passed;
zero outer-test predictions. Grouped diagnostics show no consistent residual
gain. Full comparison and decision remain incomplete; see HANDOFF.

Objective: determine whether the method merits broader evaluation.

- [x] Pre-register representative classification/regression pilots and seeds.
- [ ] Run aligned Joint Frozen, Joint TAR, direct-target gated, and residual-guided
  conditions; measure final learner scores as well as correction-head behavior.
- [ ] Save configurations, learning curves, runtime, memory, hardware, failures,
  and machine-readable metrics; analyze stability and overfit diagnostics.
- [ ] Report continue/adjust/stop recommendation to the user before scaling.
- [ ] Test result aggregation, document, commit, push, and verify.

Deliverables: reproducible pilot results and explicit decision report.
Acceptance: completed and failed runs are distinguishable, comparisons are
controlled, test scores did not drive tuning, and the decision gate is reported.

## Phase 5 - Full evaluation and ablations

**Status: Not started.**

Objective: scientifically valid benchmark coverage within agreed course scope.

- [ ] Freeze the evaluation plan and clarify all scope/metric decisions.
- [ ] Run all applicable agreed datasets, baselines, seeds, and controlled ablations.
- [ ] Report per-dataset AUROC/R2, deltas, variability, justified statistical
  analysis with correct independence unit, compute cost, and negative findings.
- [ ] Save lightweight metrics, tables/figures, manifests, and failure analysis.
- [ ] Validate result completeness and provenance; document, commit, push, verify.

Deliverables: machine-readable complete run ledger, paper-ready tables/figures,
statistical analysis. Acceptance: agreed coverage is met or missing coverage is
explicitly reported as incomplete; no unsupported significance claims.

## Phase 6 - Option B (optional)

**Status: Not started.**

Objective: investigate isolated LoRA encoder adaptation only if justified.

- [ ] Confirm promising Option A results and enough time after core deliverables.
- [ ] Isolate LoRA implementation/configuration and preserve Option A reproduction.
- [ ] Compare fairly with Option A and Joint TAR; test, document, commit, push.

Deliverables: optional isolated extension and controlled results. Acceptance:
core submission is secure, comparisons are valid, and Option A remains usable.
Skipping this phase is allowed; do not let it delay Phase 7.

## Phase 7 - Report and final deliverables

**Status: Not started.**

Objective: submit a complete, reproducible Track 1 project.

- [ ] Prepare English NeurIPS LaTeX paper, <=4 pages excluding references and
  appendices; appendices contain result tables/metrics only.
- [ ] Cover title/abstract, introduction, related work, method, setup/results,
  ablations, limitations, and conclusion; cite actual reused methods.
- [ ] Prepare required Hugging Face weights/artifact inventory with provenance
  and permissions; clarify treatment of unchanged public pretrained models.
- [ ] Provide working inference and end-to-end training modes with exact commands,
  pinned dependencies, seeds, hardware, and saved-artifact reproduction.
- [ ] Test clean setup and artifact loading; final documentation/commit/push;
  submit all deliverables individually via Moodle by October 26, 2026.

Deliverables: paper, general-purpose source/lock, evaluation results, HF models,
both operating modes, and reproducibility guide. Acceptance: all §2.3/§4 items
are checked against the PDF; uploaded artifacts and commands actually work.
The PDF does not specify an exact deadline time; confirm Moodle separately.
