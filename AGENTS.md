# ReFuTab: persistent agent instructions

## Read before changing anything

Read this file, [docs/HANDOFF.md](docs/HANDOFF.md),
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md), and [README.md](README.md).
The handoff is the progress source of truth. Check `git status --short --branch`
and preserve existing user changes. Use this existing workspace; do not nest or
move the project. Read the relevant architecture, course, and leakage documents
before implementation. If the local course PDF is unavailable, request access;
its requirements are authoritative and its redistribution is not established.

## Objective and fixed research direction

Technion NLP 097215, Track 1 (Algorithm), due October 26, 2026 via Moodle.
User priority update, October 21: focus on developing a successful algorithm;
do not let the deadline or academic report preparation drive current research
choices. Defer report writing. Preserve experiment logs, reproducibility and
leakage safeguards. Controlled scaling is the next Phase 4 investigation;
see docs/SCALING_EXPERIMENT.md for its design/readiness limits.
Investigate whether residual-guided, context-aware text fusion learns predictive
information complementary to structured columns. Improvement is a hypothesis,
not a result. Joint TAR is the primary baseline because the gate uses labels.

Option A: 3-fold OOF CatBoost structured-only teacher; frozen E5-small-v2;
384-dimensional embeddings compressed by train-fitted PCA to 30 per text field;
compact structured encoding; identity-initialized gate
`z = e * (1 + tanh(G(e, s)))`; temporary additive correction head with teacher
offsets. Discard the correction head and teacher for final inference. Train the
downstream learner on original structured features plus fused text embeddings.
Hidden size 64 and dropout 0.1 are initial hyperparameters, not conclusions.
Support multiple text columns and stable field/class ordering. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Dataset scope, confirmed by the user: use only benchmark datasets whose original
schema has no image column. Exclude datasets containing an image column.
Validate eligibility against the pinned benchmark schema during Phase 1.

Training protocol, confirmed by the user: train a separate pipeline for each
dataset and outer evaluation fold, using only its permitted training partition.
Use internal validation for selection and keep the test split untouched until
locked evaluation. Dataset scope and this training rule are resolved; do not
ask the user to reconfirm them.

## Workflow and scope gates

- Work in numbered phases and read their current status. Phases 0 through 3
  are complete; Phase 4 is authorized and in progress. Report each phase before major implementation in
  the next one. Do not start expensive training implicitly.
- Reproduce a valid upstream baseline in Phase 1 before implementing ReFuTab.
- Use the historical `master` pin in `third_party/multabench.lock.json`.
  Selected SHA: `599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`.
  Never replace it with current master or `paper_version` without discussion.
- Resolve or explicitly gate course ambiguities in `docs/COURSE_REQUIREMENTS.md`
  before dependent implementation; never silently reinterpret the assignment.
- Pilot results must be reported with a recommendation before full evaluation.
  Do not silently redesign the method. Option B is optional and gated on
  promising Option A results and sufficient time.
- A phase is Completed only when all acceptance checks pass. Record partial work
  and failed experiments honestly. Never invent results or successful uploads.

## Coding and testing

- Keep original Python modules in `src/refutab/`; configs, orchestration,
  upstream code, and results remain separate. Thin Colab notebooks call source.
- Use modular Python, explicit configuration, practical type annotations, clear
  exceptions and logs. No hardcoded dataset column names or silent skips.
- Pin experiment dependencies and model/data revisions before baseline execution.
  Use `uv sync --locked` for the Phase 1 CPU environment, then
  `uv run --frozen python -m unittest discover -s tests -v`. The standalone
  `scripts/verify_phase0.py` checks documentation/publication boundaries only.
- Run meaningful unit/integration checks appropriate to each change. Before each
  milestone, run the documented checks and record their exact outcome.
- Every run records source SHA, upstream SHA, config, split IDs/hashes, seeds,
  revisions, environment, hardware, runtime, cost, failures, and artifact paths.
  Persist Colab outputs outside its temporary runtime and support resumption.

## Leakage invariants

- Test data/labels are excluded from every fitting or model-selection step,
  including PCA and structured preprocessing. Never tune against test scores.
- Every training teacher prediction is OOF from a teacher whose training rows
  exclude that example; use a distinct internal validation split for selection.
- Keep target-trained weights isolated by dataset/outer fold. Exclude row IDs,
  labels, and target-derived metadata from features; record class ordering.
- OOF teachers do not make supervised fusion embeddings cross-fitted. Audit
  downstream overfitting and representation alignment before claiming validity.
- Test row disjointness, train-only fitting, unknown categories, prediction
  alignment, frozen E5 gradients, serialization, and forbidden feature leakage
  in their implementation phases. See [docs/DATA_LEAKAGE.md](docs/DATA_LEAKAGE.md).

## Licensing, Git, and end of every work session

- Do not publish the course PDF, credentials, private/restricted data, cached
  embeddings, checkpoints, or unreviewed upstream code. Original ReFuTab work is
  MIT; third-party licenses remain separate. The historical snapshot has no
  license file; see `docs/UPSTREAM.md` before copying its code.
- Run relevant checks; update HANDOFF and PROJECT_PLAN, decisions when needed,
  and EXPERIMENT_LOG for all experiments and failures.
- Review `git status`, staged diff, ignored-file coverage, and accidental secrets.
  Commit a descriptive milestone and push; verify remote SHA and clean status.
  Never force-push. Report authentication/push failures accurately.
- HANDOFF's commit field identifies the last verified commit at edit time; use
  `git log -1` for the commit containing the handoff itself (no self-hash loop).
- End with completed work, limitations/blockers, repository URL, and exact next
  steps. Do not claim later phases are implemented based on documentation alone.
