# Reproducibility

Phase 1 adds a locked CPU environment and a runner for the original historical
Joint Frozen baseline. Phase 2 adds data/split contracts and a structured-only
OOF teacher. Phase 3 implements Option A fusion, direct-target control and
teacher/head-free prediction; benchmark usefulness is still untested. Actual
execution outcomes, including failures, belong in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Run the historical CPU baseline

The environment uses CPython 3.11 and `uv==0.11.24`. Core numerical/ML packages
match the selected source's requirements, with CPU builds of Torch 2.7.1 and
Torchvision 0.22.1. The hash-bearing `uv.lock` fixes the complete resolved
dependency closure; it deliberately excludes unrelated learner ecosystems.
This minimal, compatibility-tested closure is not identical to the full original
lock: some transitive packages are newer (for example Tokenizers 0.22.2 versus
0.22.1). Exact actual versions are saved in each run; no paper-score match is claimed.
The package manager installs into the ignored `.venv/`, leaving global packages
alone. This is a CPU baseline environment; it does not validate GPU/TAR support.

From a clean checkout of the intended full project SHA on Windows:

```powershell
py -3.11 -m pip install uv==0.11.24
py -3.11 -m uv sync --locked
py -3.11 -m uv run --frozen python scripts/prepare_phase1.py
py -3.11 -m uv run --frozen python -m unittest discover -s tests -v
py -3.11 -m uv run --frozen python scripts/run_baseline.py --config configs/phase1_frozen.json --run-id my-new-phase1-run --summary results/my-new-phase1-run.json
```

On Linux use the same `uv` commands after installing Python 3.11, or use the
[thin Colab notebook](../notebooks/phase1_baseline.ipynb). A new run ID and
summary path must be used for each attempt; existing outputs are never overwritten.
Production runs reject tracked changes and untracked publishable source files.
`--allow-dirty` exists only for explicitly recorded development failures/checks.

Preparation verifies the unmodified historical source SHA, Kaggle pilot version 1
and CSV/metadata hashes, and E5 snapshot
`ffb93f3bd4047442299a41ebb6fa998a38507c52`. Source, data, and weights are fetched
locally into ignored directories; they are not redistributed in this repository.
The small configuration fixes a 1,000-row outer training cap before viewing any
test score. The original sampler retains 509 test rows; the learner internally
uses 900 fitting rows and 100 validation rows. This is a smoke reproduction,
not a published full-benchmark score. See [source analysis](BASELINE_ANALYSIS.md)
and [dataset provenance](DATASETS.md).

Each run writes an initial `manifest.json`, logs, split IDs, PCA states, and
predictions under `results/raw/<run-id>/`. Failures retain their traceback and
failed status. `--output-root` can point to private persistent Drive storage.
The optional reviewed JSON summary contains metrics, hashes, class order,
environment/hardware, timings, sampled peak RSS, and relative artifact references;
review it before committing. Raw IDs, text, predictions, and weights stay out of Git.

The original CPU LightGBM fit is not checkpoint-resumable. A failed/interrupted
stage requires a new attempt ID. The notebook has static validation separately
from actual Colab execution; local success does not establish Colab execution.

## Run the structured-only OOF teacher (Phase 2)

Check out the source commit recorded in the desired teacher summary. Install
the locked Python 3.11 environment, then prepare the pinned source and pilot
data; E5 weights are unnecessary for this stage.

The completed smoke used source
`f805d81c0a5badc228f423b7b26befd5f137bafd`; see its unedited
[summary](../results/phase2_product_sentiment_teacher.json). Use a fresh run ID
when reproducing it; the archived attempt is preserved.

```powershell
py -3.11 -m uv sync --locked
py -3.11 -m uv run --frozen python scripts/prepare_phase1.py --skip-model
py -3.11 -m uv run --frozen python -m unittest discover -s tests -v
py -3.11 -m uv run --frozen python scripts/run_teacher.py --config configs/phase2_teacher.json --run-id my-new-phase2-run --summary results/my-new-phase2-run.json
```

Run from clean committed source and choose new run/summary names. The fixed
CPU budget is three OOF teachers plus one validation teacher, each with 40
iterations, depth 4, learning rate 0.1, seed 42 and two threads. This budget is
an initial smoke configuration; no hyperparameter selection or calibrated
probability fitting occurs. CatBoost 1.2.8 is pinned alongside the unchanged
Phase 1 core package versions. The lock naturally differs from the archived
Phase 1 environment; reproduce that run from its recorded source commit.

The original source recreates the exact 900/100/509/3,582 training/validation/
test/excluded partitions and verifies public Phase 1 fingerprints. Only the
900 training structured rows and labels fit teachers; the separate all-training
teacher predicts the 100 validation rows. No test predictions or scores are
produced. Validation labels are used only for diagnostics after model fitting.
All-label access is limited to original split stratification, table validation,
and content fingerprinting, never teacher fitting or model selection.

Private `results/raw/<run-id>/` contains `splits.json`, schema, run manifest,
four native `.cbm` models, aligned OOF/validation predictions and offsets,
three-fold assignments, row fit/prediction audits, class order, versions, and
file checksums. Aggregate JSON summaries contain counts/hashes and OOF/validation
diagnostics without raw IDs, data, predictions, or models. Save/load uses native
CatBoost plus NumPy `allow_pickle=False`; altered model/array checksums fail.
See [PHASE2.md](PHASE2.md) for the general API and explicit schema/split protocols.

`--output-root` accepts a private persistent artifact directory. Individual
teacher fits are not checkpoint-resumable; an interrupted run keeps its failed
manifest/trace and needs a new attempt ID. The Phase 1 Colab notebook remains a
baseline runner; no Phase 2 Colab execution is claimed.

## Train and load Option A (Phase 3)

The exact component/API and nested evaluation design are in [PHASE3.md](PHASE3.md).
The completed synthetic smoke used clean source
`23d4ce34a6e3b23f458dd9bf1d06d1c1be4035d3`; its unedited
[summary](../results/phase3_synthetic_binary_fusion.json) and separate
[cold-inference audit](../results/phase3_inference_audit.json) record the evidence.
From clean committed source with the locked environment and prepared E5 assets:

```powershell
py -3.11 -m uv run --frozen python scripts/run_fusion.py --config configs/phase3_smoke.json --run-id my-new-phase3 --summary results/my-new-phase3.json
py -3.11 -m uv run --frozen python scripts/predict.py --bundle results/raw/my-new-phase3/bundle --input results/raw/my-new-phase3/validation_features.csv --output results/raw/my-inference/predictions.csv
```

The supplied config uses original synthetic data and actual pinned E5, with
60 fitting/12 validation/12 reserved test rows. Training writes a private
inference bundle and independent training checkpoint, fitting/split provenance,
learning curves, validation predictions and reviewed aggregate summary. No
test prediction is made. The final prediction bundle omits teacher and temporary
head weights; pretrained E5 stays in the verified external model directory.
`--model-path` relocates those files for prediction. Classification CSV columns
`probability_k` correspond to the printed/saved class order.

Continuation copies a prior matching checkpoint into a **new** attempt:

```powershell
py -3.11 -m uv run --frozen python scripts/run_fusion.py --config configs/phase3_smoke.json --run-id my-retry --resume-from results/raw/failed-attempt/training.pt
```

Same-environment CPU epoch continuation includes optimizer, RNG/shuffle,
best/last weights and patience. It rejects changed data/config/schema/IDs,
validation, offsets, source context or Torch version. E5 encoding and other
preprocessing/teacher creation restart; partial E5 batches/LightGBM fits do not
resume. CPU tests establish continuation, not GPU behavior. Individual helper
artifacts and run IDs cannot be overwritten. Raw outputs are private.

## Reproduce the infrastructure checks

The first successful run used exact project commit
`3188a243064f6c0eb8624dd5eed4744ec4d3c037`; its unedited
[summary](../results/phase1_product_sentiment_frozen.json) records all inputs.
Later report changes do not imply the experiment was rerun. A subsequent metadata
fix leaves service charges unmeasured on other hosts; the recorded local run
used no external paid compute service. It does not change prediction behavior.

One metadata-only detail: `eligibility_roster_sha256` in the first run hashes
the Windows working file with CRLF endings. Git stores its LF-normalized text,
so this raw-file hash differs in a fresh LF checkout; parsed JSON contents match
the committed roster. Actual CSV, model, executable configuration, and environment
lock hashes were verified. Do not reinterpret this as a different dataset release.

Use Git and CPython 3.11 (no third-party Python packages required):

```powershell
git clone https://github.com/diyarh26/residual-guided-text-tabular.git
cd residual-guided-text-tabular
git rev-parse HEAD
py -3.11 scripts/verify_phase0.py
git status --short --branch
```

On Linux use `python3 scripts/verify_phase0.py`. For a specific historical
ReFuTab run, clone and `git checkout --detach <recorded-full-project-SHA>`
before running its recorded commands. The course PDF is deliberately absent
from public clones; requirements notes include its local hash and references.
The validator does not need the PDF or any network access.

## Historical upstream source

The immutable identity is in
[../third_party/multabench.lock.json](../third_party/multabench.lock.json),
with evidence and discrepancies in [UPSTREAM.md](UPSTREAM.md). An optional
local read-only checkout can be obtained without vendoring it into ReFuTab:

```powershell
git clone --no-checkout https://github.com/alanarazi7/MulTaBench.git third_party/MulTaBench
git -C third_party/MulTaBench checkout --detach 599bd6a5631c96f8aef297cc4cb6e4c197ae0dca
git -C third_party/MulTaBench rev-parse HEAD
```

Verify the printed SHA matches the lock. The checkout path is ignored.
Do not run current README commands against this historical tree or use
`pip install multabench` as a substitute for the pin. Historical source-license
applicability and environment compatibility need review before reuse or copying.
Never overwrite an existing checkout blindly.

## Environment provenance

Before every baseline, record exact dependencies, upstream SHA, hardware,
data/model revisions, configuration, splits, and seeds. The first environment
is a fresh `.venv` installed from `uv.lock`; runtime validation is recorded in
the handoff/log. Package versions are embedded in run manifests. Resolve
CLI-versus-paper settings explicitly; the unmodified frozen path is used here.
GPU/CUDA compatibility and TAR need separate validation before later runs.

## Required experiment manifest (planned)

Each run must include:

- Immutable project SHA, upstream SHA, dirty-tree flag (production runs require
  clean committed source), config content/hash, and command/entry point.
- Dataset identifiers and revisions, license/source, schema/column order,
  allowed feature names, row IDs/split hashes, fold assignments, seeds, and
  classification class order. Do not publish sensitive row identifiers.
- Model/tokenizer revision, prefixes/pooling/normalization, PCA and preprocessing
  fitting scope, teacher OOF fit scopes, metric definitions, and checkpoint
  selection criterion. See [DATA_LEAKAGE.md](DATA_LEAKAGE.md).
- Dependency lock and environment export, OS/Python, accelerator/CUDA,
  deterministic settings and known nondeterminism, runtime/peak memory/cost.
- Status, failure trace, completed steps, metrics per split/dataset/seed, and
  persistent artifact URIs/hashes. Failed and interrupted runs stay in the log.

## Training, inference, and artifacts (future acceptance criteria)

Training must reproduce saved preprocessing, fusion, and downstream artifacts
from fixed source/config/data/seed inputs. Checkpoint resumption must restore
optimizer/scheduler/RNG state and verify provenance/schema compatibility.
Inference must load the complete saved bundle and produce predictions without
a teacher or correction head; no preprocessing may be refitted on query data.
Document exact executable commands when those APIs exist, then test a clean
round trip. Bitwise GPU determinism must be measured rather than promised.

Publish required permitted learned weights to Hugging Face with versioned
metadata and load instructions. Keep datasets, cached embeddings, and large
weights outside Git. Confirm handling of unchanged public E5 weights against
the course requirement. Use [COLAB.md](COLAB.md) for persistent execution.
