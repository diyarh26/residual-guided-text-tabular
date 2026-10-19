# Reproducibility

Status: Phase 0 infrastructure only. No training/inference entry point or
validated ML dependency lock exists yet. Do not treat proposed commands as
reproduced experiments. Actual scores and failures belong in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Reproduce the infrastructure checks

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

## Phase 1 environment gate

Before a baseline, record and validate a compatible Python version, exact
dependencies and transitive lock, PyTorch/CUDA combination if used, upstream
dependencies, hardware, and data/model revisions. A fresh environment must
install from that lock and pass a smoke test. Resolve CLI-versus-paper settings
explicitly, including what the source actually passes to TAR. No GPU package
installation or large model download has occurred in Phase 0.

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
