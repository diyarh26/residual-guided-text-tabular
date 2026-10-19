# ReFuTab

**Residual-Guided Fusion for Text-Tabular Learning**

ReFuTab investigates whether a structured-only teacher can guide a text fusion
network to learn information that complements numerical and categorical
features. This is a Technion NLP (097215) Track 1 algorithm project, due
**October 26, 2026**. The research hypothesis is untested; no benchmark results
or performance improvements are claimed.

## Proposed method

1. Train a structured-only CatBoost teacher with 3-fold out-of-fold predictions.
2. Encode each text field with frozen E5-small-v2 (384 dimensions), then apply
   training-only PCA to 30 dimensions per field.
3. Condition an identity-initialized gate on text and compact structured
   features: `z = e * (1 + tanh(G(e, s)))`.
4. Train the gate with a temporary correction head and fixed teacher offsets
   (regression predictions, binary logits, or multiclass log-probabilities).
5. Discard the correction head; train a downstream tabular learner on original
   structured columns plus fused text embeddings. The teacher is unnecessary
   at final inference.

Unlike MulTaBench's Joint Frozen condition, this learns supervised text
representations. Unlike Joint TAR's encoder adaptation, Option A freezes E5
and trains a small, structured-context gate using the teacher's mistakes.
**Joint TAR is the primary comparison**, with Joint Frozen and direct-target
gated fusion as additional controls. Option B (LoRA) is deferred.

## Current status

**Phase 2 completed:** general dataset/split contracts, a structured-only
3-fold OOF CatBoost teacher, checked model persistence, and **43 passing tests**.
The real Product Sentiment teacher smoke produced 900 aligned OOF predictions
and 100 validation predictions, with **zero test predictions or scores**. See
its [recorded summary](results/phase2_product_sentiment_teacher.json),
[API and safeguards](docs/PHASE2.md), and [experiment log](docs/EXPERIMENT_LOG.md).

Phase 1 reproduced original Joint Frozen LightGBM on that dataset: macro OVR
AUROC **0.83935** on 509 test rows, one fold with 900 fitting/100 validation rows.
Its [result](results/phase1_product_sentiment_frozen.json) is preserved.
Full ReFuTab fusion and Joint TAR are unevaluated; Phase 3 fusion is next in the
[project plan](docs/PROJECT_PLAN.md).

The historical MulTaBench `master` pin is
`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca` (July 26, 2026).
See [source provenance and discrepancies](docs/UPSTREAM.md) and the
[machine-readable pin](third_party/multabench.lock.json).

Dataset scope is confirmed: use only benchmark datasets whose original schema
has no image column. Datasets containing an image column are excluded.
The user also confirmed separate training on each dataset's training split,
with internal validation and an untouched test split. Both course clarifications
are resolved; Phase 1 validated split boundaries and historical multiclass
metrics. The PDF also has an incorrect
MulTaBench arXiv identifier. These are recorded in
[course requirements](docs/COURSE_REQUIREMENTS.md). The historical snapshot
has no license file; no upstream code is redistributed here.

## Setup and checks

Prerequisites: Git, Python 3.11, and `uv==0.11.24`. The infrastructure validator
uses only the standard library. The Phase 1 baseline uses the locked environment.

```powershell
git clone https://github.com/diyarh26/residual-guided-text-tabular.git
cd residual-guided-text-tabular
py -3.11 scripts/verify_phase0.py
py -3.11 -m pip install uv==0.11.24
py -3.11 -m uv sync --locked
py -3.11 -m uv run --frozen python scripts/prepare_phase1.py
py -3.11 -m uv run --frozen python -m unittest discover -s tests -v
py -3.11 -m uv run --frozen python scripts/run_baseline.py --run-id my-new-run --summary results/my-new-run.json
```

On Linux use `python3` and `uv`, or use the
[thin CPU Colab runner](notebooks/phase1_baseline.ipynb). Preparation downloads
the immutable source, pilot data, and E5 weights into ignored local directories.
Run only from clean committed source and use a new run ID for each attempt.
See [exact reproduction and artifact details](docs/REPRODUCIBILITY.md).

The Phase 2 teacher needs only the pinned source/data, without E5 weights:

```powershell
py -3.11 -m uv run --frozen python scripts/prepare_phase1.py --skip-model
py -3.11 -m uv run --frozen python scripts/run_teacher.py --run-id my-new-teacher-run --summary results/my-new-teacher-run.json
```

## Reproduction and development

Read [AGENTS.md](AGENTS.md) and [the handoff](docs/HANDOFF.md) at the start of
every session. Work one phase at a time; run appropriate checks, update docs,
commit, push, and verify synchronization at each completed milestone.

Experiments will record exact project/upstream/data/model revisions, splits,
seeds, configurations, hardware, metrics, runtime, failures, and persistent
artifact locations. See [reproducibility](docs/REPRODUCIBILITY.md),
[Colab workflow](docs/COLAB.md), and [leakage protocol](docs/DATA_LEAKAGE.md).
Completed/failed executions and their exact provenance belong in
[the experiment log](docs/EXPERIMENT_LOG.md).

## Repository layout

```text
configs/                Locked pilot inputs and dataset inventory
docs/                   Requirements, plan, architecture, decisions, handoff
src/refutab/            Baseline, data/splits, OOF teacher; fusion pending
scripts/                Infrastructure, preparation, baseline/teacher entry points
tests/                  Protocol, fit-scope, teacher/leakage, persistence checks
notebooks/              Thin Colab CPU baseline runner
third_party/            Upstream pin and attribution; checkout ignored
data/                   Local datasets (ignored)
artifacts/              Local model outputs (ignored)
results/                Reviewed aggregate summaries; raw artifacts ignored
reports/                Future paper, tables, and figures
```

## Research and documentation

- [MulTaBench paper](https://arxiv.org/abs/2605.10616) and
  [historical paper v1](https://arxiv.org/abs/2605.10616v1).
- [Original benchmark](https://github.com/alanarazi7/MulTaBench), by Arazi et al.
- [Architecture](docs/ARCHITECTURE.md), [decisions](docs/DECISIONS.md),
  [course requirements](docs/COURSE_REQUIREMENTS.md), and
  [project plan](docs/PROJECT_PLAN.md).

[MIT](LICENSE) covers original ReFuTab work only. Dataset, pretrained model,
dependency, upstream-source, and course-material terms are separate; see
[third-party notices](THIRD_PARTY_NOTICES.md). The local course PDF is ignored
and has not been published.
