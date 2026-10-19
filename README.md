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

**Phase 0 completed:** repository, academic requirements, research plan, historical-source
pin, and persistent handoff. No algorithm, baseline reproduction, dependency
lock for ML, or training/inference API exists yet. Later phases are explicitly
tracked in [the project plan](docs/PROJECT_PLAN.md).

The historical MulTaBench `master` pin is
`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca` (July 26, 2026).
See [source provenance and discrepancies](docs/UPSTREAM.md) and the
[machine-readable pin](third_party/multabench.lock.json).

Before dependent implementation, clarify the PDF's dataset-held-out wording,
text/image coverage, and multiclass metric. The PDF also has an incorrect
MulTaBench arXiv identifier. These are recorded in
[course requirements](docs/COURSE_REQUIREMENTS.md). The historical snapshot
has no license file; no upstream code is redistributed here.

## Setup and checks

Prerequisites: Git and Python 3.11. Phase 0's validator uses only the standard
library and downloads no models or data.

```powershell
git clone https://github.com/diyarh26/residual-guided-text-tabular.git
cd residual-guided-text-tabular
py -3.11 scripts/verify_phase0.py
```

On Linux/Colab use `python3 scripts/verify_phase0.py`. These commands validate
infrastructure only. Installation of pinned ML dependencies, baseline commands,
and executable experiment entry points belong to Phase 1 and later; do not
interpret the proposed method as an available API.

## Reproduction and development

Read [AGENTS.md](AGENTS.md) and [the handoff](docs/HANDOFF.md) at the start of
every session. Work one phase at a time; run appropriate checks, update docs,
commit, push, and verify synchronization at each completed milestone.

Experiments will record exact project/upstream/data/model revisions, splits,
seeds, configurations, hardware, metrics, runtime, failures, and persistent
artifact locations. See [reproducibility](docs/REPRODUCIBILITY.md),
[Colab workflow](docs/COLAB.md), and [leakage protocol](docs/DATA_LEAKAGE.md).
There are no experiment scores yet; [the log](docs/EXPERIMENT_LOG.md) records
that explicitly.

## Repository layout

```text
configs/                Future experiment configurations
docs/                   Requirements, plan, architecture, decisions, handoff
src/refutab/            Future original algorithm modules
scripts/                Infrastructure and future orchestration entry points
tests/                  Future algorithm and leakage tests
notebooks/              Thin Colab runners (workflow documented; runner pending)
third_party/            Upstream pin and attribution; checkout ignored
data/                   Local datasets (ignored)
artifacts/              Local model outputs (ignored)
results/                Reviewed lightweight summaries; no results yet
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
