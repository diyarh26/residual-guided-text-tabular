# ReFuTab

**Residual-Guided Fusion for Text-Tabular Learning**

ReFuTab investigates whether a structured-only teacher can guide a text fusion
network to learn information that complements numerical and categorical
features. This is a Technion NLP (097215) Track 1 algorithm project, due
**October 26, 2026**. The research hypothesis remains unestablished; no ReFuTab
benchmark improvement is claimed.

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

**Next: controlled scaling.** The separate
[scaling notebook](notebooks/phase4_scaling.ipynb) prepares nested
900/2,700/largest fitting sets with fixed grouped validation/assessment, four
methods and three replicates (72 fits). The real-data split preflight passes;
GPU scaling has not started. Setup must pass the saved-pilot artifact audit
before launch. See [execution steps](docs/SCALING_EXPERIMENT.md).

**Phase 4 GPU execution completed:** all **36 fitting stages and 24 outer-test
evaluations**, including six Joint TAR conditions, finished on Tesla T4 using
the registered checkpointed execution mode. A direct Drive review passed 209
metadata/aggregation checks. **TAR had the highest mean on both tasks; residual
fusion did not show a consistent advantage over direct fusion.** Grouped
diagnostics were weaker for residual. The registered recommendation is to
analyse failures and overfitting before scaling. See the
[results report](docs/PHASE4_RESULTS.md) and
[reviewed metrics](results/phase4_gpu_pilot_review.json).
Phase 4 remains open for independent saved-model/teacher/prediction verification
and targeted analysis. Do not repeat the finished run or begin full evaluation
from this result. The earlier CPU preflight and original failure records are
preserved; see the [pilot protocol](docs/PHASE4.md) and
[GPU handoff](docs/PHASE4_GPU.md).

A [research shortlist for the next discussion](docs/RESEARCH_IDEAS_2026-10-21.md)
contains seven untested directions with prior work, novelty limits and proposed
mechanism checks. It does not change the registered method or reported results.

**Phase 3 completed:** Option A fusion, the direct-target control, checkpoint
continuation and teacher/head-free inference, with **86 passing tests**. A real
frozen-E5 synthetic smoke completed on CPU in **47.23 seconds**, with no test
predictions or scores. Standalone CSV inference reproduced the saved outputs.
See the [run](results/phase3_synthetic_binary_fusion.json),
[readback audit](results/phase3_inference_audit.json),
[Phase 3 API](docs/PHASE3.md), and [experiment log](docs/EXPERIMENT_LOG.md).

Phase 2 implemented general data/split contracts and a 3-fold OOF CatBoost
teacher. Its [Product Sentiment smoke](results/phase2_product_sentiment_teacher.json)
produced 900 OOF/100 validation predictions and no test predictions or scores.

Phase 1 reproduced original Joint Frozen LightGBM on that dataset: macro OVR
AUROC **0.83935** on 509 test rows, one fold with 900 fitting/100 validation rows.
Its [result](results/phase1_product_sentiment_frozen.json) is preserved.
The registered two-dataset GPU pilot now includes Joint TAR; broader benchmark
evaluation has not started. The [project plan](docs/PROJECT_PLAN.md) keeps Phase 4
open for remaining artifact verification and analysis. CPU was sufficient for
Phase 3; the completed Phase 4 comparison used a T4 GPU.

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

For Phase 4, prepare the two immutable pilot tables with
`uv run --frozen python scripts/prepare_phase4.py`. Use the
[GPU pilot notebook in Colab](https://colab.research.google.com/github/diyarh26/residual-guided-text-tabular/blob/main/notebooks/phase4_pilot.ipynb)
for the complete comparison. Select a GPU runtime, keep the prefilled reviewed
implementation SHA, and follow the
[GPU instructions](docs/PHASE4_GPU.md). The notebook installs a separate locked
CUDA environment, verifies the actual device, saves outputs to private Drive,
and waits for its explicit launch switch. A full-size capacity check tests the
checkpointed FP32/batch-256 workload before pilot fitting. That capacity check
passed on the user's T4, and the registered pilot has completed. Preserve that
run; the notebook remains available for reproduction, not an automatic rerun.
No automatic batch-size change is applied.
CPU stages and GPU stages require separate run bindings.

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
src/refutab/            Baseline, teacher, fusion, TAR, pilot and reporting modules
scripts/                Preparation, training and CSV inference entry points
tests/                  Protocol, fit-scope, teacher/leakage, persistence checks
notebooks/              Thin Colab CPU baseline and GPU pilot runners
environments/gpu/       Separate Linux CUDA dependency lock
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
