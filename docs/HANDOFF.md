# Project handoff

Updated: October 19, 2026 (Asia/Jerusalem).

## Current phase and status

**Phase 1 - Completed: historical source analysis and one real baseline reproduction.**
**Phase 2 - Not started.** Phase 0 remains complete. No ReFuTab teacher, fusion
network, Joint TAR experiment, or full benchmark evaluation is implemented or
claimed. Stop at the Phase 1 report; begin the next phase when authorized.

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
  without image columns; 20 image-column datasets excluded**. Only the product
  sentiment CSV was downloaded; the other 19 eligible CSVs still need content
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
**The held-out test partition was predicted and evaluated once.** No test-score
tuning or subsequent model experiment was performed.

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
  `3188a243064f6c0eb8624dd5eed4744ec4d3c037`
  (`feat: add pinned historical baseline runner and reproducible CPU setup`).
- The result summaries and completion documentation accompany this handoff in
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

## Exact next steps: Phase 2

1. Read AGENTS, this handoff, PROJECT_PLAN, COURSE_REQUIREMENTS, ARCHITECTURE,
   DATA_LEAKAGE, and the Phase 1 analysis. Preserve existing changes.
2. Build the general dataset/schema interface, including stable field/class
   ordering, missing values, regression/binary/multiclass support, and an
   explicit audit excluding IDs, targets, split markers, and target-derived
   metadata from features. The Phase 1 full-row upstream adapter reproduces a
   baseline; it does not certify every other dataset's feature safety.
3. Persist official outer and internal-validation assignments and fingerprints;
   retain historical split semantics explicitly or document a justified change.
   Pin and hash each additional dataset before executing it.
4. Implement a structured-only **3-fold OOF CatBoost teacher**, isolated by
   dataset/outer fold. Every OOF row must be excluded from its teacher's fitting,
   preprocessing, calibration, and selection; validation uses a separate
   training-only teacher.
5. Test split disjointness, forbidden features, train-only fitting, exactly one
   aligned OOF prediction per row, stable class probabilities, unknown values,
   and deterministic failures. Record checks, update the handoff/plan/log,
   commit, push, and verify. Fusion remains Phase 3 work.

## Remaining technical boundaries

- Joint TAR remains the primary comparison for supervised ReFuTab; its training,
  local model-path lookups, and GPU execution have not been validated. No full
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
