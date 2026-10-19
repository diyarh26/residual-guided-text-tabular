# MulTaBench provenance and historical source pin

Inspected on **October 19, 2026**. Phase 0 established provenance; Phase 1 added
the direct-source analysis in [BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md) and a
completed frozen smoke baseline in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
No upstream code is vendored and no submodule is installed. The machine-readable pin is
[../third_party/multabench.lock.json](../third_party/multabench.lock.json).

## Selection and evidence

Repository: [alanarazi7/MulTaBench](https://github.com/alanarazi7/MulTaBench),
branch `master`. Interpret the requested reference date as the whole day
**July 26, 2026 UTC**, with an inclusive cutoff of `2026-07-26T23:59:59Z`.
The selected immutable commit is
[`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`](https://github.com/alanarazi7/MulTaBench/commit/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca):
“Add TabPFN-v3 and TabFM results on the 40 paper datasets (#12)”. Both its author
and committer timestamps are `2026-07-26T18:26:40Z`
(`2026-07-26T21:26:40+03:00`).

| Position in reachable `master` first-parent history | Commit | Committer timestamp (UTC) |
| --- | --- | --- |
| Selected commit's parent | [`8bf543913c7ac1cdd0d223a509ed8a1b00249310`](https://github.com/alanarazi7/MulTaBench/commit/8bf543913c7ac1cdd0d223a509ed8a1b00249310) | `2026-07-26T11:38:30Z` |
| Selected commit | [`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`](https://github.com/alanarazi7/MulTaBench/commit/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca) | `2026-07-26T18:26:40Z` |
| Next commit, a direct child | [`8ff8fc5647594077b4fcee6e80d184951fae2c64`](https://github.com/alanarazi7/MulTaBench/commit/8ff8fc5647594077b4fcee6e80d184951fae2c64) | `2026-09-01T12:13:35Z` |

Using the whole July 26 in `Asia/Jerusalem` instead selects the same SHA. A
start-of-day interpretation would select an earlier commit, so the whole-day
assumption is explicit. The next reachable commit is in September, leaving no
end-of-day timezone ambiguity. The pinned source contains the E5, PCA,
fine-tuning, dataset-loading and evaluation paths needed for later reproduction.
Its final commit adds result CSVs; it is selected by history, not by cherry-picking
an implementation change.

Evidence was checked with `git ls-remote --heads`, a fresh history clone outside
this workspace, `git log --first-parent` through the cutoff, and the
[GitHub commits API with the same cutoff](https://api.github.com/repos/alanarazi7/MulTaBench/commits?sha=master&until=2026-07-26T23%3A59%3A59Z&per_page=3).
The inspected current `master` was
[`19225a546259256431654ec19fa229fa5f8a5a1c`](https://github.com/alanarazi7/MulTaBench/commit/19225a546259256431654ec19fa229fa5f8a5a1c),
committed `2026-10-19T12:39:05Z`.

**Historical-ref limitation:** this reconstructs the latest eligible commit from
currently reachable `master` ancestry and commit timestamps. It does not prove
which ref GitHub served on July 26, because no contemporaneous ref archive or
complete branch-update event log was obtained. A past force-push could change
that history. Pinning the full SHA makes our selected source unambiguous. Neither
the `paper_version` tag nor `official-paper-version` branch was selected.

## Source map and behavior at the pin

All links in this table use the immutable selected SHA.

| Concern | Source and observed behavior |
| --- | --- |
| CLI and conditions | [`benchmark.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/benchmark.py): `all` keeps structured and text inputs with frozen encoders; `ft` also activates E5 tuning on text datasets. `non_txt` is the structured-only text-dataset condition; `txt` is text-only. |
| E5 representation | [`multabench/e5/constants.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/e5/constants.py) and [`e5_finetune.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/e5/e5_finetune.py): `intfloat/e5-small-v2`, 384 dimensions, input `passage: col_name: col_val`, masked mean pooling then L2 normalization for encoding, truncation/padding at 512 tokens. |
| TAR and projection | [`multabench/baselines/preprocessing/text_embeddings.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/preprocessing/text_embeddings.py): one E5 model tuned across row-column text examples without structured features; per-column PCA, default 30 components, fitted on training embeddings. Regression labels become up to 20 quantile bins, with fewer bins if duplicate edges collapse. |
| E5 tuning settings | [`multabench/finetune/train_args.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/finetune/train_args.py): CLI defaults rank 16, last 3 layers, learning rate `1e-4`, 50 epochs, patience 3, batch size 256, weight decay `0.001`. The E5 implementation uses alpha twice rank, dropout `0.1`, and best validation-loss checkpoint selection. |
| Fit boundaries | [`multabench/baselines/abstract_model.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/abstract_model.py): learners needing validation first hold it out, then fit encoders and preprocessing on their remaining training rows. TAR has another internal validation split. Test rows only pass through the fitted transformations. |
| Evaluation and sample cap | [`multabench/baselines/benchmarks/evaluate.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/benchmarks/evaluate.py) and [`preprocessing/sampling.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/preprocessing/sampling.py): default training cap 10,000; fold controls sampling/split seeds; evaluator exposes `FOLDS = 10`, while the paper reports five seeds. |
| Metrics | [`multabench/baselines/training/metrics.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/training/metrics.py): AUC for binary classification, macro one-vs-rest AUC for multiclass with a present-class fallback, and R-squared for regression. Other metrics may be logged but are not the primary score. |
| Curated text datasets | [`multabench/datasets/all_datasets.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/datasets/all_datasets.py) and [`multabench/benchmark/load.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/benchmark/load.py): 20 curated text datasets in `MulTaBenchDatasetID`; Kaggle `data.csv` plus metadata loading. No dataset version is supplied to `kagglehub.dataset_download`. |

### Split dependency audit

The outer evaluator and sampling module import split functions/constants from
**TabSTAR**, not the similarly named local module. Learner validation also
imports TabSTAR's `split_to_val`; E5's internal validation uses
[`multabench/preprocessing/splits.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/preprocessing/splits.py).
The snapshot's [`requirements.txt`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/requirements.txt)
locks `tabstar==1.1.15`, although `requirements.in` permits newer versions.

The [PyPI 1.1.15 wheel metadata](https://pypi.org/pypi/tabstar/1.1.15/json)
and wheel contents were inspected without installing or importing the package.
The downloaded `tabstar-1.1.15-py3-none-any.whl` SHA-256 is
`9c4bf7392101813aac9c0209edbe664e1091c4463ddb4628869a9569d75c1d4e`.
Its `tabstar/preprocessing/splits.py` matches the local duplicate, and
`tabstar/constants.py` sets `SEED = 42`. Splits use `42 + fold`, a nominal
10% test holdout capped at 2,000 before training-cap handling, and nominal 10%
validation capped at 1,000. Classification uses stratification with rare-class
handling. The binary helper additionally inserts one selected row per class into
the test/validation side, adding two rows beyond the requested holdout count.
Consequently exact row counts need executable Phase 1 checks; “90/10” alone is
not a sufficient split specification. These are repeated seeded holdouts, not
the later six-fold protocol.

### Baseline commands for Phase 1

These commands were checked statically against the selected CLI, model keys,
dataset enum and condition mapping. **They have not been executed**; dependency,
credential, data-version and hardware validation belongs to Phase 1. Run from a
separate checkout at the full SHA, using its locked requirements, after the
historical licensing question below is resolved for the intended source use.

```bash
# Joint Frozen, text-tabular pilot
python benchmark.py --model light --dataset_name MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --train_examples 10000 --multimodal_state all --e5_model e5-small --pca_components 30

# Joint TAR with the snapshot's explicit CLI defaults
python benchmark.py --model light --dataset_name MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --train_examples 10000 --multimodal_state ft --e5_model e5-small --pca_components 30 --e5_lr 1e-4 --e5_rank 16 --e5_text_layers 3 --e5_epochs 50 --e5_patience 3 --e5_weight_decay 0.001 --e5_batch_size 256
```

The actual parser uses ASCII `ft`; some historical README examples show
`all` plus a fire emoji, which is not accepted by this parser. The script starts
WandB and expects the environment described in its setup. Model/tokenizer
downloads lack immutable revision pins. Capture actual dataset/model revisions,
data hashes, split row IDs and complete configuration before any reproducibility
claim. Text-field identification and the E5 CUDA environment assertion also need
runtime verification.

## Paper comparison and later drift

The unversioned [paper link](https://arxiv.org/abs/2605.10616) currently resolves
to v2, revised September 27, 2026. The version contemporaneous with the July
source is [v1, submitted May 11](https://arxiv.org/abs/2605.10616v1).

Both versions describe frozen E5-small-v2, PCA to 30, and target-only LoRA on
the last three layers with a shared text encoder. The pinned implementation
contains those paths. [v1 Appendix A](https://arxiv.org/html/2605.10616v1#A1)
reports five seeds, a 10,000 training-row cap, AUC/R-squared metrics, 50 E5
epochs and batch size 256. It specifies weight decay `0.01`; the pinned CLI
passes `0.001`. [v2 Appendix A](https://arxiv.org/html/2605.10616v2#A1)
retains those settings, including `0.01`. Do not silently change the source
default to claim paper reproduction: preserve the pin's setting, record the
discrepancy, and decide/document any separate paper-aligned comparison in Phase 1.

The source's E5 function defaults are 40 epochs and batch size 32; the CLI
explicitly overrides these to 50 and 256. Direct calls must pass the intended
configuration. The source text detector accepts either at least 100 distinct
values **or** a unique-value ratio of at least 0.8, whereas the paper describes
the first rule only. Date handling and inferred text fields must therefore be
audited against the selected dataset and split.

The paper describes TAR as reusable preprocessing across learners. This source
fits encoders inside each learner's `fit`, after any learner validation holdout.
Consequently encoder-training rows can depend on whether a learner requests
validation; equality of learned representations across learners is not assumed.
Record these boundaries when reproducing the baseline.

Current `master` is substantially different:

| Later change | Immutable evidence |
| --- | --- |
| Different metrics: binary AUC, multiclass log loss, regression RMSE | [`e6ecfdead5df8d9aa6be77edc71fb3476d9661b0`](https://github.com/alanarazi7/MulTaBench/commit/e6ecfdead5df8d9aa6be77edc71fb3476d9661b0), September 29 |
| Three-fold splitting repeated twice | [`70f86787a14f65bd519c01961a3952422ad398e0`](https://github.com/alanarazi7/MulTaBench/commit/70f86787a14f65bd519c01961a3952422ad398e0), September 30 |
| Eighty typed Parquet datasets on Hugging Face, replacing Kaggle loader | [`2d8f042de15e5dde358127e47e5f3be9bb6f6f47`](https://github.com/alanarazi7/MulTaBench/commit/2d8f042de15e5dde358127e47e5f3be9bb6f6f47), October 2 |
| Shared LoRA loop and encoder refactoring | [`54b7c09dd467590de09ee814ca40505f0d11f545`](https://github.com/alanarazi7/MulTaBench/commit/54b7c09dd467590de09ee814ca40505f0d11f545), October 17 |
| Cached frozen and tuned representations, new model roster and leaderboard protocol | [`docs/protocol.md` at inspected master](https://github.com/alanarazi7/MulTaBench/blob/19225a546259256431654ec19fa229fa5f8a5a1c/docs/protocol.md) and [`docs/embeddings.md`](https://github.com/alanarazi7/MulTaBench/blob/19225a546259256431654ec19fa229fa5f8a5a1c/docs/embeddings.md) |

The current protocol documentation itself states that `master` has moved away
from the paper protocol. Its recommendation of `paper_version` is recorded as
upstream context only; it does not override this project's historical-master
requirement. Results from these protocols must not be mixed as equivalent runs.

## License and attribution boundary

The [selected tree](https://github.com/alanarazi7/MulTaBench/tree/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca)
contains no `LICENSE`, `COPYING` or `NOTICE` file, and its README contains no
license declaration. The repository later added an MIT license at
[`21382f806df3d4a3c27df46bd25467796301be55`](https://github.com/alanarazi7/MulTaBench/commit/21382f806df3d4a3c27df46bd25467796301be55),
committed **October 14, 2026 at `22:08:47Z`** (October 15 locally). The later
[`LICENSE`](https://github.com/alanarazi7/MulTaBench/blob/19225a546259256431654ec19fa229fa5f8a5a1c/LICENSE)
names “Copyright (c) 2026 Alan Arazi”. That current declaration is not recorded
here as proof of the July snapshot's license coverage.

Phase 0 therefore publishes only source references, this provenance report and
the SHA lock. It copies no upstream implementation or result files. Before
vendoring or redistributing historical code, establish its applicable license
and retain all required notices; record the evidence and the user's decision.
Original ReFuTab work has its own MIT license. Dataset terms remain separate
from code terms and must be checked individually before use or redistribution.

## Phase 1 acceptance work

1. Confirm the intended historical-source usage and license evidence; preserve
   the full selected SHA and record any approved deviation explicitly.
2. Resolve source-versus-paper settings without changing provenance; pin the
   runtime dependency environment, E5 revision and dataset versions/hashes.
3. Reproduce at least one original baseline, mapping both Frozen and TAR paths;
   use identical recorded outer splits when comparing them. Verify feature
   columns, class order, holdout sizes, PCA fit boundaries and train-only encoder
   fitting, including the binary split special case.
4. Record actual execution outcomes and failures. Static source inspection is
   not successful baseline reproduction and is not a training result.
