# Historical baseline source analysis

Inspected October 19, 2026. Source: MulTaBench historical `master`, immutable
commit `599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`. This document records inspected
behavior; execution outcomes belong in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
No ReFuTab implementation is implied by the integration discussion below.

## Smallest original Joint Frozen path

For the Phase 1 smoke experiment, invoke the original
[`LightGBM`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/lgbm.py)
class with `tune_e5=False`, `pca_components=30`, and an explicit
`torch.device("cpu")`. Retain all original structured and text features from an
eligible text dataset, use the original sampling/split functions, call its
unchanged `fit`, then its unchanged `score_all_metrics` on the outer test data.
This uses the original preprocessing and learner rather than a reimplementation.
A reduced training cap is a smoke configuration; it is not the paper's full
10,000-row experiment or a matched published score.

The full upstream CLI equivalent for the selected example task is:

```bash
python benchmark.py --model light --dataset_name MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --train_examples 10000 --multimodal_state all --e5_model e5-small --pca_components 30
```

That CLI imports every registered model ecosystem unconditionally, including
AutoGluon, ConTextTab, TabPFN, TabICL, TabDPT, and TabM. Its logging function also
refuses to run without `WANDB_API_KEY` and `WANDB_ENTITY`. Calling the original
LightGBM class from an external harness avoids these unrelated imports and
account requirements. The harness must supply persistent local provenance and
logging; the learner, embedding math, metrics, and split functions remain the
original implementations.

The upstream
[`evaluate_on_loaded_dataset`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/benchmarks/evaluate.py#L25)
does the same sequence: `subsample_dataset`, `split_to_test`, model construction,
`fit`, then `score_all_metrics`. Its standard result dictionary does not save
full split row IDs, immutable model/data revisions, or the fitted model. Those
are additional provenance responsibilities of the ReFuTab harness.

## Source flow and fitting boundaries

| Concern | Inspected source behavior |
| --- | --- |
| Data | `multabench/benchmark/load.py::load_multabench_dataset` loads Kaggle `data.csv` and `metadata.json`; separates `meta["target"]`; `meta["image_col"]` identifies image scope. It does not pin a Kaggle version. The dataset inventory separately establishes eligible schemas and data revisions. |
| Outer sampling | `multabench/baselines/preprocessing/sampling.py::subsample_dataset` calls **TabSTAR 1.1.15** split helpers, with seed `42 + fold`. The retained nominal budget is requested training cap plus the original dataset's 10% holdout capped at 2,000. |
| Outer evaluation split | `tabstar/preprocessing/splits.py::split_to_test` uses `42 + fold`. These are repeated seeded holdouts, not disjoint K-fold CV. `FOLDS=10` in the upstream evaluator; the contemporaneous paper reports five seeds. |
| Learner validation | `multabench/baselines/abstract_model.py::TabularModel.fit` holds out validation **before fitting any preprocessing**, because LightGBM has `USE_VAL_SPLIT=True`. The original TabSTAR `split_to_val` receives no explicit fold, so its default `fold=-1` gives seed **41**. Its requested size is 10% capped at 1,000. |
| Schema/preprocessing | `do_model_agnostic_preprocessing` fits date encoders, detects numerical/text/categorical features, and fits target labels using the learner-fit subset. LightGBM leaves numerical missing values native and ordinal-encodes categorical values; unseen categories map to the reserved missing value. |
| Frozen E5 | `text_embeddings.py::fit_text_encoders_vanilla` shares one vanilla E5 model across text columns and fits a distinct PCA per column. Text fields are processed in sorted order. No labels enter frozen E5 or PCA. |
| PCA scope | PCA is fitted on only the learner-fit embeddings, after validation removal. The same PCA transforms learner validation and outer test data. Default 30 components and `random_state=42`; insufficient fitting rows must fail rather than silently lower dimensions. |
| Final feature layout | Each original text column is removed and its 30 PCA columns appended in sorted text-field order. Original non-text columns retain their order. Categorical names are resolved after text expansion, avoiding stale positional indices. |
| LightGBM fitting | `LGBMClassifier`/`LGBMRegressor` uses upstream defaults plus `random_state=42` and `verbose=-1`. Validation is supplied as `eval_set`, but no early-stopping callback is configured: validation does not select the number of trees in this implementation. |
| Test path | `score_all_metrics` uses the saved target encoder, then `predict` applies saved preprocessors and calls the fitted learner. It performs no PCA/model/preprocessor fitting on test data. |

The exact split row counts must be measured. TabSTAR's binary helper first
reserves one row per class for the holdout and then requests the original
holdout size from the remaining rows. It appends those two reserved rows, making
the actual binary test/validation count two greater than requested. The same
helper is used during binary subsampling, so neither the requested cap nor a
simple 90/10 calculation is a reliable statement of actual fit counts.
Multiclass singleton classes are kept in training while the remaining classes
are stratified. Preserve these historical semantics for reproduction and save
the actual row IDs and class coverage of every partition.

Text detection in `multabench/preprocessing/feat_types.py` classifies a
non-numerical column as text when it has at least 100 valid distinct values **or**
a valid-value uniqueness ratio of at least 0.8. Detection occurs on the
learner-fit subset, so the detected schema can depend on the smoke training cap.
Save the detected columns, including all PCA feature names, rather than assuming
metadata labels alone determine the learner's representation.

## Encoding and TAR details

[`multabench/e5/e5_finetune.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/e5/e5_finetune.py#L185)
encodes `passage: column_name: value` with E5-small-v2 (384 dimensions), padding
to 512 tokens, truncation at 512, batch size 32, masked mean pooling, then L2
normalization. It calls `model.eval()` and uses `torch.no_grad()`. Thus frozen
inference disables dropout and gradient recording even though source loading
does not set every parameter's `requires_grad` flag to false. PCA is applied
directly to these normalized embeddings without standardization. The original
fit path encodes the fitting texts once for PCA and again when building learner
features; it contains no embedding cache.

Joint TAR uses the CLI state `ft` for text datasets, or the explicit
`tune_e5=True` class setting. `fit_text_encoders_tuned` splits the learner-fit
rows again using the local duplicate of `split_to_val` (also default seed 41),
then expands each row into one labeled example per sorted text column. Therefore
all text fields of a row remain on the same side of the TAR split. A single
shared E5 model learns target labels using LoRA and a temporary linear
classification head. Regression labels become up to 20 quantile bins before
the TAR split; this binning stays within the learner-fit partition. The best
validation-loss checkpoint is selected, and per-column PCA is then fitted on
the entire learner-fit partition using the selected encoder, including its
internal TAR validation rows. Outer validation and test remain excluded.

The CLI explicitly passes rank 16, last 3 layers, learning rate `1e-4`, 50
epochs, patience 3, batch size 256, weight decay `0.001`; LoRA uses alpha 32 and
dropout 0.1. The direct function defaults differ (40 epochs, batch size 32,
weight decay `0.01`), so a future direct TAR runner must explicitly supply the
historical CLI settings. The paper's weight decay `0.01` differs from the CLI.
Preserve and report the selected source settings rather than blending them.

## Class order and metric semantics

`multabench/baselines/preprocessing/target.py` delegates to TabSTAR's
`LabelEncoder` fitted on learner-fit labels. Save `target_transformer.classes_`
and the fitted classifier's `classes_`; probability column `k` corresponds to
the saved encoded class `k`. Binary inference returns probability column 1.
Do not reconstruct class order from a held-out partition. Unseen held-out
labels should be surfaced as a failure, not silently reassigned.

The actual downstream metric source is
[`multabench/baselines/training/metrics.py`](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/baselines/training/metrics.py):

| Task | Primary score | Additional behavior |
| --- | --- | --- |
| Regression | R-squared | Also logs MSE and RMSE in original target units; downstream regression targets are not standardized. |
| Binary | AUROC | Uses probability of the second saved class. |
| Multiclass | Macro one-vs-rest AUROC | First calls `roc_auc_score(..., multi_class="ovr", average="macro")`. On `ValueError` or `AxisError`, averages one-vs-rest AUROC for classes present in `y_true`, indexing probability columns by encoded class ID. Also attempts log loss; failure becomes NaN. |

The present-class fallback changes the averaging population when a test class
is absent. Record class coverage and whether this fallback was required. A
single-class holdout can produce an undefined AUROC; finite-score checks must
reject it as an unsuccessful metric rather than report a valid result. Later
master's multiclass log loss and regression RMSE are different protocols.

## Environment and native Windows execution

The direct LightGBM import still eagerly imports image helpers, DINOv3 model
classes, E5 tuning classes, and PEFT even for text-only frozen runs. The
inspected historical requirements pin these relevant packages:

| Package | Historical version |
| --- | --- |
| torch / torchvision | 2.7.1 / 0.22.1 |
| transformers / tokenizers | 4.57.3 / 0.22.1 |
| peft / accelerate | 0.18.0 / 1.12.0 |
| huggingface-hub / safetensors | 0.36.0 / 0.7.0 |
| numpy / scipy / pandas | 2.3.5 / 1.16.3 / 2.3.3 |
| scikit-learn / skrub | 1.6.1 / 0.6.2 |
| lightgbm / tabstar / pillow | 4.6.0 / 1.1.15 / 11.3.0 |

These are the inspected relevant upstream pins, not a claim that this table is
a complete transitive lock or that runtime compatibility was established by
static inspection. Model/data download and provenance tools add their own
requirements. The full upstream environment includes many unrelated learners
and is unnecessary for direct LightGBM use.

Frozen E5 encoding has **no CUDA assertion** and accepts CPU without source
changes. Native Windows CPU execution is therefore possible in the inspected
code path; actual speed depends on the machine and sample size. Supply an
explicit `torch.device` rather than TabSTAR's automatic GPU picker, whose
free-GPU detection uses a Linux `grep` subprocess. CPU E5 uses the same pooling,
normalization, PCA, and learner math, subject to ordinary numerical differences
between platforms/devices. Limit thread counts in the harness and record them.

Only `finetune_e5_with_lora` asserts that `CUDA_VISIBLE_DEVICES` exists. This is
an environment-variable check, not an inference restriction. No removal or dummy
GPU setting is necessary for the Frozen smoke baseline. TAR is substantially
heavier and should be run separately on suitable hardware with explicit
historical settings, without treating a Frozen result as TAR reproduction.

The vanilla loader accepts a local immutable Hugging Face snapshot path as
`e5_model_name`, so a downloaded revision can be pinned without modifying source
for Frozen. TAR additionally indexes `E5_DIM[model_name]` and
`E5_NUM_LAYERS[model_name]`, which require the original model identifier. A future
TAR harness must pin the loader revision while preserving that identifier;
passing a local path directly is insufficient for TAR.

## Later ReFuTab integration seam

The narrow seam is the text-transformer boundary in `TabularModel`: preserve
the original structured preprocessing and final learner, replace the fitted
per-field text representation with saved ReFuTab fused blocks, and maintain
the same row indices and deterministic column ordering. The fusion module also
needs original structured inputs, which the existing per-column
`E5ColumnEncoder.encode_texts(texts, device)` interface does not receive. Merely
subclassing that wrapper cannot express context-aware gating. Use an explicit
external adapter or original subclass that receives the full row and records
its own fitting boundaries; do not monkey-patch a context-free wrapper and
claim it implements ReFuTab.

This seam is a design observation for later phases. Phase 1 must first establish
an unchanged upstream baseline. All OOF teacher construction, supervised-gate
training, correction heads, and alignment diagnostics remain deferred to their
implementation phases under [DATA_LEAKAGE.md](DATA_LEAKAGE.md).
