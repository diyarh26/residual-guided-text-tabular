# Phase 2 data and structured teacher

The general dataset interface and structured-only three-fold OOF CatBoost
teacher are implemented. They supply training signals for the planned fusion
stage; they do not implement the gate, correction-head training, final learner,
or a ReFuTab improvement result. Phase status and actual executions are recorded
in [HANDOFF.md](HANDOFF.md) and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Dataset and row contract

[`DatasetSchema`](../src/refutab/data.py) explicitly declares `dataset_id`,
`task` (`regression`, `binary`, or `multiclass`), `target_column`, and ordered
`text_columns`, `numerical_columns`, and `categorical_columns`. `id_column` is
optional. Declare excluded metadata in `excluded_columns` or
`forbidden_columns`; target, ID, excluded, and forbidden columns cannot become
features. Every input column must have a declared role. The contract cannot
discover an undisclosed target-derived field: reviewing the original schema
and feature provenance remains necessary before declaring its features safe.

`image_columns` records original dataset eligibility. A nonempty declaration
rejects the dataset even if someone already dropped the image column. The
benchmark runner also checks the pinned original metadata's `image_col` and
target against the explicit configuration. [DATASETS.md](DATASETS.md) records
the benchmark roster; other eligible CSVs require their own content hashes and
feature audits before execution.

`DatasetTable(frame, schema, row_ids=None)` owns a copy and normalizes its index
to stable row IDs. Without `id_column` or supplied IDs, the source DataFrame
index supplies identity. IDs must be unique, non-null, and consistently either
strings or integers; floats, booleans, mixed types, and empty strings fail.
Provided IDs must agree with a declared ID column. Source index order is
retained in `source_index` for auditing.

`rows(ids)`, `features(ids)`, `structured(ids)`, and `targets(ids)` retain the
requested ID order. Feature order is text, numerical, then categorical in each
declared sequence; the structured view is numerical then categorical. Multiple
text fields and their original missing values are preserved. Loading performs
no learned imputation, scaling, vocabulary fitting, or class encoding.

`DatasetTable.load_csv(path, schema, expected_sha256=...)` verifies the supplied
CSV hash before returning a table and records `source_path` and `source_sha256`.
The runner requires the CSV hash in its configuration. `fingerprint()` binds
schema, IDs/order, column order/dtypes, and typed scalar contents to SHA-256;
integer `1` and string `"1"` remain distinguishable even in object columns.

## Persisted split contract

[`SplitAssignments`](../src/refutab/splits.py) stores ordered `train_ids`,
`validation_ids`, `test_ids`, `excluded_ids`, a seed, protocol metadata, and an
optional dataset-content fingerprint. Validation rejects duplicate, overlapping,
unknown, or omitted IDs. Reduced training caps must record the unselected rows
in `excluded_ids`; they do not silently disappear. `save()` writes full private
assignments and refuses an existing file by default; `load()` restores them.

Class-coverage checks derive classes from the training partition and reject
validation classes absent from training. Typed labels remain distinct, including
`True` versus `1`. Test labels do not establish class order or coverage. A
classification OOF teacher additionally requires at least three training rows
per class and complete class coverage in every fold's fitting complement.

For the benchmark pilot, `historical_phase1` reconstructs unchanged pinned
MulTaBench/TabSTAR 1.1.15 assignments and checks their ID fingerprints against
Phase 1. This retains the source's repeated-holdout, internal seed-41, and
binary holdout-size behavior. It does not substitute a new split algorithm.
The `explicit` runner protocol loads previously saved row IDs. The optional
`local` protocol calls `generate_splits`: a deterministic random holdout for
regression or stratified holdout for classification, with declared full-table
validation/test fractions. It is a separate protocol and fails when class
counts cannot support stratification. All-label access in split construction
is confined to stratification; it does not fit a predictor or class encoder.
Neither generic local splits nor generic OOF folds enforce grouping or temporal
constraints. Datasets needing those constraints require audited explicit
assignments and a compatible supplied OOF schedule before execution.

## Teacher fitting and preprocessing

[`fit_oof_teacher`](../src/refutab/teacher.py) accepts an audited structured-only
DataFrame, training targets, categorical names, matching ordered `train_ids`,
`TeacherConfig`, and optional validation features/IDs. There is no validation
or test target argument. Pandas feature/target indices must match the declared
IDs, and later teacher prediction requires the exact saved feature order.

The teacher constructs three shuffled folds: `KFold` for regression and
`StratifiedKFold` for classification, using the configured seed. A saved
`fold_ids` schedule may instead be supplied; it must assign each training row
exactly once to folds 0, 1, or 2 and pass fitting-size/class-coverage checks.
Each fold model fits only the other two folds. A fourth model fits all inner
training rows and supplies validation offsets. All four models have a fixed
budget; no `eval_set`, early stopping, calibration, or hyperparameter search
is used. `fit_observer` exposes the fitting/prediction row audit before each fit.

The initial configuration is CPU CatBoost **1.2.8**, 40 iterations, depth 4,
learning rate 0.1, seed 42, and two threads. These values are initial fixed
settings, not selected optima. Regression uses `RMSE`, binary classification
uses `Logloss`, and multiclass classification uses `MultiClass`.

| Input | Stateless preparation | Learned processing |
| --- | --- | --- |
| Numerical | Convert to float64; retain NaN; reject infinity and nonnumeric values | Each CatBoost model learns its own borders/trees; `nan_mode="Min"` handles missing values |
| Categorical | Type-prefixed JSON scalar tokens; dedicated `missing:` token | Each CatBoost model hashes tokens and learns its own category statistics/trees |

Numeric/string/bool categories and sentinel-looking strings remain distinct.
Unseen categorical tokens use CatBoost's native hashing; there is no globally
fitted category vocabulary or reserved unknown-category mapping in the teacher.
The planned fusion encoder's train-fitted missing/unknown vocabulary is a
separate Phase 3 component. Teacher preprocessing must never reuse it.

Class order is derived only from training labels, using sorted typed tokens,
or supplied explicitly with exactly the training classes once each. Do not
assume numerical or alphabetical order. Each model's probability columns are
aligned to this saved order. Binary probabilities have two columns, and the
positive class is saved index 1, including when an explicit order is supplied.

## Predictions and additive offsets

`TeacherArtifacts.oof_predictions` is aligned to `train_ids` and contains
exactly one excluded-row prediction per row. `validation_predictions` comes
from the separate all-inner-training model. Raw predictions and additive
offsets are distinct arrays:

| Task | Native predictions | Offset from `prediction_offsets` |
| --- | --- | --- |
| Regression | Vector in original target units | Same vector, shape `(n,)` |
| Binary | Two class probabilities, shape `(n, 2)` | `log(p) - log(1-p)` for saved class index 1 after clipping, shape `(n,)` |
| Multiclass | Saved-order class probabilities, shape `(n, c)` | Clip each probability, renormalize each row, then take its logarithm, shape `(n, c)` |

The recorded clipping interval is `[epsilon, 1-epsilon]`, with initial
`epsilon=1e-7`. Finite, normalized probability checks precede conversion.
Native CatBoost probabilities are not fitted calibration estimates. The
correction head and its losses remain Phase 3 work.

`predict(frame)` and `offsets(frame)` use the all-inner-training model for new
IDs. They reject training-ID overlap by default; `allow_in_sample=True` is an
explicit diagnostic escape hatch and must not replace saved OOF training
signals. `validate()` checks OOF coverage, audit alignment, score conventions,
and saved feature names. The models support later inference, but the Phase 2
benchmark runner deliberately produces no held-out test predictions or scores.

## Usage and private artifacts

Run from the repository root after the locked environment and Phase 1 data/source
preparation. Use a clean committed source tree and a unique run ID:

```powershell
uv sync --locked
uv run --frozen python scripts/run_teacher.py --config configs/phase2_teacher.json --run-id my-phase2-teacher --summary results/my_phase2_teacher.json
```

`--output-root` selects a persistent private artifact directory. The default is
ignored `results/raw/`. `--allow-dirty` is development-only and records the dirty
source state; it does not establish a clean reproducible source revision.
Source/metadata/CSV hashes, schema, partition fingerprints, environment, hardware
available to the runner, runtime, configuration, failures, and artifact hashes
are recorded in the manifest. Cost is recorded as unmeasured unless separately
established; the script does not infer a service charge.

An API caller passes only the explicit structured views to the teacher:

```python
from refutab.teacher import TeacherConfig, fit_oof_teacher

# table and splits are previously declared, pinned, and audited.
splits.validate(table)
teacher = fit_oof_teacher(
    table.structured(splits.train_ids), table.targets(splits.train_ids),
    categorical_columns=table.schema.categorical_columns,
    train_ids=splits.train_ids,
    validation_X=table.structured(splits.validation_ids),
    validation_ids=splits.validation_ids,
    config=TeacherConfig(task=table.schema.task),
)
teacher.save("results/raw/my-private-run/teacher")
```

The runner also persists full `splits.json` and `schema.json`. Teacher bundles
contain three `fold_*.cbm` models, `validation_teacher.cbm`, `predictions.npz`
(fold IDs, OOF/validation predictions and offsets), and `metadata.json` (class
order, full fit/prediction IDs, audits, configuration, and file checksums).
`TeacherArtifacts.load()` verifies the file inventory/checksums and artifact
invariants; model and score round trips are tested. Existing bundle directories
and run IDs cannot be overwritten.

Full row IDs, predictions, models, raw data, and source patches remain private.
Only reviewed aggregate summaries/configurations belong in Git. Preserve
datasets, models, and restricted-material exclusions in [.gitignore](../.gitignore)
when choosing an external output path. The summary is a diagnostic record of
teacher signals, not a final test result or a fusion result.

## Verification boundaries

Targeted checks currently cover schema exclusions/ordering/missing values,
IDs and split persistence, typed labels, all three real CatBoost task paths,
actual fitting-row audits, exactly-once OOF alignment, own-fold target
perturbation with a fixed schedule, held-out-label/validation-feature
perturbation, unknown categories, class order, finite offsets, deterministic
failures, and serialization/corruption detection. Exact final check outcomes
and executed smoke evidence belong in the handoff and experiment log.

The teacher does not validate an entire future end-to-end pipeline. OOF teacher
outputs do not make supervised fusion embeddings cross-fitted. The nested
pipeline diagnostic, frozen E5/PCA integration, compact structured encoder,
gate, correction head, and downstream inference are still required in their
respective phases; see [DATA_LEAKAGE.md](DATA_LEAKAGE.md) and
[ARCHITECTURE.md](ARCHITECTURE.md).
