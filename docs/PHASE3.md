# Phase 3 Option A implementation

The frozen text path, compact structured encoder, shared residual gate,
temporary correction losses, direct-target control and final LightGBM adapter
are implemented. Exact milestone status, tests and executed smoke evidence
belong in [HANDOFF.md](HANDOFF.md) and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
No benchmark improvement or Joint TAR result is established by this phase.

## Modules and fitting scope

- `features.py`: frozen E5, independent per-text-field PCA-30 and structured
  preprocessing. Production E5 verifies all six files of revision
  `ffb93f3bd4047442299a41ebb6fa998a38507c52`, loads locally with safetensors,
  freezes every parameter and uses evaluation/inference mode. Encoding matches
  the historical `passage: {field}: {value}` prefix, 512-token padded/truncated
  batches, masked mean pooling and L2 normalization. Missing text becomes an
  empty string before prefixing. Injected substitutes exist only for unit tests.
- `fusion.py`: trainable categorical embeddings, structured projection, shared
  identity-initialized gate, linear temporary head, losses and epoch checkpoints.
- `downstream.py`: original structured features followed by frozen fused text
  coordinates. It calls the unchanged pinned categorical helpers at fit time,
  after stateless typed token conversion, and preserves numerical NaN. Inference
  loads saved category classes and needs no upstream checkout.
- `pipeline.py`: selects fitting/validation views by stable row IDs, fits PCA
  and structured preprocessing on fitting rows only, trains/selects the gate,
  discards the head, fits the downstream learner on fitting rows and saves a
  complete prediction bundle. No outer test feature view is requested during fit.
- `fusion_runner.py` and `scripts/run_fusion.py`: configured runs, immutable
  attempt directories, failures, source/data/environment provenance and summaries.
  `scripts/predict.py` writes predictions from a saved bundle for new CSV rows.

High-level Option A requires at least one text and one structured field. The
lower-level modules also support empty numeric or categorical blocks. At least
30 fitting rows are required for PCA-30; the code never reduces dimensionality
silently. PCA is fitted in float64 with scikit-learn's seeded PCA and transformed
features become float32 for the network. Fields retain explicit schema order.

Numerical fusion inputs use training medians (zero for an all-missing column),
then the mean and population standard deviation of imputed training values;
constant columns use scale one. Category vocabularies use typed scalar tokens:
missing code 0, unseen code 1, known training categories from code 2. Every
categorical field has an 8-dimensional trainable embedding. Numeric values and
these embeddings feed a linear projection/ReLU to 16 context coordinates.
These choices are initial settings, not tuned conclusions.

For each text field, concatenate its PCA-30 vector, the structured context and
a fixed one-hot field identity. A shared 64-wide ReLU/dropout-0.1 MLP outputs
30 gates. Its final layer starts at zero, giving exactly
`z = e * (1 + tanh(G(e,s,c))) = e` initially. Concatenate the fields' fused
vectors in saved order. Scaling remains between zero and two; the gate cannot
create arbitrary new coordinates.

The temporary head is one linear layer: one scalar for regression/binary, or
one score per class for multiclass. Only the structured encoder, gate and head
are optimized. External text/features, targets and offsets are detached.
Regression uses MSE on teacher value plus correction; binary uses BCE-with-logits
on teacher logit plus correction; multiclass uses cross-entropy on teacher
log-probabilities plus correction. Class order comes only from training data.

`TrainConfig(objective="direct")` keeps the same architecture, preprocessing,
selection and downstream capacity but learns the original target without any
teacher fit or offsets. It trains its own downstream learner. Both objectives
must be compared on matched partitions and budgets in Phase 4.

Initial training settings are AdamW, learning rate 0.001, weight decay 0.0001,
batch size 32, ten epochs maximum, patience three, seed 42 and two CPU threads.
Select the minimum internal-validation correction loss. The bounded real-E5
smoke instead preregisters five epochs/batch 20. LightGBM uses the historical
default 100 trees and seed 42, with two explicit CPU threads. Its original
unused logging eval set is omitted; no downstream early stopping or second
automatic split is introduced. This adapter is an original ReFuTab integration,
not a claim that the full upstream baseline was re-executed unchanged.

## API and commands

For direct Python API use, add `src/` to `PYTHONPATH`; the scripts set their
import path automatically.

```python
from refutab.features import FrozenE5Config, FrozenE5Encoder
from refutab.pipeline import PipelineConfig, ReFuTabPipeline, fit_pipeline

encoder = FrozenE5Encoder(FrozenE5Config("artifacts/e5-small-v2"))
# table and splits are declared and audited DatasetTable/SplitAssignments.
model = fit_pipeline(table, splits, encoder, config=PipelineConfig(),
                     checkpoint_path="results/raw/new-attempt/training.pt")
model.save("results/raw/new-attempt/bundle")
loaded = ReFuTabPipeline.load("results/raw/new-attempt/bundle",
                              model_path="artifacts/e5-small-v2")
probabilities = loaded.predict(new_feature_frame)
```

Prediction inputs contain exactly the saved feature columns in schema order,
with unique stable IDs. Regression returns a vector in original target units;
classification returns one probability column per saved class. Training IDs
are rejected by default; `allow_in_sample=True` exists for diagnostics.
The API keeps targets and ID columns outside the inference feature matrix.

```powershell
py -3.11 -m uv sync --locked
py -3.11 -m uv run --frozen python scripts/prepare_phase1.py
py -3.11 -m uv run --frozen python scripts/run_fusion.py --config configs/phase3_smoke.json --run-id my-new-phase3 --summary results/my-new-phase3.json
py -3.11 -m uv run --frozen python scripts/predict.py --bundle results/raw/my-new-phase3/bundle --input results/raw/new-rows.csv --output results/raw/predictions.csv
```

The supplied configuration generates original synthetic data: 60 fitting,
12 validation and 12 reserved test rows, two text fields, two numerical fields
and one categorical field. It uses actual pinned E5 weights. The runner makes
no outer test prediction or score. Synthetic validation metrics check execution;
they do not establish benchmark usefulness. Unit/integration tests instead use
explicit deterministic text substitutes, while CatBoost, PyTorch and LightGBM
are real. No network/model download occurs inside tests.

For arbitrary CSVs, set `data.kind="csv"`, `data.path`, `data.sha256` and the
explicit `data.schema`; set `splits` to the Phase 2 explicit/local/historical
protocol. Original benchmark metadata paths/hashes can also be supplied and
must show no image column. Group/time restrictions require audited assignments
and a compatible supplied OOF schedule; the high-level runner does not infer
these restrictions. Do not use it on such a dataset without that extension.
See [PHASE2.md](PHASE2.md) for declaration requirements.

## Persistence and resumption

The inference bundle has schema, typed class mapping, PCA arrays, numerical
statistics/vocabularies, gate/structured weights, native LightGBM, pretrained
E5 reference/settings, fitting IDs and checksums. It contains no teacher model
or temporary-head weights. E5 files are referenced externally; override
`model_path` when relocating a bundle. Every component's field order, dimensions,
class order and fitting scope are cross-checked on loading. JSON/NPZ/native
model text and `torch.load(weights_only=True)` avoid arbitrary pickle loading.
Models and raw row metadata stay in ignored private storage.

The separate training checkpoint retains last-epoch model/optimizer state,
shuffle and Torch RNG, best weights/loss, patience/history, config, input
content fingerprints, context and Torch version. It is written atomically at
each completed epoch and checked for accidental corruption. Same-environment
CPU continuation with dropout/shuffling is tested against uninterrupted runs.
Resume rejects changed features, targets, offsets, validation, schema/IDs,
source/config context or budget. GPU continuation is unverified.

```powershell
py -3.11 -m uv run --frozen python scripts/run_fusion.py --config configs/phase3_smoke.json --run-id a-new-attempt --resume-from results/raw/failed-attempt/training.pt
```

The runner copies the checkpoint into the new attempt, preserving the old
attempt. Encoding, PCA/preprocessing and teacher creation are rebuilt and bound
to the checkpoint; partial E5 batches or partial LightGBM fits do not resume.
An interruption before the first complete epoch has no resumable checkpoint.
The environment lock is CPU-only. No GPU execution, Colab run or expensive
training is established by a passing local smoke.

## Nested diagnostic required before final claims

OOF teachers exclude each training row; the gate still learns that row's target.
Its resulting training embeddings are supervised in-sample representations.
Teacher OOF alone does not resolve downstream representation overfitting.

Phase 4 must lock a diagnostic entirely within benchmark training data:

1. Make three outer diagnostic folds within benchmark training. For each held
   fold D, split its complement into fitting A and independent selection V.
   D must fit or select no component, including PCA, category vocabulary,
   teacher, gate or downstream learner. Preserve any group/time restrictions.
2. Inside A, create a fresh three-fold OOF teacher, fit preprocessing/PCA only A,
   train the residual/direct control on A, select the gate on V and fit its own
   downstream learner on A. Use V labels only for selection. Do not refit on V
   implicitly; a final-refit variant needs its own declared complete protocol.
3. Evaluate final downstream predictions on D. Also record diagnostic train
   scores with an explicit in-sample flag, validation scores, temporary-head
   losses, and effective sample sizes. Compare residual and direct control on
   the same folds and budgets; retain all failures and uncertainty.
4. Keep each fold's PCA, vocabulary, gate and downstream coordinates together.
   Aggregate aligned task predictions/metrics; never stack independently fitted
   latent vectors into a common downstream training table. Report reduced
   sample sizes and any ensemble effects. The untouched benchmark test remains
   reserved until the method and reporting plan are locked.

This is a documented evaluation design, not an executed nested diagnostic.
Large pilots, Joint TAR, full evaluation and LoRA remain later phases.
