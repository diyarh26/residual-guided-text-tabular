# Splitting and leakage safeguards

Status: Phase 1 validated the original frozen baseline's split disjointness,
train-only PCA fitting, frozen E5 weights, and original multiclass metrics.
Phase 2 implements audited dataset/split contracts and a structured-only
three-fold OOF CatBoost teacher; its targeted leakage/serialization checks pass.
Phase 3 implements the supervised fusion pipeline and its synthetic boundary
tests. The nested benchmark diagnostic remains unexecuted. See
[PHASE2.md](PHASE2.md), [PHASE3.md](PHASE3.md) for implemented APIs and
[HANDOFF.md](HANDOFF.md) for final milestone verification and execution status.
The authoritative academic wording and user clarifications are recorded in
[COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md). The baseline's
[result](../results/phase1_product_sentiment_frozen.json) and independent
[split/content audit](../results/phase1_content_audit.json) document the checked scope.

## Evaluation boundary

Use the official benchmark's held-out test split and task protocol where available. Train separately for each dataset and outer evaluation fold; keep all fitted artifacts and target-trained weights isolated to that dataset/fold. Let `B_train` be its training data and `B_test` its untouched test data. Within `B_train`, create an independent inner training partition `A` and selection partition `V`. Preserve official grouping or temporal constraints rather than applying random row splitting indiscriminately. Save stable row identifiers and the exact assignments before fitting anything.

The proposed sequence is:

1. Fit all learned feature preprocessing, PCA, teachers, fusion, and downstream learners using `A`, with the OOF rules below. Use `V` for early stopping and model selection only.
2. Lock the method, hyperparameters, seeds, training budget, metrics, and planned reporting before looking at `B_test` scores.
3. If a final refit is part of the benchmark protocol, rebuild the entire pipeline on `B_train`, including new three-fold OOF teacher predictions, preprocessing, PCA, fusion, and downstream training. Use the training budget selected inside `B_train`; do not use `B_test` for checkpoint selection.
4. Run the locked evaluation on `B_test` and retain all prespecified runs, failures, and uncertainty. Do not inspect repeated test scores to select the best method or seed.

No held-out test labels or test features may be used to fit PCA, scaling, vocabulary/category mappings, calibration, encoders, teachers, gates, or downstream predictors. Frozen E5 may transform held-out text for prediction without fitting. Its external pretraining provenance must be reported; we cannot certify that public pretrained weights never encountered any benchmark text without evidence.

The Phase 2 runner loads the pinned table for content verification and original
split construction; all-label access for constructing splits is limited to
stratification. Hashing data or recording IDs does not fit a predictor. Only
declared structured training/validation views reach the teacher. It produces
zero held-out test predictions and scores. Validation labels are excluded from
teacher fitting and used only for coverage auditing and post-fit diagnostics;
they do not select teacher iterations or other settings.

**Training interpretation resolved by the user on October 19, 2026:** the user confirmed that training separately on each dataset's training split while keeping its test split untouched is acceptable ("no that is good to train separately"). This establishes the project's per-dataset supervised protocol; it is a user clarification, not a claim that course staff independently confirmed or rewrote PDF Section 2.2, page 3. Train-only fitting, independent inner validation, isolated dataset/outer-fold models, and all safeguards below remain required.

## Three-fold teacher protocol (implemented)

Partition `A` into three folds `F_1`, `F_2`, and `F_3`. For each `k`:

1. Fit a structured-only teacher pipeline on `A minus F_k`.
2. Fit any teacher imputation, category mapping, target transform, tuning, early stopping, and probability calibration using only that allowed fold-training data and any splits nested inside it. A label in `F_k` must not influence its teacher through hyperparameter selection or calibration.
3. Predict each row of `F_k` exactly once and write its prediction, row ID, fold ID, teacher-fit row-set fingerprint, and score/class conventions into the OOF artifact.
4. Assemble predictions in original row order. Verify that every training row has exactly one prediction and belongs to none of its teacher's fitting, tuning, or calibration row sets.

Use stratification where appropriate and feasible for classification. Check class coverage explicitly; fail or document a revised split if a fold cannot support the task. A saved global class order must govern every probability column and correction head. Regression offsets must use the same target scale as the correction objective. Clip classification probabilities with a recorded epsilon before score conversion.

For validation offsets, fit a separate teacher on `A` only, with any learned choices confined to `A`, and predict `V`. Never fit that teacher on `V`. Validation labels may score and select a candidate pipeline but may not train its teacher, preprocessing, gate, or correction head. When refitting on all `B_train`, repeat the OOF construction there; do not reuse earlier teachers that violate the new fold assignments.

Teacher preprocessing is distinct from the fusion preprocessing. A PCA or structured encoder fitted on all `A` for the fusion path must not accidentally be passed into a teacher whose allowed training set is only `A minus F_k`.

The Phase 2 implementation uses CatBoost 1.2.8 with a fixed iteration/depth/rate
budget, no `eval_set`, no early stopping, no search, and no calibration. Its
numeric conversion and typed categorical/missing token conversion are
stateless. Numerical borders, categorical statistics, and trees are learned
independently inside each CatBoost model's fitting partition. Missing numerics
use native `nan_mode="Min"`; unknown categories use native token hashing.
No full-training category vocabulary, median, scale, or PCA is fitted and reused
by the OOF teachers. A future fusion vocabulary with reserved missing/unknown
entries remains separate Phase 3 work.

The teacher saves class order derived solely from `A` labels, or validates an
explicit order against exactly those classes. Typed classes remain distinct;
do not assume labels are numerically or alphabetically sorted. Each class
needs at least three training rows, and every OOF fitting complement must
contain all classes. Binary offsets use clipped saved-index-1 logits;
multiclass offsets use clipped, renormalized log-probabilities, initially with
`epsilon=1e-7`. Regression stays in original target units. The score conventions,
fold IDs, and full fitting/prediction row audits accompany the saved models.

Split validation checks complete membership across train, validation, test,
and explicit excluded rows, including reduced-cap exclusions. The generic
local holdout and OOF generators do not infer groups or temporal constraints.
For tasks requiring those constraints, provide audited explicit partitions
and a compatible OOF schedule; row disjointness alone is insufficient.

## Feature and supervised-representation fitting

| Component | Candidate-pipeline fitting data | Transformation of `V` / `B_test` |
| --- | --- | --- |
| E5-small-v2 | No fitting; exact pretrained revision frozen | Saved deterministic encoding procedure |
| Per-text-column PCA to 30 dimensions | `A` text features only | Reuse the matching saved column PCA |
| Fusion numerical/categorical preprocessing | `A` structured features only | Saved imputation/scales/categories; unknown category handling |
| Three OOF teachers and teacher preprocessing (implemented) | Respective `A minus F_k` only | Training OOF predictions only; separate `A`-fitted teacher for `V`; Phase 2 runner does not predict `B_test` |
| Fusion gate, structured encoder, correction head | `A` features, targets, and fixed OOF offsets | Frozen gate/encoder; head may score `V` for selection, then is discarded before final inference |
| Downstream learner and its preprocessing | `A` original structured columns, learned fused embeddings, original targets | Saved representation and downstream model |

The table describes inner model selection for the planned full pipeline. Only
the teacher row is implemented for Phase 2; Phase 1 separately tested the
original frozen E5/PCA baseline. A locked final refit uses `B_train` in place of
`A`; `B_test` remains transformation-only. Do not combine supervised artifacts
from different datasets, evaluation folds, or incompatible schemas. Save
feature order and reject missing, duplicated, or reordered model inputs rather
than silently misaligning fields. The dataset view selects explicit named
columns in declaration order; the teacher prediction API requires that exact
saved order.

Keep labels, identifiers, target-derived columns, teacher predictions, residuals, and split metadata outside the inference feature schema. OOF offsets are permitted only in the temporary correction loss. They must not become downstream input columns. Audited schema declarations and train/validation/test row-set checks are required for arbitrary datasets.

## OOF teachers are not OOF fusion embeddings

Every row's teacher excludes that row, but the shared gate and correction head still train on that row's target. Consequently, its training embedding is a supervised in-sample representation. A downstream learner may exploit representation overfitting even when the teacher OOF procedure is correct. Do not describe these embeddings as cross-fitted or conclude that OOF teachers alone prevent this effect.

Fitting the whole supervised pipeline on `A` and evaluating on unseen `V` and untouched `B_test` can still provide a valid generalization estimate. The concern is training-representation optimism and its effect on downstream performance, not an automatic claim that all supervised representation learning leaks the outer test labels.

Before making final claims, run and record a **nested end-to-end fold diagnostic** within benchmark training data:

- For each diagnostic held-out fold, fit the complete pipeline on its complement: train-only PCA/preprocessing, an internal three-fold OOF teacher construction, fusion, and downstream learner. Use only a further inner selection split for tuning. Evaluate final predictions on the diagnostic fold, which has trained none of those components.
- Compare the full pipeline and the directly supervised gated ablation on the same folds; report final downstream train-versus-held-out behavior and the selected fitting protocol. The temporary head alone is insufficient evidence of a useful representation.
- Keep each diagnostic model's PCA/gate/downstream coordinates together. If averaging across separately fitted pipelines, combine aligned task predictions or probabilities, never arbitrary latent coordinates. Record the reduced fitting sample sizes and ensemble effects as limitations of that comparison.

If a dedicated cross-fitted-embedding experiment is needed, specify and validate its coordinate alignment and deployment mapping first. Fold-specific PCAs have sign/rotation differences, and independently trained gates can transform coordinates differently. Naively stacking their OOF embeddings and then using one full-data gate at inference does not define a consistent representation. A common reference/explicit alignment method or a different complete-pipeline design needs a documented decision and tests before use. Do not silently substitute it for Option A.

## Implemented data and teacher checks

Targeted data/split and real-CatBoost tests check:

- Declared feature exclusions, explicit roles, image scope, stable IDs/field
  order, missing values, typed fingerprints, and complete persisted partitions.
- Actual teacher fitting indices against saved audits; disjoint fit/prediction
  row sets; exactly one aligned finite OOF prediction per training row; separate
  validation-teacher fitting; and train-derived stable class probabilities.
- Own-fold target perturbations with a fixed saved schedule, which leave that
  fold's excluded-row predictions unchanged; validation-feature perturbations
  and held-out-label perturbations leave fitted training outputs unchanged.
- Missing/unseen categories, finite clipped offsets, typed class labels,
  deterministic failures before fitting, feature-order rejection, in-sample
  prediction protection, and model/score serialization with checksum failure
  detection.

These checks establish the tested data/teacher scope. They do not certify
undisclosed feature provenance, group/time constraints, or a future fusion
pipeline. Exact final check counts, full-suite results, and executed runs belong
in [HANDOFF.md](HANDOFF.md) and [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Implemented fusion checks and remaining evaluation

Phase 3 tests verify:

- Frozen E5 evaluation/no-gradient behavior and unchanged state; detached
  teacher offsets and inputs; gradients only in gate/structured encoder/head.
- Fitting-only PCA/structured statistics and vocabulary, missing/unseen category
  handling, field identities/order, exact initial identity gate and all losses.
- Regression/binary/multiclass end-to-end execution; direct control without a
  teacher; inference bundles without teacher/head weights and checked save/load.
- Changed outer-test features/labels leave fitted PCA/gate and validation
  predictions unchanged. Changed validation can legitimately select a different
  gate; fixed-epoch neural tests separate optimizer updates from selection.
- Exact same-environment CPU dropout/shuffle checkpoint continuation, with
  changed data/config/offset/context or corrupted state rejected.

Tests use tiny synthetic data and explicit text substitutes except the bounded
real-E5 smoke recorded separately. They establish software boundaries, not
benchmark generalization. The nested complete-pipeline diagnostic is specified
in [PHASE3.md](PHASE3.md) and must run in Phase 4 before final improvement
claims. Any future cross-fitted alignment still needs deliberate coordinate
permutation/sign tests; independently fitted latent vectors cannot be stacked.

Every run must save source commit, configuration, data/version references, split IDs, seeds, component fit scopes, class mapping, metric definitions, selected checkpoint, runtime/hardware, and failures. Large/raw data and checkpoints must stay outside Git. Scores are valid for reporting only when these checks pass, dataset eligibility is verified against the pinned schema, and the historical metric convention in [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md) has been validated.
