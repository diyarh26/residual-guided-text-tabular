# Planned splitting and leakage safeguards

Status: the Phase 1 original frozen baseline has validated split disjointness,
train-only PCA fitting, frozen E5 weights, and original multiclass metrics.
Teacher OOF and supervised fusion stages below remain planned and unimplemented.
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

**Training interpretation resolved by the user on October 19, 2026:** the user confirmed that training separately on each dataset's training split while keeping its test split untouched is acceptable ("no that is good to train separately"). This establishes the project's per-dataset supervised protocol; it is a user clarification, not a claim that course staff independently confirmed or rewrote PDF Section 2.2, page 3. Train-only fitting, independent inner validation, isolated dataset/outer-fold models, and all safeguards below remain required.

## Three-fold teacher protocol

Partition `A` into three folds `F_1`, `F_2`, and `F_3`. For each `k`:

1. Fit a structured-only teacher pipeline on `A minus F_k`.
2. Fit any teacher imputation, category mapping, target transform, tuning, early stopping, and probability calibration using only that allowed fold-training data and any splits nested inside it. A label in `F_k` must not influence its teacher through hyperparameter selection or calibration.
3. Predict each row of `F_k` exactly once and write its prediction, row ID, fold ID, teacher-fit row-set fingerprint, and score/class conventions into the OOF artifact.
4. Assemble predictions in original row order. Verify that every training row has exactly one prediction and belongs to none of its teacher's fitting, tuning, or calibration row sets.

Use stratification where appropriate and feasible for classification. Check class coverage explicitly; fail or document a revised split if a fold cannot support the task. A saved global class order must govern every probability column and correction head. Regression offsets must use the same target scale as the correction objective. Clip classification probabilities with a recorded epsilon before score conversion.

For validation offsets, fit a separate teacher on `A` only, with any learned choices confined to `A`, and predict `V`. Never fit that teacher on `V`. Validation labels may score and select a candidate pipeline but may not train its teacher, preprocessing, gate, or correction head. When refitting on all `B_train`, repeat the OOF construction there; do not reuse earlier teachers that violate the new fold assignments.

Teacher preprocessing is distinct from the fusion preprocessing. A PCA or structured encoder fitted on all `A` for the fusion path must not accidentally be passed into a teacher whose allowed training set is only `A minus F_k`.

## Feature and supervised-representation fitting

| Component | Candidate-pipeline fitting data | Transformation of `V` / `B_test` |
| --- | --- | --- |
| E5-small-v2 | No fitting; exact pretrained revision frozen | Saved deterministic encoding procedure |
| Per-text-column PCA to 30 dimensions | `A` text features only | Reuse the matching saved column PCA |
| Fusion numerical/categorical preprocessing | `A` structured features only | Saved imputation/scales/categories; unknown category handling |
| Three OOF teachers and teacher preprocessing | Respective `A minus F_k` only | Training OOF predictions only; separate `A`-fitted teacher for `V` |
| Fusion gate, structured encoder, correction head | `A` features, targets, and fixed OOF offsets | Frozen gate/encoder; head may score `V` for selection, then is discarded before final inference |
| Downstream learner and its preprocessing | `A` original structured columns, learned fused embeddings, original targets | Saved representation and downstream model |

The table describes inner model selection. A locked final refit uses `B_train` in place of `A`; `B_test` remains transformation-only. Do not combine supervised artifacts from different datasets, evaluation folds, or incompatible schemas. Save feature order and reject missing, duplicated, or reordered inputs rather than silently misaligning fields.

Keep labels, identifiers, target-derived columns, teacher predictions, residuals, and split metadata outside the inference feature schema. OOF offsets are permitted only in the temporary correction loss. They must not become downstream input columns. Audited schema declarations and train/validation/test row-set checks are required for arbitrary datasets.

## OOF teachers are not OOF fusion embeddings

Every row's teacher excludes that row, but the shared gate and correction head still train on that row's target. Consequently, its training embedding is a supervised in-sample representation. A downstream learner may exploit representation overfitting even when the teacher OOF procedure is correct. Do not describe these embeddings as cross-fitted or conclude that OOF teachers alone prevent this effect.

Fitting the whole supervised pipeline on `A` and evaluating on unseen `V` and untouched `B_test` can still provide a valid generalization estimate. The concern is training-representation optimism and its effect on downstream performance, not an automatic claim that all supervised representation learning leaks the outer test labels.

Before making final claims, run and record a **nested end-to-end fold diagnostic** within benchmark training data:

- For each diagnostic held-out fold, fit the complete pipeline on its complement: train-only PCA/preprocessing, an internal three-fold OOF teacher construction, fusion, and downstream learner. Use only a further inner selection split for tuning. Evaluate final predictions on the diagnostic fold, which has trained none of those components.
- Compare the full pipeline and the directly supervised gated ablation on the same folds; report final downstream train-versus-held-out behavior and the selected fitting protocol. The temporary head alone is insufficient evidence of a useful representation.
- Keep each diagnostic model's PCA/gate/downstream coordinates together. If averaging across separately fitted pipelines, combine aligned task predictions or probabilities, never arbitrary latent coordinates. Record the reduced fitting sample sizes and ensemble effects as limitations of that comparison.

If a dedicated cross-fitted-embedding experiment is needed, specify and validate its coordinate alignment and deployment mapping first. Fold-specific PCAs have sign/rotation differences, and independently trained gates can transform coordinates differently. Naively stacking their OOF embeddings and then using one full-data gate at inference does not define a consistent representation. A common reference/explicit alignment method or a different complete-pipeline design needs a documented decision and tests before use. Do not silently substitute it for Option A.

## Planned automated checks and audit artifacts

These are future acceptance criteria for Phases 2 and 3, not completed tests:

- Assert disjoint outer/inner/fold row sets, including group constraints; reject feature schemas containing the label or declared forbidden fields. Persist dataset/split fingerprints and seeds.
- Instrument teacher fitting, preprocessing, calibration, and tuning so each OOF row can be checked against all row sets influencing its predictor. Assert exactly one finite OOF prediction per row and stable class order.
- Perturb held-out labels and features and verify fitted training artifacts stay unchanged when the training configuration and selected checkpoint are fixed. Test unseen categories, missing values, and text-column ordering. Keep selection tests separate, because changing validation labels can legitimately change model selection.
- Verify E5 parameters receive no gradients and remain unchanged; verify teacher offsets are detached; verify only authorized modules update during correction training.
- Verify initial identity gating, saved-versus-loaded prediction equivalence, and a final prediction path that needs neither teacher nor temporary head.
- Record the nested diagnostic's protocol and results before interpreting final gains. Test any future cross-fitted alignment against deliberate coordinate permutations/sign changes before relying on it.

Every run must save source commit, configuration, data/version references, split IDs, seeds, component fit scopes, class mapping, metric definitions, selected checkpoint, runtime/hardware, and failures. Large/raw data and checkpoints must stay outside Git. Scores are valid for reporting only when these checks pass, dataset eligibility is verified against the pinned schema, and the historical metric convention in [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md) has been validated.
