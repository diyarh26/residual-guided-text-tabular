# Phase 4: controlled scaling experiment design

Prepared October 21, 2026. Status: **design and non-fitting feasibility preview
only; scaling runner, nested-subset tests and GPU execution are not complete.**
The user now prioritizes developing a successful algorithm over the October 26
deadline or academic report preparation. The date and course requirements remain
historical facts, not an experiment scheduling constraint. This is a Phase 4
diagnostic follow-up, not full benchmark evaluation or an architecture change.

## Question and scope

Does increasing permitted fitting data improve residual-guided fusion relative
to direct fusion and Joint TAR under one fixed training recipe? Distinguish
absolute improvement from improvement in the paired method gap. Improving with
more data while the baselines improve faster does not rescue the method.

Use the same pinned Product Sentiment and Video Games Sales v1 tables, E5 model
revision, original no-image feature schemas and four methods. Preserve all
earlier runs and scores. Do not select one of the seven architecture proposals
as part of this experiment. Do not spend effort on an academic report now;
retain only operational documentation, provenance and results needed for valid
algorithm development.

## Why changing the old cap is insufficient

The historical sampler first subsamples the table using the requested cap,
then creates outer test and validation partitions. Calling it with a different
cap can change their row identities. The existing pilot runner therefore cannot
produce the required controlled curve merely by editing `train_examples`.
Its original 36-stage design/aggregation is also fixed. A separately named
scaling runner/protocol must accept saved fixed assignments and distinguish
training-side assessment from outer-test evaluation.

Reconstruct each historical fold-0/1/2 **at the original cap of 1,000** solely
to anchor the exact known outer-test IDs. Verify its fingerprint against the
reviewed pilot. For that replicate, reserve those IDs and restore the other
previously cap-excluded rows to the available outer training pool. Do not
regenerate the outer test at the larger cap. Exclusion is replicate-specific;
the three historical repeated holdouts overlap and are not independent units.

## Fixed held-out development partitions

Within each available training pool, reserve grouped assessment `D` once, then
validation `V` once. Use the existing explicit text groups: Product Description
for sentiment, game Name for sales. `A`, `V` and `D` must share no group.

The feasibility preview used 100 prespecified `GroupShuffleSplit` candidates:
20% assessment groups with seed `42 + outer_fold`, then 10% of remaining groups
for validation with seed `142 + outer_fold`. For classification, select the
first candidate covering all training-pool classes in A, V and D, with feasible
three-fold grouped teacher assignments in A. No model scores select candidates.
All six previews selected candidate zero at both levels. Fractions specify
groups, so exact row proportions vary. Any final implementation must reproduce
these counts or explicitly revise this design before fitting.

| Dataset | Outer replicate | Largest A | Fixed V | Fixed D | Reserved old outer test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product Sentiment | 0 | 3,298 | 366 | 918 | 509 |
| Product Sentiment | 1 | 3,297 | 367 | 918 | 509 |
| Product Sentiment | 2 | 3,298 | 366 | 918 | 509 |
| Video Games Sales | 0 | 10,756 | 1,223 | 2,960 | 1,659 |
| Video Games Sales | 1 | 10,753 | 1,227 | 2,959 | 1,659 |
| Video Games Sales | 2 | 10,736 | 1,221 | 2,982 | 1,659 |

These are **actual non-fitting preview counts**, not training results or a claim
that all full-size training will fit in GPU/host memory. 'Largest' means every
permitted A row after holding out V, D and the old outer test; it does not mean
fitting on validation/test labels. D is a development assessment set within
benchmark training, not a new blind benchmark test. It may contain rows used
in earlier training; all new method/size pipelines must exclude it themselves.
Old outer-test group overlap remains a historical row-split limitation; D is
group-disjoint from A/V regardless of that fact.

## Training sizes and paired comparison

For each dataset/replicate, use **900, 2,700 and all A rows**. Make the 900-row
set a subset of the 2,700-row set, itself a subset of largest A. Use one saved
deterministic sampling order per replicate with classification stratification
and explicit class/group feasibility checks. Do not independently resample at
each size. Partial sampling of a group inside A is permitted; A/V/D group
separation and grouped teacher folds remain mandatory. Never split a group
between teacher training and its held-out teacher fold. Missing classes or an
infeasible sample cause a visible preflight failure, not silent skipping or
score-directed resampling. The exact nested sampling algorithm still needs
implementation and testing; the largest-size preview does not validate it.

At every point fit Frozen, Direct, Residual and TAR on matched rows. PCA,
structured preprocessing, teachers, gates, TAR, and downstream learner are
rebuilt for that size/replicate; no supervised warm-start from a larger sample
or another fold. Reuse only verified frozen raw embeddings with exact model/
content/order binding. TAR learned representations stay size-specific.

The full matrix is **2 datasets x 3 replicates x 3 sizes x 4 methods = 72 fitted
pipelines**, with 18 TAR fits. Each pipeline has train/V/D scalar score sets.
The old benchmark outer-test prediction count is **zero** for this development
experiment. Reports/aggregation must require complete coverage and expose any
missing/failed points rather than compare a conveniently selected subset.

The new 900-row point is not the old pilot's 900-row point: assessment/selection
design and row identities differ. Fit it again under this single matched
scaling binding. Never splice the old small run into the new learning curve.

## Fixed recipe and compute accounting

Keep the registered model recipe and objective constant initially: E5-small-v2,
PCA-30 per field, hidden size 64/dropout 0.1, three-fold CatBoost teacher with
40 iterations, 100-estimator LightGBM, gate maximum 10 epochs/batch 32/patience
3, and original TAR maximum 50 epochs/batch 256/FP32 with non-reentrant
checkpointing. Use paired optimizer seed 42 as before; the three replicates
measure split/sample sensitivity, not a separate optimization-seed study.

Training-size growth increases optimizer steps per epoch and total compute.
This is a **fixed-recipe sample-size curve**, not a fixed-FLOP experiment or a
pure causal isolation of data from optimizer updates. Record actual epochs,
steps, early stopping, loss history, row counts, total runtime, host/GPU peaks
and artifact sizes for every point. Do not silently hold TAR to 200 steps as
its row count grows. Conversely, do not tune architectures or teacher/tree
budgets separately at each point.

If all methods underfit at larger sizes, or teacher/gate losses remain far from
converged, this experiment cannot establish architectural impossibility. A
separate matched optimization-budget study may be needed, with a fresh declared
plan and training-side selection. Checkpointed T4 execution must be validated
again for actual larger data/cache/artifact workloads. The previous synthetic
batch-memory pass does not guarantee end-to-end capacity or runtime. Do not
promise an unmeasured completion time or automatically change precision/batches.

## Reading the outcome before seeing it

Primary scores: macro OVR AUROC for sentiment and R2 for sales on D; report
train and V separately. Compare paired residual-minus-direct and residual-minus-
TAR at each size, along with their change from the new 900-row point. Plot actual
sample counts, including the slightly different 'all' counts across replicates.
Keep the two task metrics separate; means/SD across overlapping replicates are
descriptive, not significance tests.

- **Evidence for investigating data limitation:** residual improves on D as
  size grows and its gap to the relevant controls improves. Absolute gain alone
  is insufficient. State which dataset supports this explanation.
- **Reason to retain this recipe for further study on a dataset:** at the
  largest size its mean residual-minus-direct gap is positive, with at least
  two of three positive paired replicates. Surpassing TAR is a stronger,
  separately reported criterion; beating direct alone is not project success.
- **Evidence against this unchanged recipe:** it still fails that largest-size
  direct-control screen, has persistent grouped optimism, and does not close
  the TAR gap. Investigate optimization/teacher diagnostics before attributing
  the result to the entire residual-learning idea.
- **Mixed findings:** keep task-specific conclusions. A tiny positive mean or
  one favorable split is not a robust algorithmic success claim. None of these
  sign screens establishes statistical significance or all-benchmark benefit.
- **Incomplete or invalid run:** make no architecture decision from missing
  comparators, changed assessment rows, failed provenance or nonfinite scores.

Freeze the recipe and decision criteria before opening the new D score matrix.
V may select epochs according to the fixed recipe, but D must select no model
weights or hyperparameters. A later design informed by D treats it as development
data; additional untouched dataset/test evidence is needed for broader claims.

## Execution prerequisites and current next step

Prepare and test the dedicated nested split/scaling runner; preserve the old
pilot runner and report assumptions. In parallel, finish the targeted saved
prediction and teacher-fit-scope audit from the completed pilot before a long
new GPU launch. Tests must cover fixed V/D/outer exclusions across sizes,
nestedness, class ordering, train-only fitting, teacher groups, changed-held-out-
label invariance, run binding, failure reporting and resumption.

Bind all assignments, seeds, config, source/asset/lock hashes and hardware into
new persistent Drive outputs. Resume only compatible size-specific checkpoints
in preserved attempts. No automatic purchase or paid service allocation is
authorized by this plan. GPU access will be needed for TAR; no GPU job is running
from this planning work. Implementation readiness and an explicit user-facing
launch command must precede that execution. Reports for academic submission
are deferred; experiment records and checks remain necessary.
