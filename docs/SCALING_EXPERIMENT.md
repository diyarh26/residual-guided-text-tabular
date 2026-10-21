# Phase 4: controlled scaling experiment design

Prepared October 21, 2026. Status: **dedicated runner and nested-subset checks
implemented; real-data split preflight passes. GPU execution has not started.
The new standalone execution defers the prior artifact review separately.**

Latest user direction supersedes the earlier transfer prerequisite: use
[ReFuTab_Run_All.ipynb](../notebooks/ReFuTab_Run_All.ipynb) on a fresh Drive,
select GPU and Run all. No old manifest is required. The explicitly named
`phase4_scaling_standalone.json` preserves all recipe/split settings and records
`prior_audit_policy: deferred_separate_review`. This does not declare the earlier
pilot audited or complete Phase 4. Current-run data/split/teacher/fit/inventory
checks and the all-fit D barrier are unchanged. The older notebook/transfer
instructions below are retained for provenance, not required by this handoff.

The run-all notebook requests a GitHub token through hidden input, verifies a
real repository write before training, then serially updates only scalar
`summary.json` and `scores.csv` in `results/scaling_runs/<run-id>` on branch
`results/<run-id>`. Raw IDs, predictions, checkpoints, console logs and tokens
are excluded from GitHub exports. API failure retries during execution; private
exports/checkpoints stay in Drive. Final upload failures are reported rather
than claimed successful. A Drive session pointer reuses the run ID for compatible
resumption when the same pinned notebook is rerun. Full GPU execution and live
GitHub publication still require the user's runtime/authentication; local tests
do not establish those external outcomes.
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
score-directed resampling. The initial design left sampling implementation
open. The implemented algorithm
shuffles each typed class with one seeded RNG, emits one row per class, then
merges remaining queues by fractional within-class position (ties by typed
class order). Regression uses one seeded permutation. Seed is `42 + fold`.
Every requested prefix must pass class coverage and three grouped OOF folds;
there is no retry of failed sampling orders. All 18 real-data prefixes passed.

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

The dedicated nested split/scaling runner and tests are implemented; preserve
the old pilot runner and report assumptions. Finish the targeted saved
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

## Implemented execution and next action

Use the separate [scaling notebook](../notebooks/phase4_scaling.ipynb), opened
from the repository in Colab, and **File > Save a copy in Drive** before edits.
The old six-cell pilot notebook remains for the completed run; do not relaunch
it. The new notebook has its own checkout `/content/refutab-scaling`, a reviewed
immutable source pin, the same GPU lock, durable logs and a default-off
`LAUNCH_SCALING` switch. This new notebook has only been checked locally, not
executed in Colab.

1. Select GPU, run settings and mount Drive. `PRIOR_RUN` points to the completed
   checkpointed pilot in `MyDrive/ReFuTab/runs`; correct the path if it was moved.
   If using another Drive, first [transfer its audit inputs](DRIVE_AUDIT_TRANSFER.md).
2. Run source/environment setup and preparation. These verify assets, run tests,
   check synthetic TAR GPU capacity, validate all 18 nested cases, then audit
   108 saved prediction score sets, 12 teacher bundles and 96 component fit
   scopes from the prior run. No new predictor is fitted by this artifact audit.
   It does not load all neural weights or complete the broader Phase 4 audit.
   A missing/corrupt artifact visibly stops setup; preserve the error for review.
3. After setup passes, set `LAUNCH_SCALING = True` in the last cell and run it.
   This starts the 72 fits. The first D prediction waits until **all 72** fits
   finish. Outputs stay in a new Drive run; old outer-test predictions are zero.
4. Save the printed scaling directory. For interruption recovery use the same
   notebook pin/environment and set `RESUME_RUN` to that directory. Failed and
   interrupted attempts remain; completed compatible stages are verified/reused.
   A changed source/config/data/device/environment requires a separate run.

The preflight-only command from a prepared checkout is:

```bash
uv run --project environments/gpu --frozen python scripts/run_scaling.py \
  --prior-run /content/drive/MyDrive/ReFuTab/runs/phase4-checkpointed-20261020T145755Z-dd78ef55 \
  --output /content/drive/MyDrive/ReFuTab/runs/_bootstrap/scaling-preflight-UNIQUE.json
```

The launch adds `--launch --run-id <unique-id> --output-root <private-runs>`.
Resume instead adds `--launch --resume-run <existing-scaling-directory>`.
Keep `MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0`
and pass the notebook's capacity check before launch. No paid allocation is
created by these commands. GPU workload capacity/runtime remains unmeasured.

Historical split portability: six original split JSONs were read from Drive,
verified against the reviewed pilot fingerprints, and compared with the local
reconstruction. Every ordered row assignment matched; only the sampling source
byte hash differed with checkout line endings. The public
[anchor evidence](../results/phase4_anchor_rows.json) contains hashes/counts and
historical metadata, with no row IDs. Preflight compares every ordered partition
hash and the full fingerprint after substituting only those two source-byte
metadata fields; immutable upstream/version checks still run independently.
