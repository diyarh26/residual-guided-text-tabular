# Seven research directions after the ReFuTab pilot

Prepared October 21, 2026 for discussion on October 22. **Research proposals,
not implemented methods or new experimental results.** The current Option A,
its negative/mixed findings, original source pin and evaluation records remain
unchanged. No GPU training or paid API work was launched for this research.

My recommendation: discuss **conditional innovation subspaces**, **tree-readable
residual codes**, and **metadata-query token pooling** first. They attack three
different possible bottlenecks: compression, downstream compatibility, and
information extraction. The other four are distinct alternatives, not extra
modules to combine into one oversized system. None has established top-tier
novelty or a performance advantage. All seven use targets during representation
learning, so **Joint TAR remains their required primary comparator**.

## 1. What the assignment and our evidence actually support

I rechecked the local authoritative eight-page course PDF against its existing
text extraction and visually inspected all eight rendered pages. Its SHA-256 matches
the previously reviewed file: `406ca54fc55e77d54058b0b14fa32ffddf3bb24d5a41ddc69ac08880399a8811`.
The PDF remains private/ignored; [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md)
records the requirements without republishing it.

For Track 1, targets used in text representations require comparison against
Joint TAR. Creativity and deep analysis are valued even without a decisive win;
simply changing to a bigger encoder is insufficient. General-purpose code,
pinned dependencies, training/inference modes, model artifacts and an English
NeurIPS-format paper of at most four pages are deliverables. The deadline is
October 26; no submission clock time is established. The user's resolved scope
remains original schemas without image columns and separate training per
dataset/outer fold. The Track 2 standard/outstanding dataset guidance does not
redefine our Track 1 project.

The [completed GPU pilot](PHASE4_RESULTS.md) provides these observations:

| Observation | What it suggests investigating | What it does not establish |
| --- | --- | --- |
| Residual mean AUROC 0.847469 vs TAR 0.875176 | Whether the representation loses task information before fusion | That residual learning is impossible |
| Residual mean R2 0.149319 vs TAR 0.227602 | Regression supervision, compression and downstream alignment | That any particular replacement will work |
| Residual beats direct on only 1/3 outer holdouts per task | Variance, teacher dependence and excessive flexibility | A statistically significant difference |
| Grouped residual-minus-direct means -0.017958 AUROC and -0.064301 R2 | Generalization beyond related entities and training optimism | Proof of leakage or a single identified cause |
| Sentiment diagnostic training AUROC is 1.0 | Honest held-out pipeline evaluation is essential | That every fitted component is defective |

Full independent saved-model/teacher/prediction audit is still outstanding.
The registered decision remains analysis before scaling. The completed scores
are exploratory evidence used to generate these hypotheses, not a fresh
selection set for trying seven methods until one looks good.

## 2. Three structural lessons worth testing

**The current gate cannot recover a text distinction already erased by its
input.** If two texts have the same 30-dimensional PCA vector and the same
structured inputs, every deterministic gate receiving only those inputs must
produce the same output. Multiplicative reweighting can reshape retained
information; it cannot distinguish this collision. This is a simple limitation
of our architecture, not a newly discovered theorem or evidence that such
collisions caused the measured deficit. MulTaBench also reports TAR gains in a
no-PCA ablation, so removing PCA alone is not a supported universal solution.
[MulTaBench, Appendix F.5](https://arxiv.org/html/2605.10616v1)

**A useful correction head need not produce useful features for LightGBM.**
The temporary head and final learner optimize different models. A small head
may exploit an oblique combination that needs many axis-aligned tree splits.
This is a proposed mechanism, not something our current audit has measured.

**A large teacher residual need not be learnable from text.** It can arise from
missing structured interactions, noise, sparse groups or teacher error.
OOF construction removes a row from its teacher fit; it does not classify the
cause of the residual or make the target-trained representation OOF.

## 3. Shortlist and practical ranking

Ranks are qualitative judgments, not measured success probabilities. CPU means
small pilot training from cached frozen embeddings, not a guarantee about full
benchmark runtime. Any token extraction would need a bounded memory check.

| ID | Proposal | Change from current pipeline | Main appeal | Closest-work risk | Pilot compute / build risk | Discussion priority |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | Conditional innovation subspaces | Learn a few conditional supervised directions before compression | Direct, inexpensive test of lost complementary signal | High: supervised PCA, PLS and orthogonalization exist | CPU after E5 caching / low-medium | First practical probe |
| I2 | Matched-context difference learning | Train on residual differences between structurally similar rows | Learn distinctions inside a context rather than global class clusters | High: conditional contrastive learning exists | CPU cached embeddings / medium | Strong conceptual candidate |
| I3 | Metadata-query token pooling | Structured context chooses token evidence before pooling/PCA | Recover information unavailable to a pooled-vector gate | High: contextual attention and pooling exist | T4 extraction; small pooler / medium-high | Largest architectural upside |
| I4 | Multi-teacher residual agreement | Train on cross-fitted residuals weighted by teacher-family agreement | Separate stable correction targets from teacher-specific errors | High: ensembles and reweighting exist | CPU cached embeddings; extra teacher fits / low-medium | Cheapest modification of Option A |
| I5 | Tree-readable residual codes | Replace a free latent gate with bounded soft split features | Match representation geometry to the final tree learner | High: differentiable trees and distillation exist | CPU cached embeddings / medium | Best downstream-alignment candidate |
| I6 | Learn features through a held-out solver | Optimize gate through a differentiable fitted learner | Reward transfer across groups instead of in-sample head fit | High: bilevel feature learning exists | CPU small solves / medium-high | Scientifically attractive, more engineering |
| I7 | Interaction features with conditional negative controls | Keep sparse text-structure interactions that survive null comparisons | Test conditional contribution before granting feature capacity | High: conditional importance and stability selection exist | CPU cached embeddings / medium | Strong interpretability/analysis angle |

No row in this table is a verified globally novel algorithm. The potential
contribution is the precise combination, objective, evidence and controlled
analysis described below. Search-result absence is not proof of originality.

## I1. Conditional innovation subspaces

October 22 discussion update: after the partial scaling validation review,
I recommend this as the next small mechanism probe, not a proven replacement.
More data improved residual AUROC without producing a mean advantage over
Direct, motivating a different representational mechanism. Retain the ordinary
supervised projection control: a gain over unsupervised PCA alone would not
establish the conditional contribution. A full-384 control tests practical
representation capacity but cannot by itself prove that PCA caused the gap.
The benchmark's no-PCA ablation still reports TAR gains, so removing PCA is
not a general solution. No new method fitting or GPU execution occurred during
this discussion. Primary sources rechecked October 22:
[supervised PCA](https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf),
[cross-fitting in DML](https://arxiv.org/abs/1608.00060),
[MulTaBench F.5](https://arxiv.org/html/2605.10616v1).

**Plain-language idea:** give the model a few text directions that explain what
the structured columns cannot, instead of allocating every dimension to the
largest text-variance directions.

Let `h` be the full frozen 384-dimensional embedding and `s` the structured row.
Within training data, estimate `m(s) = E[h | s]` using regularized multi-output
regression. For each training row use a nuisance model fitted without its fold
to form `u_i = h_i - m_minus_fold(s_i)`. Form response residuals from the OOF
structured teacher: `r_i = y_i - t_minus_fold(s_i)` for regression, or
`r_i = one_hot(y_i) - p_minus_fold(s_i)` for classification.

Learn a heavily regularized predictive subspace from the cross-covariance of
`u` and `r`, with bootstrap/group stability checks. A concrete candidate uses
left singular vectors of `(Cov(u) + lambda I)^(-1/2) Cov(u,r)` and maps them
back to embedding coordinates. All centering, covariance and shrinkage choices
are training-only. Keep `30-k` normal PCs plus `k` innovation directions, with
`k` at most five, in a **fixed 30-feature budget**.
For scalar regression, this simple cross-covariance has rank at most one; the
first honest version therefore uses **29 PCs plus one** supervised direction, not five
imaginary independent ones. Multiclass directions also have limited rank.
Higher-order residual bases would be a separate, preregistered extension.

Use a single declared deployment map for the representation. One option is an
average of the training-fold nuisance predictors for both downstream training
and inference; its in-sample contribution must be diagnosed, because it differs
from the OOF nuisance values used to estimate the subspace. An alternative
keeps one honest nuisance-training partition and applies that fixed map to all
representation/downstream fitting rows. Do not silently refit the nuisance map
and claim the representations remain cross-fitted. No supervised coordinate
stacking across incompatible PCA/gate spaces is permitted.

**Why it might work here:** it tests a clear bottleneck with far fewer learned
parameters than a free gate. **Why it might fail:** residualizing text can erase
useful redundant context or pure interactions; the conditional mean estimate
can be inaccurate with 900 fit rows. Orthogonalization alone does not deliver
conditional independence, causal identification or DML inference guarantees.

**Closest work / possible contribution:** supervised PCA already selects
outcome-related directions; DML already uses cross-fitting and nuisance
residualization. The candidate contribution is a fixed-budget conditional
innovation channel for text-tabular fusion with explicit deployment-map and
grouped-transfer audits. It needs comparison to ordinary supervised PCA/PLS,
not merely to unsupervised PCA.
[Supervised PCA](https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf),
[DML](https://arxiv.org/abs/1608.00060)

**First falsifiable experiment:** on training-side grouped folds only, compare
PCA-30, ordinary supervised-direction rescue, and conditional rescue, all at
equal dimensions and identical downstream settings. Also check full frozen
384-dimensional features as a diagnostic, not an equal-dimension main result.
Reject the conditional mechanism if it adds nothing beyond supervised rescue
or is unstable under group resampling. **GPU:** none for the subspace probe once
the full frozen embeddings have been extracted.

## I2. Matched-context difference learning

**Plain-language idea:** compare two rows with similar structured information
and ask what their texts explain about their different outcomes.

Build a neighbor graph using training-only structured distances and explicit
unknown/missing handling. Pair distinct entities rather than near-duplicate
texts. Regression can use a loss such as
`Huber((a(z_i)-a(z_j)) - (r_i-r_j))` for OOF teacher residuals `r`, plus a small
pointwise anchor loss to identify location/scale. Classification can fit vector
differences in OOF probability residuals with a bounded squared/Brier-style
pair loss plus pointwise classification supervision. No arbitrary numeric
ordering of nominal classes is introduced. An alternative contrastive loss
would be a distinct ablation, not silently mixed into the first version.

Only real observed pairs supply supervision: **do not swap a text onto another
row and keep its label as if that were a valid synthetic example**. Neighbor
matching uses features, never held-out labels. At inference the shared frozen
representation transforms one row; it does not need another row's label or a
retrieval database.

**Why it might work here:** it encourages distinctions within a structured
context and reduces the benefit of memorizing global product/game categories.
**Why it might fail:** sparse neighborhoods, poor mixed-type distance, noisy
pair differences and group confounding can dominate. Similar structured rows
are not randomized counterfactuals; no causal-effect claim is justified.

**Closest work / possible contribution:** SupCon and Conditional Contrastive
Networks already use supervised or structured notions of similarity. Our
proposed distinction is a teacher-adjusted **outcome-difference objective** on
matched, distinct-entity rows, rather than learning attribute similarity alone.
That distinction must survive an exact method comparison before a novelty claim.
[SupCon](https://arxiv.org/abs/2004.11362),
[Conditional Contrastive Networks](https://table-representation-learning.github.io/assets/papers/conditional_contrastive_networ.pdf)

**First falsifiable experiment:** compare ordinary pointwise residual fitting,
random pairs, matched pairs without teacher adjustment, and matched residual
pairs at equal capacity. Require improvement on unseen training-side groups,
and report pair coverage by class/group. Stop if matching offers no advantage
over random pairing or only helps duplicate entities. **GPU:** none for a small
cached-vector projection/gate pilot.

## I3. Metadata-query token pooling

**Plain-language idea:** let the structured row ask the text encoder which
words matter, before the sentence becomes one vector.

Keep E5 frozen and retain its token states `H = [h_1,...,h_L]`. A tiny structured
encoder produces one or two queries `q_k(s)`. For each text field, compute
`alpha_k = softmax(q_k(s)^T W_k H / sqrt(d))` with correct padding masks, then
`v_k = sum_l alpha_kl h_l`. A small bottleneck combines these readouts with a
preserved mean-pooling channel and produces a fixed-size feature block. Train
the pooler on permitted labels with the same teacher-offset correction task;
compare a direct-supervision version as well. The teacher and temporary head
can still be discarded at inference; the pooler, structured encoder and frozen
E5 remain.

**Why it might work here:** a post-PCA gate never sees token-specific evidence.
A contextual pooler could read a phrase differently depending on metadata.
**Why it might fail:** frozen token states may still omit the needed semantics;
attention can memorize and its weights are not faithful explanations. Context
may add little when a dataset has only a coarse structured feature.

**Closest work / possible contribution:** TabSTAR learns text-tabular
interactions; TabAttention already conditions attention on tabular inputs in
an imaging setting; TILE already studies row context and column pooling.
Neither generic cross-attention nor contextualizing text is new. The candidate
is a small **residual-supervised, metadata-query readout of frozen token states**
with a fixed feature budget and controlled unconditional-pooling ablations.
[TabSTAR](https://arxiv.org/html/2505.18125v2),
[TabAttention](https://arxiv.org/abs/2310.18129),
[TILE](https://arxiv.org/html/2610.17065v1)

**First falsifiable experiment:** matched mean pooling, learned global-query
pooling, structured-query pooling, and structured-query pooling with shuffled
metadata as a diagnostic. Hold output dimension, encoder and training budget
fixed. Require a real advantage over global queries on unseen groups; otherwise
the metadata-conditioning story fails. An attention visualization alone does
not count. **GPU:** a T4 would be useful for new token extraction. Budget storage
first: 1,000 x 512 x 384 FP32 token states take about 0.73 GiB per field before
metadata; actual padded lengths vary. Stream/cache privately and bound batches.

## I4. Multi-teacher residual agreement

**Plain-language idea:** learn a correction when several different structured
models agree about what they are missing; downweight errors peculiar to one
teacher.

Construct matched OOF predictions from a small, fixed family of structured
teachers, such as CatBoost, LightGBM and regularized linear models. Class order,
row order and prediction scale must agree. Let `r_i^(m)` denote aligned outcome
or probability residuals. Estimate a consensus `r_bar_i` and disagreement
`v_i = mean_m ||r_i^(m)-r_bar_i||^2`, then use a bounded weight such as
`c_i = ||r_bar_i||^2 / (||r_bar_i||^2 + v_i + epsilon)` in the correction loss.
The ensemble supplies the fixed training offset; the weight is training-only.
Regression residual scales need training-only normalization. No per-test-row
reliability weight may be inferred from its unknown label.

This first version measures **teacher-family disagreement**, not a proven
probability that an error is learnable. Agreement can occur on pure noise or
shared bias. In particular, identical teacher predictions imply identical
residuals even for a corrupted label. Any claim of noise filtering therefore
requires a separate noise-control experiment.

**Why it might work here:** it could reduce dependence on one weak or
misspecified structured teacher. **Why it might fail:** related trees may make
correlated errors, and the weighting can suppress hard informative examples.

**Closest work / possible contribution:** ensemble uncertainty, multimodal
boosting and example reweighting are established. The candidate is using
heterogeneous OOF residual disagreement to decide the training strength of a
teacher-free text representation. It is a plausible small extension, with a
weaker novelty case than I2/I3 unless the mechanism is convincingly isolated.
[Deep ensembles](https://arxiv.org/abs/1612.01474),
[ReconBoost](https://arxiv.org/html/2405.09321v1),
[Learning to Reweight](https://arxiv.org/abs/1803.09050)

**First falsifiable experiment:** compare one teacher, an unweighted ensemble
offset, and the same ensemble with agreement weights. Equalize teacher fits
across the last two conditions. If only the ensemble offset helps, the new
weighting claim fails. Use synthetic teacher misspecification and label noise
separately. **GPU:** none beyond existing frozen embedding extraction.

## I5. Tree-readable residual codes

**Plain-language idea:** teach the text representation to express a handful of
useful split-like facts that a tree can read directly.

Replace unrestricted embedding reweighting with a small bank of smooth,
contextual split features, for example
`c_k(h,s) = sigmoid((a_k^T h + b_k^T phi(s) - tau_k) / T)`.
Train a bounded additive temporary head `t_OOF(s) + sum_k beta_k c_k(h,s)`.
Reserve an unchanged frozen-text channel. Then discard the temporary head and
teacher and fit the same LightGBM on structured columns plus the retained
frozen/code features. Use the same total feature count in the main comparisons.
Do not call these codes calibrated probabilities or linguistic concepts.

Constrain split temperature and code redundancy using training-side choices;
otherwise codes can saturate, clone each other or memorize rows. A simple
initial bank could use 25 ordinary PCs plus five codes, with another 30-feature
control using learned linear directions. These are proposed starting budgets,
not empirically selected settings.

**Why it might work here:** the representation can expose decision boundaries
as individual coordinates rather than hiding them in a linear-head combination.
**Why it might fail:** LightGBM might learn equivalent boundaries from a simple
projection, and soft splits can still overfit. The head-discard mismatch is
reduced in intent, not mathematically eliminated.

**Closest work / possible contribution:** NODE already learns differentiable
tree ensembles; DeepGBM already connects trees and neural representation
learning. Our candidate is a **small residual-trained code interface for frozen
text and structured context**, followed by the original learner. A result must
beat a direct-trained code bank and a parameter-matched projection before
arguing that residual codes contribute anything specific.
[NODE](https://arxiv.org/abs/1909.06312),
[DeepGBM](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf)

**First falsifiable experiment:** compare current gated vectors, rotated/
linear-projected frozen vectors, direct codes and residual codes. Keep feature
count and final learner fixed; report held-out performance, code saturation
and split usage. Check one additional learner as a diagnostic to distinguish
tree-interface benefit from general representation benefit. **GPU:** none for
a small bank learned from cached pooled embeddings.

## I6. Learn features through a held-out solver

**Plain-language idea:** update a representation only when a model trained on
one group of rows benefits on different groups.

Create support/query episodes *inside the representation-training partition*.
Fit an inner ridge predictor on support representations and teacher residuals,
`w*(theta) = argmin_w ||Z_support(theta) w - R_support||^2 + lambda ||w||^2`.
Backpropagate query prediction loss through this differentiable solve to update
the gate/projection. For classification use aligned one-hot/probability
residuals in the small ridge surrogate; a direct-supervision control is needed.
Keep structured context in the feature map or as a declared ridge input.
After meta-training, freeze one shared representation and train the standard
downstream learner on its permitted fitting rows.

**Partition distinction:** query labels influence the meta-gradient, so those
query rows are **training data**, not honest diagnostic D or selection V.
Reserve a separate V for selection and a further grouped assessment partition
for evaluating the complete pipeline. Rotating support/query roles does not
turn the shared gate into an OOF representation. Reuse no outer test labels.

**Why it might work here:** the objective penalizes representations whose
apparent benefit only exists where the head was fitted. **Why it might fail:**
the ridge surrogate can disagree with LightGBM, small query groups make noisy
meta-gradients, and extra partitioning leaves less data.

**Closest work / possible contribution:** differentiable closed-form solvers
and validation-directed meta-gradients already exist. The proposed variant
uses a conditional residual task and **within-dataset group transfer episodes**
to train a reusable text-tabular feature map. There is no new general bilevel
optimization principle here.
[Differentiable solvers](https://arxiv.org/abs/1805.08136),
[Learning to Reweight](https://arxiv.org/abs/1803.09050)

**First falsifiable experiment:** equal-capacity pointwise training, random-row
episodes and group-disjoint episodes. Evaluate a complete pipeline on a third,
untouched training-side group partition. If improvement stops at the ridge
surrogate and does not transfer to LightGBM, reject this version. **GPU:** not
required for small cached-vector episodes; engineering and validation cost are
higher than I1/I4/I5.

## I7. Interaction features with conditional negative controls

**Plain-language idea:** a text-structure interaction earns feature capacity
only if it is more useful than plausible, conditionally shuffled alternatives.

Construct a small low-rank interaction bank
`psi_k(h,s) = (a_k^T h) * (b_k^T phi(s))`, alongside unmodified structured and
frozen text features. Generate negative-control text embeddings by shuffling
whole text rows within training-only structured neighborhoods/subgroups. Keep
all fields of a text row together when evaluating multi-field contribution.
Score interaction utility on an internal selection partition, compare it with
the corresponding negative-control utility, and retain only a small stable
bank under group resampling. Fit a fresh downstream learner after selection.
Use an untouched training-side assessment set to measure the selected system.

Approximate neighborhood shuffling is not exact sampling from `p(text | s)`;
it can produce implausible combinations or break dependence. Therefore call
this a **negative-control screen**, not a knockoff-certified procedure, causal
test, FDR guarantee or proof of unique information. Include negative-control
fidelity and subgroup-size diagnostics. Reusing its selection partition for
the final performance claim would invalidate the screen's evaluation.

**Why it might work here:** it limits free interaction capacity and asks whether
the text adds information within a context rather than merely repeating it.
**Why it might fail:** conditional samples are unreliable at this pilot size,
many interactions can overfit the screen, and nuisance grouping may remove
genuine signal. Stable selection can still select a stable confounder.

**Closest work / possible contribution:** conditional predictive impact,
knockoffs and subgroup permutation methods already evaluate conditional
importance. The proposed contribution is their cautious use as a training-side
admission rule for a budgeted text-tabular interaction representation. A valid
statistical guarantee would require stronger assumptions and additional theory.
[Conditional importance](https://arxiv.org/abs/2210.03047),
[Conditional subgroup permutation](https://arxiv.org/abs/2006.04628)

**First falsifiable experiment:** compare the same bank with no selection,
ordinary marginal selection and conditional negative-control selection. Require
synthetic recovery when only interactions carry signal, rejection when text
only duplicates structured information, and honest grouped assessment after
selection. **GPU:** none for a cached-vector prototype.

## 4. Prior-work collisions that change the pitch

These are source-grounded boundaries, not reasons to abandon research:

- [ReconBoost](https://arxiv.org/html/2405.09321v1) already formulates multimodal
  learning using alternating, boosting-related corrections. We cannot pitch
  'learn one modality from another modality's errors' as new by itself.
- [MARS](https://arxiv.org/abs/2606.30355) uses missingness-induced representation
  residuals to guide experts and distills routing into an inference-available
  router. Its setting differs from our label residuals, but residual-guided
  specialization or a privileged training router is not an empty literature.
- [TabSTAR](https://arxiv.org/html/2505.18125v2) and
  [CARTE](https://proceedings.mlr.press/v235/kim24d.html) already address contextual
  representations of table content. 'Make text tabular-aware' needs a much
  narrower algorithmic claim.
- [TILE](https://arxiv.org/html/2610.17065v1), available in October 2026, directly
  studies the representation interface: row context, intermediate states,
  column pooling and bypass features. Generic row serialization or smarter
  pooling is therefore a weak novelty claim for I3.
- [Conditional Contrastive Networks](https://table-representation-learning.github.io/assets/papers/conditional_contrastive_networ.pdf)
  already use structured notions of similarity. I2 must distinguish outcome
  differences and teacher adjustment, not merely add a contrastive objective.
- [Multimodal AutoML](https://arxiv.org/abs/2111.02705) already evaluates stacked
  multimodal/tree predictions. An ensemble of existing conditions is a useful
  sanity baseline, not automatically a new representation-learning algorithm.

The literature check used primary papers, author-hosted PDFs and proceedings.
Some sources were checked at abstract level and others at relevant method/
ablation sections; this is targeted research, not a systematic review or a
complete code reproduction. Two guessed bibliographic IDs resolved to unrelated
papers and were excluded after title verification. Unverified secondary
summaries and public hypothesis pages are not evidence that a method works.

## 5. Stronger novelty and feasibility tests for tomorrow

Before choosing a method, state one sentence that existing methods do not
already claim. For I1: 'Reserve a fixed representation budget for conditional
innovation and demonstrate that it survives honest grouped transfer.' For I3:
'Read frozen token evidence with metadata queries trained on complementary
prediction, rather than adapting encoder weights.' For I5: 'Train a compact
text-structure correction interface whose coordinates are directly usable by
the final tree learner.' These are **candidate claims to challenge**, not
accepted novelty statements.

Five checks are more valuable than an impressive name:

1. Can it solve a controlled synthetic problem that the current gate cannot?
2. Does its defining component beat the closest simpler ablation?
3. Does the gain survive complete-pipeline group-disjoint assessment?
4. Can it beat TAR under a fair resource/feature-budget comparison?
5. Can we implement, verify, generalize and explain it before the deadline?

Useful synthetic cases, all newly generated and separate from benchmark tests:
a low-variance text signal removed by PCA; a pure text-structure interaction;
text that only repeats structured features; independent label noise; entity
shortcuts that break across groups; and an oblique boundary versus split-coded
features. Synthetic success is a mechanism check, never benchmark evidence.

## 6. Evaluation rules for any selected follow-up

The two pilot datasets have already influenced our thinking. Their old test
scores are not clean confirmatory evidence for a redesign. Do not repeatedly
evaluate seven methods on those outer tests to pick a winner. Development and
selection belong inside permitted training data; any new comparison on these
two tasks must be disclosed as exploratory.

After selecting at most one main candidate and a small fixed ablation set,
freeze the method and protocol before evaluating additional eligible datasets
whose test outcomes have not informed the design. Fit each dataset/fold only
on its allowed training partition, as the user authorized. Audit dataset/model
pretraining overlap where relevant. Keep all label-trained artifacts isolated.
This adds untouched evidence without silently changing the pinned benchmark.

Retain TAR as the primary baseline for all seven ideas, retain direct
supervision as the mechanism control, and preserve original structured
features. Match feature count where relevant; separately report extra compute,
teacher fits, encoder accesses and inference dependencies. Do not reuse final
test labels for PCA, residual subspaces, pair construction, teacher selection,
feature admission, meta-gradient training, calibration or stopping decisions.
Report AUROC and R2 separately and do not convert three overlapping holdouts
into independent-sample significance evidence.

Each selected method changes the registered architecture and needs an explicit
new config/run, a documented decision and validation before fitting. The user's
research mission authorizes investigation and proposals; it does not select
one of these seven as the new project method. The current method and its
negative findings remain valid deliverable material.

## 7. A realistic discussion for October 22

Suggested order:

1. Revisit the actual pilot table and outstanding artifact audit.
2. Decide whether preserving a rigorous negative-result project is preferable
   to a bounded extension; the course values motivated analysis.
3. Compare I1, I5 and I3 for information gained per implementation hour. I1 is
   the cheapest structural probe; I5 targets the learner interface; I3 makes
   the clearest architectural departure and needs new token extraction.
4. If one direction is selected, specify its single decisive training-side
   experiment and stop condition before writing implementation code.
5. Protect time for the general-purpose inference/training deliverables,
   artifact publication review and four-page report. Do not attempt all seven.

My provisional preference is **I1 for a quick mechanism diagnosis, I5 for a
feasible method extension, and I3 for a larger research bet**. I2 is a close
alternative if we prioritize a pairwise complementarity story. I4 is the most
incremental rescue; I6/I7 offer strong methodological questions but more
partitioning/estimation risk at the current sample size. No option is promised
to beat TAR, and none has yet been tested.
