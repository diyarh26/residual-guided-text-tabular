# Decisions and unresolved questions

Updated October 19, 2026. Decisions describe the research plan, not implemented
functionality. Course requirements override preferences; conflicts remain open
until clarified. See [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md).

| ID | Decision | Reason / alternatives |
| --- | --- | --- |
| D001 | Keep the existing workspace; create public `diyarh26/residual-guided-text-tabular`. | User requested this name and portfolio visibility; no pre-existing Git repository or same-name GitHub repository was found. |
| D002 | Complete Phase 0 and report before Phase 1. | User explicitly requires phased work; infrastructure is not baseline reproduction. |
| D003 | Option A freezes E5 and trains a residual gate with OOF teacher offsets. | Test complementarity with modest compute; direct-target gated fusion isolates objective effects. LoRA is deferred to optional Phase 6. |
| D004 | Treat Joint TAR as primary baseline. | The gate uses target labels; frozen E5 alone does not make representations unsupervised (§2.1). |
| D005 | Pin historical master at `599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`. | Latest reachable master commit through July 26, 2026; `paper_version` and today's master do not satisfy the tutor's requested provenance. See UPSTREAM for cutoff/evidence limitations. |
| D006 | MIT for original work; publish no upstream implementation or course PDF in Phase 0. | Selected historical tree lacks a license file; later MIT addition is documented separately. Course redistribution is unverified. |
| D007 | Keep source modules in repo and use thin Colab orchestration with external persistent artifacts. | GitHub is code source; ephemeral Colab storage cannot hold irreplaceable outputs. |
| D008 | No fabricated ML lock, baseline command, or scores in infrastructure phase. | Phase 1 must establish a real compatible environment and run a baseline before algorithm implementation. |
| D009 | Separate OOF teacher correctness from supervised-fusion overfit. | Teacher OOF does not make gate features OOF. Compare a nested diagnostic before final claims; independently fitted latent spaces cannot simply be concatenated across rows. |
| D010 | Run local and clean-clone infrastructure checks; defer optional GitHub Actions configuration. | The authenticated OAuth token lists repo/read:org/gist scopes but no workflow scope. CI is not a Phase 0 user requirement; do not block the authorized repository setup on extra account permissions. |
| D011 | Use only benchmark datasets whose original schema has no image column. | User clarified on October 19, 2026: "the data sets that dont have images column". Datasets containing an image column are excluded. Phase 1 will enumerate and validate the eligible roster against the pinned benchmark schema. |
| D012 | Train separately per dataset and outer evaluation fold on permitted training rows, with internal validation and untouched test data. | On October 19, 2026, the user replied "no that is good to train separately" to the proposed per-dataset training/test protocol. This resolves the project interpretation of the PDF leakage wording; OOF teacher and all train-only fitting safeguards remain required. |
| D013 | Invoke unchanged historical LightGBM/split/metric implementations from an external harness. | The full CLI eagerly imports unrelated learner ecosystems and requires WandB credentials. Original source stays in an ignored checkout; no upstream implementation is redistributed or changed. |
| D014 | First baseline is CPU Joint Frozen, Product Sentiment v1, fold 0, cap 1,000. | Chosen before test scores to bound local computation; the original sampler yields 900 fit/100 validation/509 test rows. This smoke result cannot establish benchmark improvement or replace full evaluation. |
| D015 | Lock a minimal compatible CPU dependency closure, including Joblib 1.5.2. | Core versions match the historical requirements. New Joblib 1.6.0 broke old Skrub imports during setup; the upstream-compatible pin resolves that observed failure. |

## Resolved requirements and technical follow-up

1. **Dataset scope resolved:** the user confirmed datasets must have no image
   column in their original schema. Enumerating the eligible datasets is Phase 1
   verification work, not an outstanding question for the user.
2. **Training rule resolved:** the user confirmed ordinary separate training
   on each dataset's training partition, with internal validation and an untouched
   test split. Retain dataset/fold isolation and all documented leakage safeguards.
3. **Classification metrics:** course requires AUROC; historical implementation
   and multiclass conventions must be mapped before evaluation. Preserve
   official metrics and separately report course metrics after fixing averaging
   and class ordering; never relabel accuracy/log-loss as AUROC.
4. **Source license:** establish applicability of later MIT terms to historical
   files before copying/redistributing them; retain immutable source pin.
5. **Weights:** confirm whether existing unchanged public E5 weights require
   re-upload under §2.3; list and publish original trained artifacts as required.
6. **Metadata typo:** PDF reference [1] uses arXiv:2605.12292, which is STRABLE;
   the named MulTaBench paper and user's URL are arXiv:2605.10616. Record the
   mismatch explicitly; use the verified named source for research provenance.

User clarification during the Phase 0 follow-up resolved both dataset eligibility
and separate per-dataset training. No further user answer is needed on these
points; do not reopen them without new conflicting evidence.
Other technical provenance checks will be handled by source inspection rather
than treated as questions the user must answer. No implementation has started.
