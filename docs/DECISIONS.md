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
| D011 | Exclude photos from the project inputs. | User clarification on October 19, 2026: "we are suppposed to use the data set without the photos". The exact dataset roster is still being confirmed; this does not by itself choose between the text-tabular subset and all tables with image columns removed. |

## Questions that gate dependent work

1. **Course dataset scope:** the user confirmed photos are excluded. Confirm
   whether this means the official text-tabular subset or all benchmark tables
   with image columns removed; do not assume an exact roster from that wording.
2. **Course leakage wording:** confirm whether "never seen" means unseen rows
   with per-dataset/fold fitting or an entirely held-out dataset. Ordinary
   per-dataset supervision is proposed, not asserted as course approval.
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

User clarification during the Phase 0 follow-up resolved exclusion of photos.
Two concise follow-up questions cover the precise dataset roster and whether
ordinary per-dataset training with untouched test splits is the intended rule.
Other technical provenance checks will be handled by source inspection rather
than treated as questions the user must answer. No implementation has started.
