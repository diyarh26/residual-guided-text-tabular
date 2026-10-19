# Project handoff

Updated: October 19, 2026 (Asia/Jerusalem).

## Current phase and status

**Phase 0 - Completed: documentation, provenance, checks, and public repository.**
Phase 1 has not started. The full algorithm, baseline, and all experiments are
unimplemented. This follow-up records a scope clarification only.

## Latest user clarification

On October 19, 2026, the user confirmed the project uses data without photos.
Photos are excluded from project inputs. Two follow-up questions are pending:
whether the roster is the official text-tabular subset or all tables with image
columns removed, and whether ordinary per-dataset training with untouched test
splits is the intended course leakage rule. Do not infer answers from silence.

## Completed work

- Inspected the existing workspace, originally containing only the course PDF.
  Kept the folder in place and initialized `main` without nesting.
- Reviewed all eight PDF pages (text and rendered views), recorded its SHA-256,
  and extracted authoritative requirements with page/section references.
- Verified GitHub authentication as `diyarh26`, Git identity, and that the
  requested repository did not already exist before creation.
- Investigated historical MulTaBench master; pinned the July 26 commit in
  `third_party/multabench.lock.json`. Preliminary source compatibility and
  current-source/paper/license discrepancies are in `docs/UPSTREAM.md`.
- Created persistent instructions, required documentation, directory roles,
  MIT license for original work, private-file exclusions, and an infrastructure
  validator. No upstream implementation is copied into the repository.

## Verified work

PDF SHA-256: `406ca54fc55e77d54058b0b14fa32ffddf3bb24d5a41ddc69ac08880399a8811`.
The source PDF remains local and is ignored along with extraction/rendering
intermediates. Python 3.11 is available via `py -3.11`; `python` is a Windows
Store alias in the current shell.

- `py -3.11 scripts/verify_phase0.py`: PASS (required files, local Markdown
  links, source-pin chronology, and ignored publication boundaries).
- `git diff --cached --check` before the initial commit and
  `git show --format= --check HEAD` in a fresh clone: PASS.
- Reviewed staged paths/diff and searched common token/private-key patterns:
  no PDF, upstream source, credentials, datasets, weights, or large files staged.
- Fresh public clone at `e205946b9a0d7cb410449445cf0ca83ada37de6d`:
  validator and whitespace checks PASS; clean status; no course PDF required.
- Initial GitHub push succeeded; `git ls-remote` matched local HEAD; GitHub
  reports PUBLIC, nonempty repository with default branch `main`.
- Optional Actions CI was deferred because the authenticated token lacks
  workflow scope; no extra account permission is needed for these local checks.

During development, the validator caught a missing SHA mention and Windows
CRLF handling in Git stdin probes; both were corrected before publication.
Infrastructure validation does not establish ML or leakage correctness.

No baseline or method tests exist because no training code exists. No model
scores, checkpoints, failed ML experiments, or completed benchmark runs exist.

## Git and repository

- Public repository: https://github.com/diyarh26/residual-guided-text-tabular
- Branch: `main`.
- Latest verified commit at edit time:
  `33bfeb2325ee7a2b442220d03436af788760158e`
  (`docs: finalize Phase 0 verification and handoff`).
- That commit is pushed and verified against `origin/main`. This clarification
  update follows it in a separate documentation commit; compare actual HEAD
  with the remote using the commands below for the final session tip.
- To identify the commit containing this handoff: `git log -1 --format="%H %s"`.
  A handoff cannot store its own commit SHA; its field records the previously
  verified commit and final remote checks must compare actual HEAD.

## Exact next steps for Phase 1 (after Phase 0 report)

1. Read AGENTS, this handoff, PROJECT_PLAN, COURSE_REQUIREMENTS, UPSTREAM,
   ARCHITECTURE, and DATA_LEAKAGE. Review current Git status and preserve changes.
2. Confirm the precise non-image dataset roster and course training rule.
   Photos are already excluded. Validate historical multiclass metric behavior
   and source-license applicability through technical inspection; do not silently
   choose a course interpretation or ask the user to resolve engineering details.
3. Inspect a local detached copy of the pinned historical source. Map the exact
   E5/PCA/TAR/data/split/evaluation flows and identify a minimal integration seam.
4. Build and lock a compatible environment. Pin model/tokenizer/data revisions;
   record differences between source CLI settings and paper descriptions.
5. Add a small baseline/smoke configuration and thin Colab runner, run one real
   original text-tabular baseline, and persist its full provenance and metrics.
6. Record the actual command, hardware, runtime, results/failures, and smoke tests;
   update docs, commit, push, verify. Only then begin Phase 2 teacher work.

## Important paths and commands

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Mandatory future-agent rules |
| `docs/PROJECT_PLAN.md` | All eight numbered phases and acceptance criteria |
| `docs/COURSE_REQUIREMENTS.md` | Authoritative PDF extraction and open conflicts |
| `docs/UPSTREAM.md`, `third_party/multabench.lock.json` | Historical source provenance |
| `docs/ARCHITECTURE.md`, `docs/DATA_LEAKAGE.md` | Proposed method and safeguards |
| `docs/DECISIONS.md`, `docs/EXPERIMENT_LOG.md` | Decisions, open questions, honest run log |
| `docs/REPRODUCIBILITY.md`, `docs/COLAB.md` | Setup, future run manifests, persistence |
| `scripts/verify_phase0.py` | Standard-library infrastructure check |

```powershell
py -3.11 scripts/verify_phase0.py
git status --short --branch
git log -1 --format="%H %s"
git ls-remote origin refs/heads/main
```

## Decisions, blockers, and next-agent warnings

- Joint TAR is mandatory primary comparison for this supervised gate despite
  frozen E5; direct-target gating is an essential controlled ablation.
- Historical master is `599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`, not
  `paper_version` or current master. No license file exists in the selected
  tree; later MIT licensing is recorded without vendoring/relicensing it here.
- User confirmed photos are excluded. The exact dataset roster and "never seen"
  training rule remain questions for the user. Multiclass metric behavior,
  source licensing, and artifact inventory remain technical follow-up work.
- PDF reference [1] points to STRABLE's identifier; the named MulTaBench source
  is arXiv:2605.10616. July 26 is tutor/user-provided, not a PDF publication date.
- OOF teacher outputs do not make fusion embeddings OOF. Audit supervised-gate
  optimism with the planned nested diagnostic; do not mix unaligned fold spaces.
- No experiment dependencies are locked yet. Do not install latest upstream or
  run today's README against the historical source and call it reproduction.
- Preserve the original PDF locally; no redistribution permission is recorded.
  Do not commit datasets, tokens, model weights, or large embedding caches.
- User originally requested Phase 0 first; it is complete. This follow-up asks
  for clarification only. Do not treat it as an instruction to start training,
  expensive experiments, Option B, or major implementation.
