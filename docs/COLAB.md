# Colab execution workflow

Status: documented workflow only. A thin runnable notebook, tested dependency
lock, experiment command, and resume API will be added in Phase 1 and subsequent
implementation phases. Phase 0 requires no GPU or training job.

VS Code is the development workspace. GitHub is the code source. Colab is an
execution environment, not the canonical source or sole artifact store.

## Before every later Colab experiment

1. Commit and push the intended source/configuration; choose the full 40-character
   project commit, not a moving branch. Obtain the upstream SHA from its lock.
2. Select hardware appropriate to the locked baseline/experiment. Record runtime
   type, accelerator, memory, Python, CUDA, and dependency-lock fingerprint.
3. Authenticate/mount Google Drive interactively or use another approved
   persistent artifact store. Keep credentials out of code, notebook outputs,
   logs, and Git. Do not put access tokens into clone URLs.
4. Clone ReFuTab into Colab's temporary filesystem and detach at the recorded
   project SHA. Verify `git rev-parse HEAD` and a clean working tree. A bootstrap
   cell can use the following commands with a real recorded SHA:

   ```bash
   git clone https://github.com/diyarh26/residual-guided-text-tabular.git /content/refutab
   git -C /content/refutab checkout --detach <FULL_PROJECT_SHA>
   git -C /content/refutab rev-parse HEAD
   ```

5. Install from the environment lock belonging to that commit. Resolve upstream
   using the pinned historical source and verify its SHA. Current `master` and
   the latest published package are not substitutes. Run the commit's smoke test
   before a long run.
6. Give the run a unique ID; persist its manifest before fitting. The source
   runner must take a configuration and explicit persistent output directory.
   Do not start an experiment until its executable command exists and is tested.

## Persistence and resumption

Use a private persistent folder such as
`/content/drive/MyDrive/ReFuTab/runs/<run-id>/` for configs, logs, incremental
metrics, split provenance, checkpoints, and final models. Never place private
data in a shared/public folder by default. Stage large temporary files locally
if useful, but periodically flush complete checkpoints and manifests to Drive.
Verify that a saved artifact can be read back; mounting Drive alone is not a
backup policy. Use atomic checkpoint replacement where supported and retain a
previous valid checkpoint when transferring to remote storage.

Record a run as interrupted if the runtime disconnects. Resume only when its
source SHA, data/model revisions, split IDs, config hash, feature/class order,
and environment match. Restore optimizer/scheduler/RNG state; do not silently
restart under the same run ID with different settings. If the chosen learner
does not support resumption, report that limitation and rerun the incomplete
stage with an explicit new attempt ID.

Return reviewed lightweight summaries and figures to GitHub with a descriptive
commit; keep raw datasets, embedding caches, and checkpoints in their artifact
store. Phase 1 must test these steps on one small baseline before scaling.
