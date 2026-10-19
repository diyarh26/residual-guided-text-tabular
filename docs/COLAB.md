# Colab execution workflow

[phase1_baseline.ipynb](../notebooks/phase1_baseline.ipynb) is a thin runner for
the original historical Joint Frozen LightGBM baseline on CPU. Its cells call
repository source; no model, embedding, splitting, or metric logic lives in the
notebook. The notebook has no saved outputs. Local notebook syntax checks do
not establish that it has been executed successfully in Google Colab; actual
execution outcomes must be recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

VS Code is the development workspace. GitHub is the code source. Colab is an
execution environment; Google Drive is the persistent output store for this
runner. A Colab filesystem or notebook output alone is insufficient persistence.

## Run the CPU baseline

1. Commit, push, and review the intended source/configuration. Copy the full
   lowercase 40-character ReFuTab commit into the notebook's `PROJECT_SHA`.
   The default placeholder raises an exception before downloads or mounting
   Drive. A branch name, short SHA, or moving `main` is not accepted.
2. Open the notebook in Colab and select **Runtime > Change runtime type > CPU**.
   The environment lock installs CPU Torch and the baseline explicitly uses CPU.
   This notebook does not exercise a GPU, Joint TAR, or ReFuTab training.
3. Run the configuration and interactive Google Drive mount cells. Use a private
   Drive folder. The notebook generates a unique run ID and immediately creates
   a persistent console log at
   `/content/drive/MyDrive/ReFuTab/runs/_bootstrap/<run-id>.log`.
   Do not embed credentials or tokens in cells, clone URLs, or outputs.
4. Run setup. The notebook installs `uv==0.11.24` using the Colab kernel's pip,
   clones the public repository into `/content/refutab`, checks out the selected
   commit in detached mode, verifies the exact HEAD, and requires a clean tree.
   If that directory already exists, setup fails without deleting it; use a
   fresh runtime or explicitly move the old checkout before retrying.
5. Install the commit's dependency lock, prepare the pinned source/data/model
   inputs, then run tests. The source commands are:

   ```bash
   uv sync --locked --python 3.11
   uv run --frozen python scripts/prepare_phase1.py
   uv run --frozen python -m unittest discover -s tests
   ```

   Preparation fetches the historical upstream SHA from the project lock, the
   configured pilot data revision, and the pinned frozen E5 snapshot. It must
   verify inputs rather than substitute latest versions. The original upstream
   checkout, data, and model cache stay outside tracked source. The notebook
   checks the working tree again before fitting.
6. Run the baseline cell. It calls this interface with the generated run ID:

   ```bash
   uv run --frozen python scripts/run_baseline.py \
     --config configs/phase1_frozen.json \
     --run-id <unique-run-id> \
     --output-root /content/drive/MyDrive/ReFuTab/runs
   ```

   The configuration fixes the dataset, training cap, fold, device, and other
   settings. Keep the upstream implementation and pinned revisions intact.
   Inspect the actual saved split sizes; a requested cap is not the number of
   learner-fitting rows. Do not tune settings after inspecting test scores.
7. Review the saved run status, metrics, class order, split provenance,
   source/upstream/data/model revisions, environment, hardware, runtime, and
   artifact paths. Read back the saved manifest and artifacts before treating
   the run as persisted. Record the real Colab execution, limitations, and
   actual runtime/cost; do not infer a successful reproduction from notebook
   creation or a passing installation cell.

The notebook streams installation, preparation, test, and baseline console
output to the persistent bootstrap log. Nonzero command exits stop execution.
Runner artifacts are created separately under
`/content/drive/MyDrive/ReFuTab/runs/<run-id>/`; keeping the bootstrap log outside
that directory lets the runner require a new, unused run directory.

## Phase 3 CPU architecture run

Use the same pinned-commit checkout, Python 3.11 locked environment and verified
asset preparation above. After mounting Drive, call the source entry point:

```bash
uv run --frozen python scripts/run_fusion.py --config configs/phase3_smoke.json \
  --run-id <unique-run-id> --output-root /content/drive/MyDrive/ReFuTab/runs
```

The new private run contains `bundle/`, `training.pt`, splits, learning curves
and manifest. For retry, use a new run ID and pass the prior persistent
`training.pt` with `--resume-from`. Preserve the original attempt. The fusion
epoch state resumes; frozen embedding batches, preprocessing and teacher fits
restart. A checkpoint is available only after a complete fusion epoch.

The current lock runs on CPU, including in a Colab runtime with a GPU attached.
Phase 3 needs only a bounded synthetic architecture check; later GPU/TAR runs
require a separately pinned and verified GPU environment. This documented
workflow has not been executed in Colab. Local smoke execution is recorded
separately in EXPERIMENT_LOG.

## Persistence and retry limitations

The runner saves a run manifest before fitting and retains failure records and
any artifacts already written. Google Drive preserves those files across
runtime resets; large input downloads and temporary caches in `/content` may
need preparation again. Keep data, embeddings, fitted models, and raw run
outputs in the private artifact store. Return only reviewed lightweight
summaries and figures to GitHub with descriptive commits.

The current LightGBM `fit` stage is not checkpoint-resumable. A disconnected
runtime may leave an incomplete manifest or only the flushed console log;
neither is evidence of completion. Retain the failed/interrupted attempt and
retry the complete baseline with a **new run ID**. Do not overwrite an old run
or describe a full restart as checkpoint resumption. Check the actual process
and saved status after a browser disconnection, since disconnection alone does
not prove the runtime stopped.

Future resumable training must verify the same source SHA, data/model revisions,
split IDs, configuration hash, feature/class order, and environment before
restoring model, optimizer, scheduler, and random state. No such optimizer
resume API is claimed by this Phase 1 notebook. GPU/TAR execution and ReFuTab
training require their own later configuration, tests, and execution record.
