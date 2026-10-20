# Phase 4 GPU execution

Status: the separate dependency lock and thin Colab entry point are prepared.
They have not been installed or executed in Colab, and no GPU training result
is claimed. Local tests inspect the lock, notebook guards, CPU rejection and
durable failure logging without using a GPU.

Phase 4 compares Joint Frozen, Joint TAR, direct-target gated fusion and
residual-guided fusion on the committed pilot protocol. Joint TAR is the
target-aware primary baseline. Pilot results and a recommendation must be
reported before full evaluation; notebook preparation does not complete Phase 4.
The [registered protocol](PHASE4.md) requires 36 completed fitting stages before
24 main outer-test prediction/evaluation calls. It fixes two datasets, three
overlapping outer holdouts and grouped nested diagnostics inside outer fold-0
training. Prior Phase 1 test-score visibility makes this an exploratory pilot.

## Required resources and dependency isolation

Use a Linux x86_64 runtime, Python 3.11 and one working NVIDIA CUDA GPU. A
16 GB T4-class GPU is a practical first device to try; an A100 with more memory
provides more room. Have at least 16 GB host RAM available. These are resource
planning targets, not measured capacity requirements or a guarantee that the
historical FP32 TAR batch of 256 at 512 tokens fits. The actual GPU, driver, available
device/host memory, runtime and cost must be recorded. If the fixed settings
exhaust memory, preserve the failure and discuss any changed protocol before
running a revised comparison. No automatic batch-size reduction is applied.

The root `pyproject.toml` and `uv.lock` retain the verified CPU environment.
[environments/gpu/pyproject.toml](../environments/gpu/pyproject.toml) and its
[uv.lock](../environments/gpu/uv.lock) separately pin Torch `2.7.1+cu126` and
torchvision `0.22.1+cu126`, with all common dependency versions equal to the CPU
lock except these two build suffixes. NVIDIA runtime libraries and Triton are
transitively locked. The GPU project supports only Linux x86_64/Python 3.11.
Do not install this environment into the CPU environment or use a mutable
`pip install --upgrade torch` replacement.

The official [PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v271)
specify the 2.7.1/0.22.1 CUDA 12.6 pairing. The official
[Torch index](https://download.pytorch.org/whl/cu126/torch/) and
[torchvision index](https://download.pytorch.org/whl/cu126/torchvision/) contain
the locked Python 3.11 Linux wheels. The notebook tests their indexed hashes
and versions; actual installation, driver compatibility and training capacity
remain to be checked on the allocated GPU. `uv lock` resolution succeeded
locally without installing GPU wheels.
The Linux `uv sync --locked --dry-run` check selected 68 installable packages;
it did not install them or validate CUDA at runtime.

## Colab steps

1. [Open phase4_pilot.ipynb in Colab](https://colab.research.google.com/github/diyarh26/residual-guided-text-tabular/blob/main/notebooks/phase4_pilot.ipynb), choose a
   GPU runtime. `PROJECT_SHA` is already pinned to the reviewed, pushed pilot
   implementation `c1e7ec1cdbb79fbc2f000a060616a5e73e11c0d1`.
   Keep this pin for the registered run; malformed values fail before external work.
2. Run setup cells in order. Mount your private Drive interactively. Before
   installation or preparation, the notebook creates a unique bootstrap log
   under `MyDrive/ReFuTab/runs/_bootstrap/`. Console output and exit codes flush
   to that log while commands execute. Credentials do not belong in the notebook.
3. Setup installs `uv==0.11.24`, clones the public repository into a new
   `/content/refutab`, checks out the exact commit in detached mode and verifies
   a clean tree. An existing checkout causes failure without deleting it.
4. The separate lock installs into `environments/gpu/.venv`. Hardware preflight
   runs `nvidia-smi`, checks exact Torch builds and CUDA availability, exercises
   CUDA matrix multiplication, and writes a private hardware JSON beside the
   bootstrap log. CPU fallback is disabled. The original TAR environment
   variable selects GPU 0 only after `nvidia-smi` succeeds; CUDA availability
   and an actual operation are verified before preparation.
5. Preparation verifies pinned upstream/data/model assets, then prepares the
   additional pilot inputs. Tests and a final clean-tree check must pass before
   fitting. Source commands, run from the repository root, are:

   ```bash
   uv sync --project environments/gpu --locked --python 3.11
   uv run --project environments/gpu --frozen python scripts/prepare_phase1.py
   uv run --project environments/gpu --frozen python scripts/prepare_phase4.py
   uv run --project environments/gpu --frozen python -m unittest discover -s tests -v
   ```

6. Review the fixed pilot configuration, actual GPU memory and service cost.
   The launch cell ships with `LAUNCH_PILOT = False`, so running all cells stops
   before training. Set that variable to `True` in the launch cell when ready.
   Do not rerun the settings cell solely to launch; it generates a new run ID.
   The launch command is:

   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run --project environments/gpu --frozen python scripts/run_pilot.py \
     --config configs/phase4_pilot.json --run-id <unique-run-id> \
     --output-root /content/drive/MyDrive/ReFuTab/runs
   ```

7. Read back the run manifest and each method's artifacts/status. Record actual
   hardware, runtime, memory, cost, curves, failures and metrics. Keep full data,
   row IDs, models, predictions and caches private in Drive. Only reviewed
   lightweight summaries should return to GitHub.

The notebook contains orchestration only; model, split, metric and audit logic
remain in source modules. The local CPU environment remains suitable for tests
and bounded frozen-encoder checks. Joint TAR requires the real GPU preflight.
In a standalone shell, `CUDA_VISIBLE_DEVICES=0` must be set before Python starts;
the notebook sets it after `nvidia-smi` succeeds and verifies actual CUDA work.
Neither setting the variable nor passing preflight establishes TAR fit capacity.

The optional nine-stage Product Sentiment CPU preflight has a different device,
package-build and embedding binding. It never unlocks the registered outer tests.
Start a **new complete GPU run** for the matched comparison rather than passing
that CPU directory to `--resume-run`. Completed CPU stages are not merged with
GPU stages or used to choose which registered methods to report.

## Interruption and retry boundaries

Drive retains flushed logs, manifests and artifacts already written. `/content`
is temporary: after a reset, bootstrap the exact source/environment again and
verify assets before restoring any permitted state. A browser disconnect does
not establish whether the runtime stopped; inspect actual saved method status.

The pilot adapter adds audited persistence around the unchanged TAR Trainer.
TAR can resume from a completed epoch checkpoint with validated model,
optimizer/scheduler, Trainer/random state, callback state and provenance. Last
and selected-best checkpoints are retained and copied into a new stage attempt;
only the required saved path references are rewritten. A completed selected
representation can be reused without training when its artifact checks pass.
Actual GPU checkpoint continuation remains unverified. Without a valid saved
checkpoint, TAR restarts; a partial LightGBM fit always restarts. Preserve the
old attempt and distinguish a restart from continuation.

ReFuTab fusion has epoch-boundary checkpoints, with source/config, data/split,
feature/class order, optimizer and random-state checks. PCA/structured state
and teachers are reconstructed with the same fitting scope. Verified frozen
text packs can be reused within their exact model/settings/runtime binding;
fitted preprocessing and supervised encoders are never shared across folds.
Continue the same pilot ledger using:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project environments/gpu --frozen python scripts/run_pilot.py \
  --config configs/phase4_pilot.json \
  --resume-run /content/drive/MyDrive/ReFuTab/runs/<prior-run-id>
```

Omit `--run-id` and `--output-root` when continuing. In the notebook, set
`RESUME_RUN` in the settings cell to that existing Drive directory before setup;
the new bootstrap log remains separate. The CLI requires the same source and
configuration, data/splits, device and environment; it validates completed stages
before reusing them and creates a new attempt child for interrupted work.
Validated fusion/TAR checkpoints can continue in that new attempt; old attempts
remain intact. Do not manually copy partial or incompatible fitted state into a
new method result. No completed, failed or interrupted attempt may be
overwritten or described as completed
without artifact validation.

Service cost is initially recorded as unmeasured. No paid GPU allocation,
purchase, account action or external message was performed during preparation.
