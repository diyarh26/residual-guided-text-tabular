# Phase 4: memory-saving TAR execution

The October 20 Colab run completed the frozen, direct and residual stages
according to its control flow, then TAR exhausted a GPU reporting 14.56 GiB.
The first forward pass requested another 768 MiB with 551.81 MiB free. The
[failure record](../results/phase4_colab_tar_oom.json) preserves the reported
numbers; full private artifacts still require readback. The four-worker warning
was not the terminating error. No outer-test evaluation had been unlocked.

The user confirmed that only the current GPU is available. TPU v5e has 16 GB
of memory per chip according to [Google](https://docs.cloud.google.com/tpu/docs/v5e?hl=en),
and this project's CUDA-specific environment does not support it directly.

## Execution-only amendment

[phase4_pilot_checkpointed.json](../configs/phase4_pilot_checkpointed.json)
keeps the original datasets, methods, splits, objectives, precision and budgets.
TAR still uses FP32, batch 256, maximum 512 tokens, rank 16, last three layers,
50 epochs maximum and the same learning rate, decay, validation and patience.
The original [configuration](../configs/phase4_pilot.json) remains unchanged.

The adapter explicitly enables non-reentrant activation checkpointing on BERT
blocks containing trainable LoRA parameters. Frozen preceding blocks keep their
normal forward execution. Intermediate values are recomputed during backward
instead of all being retained. Dropout RNG is preserved; trainable parameter
scope is checked. This follows [PyTorch checkpointing](https://docs.pytorch.org/docs/2.7/checkpoint.html)
and the [pinned Transformers API](https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable).
It trades additional computation for memory, without reducing the batch or
using mixed precision. Upstream source files remain unmodified.

Native CPU BERT/PEFT tests with dropout produced exactly matching loss and all
trainable gradients, with fewer saved intermediate tensors. These establish a
software check, not GPU numerical equivalence or a capacity guarantee.
The execution flag is bound into TAR checkpoint context and its audit. A
checkpoint from the other execution mode is rejected.

## Check capacity before the new comparison

Use `scripts/run_memory_pilot.py --output-root <private-output-directory>` from
the reviewed committed source. It supplies both plotting/cuBLAS environment
settings before starting Python children. First it invokes
`scripts/check_tar_memory.py` with real verified E5 weights, a synthetic
256-by-512 token batch, FP32, LoRA on the last three layers and the frozen E5
encoder also resident on the GPU. A twenty-output head covers the pilot's TAR
regression bin count. It performs forward/backward only: zero benchmark rows,
zero optimizer updates, no test labels. GPU peaks and failures persist in a
new private `_bootstrap/*.capacity-*.json` file.

If this capacity check fails, execution stops before pilot fitting. It never
automatically changes batch size, sequence length, precision or other budgets.
A pass is evidence for this workload, not a guarantee that every training stage
will finish. The October 20 supplied log now verifies a Tesla T4 pass at
6,989,661,184 bytes (6.510 GiB) peak allocated, followed by TAR training through
epoch 9 / optimizer step 36. End-to-end completion remains pending. See the
[progress record](../results/phase4_colab_progress_epoch9.json).

The amended config/source gets a **fresh run ID**. Preserve the original run
and all three completed conditions; do not merge their learned models or
overwrite their metrics. The short non-TAR stages repeat under the new single
source/config binding before all 36-stage/24-test requirements can complete.
The original exploratory visibility and descriptive decision rule remain in
force. New results have not driven hyperparameter selection.

For later continuation of this amended run, use
`scripts/run_memory_pilot.py --output-root <private-output-directory> --resume-run <new-run-directory>`
from the same source and environment. Its capacity evidence receives a new
name; the pilot retains prior attempts and validates all completed artifacts.

## Current Colab session

Keep the existing GPU runtime and verified local assets. Add one new cell at
the bottom of the current notebook and run this reviewed update:

```python
MEMORY_FIX_SHA = "6e09955514caffa52050459377e5b4736dedca75"
run(["git", "fetch", "origin", MEMORY_FIX_SHA], cwd=PROJECT_DIR)
run(["git", "checkout", "--detach", MEMORY_FIX_SHA], cwd=PROJECT_DIR)
run(UV_PYTHON + ["scripts/run_memory_pilot.py", "--output-root", str(RUNS_ROOT)], cwd=PROJECT_DIR)
```

Use this new cell rather than the original launch cell. It preserves verified
data/model files in the existing checkout and old private outputs, tests capacity,
and starts a separate checkpointed pilot only if that check passes. Existing
model arrays are not merged into it. Source/dataset integrity checks still run.
The wrapper sets plotting/CUDA workspace variables for its children even when
the existing notebook has the earlier helper. The user has executed this cell;
capacity passed and TAR is running. Leave that job running without launching
another copy. These commands remain instructions for future sessions.

For a fresh runtime, the latest
[Colab notebook](https://colab.research.google.com/github/diyarh26/residual-guided-text-tabular/blob/main/notebooks/phase4_pilot.ipynb)
prefills this immutable source and the amended config, with the same capacity
check included in preparation. It retains the explicit launch switch.
