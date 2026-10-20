# Colab runners

[phase4_pilot.ipynb](phase4_pilot.ipynb) is the current GPU handoff for the
registered matched pilot. [Open it in Google Colab](https://colab.research.google.com/github/diyarh26/residual-guided-text-tabular/blob/main/notebooks/phase4_pilot.ipynb),
select a GPU runtime. The reviewed/pushed `PROJECT_SHA` is prefilled; malformed
values fail before external work. Setup uses the separate Linux/Python
3.11 CUDA lock, verifies a real device and CUDA operation, prepares pinned
inputs and runs tests. The default checkpointed execution mode preserves the
original TAR budgets and checks full-size synthetic GPU capacity before fitting.
Existing sessions can use the update cell in
[PHASE4_MEMORY.md](../docs/PHASE4_MEMORY.md), preserving earlier runs.
Bootstrap logs, hardware records and raw outputs remain
in private Google Drive from the start. The launch cell ships with
`LAUNCH_PILOT = False`; enable it explicitly when ready to run the fixed plan.

The complete plan has 36 fitted stages followed by 24 main outer-test
prediction calls. Grouped nested diagnostics stay within benchmark training.
A nine-stage Product Sentiment CPU preflight can produce internal diagnostics
while leaving all main tests untouched. CPU and GPU stages cannot be mixed;
start a new full GPU run after CPU preparation. `RESUME_RUN` supports verified
continuation of the same source/config/device/environment run, preserving old
attempts. See [pilot protocol](../docs/PHASE4.md) and
[GPU execution and retry guide](../docs/PHASE4_GPU.md). Neither notebook has
been executed in Colab, and no real GPU/TAR execution is claimed.

[phase1_baseline.ipynb](phase1_baseline.ipynb) is a thin CPU runner for the
historical Joint Frozen LightGBM baseline. Open it in Google Colab, select a CPU
runtime, and fill in a reviewed, pushed 40-character ReFuTab commit. The default
placeholder stops execution. The notebook mounts Google Drive interactively,
checks out that exact source, installs the locked Python 3.11 environment,
prepares pinned inputs, runs repository tests, and calls the baseline script.
It contains no model implementation and has no stored cell outputs.

The [Colab workflow](../docs/COLAB.md) describes the exact commands, persistent
logs/artifacts, and retry limitations. This older CPU notebook does not run
Joint TAR, ReFuTab, or checkpoint continuation. Both notebooks contain source
orchestration only, with no stored cell outputs or model implementation.
