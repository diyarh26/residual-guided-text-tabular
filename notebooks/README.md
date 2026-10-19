# Colab runners

[phase1_baseline.ipynb](phase1_baseline.ipynb) is a thin CPU runner for the
historical Joint Frozen LightGBM baseline. Open it in Google Colab, select a CPU
runtime, and fill in a reviewed, pushed 40-character ReFuTab commit. The default
placeholder stops execution. The notebook mounts Google Drive interactively,
checks out that exact source, installs the locked Python 3.11 environment,
prepares pinned inputs, runs repository tests, and calls the baseline script.
It contains no model implementation and has no stored cell outputs.

The [Colab workflow](../docs/COLAB.md) describes the exact commands, persistent
logs/artifacts, and retry limitations. This notebook's existence does not mean
a Colab execution has been validated. Joint TAR, GPU training, ReFuTab training,
and checkpoint resumption remain outside this notebook's scope.
