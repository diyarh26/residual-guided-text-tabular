# Restore the prior pilot audit inputs on another Drive

Cell 5 of the scaling notebook audits the completed pilot, using `PRIOR_RUN`.
The default mounted path is:

```text
/content/drive/MyDrive/ReFuTab/runs/phase4-checkpointed-20261020T145755Z-dd78ef55/manifest.json
```

If the runtime now mounts a different account's Drive, that file and its
associated artifacts will be absent unless transferred. An empty replacement
manifest or a lone copied manifest cannot pass the audit. This error precedes
new scaling fitting; it does not indicate a failed model training stage.

## Original account: create a small private transfer

Open [the transfer notebook in Colab](https://colab.research.google.com/github/diyarh26/residual-guided-text-tabular/blob/main/notebooks/phase4_transfer_audit.ipynb)
using the account whose Drive contains the completed run. Select **CPU** and
run its two cells. No GPU, ML installation, fitting or inference is needed.
If the run was moved, correct `ORIGINAL_RUN`. The notebook downloads a private
ZIP of exactly the inputs consumed by the targeted audit: original manifest,
splits, stage/component metadata, saved predictions and teacher bundles. All
source bytes are preserved and available recorded checksums are verified.

Large neural weights, Trainer checkpoints and frozen caches are omitted.
Keep the original complete run: this ZIP is for read-only auditing and cannot
replace a full model backup or resume the old pilot. Keep the ZIP private.

## Destination account: restore and finish the existing setup

Use the already prepared scaling runtime with the destination Drive mounted.
Place the downloaded ZIP on the computer from which you can upload to that
runtime. Add and run this cell there; select the ZIP when prompted:

```python
from google.colab import files
from pathlib import Path
import subprocess
import sys
import urllib.request
from uuid import uuid4

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Select exactly the private pilot audit ZIP.")
archive = Path(next(iter(uploaded))).resolve()
helper = Path("/content/refutab_audit_transfer.py")
helper_sha = "b6a34226d88664c50db42647e312598e5bf59017"
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/diyarh26/residual-guided-text-tabular/{helper_sha}/src/refutab/audit_transfer.py",
    helper,
)
subprocess.run([
    sys.executable, str(helper), "--archive", str(archive),
    "--destination", str(RUNS_ROOT),
], check=True)
assert (PRIOR_RUN / "manifest.json").is_file(), str(PRIOR_RUN)
run(UV_PYTHON + [
    "scripts/run_scaling.py", "--prior-run", str(PRIOR_RUN),
    "--output", str(RUNS_ROOT / "_bootstrap" / ("restored-audit-" + uuid4().hex[:8] + ".json")),
], cwd=PROJECT_DIR)
```

The restore verifies every packaged payload before writing and refuses to
overwrite an existing run directory. If one already exists, preserve it and
inspect whether it is a full run or an incomplete earlier copy rather than
overwriting it. The normal scaling audit still runs and must pass; packaging
itself is not an audit success claim.

This cell reruns only the interrupted audit. Cell 5 reached the missing
manifest after its earlier preparation/tests/capacity steps; do not blindly
rerun all of cell 5 because the capacity evidence path is created exclusively.
Keep the same training source pin, `RUN_ID`, `PRIOR_RUN`, and model settings.
The transfer helper runs outside the source checkout and does not change its
clean status. If the runtime was reset, prepare a fresh setup as documented in
[the scaling workflow](SCALING_EXPERIMENT.md) first.

After the restored audit passes, enable `LAUNCH_SCALING = True` in the existing
last cell and run it. New scaling outputs remain on the destination Drive.
