"""Check full-size checkpointed TAR capacity, then run the registered variant."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit source before the capacity check and pilot")
    environment = dict(os.environ, MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    run_id = (args.resume_run.name if args.resume_run else
              f"phase4-checkpointed-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}")
    capacity = args.output_root / "_bootstrap" / f"{run_id}.capacity-{uuid4().hex[:8]}.json"
    print("Capacity evidence:", capacity, flush=True)
    print("Execution settings: MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8", flush=True)
    subprocess.run([sys.executable, "scripts/check_tar_memory.py", "--output", str(capacity)],
                   cwd=ROOT, env=environment, check=True)
    command = [sys.executable, "scripts/run_pilot.py", "--config", "configs/phase4_pilot_checkpointed.json"]
    if args.resume_run:
        command += ["--resume-run", str(args.resume_run)]
    else:
        command += ["--run-id", run_id, "--output-root", str(args.output_root)]
    print("Capacity check passed. Pilot directory:", args.resume_run or args.output_root / run_id, flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
