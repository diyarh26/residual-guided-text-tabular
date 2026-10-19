"""Run the pinned original Joint Frozen baseline; no ReFuTab training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from refutab.baseline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_frozen.json"))
    parser.add_argument("--summary", type=Path, help="Write a reviewed, lightweight summary here; must not exist.")
    parser.add_argument("--run-id", help="Optional new output directory name; existing runs are never overwritten.")
    parser.add_argument("--output-root", type=Path, help="Override raw artifact root, e.g. persistent Google Drive storage.")
    parser.add_argument("--allow-dirty", action="store_true", help="Development only: record and allow tracked source changes.")
    args = parser.parse_args()
    summary = run_baseline(ROOT, args.config, args.summary, args.run_id, args.allow_dirty, args.output_root)
    print(f"Completed {summary['run_id']}: {summary['metrics']}")


if __name__ == "__main__":
    main()
