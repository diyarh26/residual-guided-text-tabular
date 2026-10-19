"""Generate audited OOF teacher signals without predicting held-out test rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from refutab.teacher_runner import run_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_teacher.json"))
    parser.add_argument("--run-id", help="Unique new attempt ID; existing directories cannot be overwritten")
    parser.add_argument("--summary", type=Path, help="New reviewed aggregate summary; no raw IDs/models")
    parser.add_argument("--output-root", type=Path, help="Private persistent raw artifact directory")
    parser.add_argument("--allow-dirty", action="store_true", help="Explicit development only; record source changes")
    args = parser.parse_args()
    result = run_teacher(ROOT, args.config, run_id=args.run_id, summary_path=args.summary,
                         output_root=args.output_root, allow_dirty=args.allow_dirty)
    print(f"Completed {result['run_id']}: {result['checks']}; held-out test predictions: 0")


if __name__ == "__main__":
    main()
