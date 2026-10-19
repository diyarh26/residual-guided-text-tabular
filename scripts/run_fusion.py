"""Train Option A or the direct-target control without scoring the outer test."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from refutab.fusion_runner import run_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_smoke.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-from", type=Path, help="Matching prior fusion checkpoint; copied to the new attempt")
    parser.add_argument("--allow-dirty", action="store_true", help="Development only")
    args = parser.parse_args()
    result = run_fusion(ROOT, args.config, run_id=args.run_id, summary_path=args.summary,
                        output_root=args.output_root, resume_from=args.resume_from, allow_dirty=args.allow_dirty)
    print(f"Completed {result['run_id']}; outer test predictions/scores: 0/0")


if __name__ == "__main__":
    main()
