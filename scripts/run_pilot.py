"""Run or continue the fixed Phase 4 pilot; CPU subsets never unlock tests."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from refutab.pilot import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_pilot.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--methods", nargs="+", choices=("frozen", "direct", "residual", "tar"))
    parser.add_argument("--dataset")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", help="Explicit CPU subset or preregistered CUDA runtime")
    parser.add_argument("--allow-dirty", action="store_true", help="Development only")
    args = parser.parse_args()
    result = run_pilot(ROOT, args.config, run_id=args.run_id, output_root=args.output_root,
                        resume_run=args.resume_run, methods=args.methods, dataset=args.dataset,
                        fold=args.fold, device=args.device, allow_dirty=args.allow_dirty)
    print(f"{result['status']}: {result['completed_stages']}/{result['planned_stages']} stages; outer-test predictions: {result['test_predictions']}")


if __name__ == "__main__":
    main()
