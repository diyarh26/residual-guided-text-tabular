"""Preflight only by default; explicitly launch the fixed scaling matrix."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab import pilot
from refutab.pilot_artifact_audit import audit_completed_pilot
from refutab.scaling import run_scaling
from refutab.scaling_protocol import build_scaling_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_scaling.json"))
    parser.add_argument("--prior-run", type=Path, help="Completed private pilot directory for saved-artifact audit")
    parser.add_argument("--output", type=Path, help="New JSON preflight evidence file; must not exist")
    parser.add_argument("--launch", action="store_true", help="Fit all 72 pipelines on CUDA after prior audit")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    tables, _ = pilot._load_tables(ROOT, config)
    cases = build_scaling_cases(ROOT, tables, config)
    evidence = {"status": "split_preflight_passed", "source_commit": pilot._git(ROOT, "rev-parse", "HEAD"),
                "working_tree_dirty": bool(pilot._git(ROOT, "status", "--porcelain", "--untracked-files=all")),
                "config_sha256": pilot.sha256_file(ROOT / args.config), "new_fits": 0, "new_predictions": 0,
                "planned_fits": sum(len(c["methods"]) for c in cases.values()), "outer_test_predictions": 0,
                "cases": {name: {"fit_rows": len(c["split"].train_ids), "validation_rows": len(c["split"].validation_ids),
                                  "assessment_rows": len(c["split"].test_ids), "split_fingerprint": c["split"].fingerprint()}
                          for name, c in cases.items()}}
    audit = audit_completed_pilot(ROOT, args.prior_run, config) if args.prior_run else None
    evidence["prior_artifact_audit"] = audit
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"Split preflight: {len(cases)} cases, {evidence['planned_fits']} planned fits; no new predictions", flush=True)
    if args.launch:
        result = run_scaling(ROOT, args.config, run_id=args.run_id, output_root=args.output_root,
                             resume_run=args.resume_run, prior_audit=audit)
        print(f"{result['status']}: {result['completed_stages']}/{result['planned_stages']} fits; "
              f"{result['completed_assessments']} development assessments; zero old-test predictions")
    elif args.run_id or args.output_root or args.resume_run:
        raise ValueError("Run/resume arguments require --launch")


if __name__ == "__main__":
    main()
