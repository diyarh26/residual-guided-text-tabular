"""Check Phase 0 documentation, source provenance, and publication boundaries.

Standard library only; no network, data downloads, or ML execution. This is an
infrastructure check, not a claim that later algorithm/leakage tests exist.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "AGENTS.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "docs/HANDOFF.md", "docs/PROJECT_PLAN.md", "docs/COURSE_REQUIREMENTS.md",
    "docs/ARCHITECTURE.md", "docs/DATA_LEAKAGE.md", "docs/DECISIONS.md",
    "docs/EXPERIMENT_LOG.md", "docs/REPRODUCIBILITY.md", "docs/COLAB.md",
    "docs/UPSTREAM.md", "third_party/multabench.lock.json",
    "third_party/README.md", "src/refutab/README.md", "configs/README.md",
    "tests/README.md", "notebooks/README.md", "data/README.md",
    "artifacts/README.md", "results/README.md", "reports/README.md",
    ".gitignore", ".gitattributes",
)


def git(*args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, input=stdin, capture_output=True,
        text=True, encoding="utf-8", check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            errors.append(f"Missing/empty required file: {relative}")

    try:
        lock = json.loads((ROOT / "third_party/multabench.lock.json").read_text(encoding="utf-8"))
        for key in ("commit", "current_master_at_inspection", "license_added_commit"):
            if re.fullmatch(r"[0-9a-f]{40}", lock[key]) is None:
                errors.append(f"Invalid immutable SHA: {key}")
        if lock["repository"] != "https://github.com/alanarazi7/MulTaBench":
            errors.append("Unexpected upstream repository")
        if lock["branch"] != "master" or lock["reference_date"] != "2026-07-26":
            errors.append("Historical master/date requirement changed")
        selected = datetime.fromisoformat(lock["committed_at"].replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(lock["cutoff_utc"].replace("Z", "+00:00"))
        next_commit = datetime.fromisoformat(lock["next_first_parent_committed_at"].replace("Z", "+00:00"))
        if not selected <= cutoff < next_commit:
            errors.append("Historical date chronology is inconsistent")
        if lock["source_vendored"] or lock["submodule_installed"]:
            errors.append("Phase 0 must not claim vendored/submodule source")
        if not lock["license_at_selected_commit"] or not lock["historical_ref_limitation"]:
            errors.append("Missing license/history limitation")
        for relative in ("README.md", "AGENTS.md", "docs/HANDOFF.md", "docs/UPSTREAM.md", "docs/DECISIONS.md"):
            if lock["commit"] not in (ROOT / relative).read_text(encoding="utf-8"):
                errors.append(f"Source pin absent/inconsistent: {relative}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Invalid source lock: {exc}")

    try:
        # Include new publishable files before the initial commit as well as
        # tracked files. Ignored PDF and local upstream trees are never scanned.
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
        paths = [ROOT / name for name in names if name]
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() in {".pdf", ".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key"}:
                errors.append(f"Forbidden publication file: {relative}")
            if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
                errors.append(f"Potential credential file: {relative}")
            if path.stat().st_size > 1_000_000:
                errors.append(f"Review large publication file: {relative}")
            if path.suffix != ".md":
                continue
            content = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)\n]+)\)", content):
                target = target.strip().strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("#"):
                    continue
                destination = path.parent / unquote(parsed.path)
                if not destination.exists():
                    errors.append(f"Broken local link in {relative}: {target}")

        probes = (
            "course-private.pdf", "tmp/pdfs/course-extracted.txt", ".env",
            ".env.local", "credentials-local.json", "data/private.csv",
            "artifacts/model.pt", "third_party/MulTaBench/benchmark.py",
            ".venv/pyvenv.cfg", "results/raw/run.json",
        )
        # NUL delimiters avoid Git path quoting and Windows CRLF translation.
        ignored = set(git("check-ignore", "--stdin", "-z", stdin="\0".join(probes) + "\0").split("\0"))
        for probe in probes:
            if probe not in ignored:
                errors.append(f"Missing ignore protection: {probe}")
    except (OSError, ValueError) as exc:
        errors.append(f"Publication check failed: {exc}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: required files, local Markdown links, historical pin chronology, and publication exclusions")
    print("Scope: infrastructure only; no ML, baseline, or algorithm leakage tests have run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
