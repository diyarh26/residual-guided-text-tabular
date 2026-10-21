"""Package a completed pilot's small audit inputs using only the standard library.

The private ZIP preserves paths and bytes for the existing read-only audit. It
excludes frozen caches, neural weights and Trainer checkpoints, and cannot be
used to resume or load the full pilot models. It never changes the source run.
This module can also be downloaded and run as a standalone Python script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_audit(run_directory: Path, output: Path) -> dict:
    run_directory, output = run_directory.resolve(), output.resolve()
    if output.is_relative_to(run_directory):
        raise ValueError("Write the transfer ZIP outside the original run")
    manifest = run_directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Completed pilot manifest missing: {manifest}; mount its original Drive or correct --run-directory")
    state = json.loads(manifest.read_text(encoding="utf-8"))
    if state.get("status") != "completed" or state.get("run_id") != run_directory.name:
        raise ValueError("Transfer requires the original completed run directory")
    files: dict[str, str] = {}

    def include(relative: str, expected: str | None = None):
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in relative or ":" in relative:
            raise ValueError("Audit artifact path escapes run")
        path = (run_directory / relative).resolve()
        if not path.is_relative_to(run_directory):
            raise ValueError("Audit artifact resolves outside run")
        digest = _sha(path)
        if expected is not None and expected != digest:
            raise ValueError(f"Audit artifact checksum differs: {relative}")
        files[relative] = digest

    include("manifest.json")
    for case in state["binding"]["split_fingerprints"]:
        if Path(case).name != case or "\\" in case or ":" in case:
            raise ValueError("Invalid split case name")
        include(f"splits/{case}.json")
    for job, entry in state["stages"].items():
        if entry["status"] != "completed":
            raise ValueError("Transfer requires every pilot stage completed")
        attempt = entry["attempt"]
        include(f"{attempt}/manifest.json", entry["manifest_sha256"])
        record = json.loads((run_directory / attempt / "manifest.json").read_text(encoding="utf-8"))
        pipeline = record["method"] in {"direct", "residual"}
        needed = {"bundle/downstream/metadata.json", "bundle/pca/metadata.json", "bundle/structured/metadata.json"} if pipeline else {
            "downstream/metadata.json", "representation/metadata.json" if record["method"] == "frozen" else "tar/representation/metadata.json"}
        needed.update(f"{role}_predictions.npy" for role in record["scores"])
        if record["method"] == "residual":
            needed.update(name for name in record["files_sha256"] if name.startswith("teacher/"))
        for name in sorted(needed):
            include(f"{attempt}/{name}", record["files_sha256"][name])
    for job, result in state["outer_tests"].items():
        if Path(job).name != job or "\\" in job or ":" in job:
            raise ValueError("Invalid outer evaluation job name")
        for name in ("request.json", "result.json", "test_predictions.npy"):
            include(f"outer_evaluation/{job}/{name}", result["prediction_sha256"] if name.endswith(".npy") else None)
    # Validate every requested input before creating an exclusive output archive.
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            archive.write(run_directory / relative, f"{run_directory.name}/{relative}")
        archive.writestr(f"{run_directory.name}/audit_transfer.json", json.dumps({
            "format_version": 1, "purpose": "private_read_only_audit_inputs_not_full_model_backup",
            "run_id": state["run_id"], "files_sha256": files}, indent=2) + "\n")
    return {"run_id": state["run_id"], "files": len(files), "zip_bytes": output.stat().st_size,
            "zip_sha256": _sha(output), "output": str(output), "new_fits": 0, "new_predictions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-directory", type=Path)
    mode.add_argument("--archive", type=Path, help="Restore a private audit archive instead of creating one")
    parser.add_argument("--output", type=Path, help="New ZIP file for packaging")
    parser.add_argument("--destination", type=Path, help="Runs parent directory for restoration")
    args = parser.parse_args()
    if args.archive:
        if args.destination is None or args.output is not None:
            parser.error("Restoration requires --destination, without --output")
        result = restore_audit(args.archive, args.destination)
    else:
        if args.output is None or args.destination is not None:
            parser.error("Packaging requires --output, without --destination")
        result = package_audit(args.run_directory, args.output)
    print(json.dumps(result, indent=2))


def restore_audit(archive_path: Path, destination: Path) -> dict:
    """Verify all private payload hashes, then restore into one new directory."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("Audit archive contains duplicate paths")
        indexes = [name for name in members if name.endswith("/audit_transfer.json")]
        if len(indexes) != 1:
            raise ValueError("Audit archive needs exactly one transfer index")
        metadata = json.loads(archive.read(indexes[0]))
        run_id = metadata["run_id"]
        if (not isinstance(run_id, str) or not run_id or run_id in {".", ".."}
                or any(c in run_id for c in ("/", "\\", ":"))
                or metadata["format_version"] != 1
                or metadata["purpose"] != "private_read_only_audit_inputs_not_full_model_backup"):
            raise ValueError("Invalid audit transfer identity")
        expected = {f"{run_id}/{name}" for name in metadata["files_sha256"]} | {f"{run_id}/audit_transfer.json"}
        if set(members) != expected:
            raise ValueError("Audit transfer file inventory differs")
        for name in members:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Audit archive path escapes destination")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination / run_id):
                raise ValueError("Audit archive path escapes run")
        for relative, expected_sha in metadata["files_sha256"].items():
            digest = hashlib.sha256()
            with archive.open(f"{run_id}/{relative}") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha:
                raise ValueError(f"Audit transfer payload checksum differs: {relative}")
        run = destination / run_id
        run.mkdir(parents=True, exist_ok=False)
        for name in members:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    return {"run_id": run_id, "restored_directory": str(run), "files": len(members),
            "purpose": "read_only_audit_inputs_not_full_model_backup", "new_fits": 0, "new_predictions": 0}


if __name__ == "__main__":
    main()
