"""Fetch immutable upstream, pilot data, and frozen E5 assets; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}; preserve/review the file before retrying")


def prepare_upstream() -> None:
    lock = json.loads((ROOT / "third_party/multabench.lock.json").read_text(encoding="utf-8"))
    destination = ROOT / "third_party/MulTaBench"
    if not destination.exists():
        subprocess.run(["git", "clone", lock["repository"], str(destination)], check=True)
    if not (destination / ".git").is_dir():
        raise ValueError(f"Expected a Git checkout at {destination}")
    if git("-C", str(destination), "status", "--porcelain"):
        raise ValueError("Upstream checkout has local changes; refusing to overwrite")
    remote = git("-C", str(destination), "remote", "get-url", "origin").removesuffix(".git")
    if remote != lock["repository"]:
        raise ValueError("Unexpected upstream remote")
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", lock["commit"]], check=True)
    if git("-C", str(destination), "rev-parse", "HEAD") != lock["commit"]:
        raise ValueError("Upstream SHA mismatch")
    print(f"Verified upstream {lock['commit']}", flush=True)


def prepare_dataset(config: dict) -> None:
    destination = ROOT / config["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    if all((destination / name).is_file() for name in config["files"]):
        for name, expected in config["files"].items():
            check_hash(destination / name, expected)
        print("Verified existing pilot data and metadata", flush=True)
        return
    archive = destination / "archive.zip"
    if not archive.exists():
        partial = destination / "archive.zip.part"
        request = urllib.request.Request(config["url"], headers={"User-Agent": "ReFuTab research baseline"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        check_hash(partial, config["archive_sha256"])
        partial.replace(archive)
    check_hash(archive, config["archive_sha256"])
    with zipfile.ZipFile(archive) as source:
        for name, expected in config["files"].items():
            # Read only the two known members; never extract arbitrary paths.
            payload = source.read(name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Unexpected archive member content: {name}")
            target = destination / name
            if target.exists():
                check_hash(target, expected)
            else:
                target.write_bytes(payload)
    print("Verified Kaggle pilot version 1", flush=True)


def prepare_model(config: dict) -> None:
    from huggingface_hub import snapshot_download

    destination = ROOT / config["directory"]
    for name in config["files"]:
        if (destination / name).exists():
            check_hash(destination / name, config["sha256"][name])
    if not all((destination / name).is_file() for name in config["files"]):
        snapshot_download(
            repo_id=config["identifier"], revision=config["revision"],
            local_dir=destination, allow_patterns=config["files"], max_workers=2,
        )
    for name in config["files"]:
        check_hash(destination / name, config["sha256"][name])
    files = {name: sha256(destination / name) for name in config["files"]}
    provenance = {"identifier": config["identifier"], "revision": config["revision"], "files": files}
    record = destination / "provenance.json"
    if record.exists() and json.loads(record.read_text(encoding="utf-8")) != provenance:
        raise ValueError("Existing model provenance differs; inspect before replacing")
    record.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Verified frozen E5 snapshot {config['revision']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true", help="Prepare only source and pilot data")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/phase1_assets.json").read_text(encoding="utf-8"))
    prepare_upstream()
    prepare_dataset(config["dataset"])
    if not args.skip_model:
        prepare_model(config["model"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
