"""Verify and prepare the preregistered, versioned Phase 4 pilot tables.

Downloads public Kaggle archives only; never downloads model weights, fits a
predictor, requests credentials, extracts arbitrary ZIP paths or replaces
existing bytes. Complete verified tables need no network connection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}; preserve/review before retrying")


def _validate_config(config: dict[str, Any], root: Path) -> Path:
    version = config["kaggle_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("kaggle_version must be a positive pinned integer")
    url = urllib.parse.urlsplit(config["url"])
    expected_path = "/api/v1/datasets/download/" + config["kaggle_ref"]
    if (url.scheme != "https" or url.netloc != "www.kaggle.com"
            or url.path != expected_path or url.fragment
            or urllib.parse.parse_qs(url.query) != {"datasetVersionNumber": [str(version)]}):
        raise ValueError("Expected the explicit version-pinned public Kaggle archive URL")
    files = config["files"]
    if set(files) != {"data.csv", "metadata.json"}:
        raise ValueError("Only the explicitly pinned data.csv and metadata.json members are allowed")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in [config["archive_sha256"], *files.values()]):
        raise ValueError("Every archive/member requires a lowercase SHA-256")
    destination = (root / config["directory"]).resolve()
    data_root = (root / "data").resolve()
    if destination == data_root or not destination.is_relative_to(data_root):
        raise ValueError("Dataset destination must be contained below this workspace's data directory")
    return destination


def validate_payloads(config: dict[str, Any], payloads: dict[str, bytes]) -> dict[str, Any]:
    """Validate original metadata and complete CSV schema without fitting anything."""
    for name, expected in config["files"].items():
        if hashlib.sha256(payloads[name]).hexdigest() != expected:
            raise ValueError(f"Unexpected archive member content: {name}")
    metadata = json.loads(payloads["metadata.json"].decode("utf-8"))
    schema = config["schema"]
    task = {"regression": "reg", "binary": "cls", "multiclass": "cls"}[schema["task"]]
    if "image_col" not in metadata or metadata["image_col"] is not None or schema.get("image_columns"):
        raise ValueError("Original metadata declares an image column or omits its eligibility declaration")
    expected = {
        "dataset_id": config["metadata_dataset_id"],
        "target": schema["target_column"], "task_type": task,
        "num_rows": config["audit"]["rows"],
        "num_features": len(config["csv_columns"]) - 1,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned metadata disagrees with the audited dataset schema")
    reader = csv.reader(io.StringIO(payloads["data.csv"].decode("utf-8-sig"), newline=""))
    columns = next(reader)
    if columns != config["csv_columns"] or len(columns) != len(set(columns)):
        raise ValueError("CSV columns/order differ from the audited schema")
    roles = [schema["target_column"], *schema.get("text_columns", []),
             *schema.get("numerical_columns", []), *schema.get("categorical_columns", []),
             *schema.get("excluded_columns", [])]
    if schema.get("id_column") is not None:
        roles.append(schema["id_column"])
    if len(roles) != len(set(roles)) or set(roles) != set(columns):
        raise ValueError("Each raw CSV column must have exactly one explicit role")
    if set(schema.get("forbidden_columns", [])) & set(roles[1:]):
        raise ValueError("Forbidden columns overlap a declared non-target role")
    rows = 0
    for row in reader:
        if len(row) != len(columns):
            raise ValueError("CSV contains a row with inconsistent field count")
        rows += 1
    if rows != config["audit"]["rows"]:
        raise ValueError("CSV row count differs from the pinned metadata")
    return {"dataset_id": schema["dataset_id"], "rows": rows,
            "image_col": None, "files": dict(config["files"])}


def prepare_dataset(config: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    destination = _validate_config(config, root)
    # Validate existing files before touching the network or writing anything.
    for name, expected in config["files"].items():
        target = destination / name
        if target.exists():
            _check_hash(target, expected)
    archive = destination / "archive.zip"
    if archive.exists():
        _check_hash(archive, config["archive_sha256"])
    if all((destination / name).is_file() for name in config["files"]):
        payloads = {name: (destination / name).read_bytes() for name in config["files"]}
        return validate_payloads(config, payloads)
    destination.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = destination / "archive.zip.part"
        # Stale partial bytes are evidence of a previous interrupted attempt.
        # Refuse to replace them; the caller must inspect/move them first.
        with partial.open("xb") as target:
            request = urllib.request.Request(config["url"], headers={"User-Agent": "ReFuTab research pilot"})
            with urllib.request.urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
        _check_hash(partial, config["archive_sha256"])
        if archive.exists():
            raise FileExistsError(f"Archive appeared during preparation: {archive}")
        partial.rename(archive)
    with zipfile.ZipFile(archive) as source:
        if sorted(source.namelist()) != sorted(config["files"]):
            raise ValueError("Unexpected or duplicate ZIP members; no archive content extracted")
        payloads = {name: source.read(name) for name in config["files"]}
    audit = validate_payloads(config, payloads)
    # Validate every member and the whole schema before materializing any file.
    for name, payload in payloads.items():
        target = destination / name
        if target.exists():
            _check_hash(target, config["files"][name])
        else:
            with target.open("xb") as handle:
                handle.write(payload)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase4_datasets.json")
    parser.add_argument("--dataset", action="append", help="Prepare only the named pilot; repeat to select more")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported Phase 4 asset configuration")
    datasets = config["datasets"]
    available = {entry["schema"]["dataset_id"] for entry in datasets}
    selected = set(args.dataset) if args.dataset else available
    if not selected <= available:
        raise ValueError(f"Unknown pilot dataset: {sorted(selected - available)}")
    for entry in datasets:
        if entry["schema"]["dataset_id"] in selected:
            print(json.dumps(prepare_dataset(entry), sort_keys=True), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
