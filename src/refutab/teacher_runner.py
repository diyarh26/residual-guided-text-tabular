"""Reproducible Phase 2 orchestration; the outer test is reserved, never scored."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

from .baseline import _PeakRSS, audit_splits, fingerprint
from .data import DatasetSchema, DatasetTable, sha256_file
from .splits import SplitAssignments, generate_splits
from .teacher import TeacherArtifacts, TeacherConfig, fit_oof_teacher, row_ids_sha256


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def _json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def build_assignments(root: Path, table: DatasetTable, settings: dict) -> SplitAssignments:
    """Import exact historical assignments, load explicit IDs, or declare a local split."""
    protocol = settings["protocol"]
    if protocol == "explicit":
        result = SplitAssignments.load(root / settings["path"])
    elif protocol == "local":
        result = generate_splits(table, seed=settings["seed"],
                                 validation_fraction=settings["validation_fraction"],
                                 test_fraction=settings["test_fraction"])
    elif protocol == "historical_phase1":
        config = json.loads((root / settings["baseline_config"]).read_text(encoding="utf-8"))
        reference = json.loads((root / settings["baseline_reference"]).read_text(encoding="utf-8"))
        upstream = root / config["upstream_path"]
        lock = json.loads((root / "third_party/multabench.lock.json").read_text(encoding="utf-8"))
        if _git(upstream, "rev-parse", "HEAD") != lock["commit"] or config["upstream_commit"] != lock["commit"]:
            raise ValueError("Historical split source differs from the immutable source pin")
        if _git(upstream, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("Historical split source must remain unchanged")
        if importlib.metadata.version("tabstar") != "1.1.15":
            raise ValueError("Historical split protocol requires TabSTAR 1.1.15")
        if table.source_sha256 != config["dataset_sha256"] or table.schema.dataset_id != config["dataset_id"]:
            raise ValueError("Historical split configuration does not match this dataset")
        sys.path.insert(0, str(upstream))
        from tabstar.preprocessing.splits import split_to_test, split_to_val
        from multabench.baselines.preprocessing import sampling
        if Path(sampling.__file__).resolve() != (upstream / "multabench/baselines/preprocessing/sampling.py").resolve():
            raise ValueError("Historical sampling module was not imported from the pinned checkout")
        subsample_dataset = sampling.subsample_dataset
        is_cls = table.schema.task != "regression"
        # The only all-label operation is the original split stratification.
        x, y = subsample_dataset(table.features(), table.targets(), is_cls,
                                  config["train_examples"], config["fold"])
        outer, test, outer_y, _ = split_to_test(x, y, is_cls, config["fold"], config["train_examples"])
        fit, validation, _, _ = split_to_val(outer, outer_y, is_cls)
        ids = {"fit": fit.index.tolist(), "validation": validation.index.tolist(), "test": test.index.tolist()}
        if audit_splits(ids, x.index.tolist()) != reference["splits"]:
            raise ValueError("Reconstructed historical assignments differ from Phase 1")
        if fingerprint(x.index.tolist()) != reference["data"]["sampled_ordered_ids_sha256"]:
            raise ValueError("Historical subsample order differs from Phase 1")
        selected = set(x.index)
        result = SplitAssignments(
            train_ids=tuple(ids["fit"]), validation_ids=tuple(ids["validation"]), test_ids=tuple(ids["test"]),
            excluded_ids=tuple(row for row in table.row_ids if row not in selected), seed=config["seed"],
            metadata={"protocol": "historical_phase1", "upstream_commit": lock["commit"],
                      "tabstar": "1.1.15", "outer_fold": config["fold"],
                      "outer_seed": config["seed"] + config["fold"], "validation_seed": 41,
                      "source_config_sha256": sha256_file(root / settings["baseline_config"]),
                      "reference_sha256": sha256_file(root / settings["baseline_reference"])},
            dataset_fingerprint=table.fingerprint(),
        )
    else:
        raise ValueError(f"Unknown explicit split protocol: {protocol!r}")
    result.validate(table)
    return result


def _encoded_targets(target: pd.Series, classes: tuple) -> np.ndarray:
    def key(value):
        if isinstance(value, np.generic):
            value = value.item()
        return type(value).__name__, json.dumps(value, ensure_ascii=False, allow_nan=False)
    mapping = {key(value): index for index, value in enumerate(classes)}
    return np.asarray([mapping[key(value)] for value in target.tolist()], dtype=np.int64)


def diagnostic_metrics(target: pd.Series, predictions: np.ndarray, artifacts: TeacherArtifacts) -> dict:
    """OOF/validation diagnostics only; never fit or select a model here."""
    if artifacts.config.task == "regression":
        y = np.asarray(target, dtype=float)
        return {"r2": float(r2_score(y, predictions)) if len(y) > 1 else None,
                "rmse": float(np.sqrt(mean_squared_error(y, predictions)))}
    y = _encoded_targets(target, artifacts.class_order)
    count = len(artifacts.class_order)
    present = np.unique(y)
    result = {"logloss": float(log_loss(y, predictions, labels=np.arange(count))),
              "present_class_indices": present.tolist(), "all_classes_present": len(present) == count}
    if len(present) < 2:
        result.update({"auc": None, "auc_convention": "undefined: fewer than two observed classes"})
    elif count == 2:
        result.update({"auc": float(roc_auc_score(y, predictions[:, 1])), "auc_convention": "binary: saved class index 1"})
    else:
        values = [roc_auc_score((y == value).astype(int), predictions[:, value]) for value in present]
        result.update({"auc": float(np.mean(values)), "auc_convention": "macro OVR: all classes" if len(present) == count else "macro OVR: present classes only"})
    return result


def run_teacher(root: Path, config_path: Path, *, run_id: str | None = None,
                summary_path: Path | None = None, output_root: Path | None = None,
                allow_dirty: bool = False) -> dict:
    root = root.resolve()
    config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported teacher runner schema_version")
    run_id = run_id or f"phase2-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one new directory name")
    directory = (root / (output_root or config["output_root"]) / run_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    summary_path = root / summary_path if summary_path is not None else None
    start = time.perf_counter()
    manifest = {"format_version": 1, "run_id": run_id, "status": "started",
                "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "config_sha256": sha256_file(config_path), "command": sys.argv,
                "cost": {"provider_service_charge_usd": None, "basis": "Not measured; no automatic billing estimate"},
                "artifacts": {"directory": directory.relative_to(root).as_posix() if directory.is_relative_to(root) else str(directory)}}
    _json(directory / "manifest.json", manifest)
    try:
        if summary_path is not None and summary_path.exists():
            raise FileExistsError("Refusing to overwrite an existing summary")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
        manifest["source"] = {"commit": _git(root, "rev-parse", "HEAD"), "working_tree_dirty": bool(dirty), "allow_dirty": allow_dirty}
        if dirty and not allow_dirty:
            raise ValueError("Commit publishable source changes before a recorded run")
        if dirty:
            (directory / "source.patch").write_text(_git(root, "diff", "HEAD", "--binary"), encoding="utf-8")
        manifest["hardware"] = {"device": "cpu", "gpu_used": False,
                                "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
                                "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total}
        manifest["environment"] = {"python": sys.version, "platform": platform.platform(),
                                   "packages": {name: importlib.metadata.version(name) for name in ("catboost", "numpy", "pandas", "scikit-learn", "tabstar")}}
        manifest["environment_lock_sha256"] = {name: sha256_file(root / name) for name in ("pyproject.toml", "uv.lock")}
        _json(directory / "manifest.json", manifest)
        schema = DatasetSchema.from_dict(config["schema"])
        table = DatasetTable.load_csv(root / config["dataset_path"], schema, expected_sha256=config["dataset_sha256"])
        if config.get("metadata_path"):
            metadata_path = root / config["metadata_path"]
            if sha256_file(metadata_path) != config["metadata_sha256"]:
                raise ValueError("Source metadata SHA-256 mismatch")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["image_col"] is not None or metadata["target"] != schema.target_column:
                raise ValueError("Original schema contains images or a different target")
        assignments = build_assignments(root, table, config["splits"])
        assignments.save(directory / "splits.json")
        _json(directory / "schema.json", schema.to_dict())
        teacher_config = TeacherConfig(task=schema.task, **config["teacher"])
        # Only declared structured fit/validation features enter the teacher.
        train_x, train_y = table.structured(assignments.train_ids), table.targets(assignments.train_ids)
        validation_x = table.structured(assignments.validation_ids)
        with _PeakRSS() as memory:
            artifacts = fit_oof_teacher(
                train_x, train_y, categorical_columns=schema.categorical_columns,
                train_ids=assignments.train_ids, validation_X=validation_x,
                validation_ids=assignments.validation_ids, config=teacher_config,
            )
            artifacts.validate()
            forbidden_ids = set(assignments.test_ids) | set(assignments.excluded_ids)
            counts = {value: 0 for value in assignments.train_ids}
            for audit in artifacts.fold_audits:
                if set(audit.fit_row_ids) & set(audit.predict_row_ids) or forbidden_ids & (set(audit.fit_row_ids) | set(audit.predict_row_ids)):
                    raise ValueError("OOF provenance includes held-out or excluded rows")
                for value in audit.predict_row_ids:
                    counts[value] += 1
            if set(counts.values()) != {1} or tuple(artifacts.validation_audit.fit_row_ids) != assignments.train_ids:
                raise ValueError("OOF coverage or validation fitting scope is invalid")
            artifacts.save(directory / "teacher")
            loaded = TeacherArtifacts.load(directory / "teacher")
            np.testing.assert_array_equal(artifacts.oof_predictions, loaded.oof_predictions)
            np.testing.assert_allclose(loaded.predict(validation_x), artifacts.validation_predictions, rtol=0, atol=1e-12)
        # Validation labels are accessed only after all model fitting/saving.
        diagnostics = {"oof": diagnostic_metrics(train_y, artifacts.oof_predictions, artifacts),
                       "validation": diagnostic_metrics(table.targets(assignments.validation_ids), artifacts.validation_predictions, artifacts)}
        manifest.update({
            "status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
            "schema": schema.to_dict(), "teacher_config": asdict(teacher_config), "class_order": list(artifacts.class_order),
            "data": {"csv_sha256": table.source_sha256, "typed_content_sha256": table.fingerprint(), "rows": len(table.row_ids)},
            "split_protocol": dict(assignments.metadata), "split_assignments_sha256": assignments.fingerprint(),
            "partitions": {name: {"rows": len(ids), "typed_ordered_ids_sha256": row_ids_sha256(ids)} for name, ids in assignments.partitions.items()},
            "folds": [{"fold": audit.fold, "fit_rows": len(audit.fit_row_ids), "predicted_rows": len(audit.predict_row_ids),
                       "fit_ids_sha256": audit.fit_row_ids_sha256, "prediction_ids_sha256": audit.predict_row_ids_sha256,
                       "selection_rows": len(audit.selection_row_ids)} for audit in artifacts.fold_audits],
            "validation_teacher": {"fit_rows": len(artifacts.validation_audit.fit_row_ids), "predicted_rows": len(artifacts.validation_ids),
                                   "fit_ids_sha256": artifacts.validation_audit.fit_row_ids_sha256},
            "diagnostics": diagnostics, "checks": {"exactly_one_oof_prediction_per_training_row": True,
                "fit_predict_row_exclusion": True, "separate_validation_teacher": True, "no_heldout_labels_in_fitting": True,
                "schema_excludes_text_target_ids": True, "serialization_round_trip": True},
            "test_predictions": 0, "test_scores": 0,
            "runtime": {"total_seconds": time.perf_counter() - start, "sampled_peak_rss_bytes": memory.peak},
            "limitations": ["Structured teacher diagnostics only; no final test, fusion, or method improvement result.",
                            "Fixed-budget native probabilities; no fitted calibration or hyperparameter selection.",
                            "OOF teachers do not imply cross-fitted supervised fusion embeddings."],
        })
        manifest["artifacts"]["files_sha256"] = {file.relative_to(directory).as_posix(): sha256_file(file) for file in sorted(directory.rglob("*")) if file.is_file() and file.name != "manifest.json"}
        _json(directory / "manifest.json", manifest)
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            _json(summary_path, manifest, exclusive=True)
        return manifest
    except BaseException as error:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "failure": {"type": type(error).__name__, "message": str(error)},
                         "runtime_seconds": time.perf_counter() - start})
        (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _json(directory / "manifest.json", manifest)
        raise
