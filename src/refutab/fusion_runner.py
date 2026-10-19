"""Configured Phase 3 training with private artifacts and no outer test scoring."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import psutil
import torch

from .baseline import _PeakRSS
from .data import DatasetSchema, DatasetTable, sha256_file
from .features import FrozenE5Config, FrozenE5Encoder
from .pipeline import PipelineConfig, ReFuTabPipeline, fit_pipeline, load_teacher_for_table
from .synthetic import make_synthetic
from .teacher import TeacherConfig, row_ids_sha256
from .teacher_runner import _git, _json, build_assignments, diagnostic_metrics


def load_data(root: Path, config: dict):
    settings = dict(config["data"])
    kind = settings.pop("kind")
    if kind == "synthetic":
        return make_synthetic(**settings)
    if kind != "csv":
        raise ValueError("data.kind must be synthetic or csv")
    schema = DatasetSchema.from_dict(settings["schema"])
    table = DatasetTable.load_csv(root / settings["path"], schema, expected_sha256=settings["sha256"])
    if settings.get("metadata_path"):
        path = root / settings["metadata_path"]
        if sha256_file(path) != settings["metadata_sha256"]:
            raise ValueError("Original metadata SHA-256 mismatch")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["image_col"] is not None or metadata["target"] != schema.target_column:
            raise ValueError("Original dataset schema has images or a different target")
    return table, build_assignments(root, table, config["splits"])


def run_fusion(root: Path, config_path: Path, *, run_id: str | None = None,
                summary_path: Path | None = None, output_root: Path | None = None,
                resume_from: Path | None = None, allow_dirty: bool = False) -> dict:
    root = root.resolve()
    config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported fusion runner schema_version")
    run_id = run_id or f"phase3-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one new directory name")
    directory = (root / (output_root or config["output_root"]) / run_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    summary_path = root / summary_path if summary_path is not None else None
    started = time.perf_counter()
    manifest = {"format_version": 1, "run_id": run_id, "status": "started",
                "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "config_sha256": sha256_file(config_path), "command": sys.argv,
                "artifacts": {"directory": directory.relative_to(root).as_posix() if directory.is_relative_to(root) else str(directory)},
                "cost": {"provider_service_charge_usd": None, "basis": "Not measured; no automatic billing estimate"}}
    _json(directory / "manifest.json", manifest)
    previous_threads = torch.get_num_threads()
    try:
        if summary_path is not None and summary_path.exists():
            raise FileExistsError("Refusing to overwrite an existing summary")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
        source = _git(root, "rev-parse", "HEAD")
        manifest["source"] = {"commit": source, "working_tree_dirty": bool(dirty), "allow_dirty": allow_dirty}
        if dirty and not allow_dirty:
            raise ValueError("Commit publishable source changes before a recorded run")
        if dirty:
            (directory / "source.patch").write_text(_git(root, "diff", "HEAD", "--binary"), encoding="utf-8")
        pipeline_config = PipelineConfig.from_dict(config["pipeline"])
        torch.set_num_threads(pipeline_config.training.threads)
        manifest["hardware"] = {"cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
                                "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                                "encoder_device": config["encoder"]["device"], "fusion_device": pipeline_config.training.device,
                                "torch_threads": torch.get_num_threads(), "gpu_available": torch.cuda.is_available()}
        manifest["environment"] = {"python": sys.version, "platform": platform.platform(),
                                   "packages": {name: importlib.metadata.version(name) for name in
                                                ("torch", "transformers", "catboost", "lightgbm", "numpy", "pandas", "scikit-learn", "tabstar")}}
        manifest["environment_lock_sha256"] = {name: sha256_file(root / name) for name in ("pyproject.toml", "uv.lock")}
        _json(directory / "manifest.json", manifest)
        table, splits = load_data(root, config)
        splits.save(directory / "splits.json")
        if config["data"]["kind"] == "synthetic":
            # Private copy ensures the generated fixture can be inspected without publishing rows.
            table.frame.to_csv(directory / "synthetic.csv", index=False)
        encoder_config = FrozenE5Config(**config["encoder"])
        encoder_config = replace(encoder_config, model_path=str(root / encoder_config.model_path))
        teacher = None
        if config.get("teacher_run"):
            teacher = load_teacher_for_table(root / config["teacher_run"], table, splits)
        checkpoint = directory / "training.pt"
        if resume_from is not None:
            resume_from = root / resume_from
            manifest["resumed_checkpoint_sha256"] = sha256_file(resume_from)
            shutil.copy2(resume_from, checkpoint)
            _json(directory / "manifest.json", manifest)
        print(f"Encoding/training {len(splits.train_ids)} rows; validation {len(splits.validation_ids)}; outer test reserved", flush=True)
        with _PeakRSS() as memory:
            encoder = FrozenE5Encoder(encoder_config)
            model = fit_pipeline(table, splits, encoder, config=pipeline_config,
                                  teacher_config=TeacherConfig(task=table.schema.task, **config.get("teacher", {})),
                                  teacher=teacher, teacher_dataset_fingerprint=table.fingerprint() if teacher else None,
                                  teacher_output=directory / "teacher", checkpoint_path=checkpoint,
                                  resume=resume_from is not None,
                                  training_context={"source_commit": source, "configuration_sha256": manifest["config_sha256"]})
            model.save(directory / "bundle")
            validation = table.features(splits.validation_ids)
            validation.rename_axis(table.schema.id_column or "row_id").to_csv(directory / "validation_features.csv")
            predictions = model.predict(validation)
            # Loading checks all files and class/feature/scope semantics; no teacher/head exists in this bundle.
            loaded = ReFuTabPipeline.load(directory / "bundle", encoder=encoder)
            np.testing.assert_allclose(predictions, loaded.predict(validation), rtol=0, atol=1e-12)
            np.save(directory / "validation_predictions.npy", predictions)
        diagnostics = diagnostic_metrics(table.targets(splits.validation_ids), predictions, model.downstream)
        encoder_metadata = encoder.metadata()
        manifest.update({"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "data": {"kind": config["data"]["kind"], "typed_content_sha256": table.fingerprint(),
                                  "csv_sha256": table.source_sha256, "rows": len(table.row_ids)},
                         "schema": table.schema.to_dict(), "class_order": list(model.class_order),
                         "partitions": {name: {"rows": len(ids), "typed_ordered_ids_sha256": row_ids_sha256(ids)}
                                        for name, ids in splits.partitions.items()},
                         "split_assignments_sha256": splits.fingerprint(), "split_protocol": dict(splits.metadata),
                         "encoder": {key: encoder_metadata[key] for key in ("kind", "dimension", "prefix", "missing_text", "pooling", "verified_files", "test_injection")},
                         "encoder_state_sha256": model.training["encoder_state_sha256"],
                         "fusion_config": asdict(model.fusion.config), "pipeline_config": asdict(pipeline_config),
                         "teacher": model.training["teacher"], "best_epoch": model.training["best_epoch"],
                         "completed_epoch": model.training["completed_epoch"], "history": model.training["history"],
                         "validation_downstream_diagnostics": diagnostics,
                         "checks": {"training_only_pca": True, "training_only_structured_preprocessing": True,
                                    "frozen_e5_weights_and_gradients": True, "teacher_signals_valid_for_objective": True,
                                    "identity_initialization_tested": True, "head_free_teacher_free_bundle": True,
                                    "saved_loaded_predictions_match": True},
                         "test_predictions": 0, "test_scores": 0,
                         "runtime": {"total_seconds": time.perf_counter() - started, "sampled_peak_rss_bytes": memory.peak},
                         "limitations": ["Architecture-stage training run; no matched-baseline improvement claim.",
                                          "Training fusion embeddings are supervised in-sample, not cross-fitted.",
                                          "CPU tested; GPU execution unverified. E5 batches and LightGBM fitting restart on retry."]})
        manifest["artifacts"]["files_sha256"] = {path.relative_to(directory).as_posix(): sha256_file(path)
                                                 for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        # Include component manifests too; only exclude this run's self-referential manifest.
        manifest["artifacts"]["files_sha256"].update({path.relative_to(directory).as_posix(): sha256_file(path)
                                                     for path in directory.rglob("manifest.json") if path != directory / "manifest.json"})
        _json(directory / "manifest.json", manifest)
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            _json(summary_path, manifest, exclusive=True)
        return manifest
    except BaseException as error:
        manifest.update({"status": "failed", "failed_utc": datetime.now(timezone.utc).isoformat(),
                         "failure": {"type": type(error).__name__, "message": str(error)},
                         "runtime_seconds": time.perf_counter() - started})
        (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _json(directory / "manifest.json", manifest)
        raise
    finally:
        torch.set_num_threads(previous_threads)
