"""Matched, resumable pilot stages with an explicit outer-test barrier.

The pilot plan fixes methods, rows and budgets before new method scores. Partial
CPU work reports validation/grouped diagnostics only. Outer tests are predicted
only after every preregistered main and diagnostic model has completed.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import psutil
import torch

from .baseline import _PeakRSS
from .data import DatasetSchema, DatasetTable, sha256_file
from .downstream import DownstreamConfig, DownstreamModel
from .features import FrozenE5Config, FrozenE5Encoder
from .pipeline import PipelineConfig, ReFuTabPipeline, _training_classes, encode_targets, fit_pipeline
from .pilot_features import CachedFrozenEncoder, FrozenRepresentation
from .pilot_protocol import (benchmark_split, grouped_diagnostics, group_overlap_counts,
                             score_predictions, teacher_fold_ids)
from .splits import SplitAssignments
from .tar import UPSTREAM_SHA, TarRepresentation, fit_tar_representation
from .teacher import TeacherConfig, fit_oof_teacher, row_ids_sha256
from .teacher_runner import _git

METHODS = ("frozen", "direct", "residual", "tar")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: dict):
    """Atomically update a stage ledger; immutable attempt/model files are separate."""
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _inventory(directory: Path):
    return {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p != directory / "manifest.json"}


def _verify_attempt(path: Path, expected_sha: str):
    if sha256_file(path / "manifest.json") != expected_sha:
        raise ValueError("Completed stage manifest was changed")
    record = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if record["status"] != "completed":
        raise ValueError("Only a completed verified stage may be reused")
    for name, digest in record["files_sha256"].items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path.resolve()) or sha256_file(artifact) != digest:
            raise ValueError("Completed stage artifact checksum/path mismatch")
    if _inventory(path) != record["files_sha256"]:
        raise ValueError("Completed stage file inventory differs")
    return record


def _load_tables(root: Path, config: dict):
    manifest = json.loads((root / config["datasets_manifest"]).read_text(encoding="utf-8"))
    tables, entries = {}, {}
    for entry in manifest["datasets"]:
        schema = DatasetSchema.from_dict(entry["schema"])
        path = root / entry["directory"]
        if sha256_file(path / "metadata.json") != entry["files"]["metadata.json"]:
            raise ValueError("Original dataset metadata SHA-256 mismatch")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata["image_col"] is not None or metadata["target"] != schema.target_column:
            raise ValueError("Original dataset has images or a different target")
        table = DatasetTable.load_csv(path / "data.csv", schema, expected_sha256=entry["files"]["data.csv"])
        tables[schema.dataset_id], entries[schema.dataset_id] = table, entry
    return tables, entries


def _schema_audit(table: DatasetTable, ids):
    # The pinned detector is a non-fitting feature-role check on A only.
    from tabstar.preprocessing.feat_types import detect_numerical_features
    from multabench.preprocessing.feat_types import classify_semantic_features
    frame = table.features(ids)
    numbers = set(detect_numerical_features(frame))
    semantic = classify_semantic_features(frame, numbers)
    if (numbers != set(table.schema.numerical_columns)
            or semantic.text_features != set(table.schema.text_columns)
            or semantic.categorical_features != set(table.schema.categorical_columns)):
        raise ValueError("Declared pilot feature roles differ from the historical fitting-only detector")
    return {"numerical": sorted(numbers), "text": sorted(semantic.text_features),
            "categorical": sorted(semantic.categorical_features), "fitting_rows": len(ids)}


def build_cases(root: Path, tables: dict, config: dict):
    """Construct and audit every main/nested split before any model score."""
    cases = {}
    diagnostics = config["diagnostics"]
    for dataset_id, table in tables.items():
        for fold in config["outer_folds"]:
            split = benchmark_split(root, table, fold, config["train_examples"])
            name = f"{dataset_id}__fold{fold}"
            groups = diagnostics["group_columns"][dataset_id]
            cases[name] = {"dataset_id": dataset_id, "outer_fold": fold, "scope": "main", "split": split,
                           "task": table.schema.task,
                           "group_columns": groups, "schema_audit": _schema_audit(table, split.train_ids),
                           "group_overlap": group_overlap_counts(table, split, groups), "methods": config["methods"]}
            if diagnostics["enabled"] and fold == diagnostics["outer_fold"]:
                nested = grouped_diagnostics(table, split, groups, diagnostics["seed"],
                                              diagnostics["n_splits"], diagnostics["validation_fraction"])
                for index, inner in enumerate(nested):
                    cases[f"{name}__diagnostic{index}"] = {
                        "dataset_id": dataset_id, "task": table.schema.task, "outer_fold": fold, "scope": "diagnostic", "diagnostic_fold": index,
                        "split": inner, "group_columns": groups, "schema_audit": _schema_audit(table, inner.train_ids),
                        "group_overlap": group_overlap_counts(table, inner, groups), "methods": diagnostics["methods"]}
    return cases


class _Baseline:
    def __init__(self, schema, representation, downstream):
        self.schema, self.representation, self.downstream = schema, representation, downstream
        self.class_order = downstream.class_order

    def predict(self, frame, *, allow_in_sample=False):
        if tuple(frame.columns) != self.schema.feature_columns:
            raise ValueError("Pilot prediction feature order differs")
        fused = self.representation.transform(frame.loc[:, list(self.schema.text_columns)])
        return self.downstream.predict(frame.loc[:, list(self.schema.structured_columns)], fused)


def _checkpoint(previous: Path | None, method: str):
    if previous is None:
        return None
    if method in {"direct", "residual"}:
        path = previous / "training.pt"
        return path if path.is_file() else None
    if method == "tar":
        candidates = [p for p in (previous / "tar/hf").glob("checkpoint-*")
                      if p.is_dir() and (p / "refutab_checkpoint.json").is_file() and p.name[11:].isdigit()]
        return max(candidates, key=lambda p: int(p.name[11:])) if candidates else None
    return None


def train_method(method, table, case, encoder, config, directory, context, previous=None):
    """Fit a single stage on its explicit A/V; never predict the outer test."""
    split, schema = case["split"], table.schema
    training = PipelineConfig.from_dict(config["pipeline"])
    train_x, train_y = table.features(split.train_ids), table.targets(split.train_ids)
    checkpoint = _checkpoint(previous, method)
    if method in {"direct", "residual"}:
        training = replace(training, training=replace(training.training, objective=method))
        teacher = None
        if method == "residual" and (case["scope"] == "diagnostic" or case.get("teacher_grouped", False)):
            schedule = teacher_fold_ids(table, split.train_ids, case["group_columns"], seed=42)
            teacher = fit_oof_teacher(table.structured(split.train_ids), train_y,
                                      categorical_columns=schema.categorical_columns, train_ids=split.train_ids,
                                      validation_X=table.structured(split.validation_ids), validation_ids=split.validation_ids,
                                      config=TeacherConfig(task=schema.task, **config["teacher"]), fold_ids=schedule)
        target = directory / "training.pt"
        if checkpoint is not None:
            shutil.copy2(checkpoint, target)
        model = fit_pipeline(table, split, encoder, config=training,
                              teacher_config=TeacherConfig(task=schema.task, **config["teacher"]),
                              teacher=teacher, teacher_dataset_fingerprint=table.fingerprint() if teacher else None,
                              teacher_output=directory / "teacher", checkpoint_path=target,
                              resume=checkpoint is not None, training_context=context)
        model.save(directory / "bundle")
        record = {"kind": "pipeline", "class_order": list(model.class_order),
                  "history": model.training["history"], "best_epoch": model.training["best_epoch"],
                  "completed_epoch": model.training["completed_epoch"], "teacher": model.training["teacher"],
                  "fit_ids_sha256": model.training["train_ids_sha256"]}
        return model, record
    classes = _training_classes(train_y, schema.task)
    if method == "frozen":
        representation = FrozenRepresentation.fit(train_x.loc[:, list(schema.text_columns)], schema.text_columns,
                                                   split.train_ids, encoder)
        representation.save(directory / "representation")
        representation = FrozenRepresentation.load(directory / "representation", encoder=encoder)
    elif method == "tar":
        if previous is not None and (previous / "tar/representation/metadata.json").is_file():
            representation = TarRepresentation.load(previous / "tar/representation", local_model_path=encoder.config.model_path,
                                                       device=config["device"])
            if tuple(representation.metadata()["pca_fit_ids"]) != split.train_ids:
                raise ValueError("Previously selected TAR representation has different fitting IDs")
            representation.save(directory / "tar/representation")
        else:
            representation = fit_tar_representation(
                train_x, schema.text_columns, pd.Series(encode_targets(train_y, schema.task, classes), index=train_x.index), task=schema.task,
                train_ids=split.train_ids, device=config["device"], local_model_path=encoder.config.model_path,
                output_directory=directory / "tar", training_context=context, resume_checkpoint=checkpoint, seed=42,
                activation_checkpointing=config.get("tar_execution", {}).get("activation_checkpointing", False))
        # Canonical persisted PCA layout is used before downstream fitting too.
        representation = TarRepresentation.load(directory / "tar/representation", local_model_path=encoder.config.model_path,
                                                  device=config["device"])
    else:
        raise ValueError(f"Unknown preregistered pilot method: {method}")
    fused = representation.transform(train_x.loc[:, list(schema.text_columns)])
    downstream = DownstreamModel(DownstreamConfig(schema.task, seed=42, n_estimators=training.downstream_estimators,
                                                  num_threads=training.downstream_threads))
    downstream.fit(train_x.loc[:, list(schema.structured_columns)], fused, encode_targets(train_y, schema.task, classes),
                    categorical_columns=schema.categorical_columns, class_order=classes, fit_ids=split.train_ids)
    downstream.save(directory / "downstream")
    return _Baseline(schema, representation, downstream), {
        "kind": method, "class_order": list(classes), "representation": representation.metadata(),
        "fit_ids_sha256": row_ids_sha256(split.train_ids)}


def load_method(table, record, directory, encoder, device):
    if record["kind"] == "pipeline":
        return ReFuTabPipeline.load(directory / "bundle", encoder=encoder)
    if record["kind"] == "frozen":
        representation = FrozenRepresentation.load(directory / "representation", encoder=encoder)
    elif record["kind"] == "tar":
        representation = TarRepresentation.load(directory / "tar/representation", local_model_path=encoder.config.model_path, device=device)
    else:
        raise ValueError("Unknown saved pilot model kind")
    downstream = DownstreamModel.load(directory / "downstream")
    if tuple(downstream.class_order) != tuple(record["class_order"]):
        raise ValueError("Stage class order differs from the saved downstream learner")
    return _Baseline(table.schema, representation, downstream)


def _prediction_score(table, ids, prediction, class_order, *, in_sample=False):
    metrics = score_predictions(table.targets(ids), prediction, table.schema.task, class_order)
    return {"rows": len(ids), "row_ids_sha256": row_ids_sha256(ids), "in_sample": in_sample, "metrics": metrics}


def _score(model, table, ids, directory, label, *, in_sample=False):
    prediction = model.predict(table.features(ids), allow_in_sample=in_sample)
    result = _prediction_score(table, ids, prediction, model.class_order, in_sample=in_sample)
    path = directory / f"{label}_predictions.npy"
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        np.save(stream, prediction, allow_pickle=False)
    os.replace(temporary, path)
    return result


def _environment(device):
    result = {"python": sys.version, "platform": platform.platform(), "device": device,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "peft", "accelerate", "catboost", "lightgbm", "numpy", "pandas", "scikit-learn", "tabstar")}}
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Joint TAR pilot needs one visible working CUDA GPU; the current CPU environment cannot run it")
        result["cuda"] = {"runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                          "capability": list(torch.cuda.get_device_capability(0)),
                          "total_memory": torch.cuda.get_device_properties(0).total_memory,
                          "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}
    return result


def _summary(state, cases):
    # Full representation metadata contains private row IDs; export metrics/counts only.
    stages = {}
    for name, value in state["stages"].items():
        stages[name] = {key: value[key] for key in ("status", "attempt", "manifest_sha256", "scores", "runtime", "failure") if key in value}
    summary = {"format_version": 1, "run_id": state["run_id"], "status": state["status"],
            "source": state["source"], "plan_sha256": state["plan_sha256"], "binding_sha256": state["binding_sha256"],
            "environment": state["environment"], "hardware": state["hardware"], "cost": state["cost"],
            "execution_history": state.get("execution_history", []), "failure": state.get("failure"),
            "planned_stages": len(state["expected_jobs"]), "completed_stages": sum(x["status"] == "completed" for x in stages.values()),
            "stages": stages, "outer_tests": state.get("outer_tests", {}),
            "test_predictions": len(state.get("outer_tests", {})), "test_barrier_satisfied": all(x in stages and stages[x]["status"] == "completed" for x in state["expected_jobs"]),
            "cases": {key: {"dataset_id": value["dataset_id"], "task": value["task"], "outer_fold": value["outer_fold"], "scope": value["scope"],
                            "diagnostic_fold": value.get("diagnostic_fold"),
                            "partitions": {name: len(ids) for name, ids in value["split"].partitions.items()},
                            "schema_audit": value["schema_audit"], "group_overlap": value["group_overlap"]} for key, value in cases.items()},
            "limitations": ["Exploratory matched-adapter pilot; original Phase 1 result is separate.",
                            "Overlapping repeated holdouts support descriptive differences, not independent-sample significance.",
                            "Grouped nested diagnostics estimate the complete unseen-row pipeline; no latent-coordinate stacking."]}
    from .pilot_report import aggregate_pilot
    summary["aggregation"] = aggregate_pilot(summary)
    return summary


def run_pilot(root: Path, config_path: Path, *, run_id=None, output_root=None, resume_run=None,
                methods=None, dataset=None, fold=None, device=None, allow_dirty=False):
    setup_start = time.perf_counter()
    root = root.resolve()
    config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config["methods"] != list(METHODS):
        raise ValueError("Pilot requires the registered four methods and format")
    execution = config.get("tar_execution", {})
    if (not isinstance(execution, dict) or set(execution) - {"activation_checkpointing"}
            or any(type(value) is not bool for value in execution.values())):
        raise ValueError("tar_execution accepts only an explicit activation_checkpointing boolean")
    selected = list(METHODS if methods is None else methods)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(METHODS):
        raise ValueError("Select only unique preregistered methods")
    effective = json.loads(json.dumps(config))
    effective["device"] = device or config["device"]
    effective["encoder"]["device"] = effective["device"]
    effective["pipeline"]["training"]["device"] = effective["device"]
    if "tar" in selected and not effective["device"].startswith("cuda"):
        raise ValueError("Joint TAR needs a CUDA GPU; use a non-TAR subset for bounded CPU diagnostics")
    environment = _environment(effective["device"])
    source = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty and not allow_dirty:
        raise ValueError("Commit publishable source before a recorded pilot")
    tables, entries = _load_tables(root, config)
    cases = build_cases(root, tables, config)
    if dataset is not None and dataset not in tables:
        raise ValueError("Requested dataset was not preregistered")
    if fold is not None and fold not in config["outer_folds"]:
        raise ValueError("Requested outer fold was not preregistered")
    jobs = [f"{name}__{method}" for name, case in cases.items() for method in case["methods"]]
    binding = {"source_commit": source, "upstream_commit": UPSTREAM_SHA, "plan_sha256": sha256_file(config_path),
               "datasets_manifest_sha256": sha256_file(root / config["datasets_manifest"]), "effective_config": effective,
               "dataset_fingerprints": {name: table.fingerprint() for name, table in tables.items()},
               "split_fingerprints": {name: case["split"].fingerprint() for name, case in cases.items()},
               "environment": environment,
               "lock_sha256": {name: sha256_file(root / name) for name in
                                (("environments/gpu/pyproject.toml", "environments/gpu/uv.lock") if effective["device"].startswith("cuda") else ("pyproject.toml", "uv.lock"))}}
    if resume_run is not None:
        if run_id is not None or output_root is not None:
            raise ValueError("Resume uses its existing run directory; omit run-id/output-root")
        directory = (root / resume_run).resolve()
        state = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if state["binding_sha256"] != _hash(binding) or state["binding"] != binding:
            raise ValueError("Pilot continuation provenance changed: source/data/config/splits/device/environment")
        for entry in state["stages"].values():
            if entry["status"] == "completed":
                _verify_attempt(directory / entry["attempt"], entry["manifest_sha256"])
    else:
        run_id = run_id or f"pilot-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be one new directory name")
        directory = (root / (output_root or config["output_root"]) / run_id).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        state = {"format_version": 1, "run_id": run_id, "status": "started", "started_utc": _now(),
                 "source": {"commit": source, "upstream_commit": UPSTREAM_SHA, "working_tree_dirty": bool(dirty), "allow_dirty": allow_dirty},
                 "plan_sha256": sha256_file(config_path), "binding_sha256": _hash(binding), "binding": binding,
                 "environment": environment, "config": config, "expected_jobs": jobs, "stages": {}, "outer_tests": {},
                 "hardware": {"cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                              "ram_bytes": psutil.virtual_memory().total, "gpu": environment.get("cuda")},
                 "cost": {"provider_service_charge_usd": None, "basis": "Record actual provider charge after execution"}}
        for name, case in cases.items():
            path = directory / "splits" / f"{name}.json"
            case["split"].save(path)
        _write(directory / "manifest.json", state)
    encoder_config = FrozenE5Config(**effective["encoder"])
    encoder_config = replace(encoder_config, model_path=str(root / encoder_config.model_path))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(effective["pipeline"]["training"]["threads"])
    state["status"], state["last_continuation_utc"] = "running", _now()
    state.pop("failure", None)
    state.setdefault("invocations", []).append({"started_utc": _now(), "argv": list(sys.argv),
                                               "methods": selected, "dataset": dataset, "fold": fold,
                                               "device": effective["device"]})
    history = state.setdefault("execution_history", [])
    for prior in history:
        if prior["status"] == "running":
            prior["status"] = "interrupted_duration_unknown"
    execution = {"started_utc": _now(), "status": "running", "setup_seconds": time.perf_counter() - setup_start,
                 "measurement_scope": "execution includes encoder loading/cache prefetch, stages, artifact verification and outer evaluation; setup precedes RSS sampling"}
    history.append(execution)
    _write(directory / "manifest.json", state)
    execution_start = time.perf_counter()
    run_memory = _PeakRSS()
    run_memory.__enter__()
    try:
        encoder = CachedFrozenEncoder(FrozenE5Encoder(encoder_config), directory / "frozen_cache")
        for name, case in cases.items():
            if (dataset is not None and case["dataset_id"] != dataset) or (fold is not None and case["outer_fold"] != fold):
                continue
            table = tables[case["dataset_id"]]
            # Cache only fixed pretrained inference; no PCA/vocabulary is shared across folds.
            pending_frozen = any(method in selected and method != "tar" and
                                 state["stages"].get(f"{name}__{method}", {}).get("status") != "completed"
                                 for method in case["methods"])
            if case["scope"] == "main" and pending_frozen:
                ids = case["split"].train_ids + case["split"].validation_ids
                print(f"Preparing frozen text: {name}, {len(ids)} benchmark-training rows", flush=True)
                encoder.encode(table.features(ids).loc[:, list(table.schema.text_columns)], table.schema.text_columns)
            for method in case["methods"]:
                if method not in selected:
                    continue
                job = f"{name}__{method}"
                previous_record = state["stages"].get(job)
                if previous_record and previous_record["status"] == "completed":
                    continue
                context = {"pilot_binding_sha256": state["binding_sha256"], "job": job,
                           "dataset_fingerprint": table.fingerprint(), "split_fingerprint": case["split"].fingerprint()}
                # Recover a finished attempt if the process stopped between its atomic
                # completion record and the following global-ledger update.
                if previous_record:
                    prior_path = directory / previous_record["attempt"]
                    prior = json.loads((prior_path / "manifest.json").read_text(encoding="utf-8"))
                    if prior["status"] == "completed" and prior["context"] == context:
                        digest = sha256_file(prior_path / "manifest.json")
                        _verify_attempt(prior_path, digest)
                        state["stages"][job] = {**previous_record, "status": "completed", "manifest_sha256": digest,
                                                "scores": prior["scores"], "runtime": prior["runtime"]}
                        _write(directory / "manifest.json", state)
                        continue
                attempt_index = 1 if previous_record is None else previous_record["attempt_index"] + 1
                attempt = directory / "attempts" / f"{job}__attempt{attempt_index}"
                attempt.mkdir(parents=True, exist_ok=False)
                previous = None if previous_record is None else directory / previous_record["attempt"]
                record = {"status": "running", "job": job, "method": method, "started_utc": _now(), "context": context}
                _write(attempt / "manifest.json", record)
                state["stages"][job] = {"status": "running", "attempt": attempt.relative_to(directory).as_posix(),
                                        "attempt_index": attempt_index}
                _write(directory / "manifest.json", state)
                start = time.perf_counter()
                memory = None
                print(f"Fitting {job}", flush=True)
                try:
                    if effective["device"].startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    with _PeakRSS() as memory:
                        model, details = train_method(method, table, case, encoder, effective, attempt, context, previous)
                        scores = {"train": _score(model, table, case["split"].train_ids, attempt, "train", in_sample=True),
                                  "validation": _score(model, table, case["split"].validation_ids, attempt, "validation")}
                        if case["scope"] == "diagnostic":
                            scores["diagnostic"] = _score(model, table, case["split"].test_ids, attempt, "diagnostic")
                        loaded = load_method(table, details, attempt, encoder, effective["device"])
                        np.testing.assert_allclose(model.predict(table.features(case["split"].validation_ids)),
                                                   loaded.predict(table.features(case["split"].validation_ids)), rtol=0, atol=1e-12)
                    record.update({"status": "completed", "completed_utc": _now(), "model": details, "scores": scores,
                                   "runtime": {"seconds": time.perf_counter() - start, "sampled_peak_rss_bytes": memory.peak,
                                               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if effective["device"].startswith("cuda") else None}})
                    del model, loaded
                    gc.collect()
                    if effective["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
                except BaseException as error:
                    record.update({"status": "failed", "failed_utc": _now(),
                                   "failure": {"type": type(error).__name__, "message": str(error)},
                                   "runtime": {"seconds": time.perf_counter() - start,
                                               "sampled_peak_rss_bytes": memory.peak if memory is not None else None,
                                               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if effective["device"].startswith("cuda") else None}})
                    (attempt / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    record["files_sha256"] = _inventory(attempt)
                    _write(attempt / "manifest.json", record)
                    state["stages"][job] = {"status": "failed", "attempt": attempt.relative_to(directory).as_posix(),
                                            "attempt_index": attempt_index, "failure": record["failure"], "runtime": record["runtime"]}
                    raise
                record["files_sha256"] = _inventory(attempt)
                _write(attempt / "manifest.json", record)
                state["stages"][job] = {"status": "completed", "attempt": attempt.relative_to(directory).as_posix(),
                                        "attempt_index": attempt_index, "manifest_sha256": sha256_file(attempt / "manifest.json"),
                                        "scores": scores, "runtime": record["runtime"]}
                _write(directory / "manifest.json", state)
                _write(directory / "summary.json", _summary(state, cases))
        ready = all(job in state["stages"] and state["stages"][job]["status"] == "completed" for job in jobs)
        if ready:
            # Locked execution barrier: every model/diagnostic exists before the first outer prediction.
            for name, case in cases.items():
                if case["scope"] != "main":
                    continue
                table = tables[case["dataset_id"]]
                for method in case["methods"]:
                    job = f"{name}__{method}"
                    if job in state["outer_tests"]:
                        prior_dir = directory / "outer_evaluation" / job
                        prior = json.loads((prior_dir / "result.json").read_text(encoding="utf-8"))
                        if (prior != state["outer_tests"][job]
                                or sha256_file(prior_dir / "test_predictions.npy") != prior["prediction_sha256"]
                                or prior["stage_manifest_sha256"] != state["stages"][job]["manifest_sha256"]):
                            raise ValueError("Previously completed outer-test artifacts differ")
                        continue
                    entry = state["stages"][job]
                    attempt = directory / entry["attempt"]
                    record = _verify_attempt(attempt, entry["manifest_sha256"])
                    evaluation_dir = directory / "outer_evaluation" / job
                    if (evaluation_dir / "result.json").is_file():
                        result = json.loads((evaluation_dir / "result.json").read_text(encoding="utf-8"))
                        if (sha256_file(evaluation_dir / "test_predictions.npy") != result["prediction_sha256"]
                                or result["row_ids_sha256"] != row_ids_sha256(case["split"].test_ids)
                                or result["stage_manifest_sha256"] != entry["manifest_sha256"]):
                            raise ValueError("Interrupted outer evaluation has different provenance")
                    else:
                        evaluation_dir.mkdir(parents=True, exist_ok=True)
                        predictions_path = evaluation_dir / "test_predictions.npy"
                        request = {"binding_sha256": state["binding_sha256"], "job": job,
                                   "stage_manifest_sha256": entry["manifest_sha256"],
                                   "row_ids_sha256": row_ids_sha256(case["split"].test_ids),
                                   "rows": len(case["split"].test_ids), "class_order": record["model"]["class_order"]}
                        request_path = evaluation_dir / "request.json"
                        if request_path.exists():
                            if json.loads(request_path.read_text(encoding="utf-8")) != request:
                                raise ValueError("Interrupted outer prediction request differs")
                        elif predictions_path.exists():
                            raise ValueError("Outer predictions have no matching provenance request")
                        else:
                            _write(request_path, request)
                        if predictions_path.exists():
                            # Durable atomic predictions survive a crash before result/ledger
                            # publication; compute their metrics without predicting again.
                            prediction = np.load(predictions_path, allow_pickle=False)
                            result = _prediction_score(table, case["split"].test_ids, prediction, request["class_order"])
                        else:
                            model = load_method(table, record["model"], attempt, encoder, effective["device"])
                            result = _score(model, table, case["split"].test_ids, evaluation_dir, "test")
                            del model
                        result["prediction_sha256"] = sha256_file(predictions_path)
                        result["stage_manifest_sha256"] = entry["manifest_sha256"]
                        _write(evaluation_dir / "result.json", result)
                    state["outer_tests"][job] = result
                    _write(directory / "manifest.json", state)
                    gc.collect()
                    if effective["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
        encoder.assert_frozen()
        state["status"] = "completed" if ready else "incomplete"
        state["updated_utc"] = _now()
        _write(directory / "manifest.json", state)
        summary = _summary(state, cases)
        _write(directory / "summary.json", summary)
        return summary
    except BaseException as error:
        state["status"], state["updated_utc"] = "failed", _now()
        state["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write(directory / "manifest.json", state)
        (directory / f"failure-{uuid.uuid4().hex[:8]}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _write(directory / "summary.json", _summary(state, cases))
        raise
    finally:
        run_memory.__exit__(None, None, None)
        execution.update(status=state["status"], completed_utc=_now(), seconds=time.perf_counter() - execution_start,
                         sampled_peak_rss_bytes=run_memory.peak)
        _write(directory / "manifest.json", state)
        _write(directory / "summary.json", _summary(state, cases))
        torch.set_num_threads(old_threads)
