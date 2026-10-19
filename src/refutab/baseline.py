"""Audited orchestration of the unchanged historical MulTaBench baseline.

The original package remains in an ignored checkout. This module only verifies
inputs, calls its split/preprocessing/learner/metric functions, and saves evidence.
"""

from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from types import MethodType
from typing import Any, Iterator
import uuid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(values: list[Any]) -> str:
    """Hash ordered IDs with an unambiguous canonical representation."""
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_splits(splits: dict[str, list[int]], expected_ids: list[int]) -> dict[str, Any]:
    """Reject duplication, overlap, and omissions before any fit."""
    seen: set[int] = set()
    result: dict[str, Any] = {}
    for name, values in splits.items():
        ids = set(values)
        if len(ids) != len(values) or seen.intersection(ids):
            raise ValueError(f"Duplicate or overlapping row IDs in {name}")
        seen.update(ids)
        result[name] = {"rows": len(values), "ordered_ids_sha256": fingerprint(values), "sorted_ids_sha256": fingerprint(sorted(values))}
    if seen != set(expected_ids):
        raise ValueError("Split IDs do not partition the selected dataset")
    return result


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def _model_files(model_path: Path) -> dict[str, str]:
    names = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
    return {name: sha256_file(model_path / name) for name in names}


def _parameter_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _PeakRSS:
    """Sample process resident memory; this is sampled RSS, not allocator peak."""

    def __init__(self) -> None:
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "_PeakRSS":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join()


@contextlib.contextmanager
def _log_to(path: Path) -> Iterator[None]:
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield


def run_baseline(
    root: Path, config_path: Path, summary_path: Path | None = None,
    run_id: str | None = None, allow_dirty: bool = False, output_root: Path | None = None,
) -> dict[str, Any]:
    """Run one locked original fold, evaluating its held-out rows exactly once.

    A failure manifest is retained, including preflight failures. Raw predictions,
    IDs, logs, and PCA state stay under the ignored output directory.
    """
    root = root.resolve()
    config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = run_id or f"phase1-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    run_dir = root / (output_root if output_root is not None else config["output_root"]) / run_id
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = root / summary_path if summary_path is not None else None
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "status": "started", "started_utc": _utc(),
        "config": config, "config_sha256": sha256_file(config_path), "allow_dirty": allow_dirty,
        "command": sys.argv, "artifacts": {"run_directory": run_dir.relative_to(root).as_posix() if run_dir.is_relative_to(root) else str(run_dir)},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "cost": {"incremental_service_charge_usd": 0, "basis": "Local existing hardware; electricity and equipment excluded."},
    }
    _write_json(run_dir / "manifest.json", manifest)
    try:
        if summary_path is not None and summary_path.exists():
            raise FileExistsError(f"Refusing to overwrite summary {summary_path}")
        source_status = _git(root, "status", "--porcelain", "--untracked-files=all")
        manifest["source"] = {"commit": _git(root, "rev-parse", "HEAD"), "working_tree_dirty": bool(source_status)}
        if not allow_dirty:
            _git(root, "ls-files", "--error-unmatch", "src/refutab/baseline.py", "scripts/run_baseline.py", str(config_path.relative_to(root)), "pyproject.toml", "uv.lock")
        if source_status and not allow_dirty:
            raise ValueError("Project changes or untracked publishable files exist; commit before a recorded experiment (or use explicit --allow-dirty for development).")
        if source_status:
            diff = _git(root, "diff", "HEAD", "--binary")
            manifest["source"]["tracked_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
            (run_dir / "source.patch").write_text(diff, encoding="utf-8")
        upstream = root / config["upstream_path"]
        upstream_sha = _git(upstream, "rev-parse", "HEAD")
        lock = json.loads((root / "third_party/multabench.lock.json").read_text(encoding="utf-8"))
        if upstream_sha != config["upstream_commit"] or upstream_sha != lock["commit"]:
            raise ValueError("Upstream checkout/config/lock do not identify the same historical commit")
        if _git(upstream, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("The original upstream source must remain clean, including untracked files")
        manifest["upstream"] = {"commit": upstream_sha, "tracked_clean": True}
        manifest["environment_lock_sha256"] = {name: sha256_file(root / name) for name in ("pyproject.toml", "uv.lock")}
        if config["condition"] != "joint_frozen" or config["device"] != "cpu":
            raise ValueError("Phase 1 harness supports only the locked CPU Joint Frozen condition")
        data_path, meta_path = root / config["dataset_path"], root / config["metadata_path"]
        manifest["data"] = {"csv_sha256": _check_hash(data_path, config["dataset_sha256"]), "metadata_sha256": _check_hash(meta_path, config["metadata_sha256"])}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["image_col"] is not None or meta["slug"] != config["dataset_slug"]:
            raise ValueError("Dataset must have no original image column and match the configured slug")
        roster_path = root / "configs/datasets.json"
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        eligible = [row for row in roster["eligible_datasets"] if row["dataset_id"] == config["dataset_id"]]
        if roster["upstream_sha"] != upstream_sha or len(eligible) != 1 or eligible[0]["image_col"] is not None or eligible[0]["metadata_sha256"] != config["metadata_sha256"]:
            raise ValueError("Dataset eligibility audit does not match this source/schema pin")
        manifest["data"]["eligibility_roster_sha256"] = sha256_file(roster_path)
        with data_path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream))
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV columns are unsupported")
        model_path = root / config["model_path"]
        model_files = _model_files(model_path)
        provenance = json.loads((model_path / "provenance.json").read_text(encoding="utf-8"))
        if provenance["revision"] != config["model_revision"] or provenance["identifier"] != config["model_identifier"]:
            raise ValueError("Local model revision does not match configuration")
        if any(provenance["files"].get(name) != actual for name, actual in model_files.items()):
            raise ValueError("Local model files no longer match their download provenance")
        manifest["pretrained_model"] = {"identifier": config["model_identifier"], "revision": config["model_revision"], "file_sha256": model_files, "tree_sha256": fingerprint(sorted(model_files.items()))}
        sys.path.insert(0, str(upstream))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import numpy as np
        import pandas as pd
        import psutil
        import torch
        from sklearn.metrics import roc_auc_score, r2_score
        from tabstar.constants import SEED
        from tabstar.preprocessing.splits import split_to_test, split_to_val
        from multabench.baselines.lgbm import LightGBM
        from multabench.baselines.preprocessing.sampling import subsample_dataset
        from multabench.baselines.preprocessing.target import transform_preprocess_y
        from multabench.datasets.all_datasets import MulTaBenchDatasetID
        from multabench.datasets.objects import SupervisedTask
        from multabench.e5.e5_finetune import get_vanilla_e5
        audited_modules = ("multabench.baselines.lgbm", "multabench.baselines.abstract_model", "multabench.baselines.training.metrics", "multabench.baselines.preprocessing.text_embeddings", "multabench.e5.e5_finetune")
        module_paths = {name: Path(sys.modules[name].__file__).resolve() for name in audited_modules}
        if any(not path.is_relative_to(upstream.resolve()) for path in module_paths.values()):
            raise ValueError("An imported upstream module came from outside the pinned checkout")
        manifest["upstream"]["imported_modules"] = {name: path.relative_to(upstream.resolve()).as_posix() for name, path in module_paths.items()}
        if config["seed"] != SEED:
            raise ValueError("Configured seed disagrees with original tabstar constant")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(config["torch_threads"])
        manifest["environment"] = {
            "python": sys.version, "platform": platform.platform(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
        }
        manifest["hardware"] = {"device": "cpu", "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"), "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total, "torch_threads": torch.get_num_threads(), "gpu_used": False}
        frame = pd.read_csv(data_path)
        target = meta["target"]
        y, x = frame[target].copy(), frame.drop(columns=[target])
        if len(frame) != meta["num_rows"] or x.shape[1] != meta["num_features"]:
            raise ValueError("CSV shape disagrees with original metadata")
        if y.isna().any() or not x.index.equals(y.index):
            raise ValueError("Missing targets or misaligned row IDs")
        dataset_id = MulTaBenchDatasetID[config["dataset_id"]]
        if dataset_id.value != meta["slug"] or "_TEXT_" not in dataset_id.name:
            raise ValueError("Dataset does not match the pinned original text-only benchmark enum")
        is_cls = meta["task_type"] != "reg"
        task = (SupervisedTask.BINARY if y.nunique() == 2 else SupervisedTask.MULTICLASS) if is_cls else SupervisedTask.REGRESSION
        manifest["data"].update({"metadata_dataset_id": meta["dataset_id"], "benchmark_enum": dataset_id.name, "version": config["dataset_version"], "rows": len(frame), "feature_columns": list(x.columns), "target": target, "task": task.name, "original_image_column": None, "identical_full_row_duplicates": int(frame.duplicated().sum())})
        x, y = subsample_dataset(x, y, is_cls, config["train_examples"], config["fold"])
        manifest["data"].update({"sampled_rows": len(x), "excluded_by_original_subsampling": len(frame) - len(x), "sampled_ordered_ids_sha256": fingerprint(x.index.tolist())})
        outer_x, test_x, outer_y, test_y = split_to_test(x, y, is_cls, config["fold"], config["train_examples"])
        fit_x, val_x, fit_y, val_y = split_to_val(outer_x, outer_y, is_cls)
        splits = {"fit": fit_x.index.tolist(), "validation": val_x.index.tolist(), "test": test_x.index.tolist()}
        manifest["splits"] = audit_splits(splits, x.index.tolist())
        manifest["seeds"] = {"base": SEED, "outer_fold": config["fold"], "outer_random_state": SEED + config["fold"], "internal_validation_fold": -1, "internal_validation_random_state": SEED - 1, "pca_random_state": SEED, "lightgbm_random_state": SEED}
        _write_json(run_dir / "splits.json", splits)
        _write_json(run_dir / "manifest.json", manifest)
        with _log_to(run_dir / "upstream.log"), _PeakRSS() as memory:
            encoder, _ = get_vanilla_e5(torch.device("cpu"), model_name=str(model_path))
            before_hash = _parameter_hash(encoder)
            model = LightGBM(problem_type=task, device=torch.device("cpu"), dataset=dataset_id, verbose=True, tune_e5=False, e5_model_name=str(model_path), pca_components=config["pca_components"], no_pca=False)
            original_fit_preprocessor = model.fit_preprocessor
            actual_fit_ids: list[int] = []
            def observed_fit_preprocessor(self: Any, x_train: Any, y_train: Any) -> Any:
                actual_fit_ids.extend(x_train.index.tolist())
                if actual_fit_ids != splits["fit"] or not x_train.index.equals(y_train.index):
                    raise ValueError("Original preprocessing fit rows differ from audited training IDs")
                return original_fit_preprocessor(x_train=x_train, y_train=y_train)
            model.fit_preprocessor = MethodType(observed_fit_preprocessor, model)
            train_start = time.perf_counter()
            model.fit(outer_x, outer_y)
            train_seconds = time.perf_counter() - train_start
            if actual_fit_ids != splits["fit"] or model.image_transformers or not model.text_transformers:
                raise ValueError("Missing expected fit observation/text transform, or unexpected image transform")
            pcas = {}
            for name, transformer in model.text_transformers.items():
                pca = transformer.encoder
                if pca.n_components_ != config["pca_components"] or pca.n_samples_ != len(fit_x):
                    raise ValueError("PCA dimensions/fitting sample count violates locked protocol")
                pcas[name] = {"components": int(pca.n_components_), "input_width": int(pca.n_features_in_), "fit_rows": int(pca.n_samples_), "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()), "parameters": pca.get_params()}
                np.savez_compressed(run_dir / f"pca-{len(pcas)}.npz", components=pca.components_, mean=pca.mean_, explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_)
            if encoder.training or any(p.grad is not None for p in encoder.parameters()) or _parameter_hash(encoder) != before_hash:
                raise ValueError("Frozen baseline encoder mode/gradients/weights audit failed")
            # Observe the original scoring method without a second prediction.
            inference_start = time.perf_counter()
            original_predict = model.predict
            captured_predictions: list[Any] = []
            def observed_predict(self: Any, x: Any) -> Any:
                if x.index.tolist() != splits["test"] or captured_predictions:
                    raise ValueError("Expected exactly one aligned original test prediction")
                result = original_predict(x)
                captured_predictions.append(result)
                return result
            model.predict = MethodType(observed_predict, model)
            metrics = model.score_all_metrics(X=test_x, y=test_y)
            if len(captured_predictions) != 1:
                raise ValueError("Original scoring did not perform exactly one test prediction")
            predictions = captured_predictions[0]
            y_encoded = transform_preprocess_y(test_y, model.target_transformer)
            independent = roc_auc_score(y_encoded, predictions, multi_class="ovr", average="macro") if model.d_output > 2 else (roc_auc_score(y_encoded, predictions) if is_cls else r2_score(y_encoded, predictions))
            if not np.isfinite(predictions).all() or not np.isfinite(metrics.score) or not np.isclose(metrics.score, independent, atol=1e-12):
                raise ValueError("Original metric failed independent validation")
            if encoder.training or any(p.grad is not None for p in encoder.parameters()) or _parameter_hash(encoder) != before_hash:
                raise ValueError("Encoder changed during test inference")
            inference_seconds = time.perf_counter() - inference_start
        np.savez_compressed(run_dir / "predictions.npz", row_ids=test_x.index.to_numpy(), y_true=np.asarray(y_encoded), predictions=predictions)
        classes = model.target_transformer.classes_.tolist() if hasattr(model.target_transformer, "classes_") else None
        manifest.update({
            "status": "completed", "metrics": {"score": metrics.score, **metrics.metrics},
            "metric_convention": "macro one-vs-rest AUROC; all classes present" if model.d_output > 2 else ("binary AUROC" if is_cls else "R2"),
            "independent_metric_check": float(independent), "class_order": classes,
            "preprocessing_fit": {"rows": len(actual_fit_ids), "ordered_ids_sha256": fingerprint(actual_fit_ids), "matches_audited_training_partition": True},
            "features": {"ordered_original": list(outer_x.columns), "text": sorted(model.text_features), "numerical": sorted(model.numerical_features), "categorical": sorted(model.categorical_features), "ordered_transformed": model.model_.feature_name_, "transformed_width": int(model.model_.n_features_in_)},
            "pca": pcas, "learner_parameters": model.model_.get_params(),
            "encoder_audit": {"eval_mode": not encoder.training, "all_gradients_none": all(p.grad is None for p in encoder.parameters()), "weights_unchanged": True, "state_sha256": before_hash, "parameters_requiring_grad": sum(p.numel() for p in encoder.parameters() if p.requires_grad), "freezing_mechanism": "Unmodified original torch.no_grad encoding; original parameters retain requires_grad flags."},
            "encoding": {"prefix": "passage: {column_name}: {value}", "batch_size": 32, "max_length": 512, "padding": "max_length", "pooling": "attention-mask mean then L2 normalization", "source_unchanged": True},
            "runtime": {"training_seconds": train_seconds, "test_inference_seconds": inference_seconds, "total_seconds": time.perf_counter() - started, "sampled_peak_rss_bytes": memory.peak, "memory_sampling_seconds": 0.1},
            "test_evaluations": 1, "completed_utc": _utc(),
            "limitations": ["One dataset, one outer fold, one original LightGBM Joint Frozen baseline; no ReFuTab or Joint TAR result.", "A reduced 1000-example outer-training cap bounds this smoke reproduction; this is not a full paper-table replication.", "Official random row splitting retains possible duplicate-content overlap; no deduplication or group split was substituted.", "The selected historical source and pinned dependency defaults are preserved."],
        })
        manifest["artifacts"]["files_sha256"] = {p.name: sha256_file(p) for p in sorted(run_dir.iterdir()) if p.is_file() and p.name != "manifest.json"}
        _write_json(run_dir / "manifest.json", manifest)
        if summary_path is not None:
            # IDs, predictions, raw text, and weights never enter this export.
            _write_json(summary_path, manifest, exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update({"status": "failed", "failed_utc": _utc(), "runtime_seconds": time.perf_counter() - started, "failure": {"type": type(exc).__name__, "message": str(exc)}})
        (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _write_json(run_dir / "manifest.json", manifest)
        raise
