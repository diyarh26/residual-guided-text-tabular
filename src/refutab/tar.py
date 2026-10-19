"""Audited runtime adapter for the unchanged historical Joint TAR text path.

Only local asset lookup and checkpoint persistence are intercepted. The pinned
source owns its supervised split, regression bins, LoRA, loss and training loop.
Selected inference artifacts contain the adapter/tokenizer and native PCA state;
the temporary supervised head is confined to private Trainer checkpoints.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

from .data import RowId, _typed_cell, normalize_row_ids, sha256_file
from .features import E5_FILE_HASHES, E5_IDENTIFIER, E5_REVISION, _columns

UPSTREAM_SHA = "599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
# Historical benchmark.py CLI's E5TrainArgs, plus unchanged function defaults.
TAR_TRAIN_KWARGS = {
    "lora_rank": 16, "text_layers": 3, "learning_rate": 1e-4,
    "epochs": 50, "patience": 3, "weight_decay": 0.001,
    "batch_size": 256, "dataloader_num_workers": 4, "max_length": 512,
    "seed": 42,
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _write(path: Path, value: Any) -> None:
    path.write_text(_json(value) + "\n", encoding="utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _inventory(directory: Path, *, excluding: Sequence[str] = ()) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.relative_to(directory).as_posix() not in excluding}


def _verify_assets(local_model_path: str | Path) -> tuple[Path, dict[str, str]]:
    directory = Path(local_model_path).resolve()
    allowed = set(E5_FILE_HASHES) | {"provenance.json", "README.md"}
    if extras := {path.name for path in directory.iterdir() if path.is_file()} - allowed:
        raise ValueError(f"Unexpected unverified E5 asset files: {sorted(extras)}")
    hashes = {name: sha256_file(directory / name) for name in E5_FILE_HASHES}
    if hashes != E5_FILE_HASHES:
        raise ValueError("TAR requires all six immutable pinned E5 asset checksums")
    return directory, hashes


def _source_modules() -> tuple[Any, Any, Any]:
    root = Path(__file__).resolve().parents[2]
    upstream = root / "third_party/MulTaBench"
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()
    if git("rev-parse", "HEAD") != UPSTREAM_SHA or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Joint TAR requires the clean historical pinned upstream checkout")
    if version("tabstar") != "1.1.15":
        raise ValueError("Historical Joint TAR requires tabstar==1.1.15")
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(upstream))
        modules = tuple(importlib.import_module(name) for name in (
            "multabench.baselines.preprocessing.text_embeddings",
            "multabench.e5.e5_finetune", "multabench.preprocessing.splits"))
        for module in modules:
            if not Path(module.__file__).resolve().is_relative_to(upstream.resolve()):
                raise ValueError("A different MulTaBench package is already imported")
        return modules
    finally:
        sys.path[:] = previous


def _training_device(device: str) -> torch.device:
    value = torch.device(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (value.type != "cuda" or value.index not in (None, 0)
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or not visible.strip() or "," in visible):
        raise RuntimeError("Joint TAR training requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES before starting Python")
    return value


def _resume_check(checkpoint: Path, context: dict[str, Any]) -> None:
    metadata = json.loads((checkpoint / "refutab_checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("context") != context or metadata.get("context_sha256") != _hash(context):
        raise ValueError("TAR resume checkpoint belongs to a different training context")
    if metadata.get("files") != _inventory(checkpoint, excluding=("refutab_checkpoint.json",)):
        raise ValueError("TAR resume checkpoint checksum or file inventory mismatch")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    best = state.get("best_model_checkpoint")
    if best and Path(best).resolve() != checkpoint.resolve():
        # The best checkpoint may be distinct from the last one. Both must remain
        # persistent and verified; Trainer retains its original best-model path.
        best_path = Path(best)
        best_meta = json.loads((best_path / "refutab_checkpoint.json").read_text(encoding="utf-8"))
        if (best_meta.get("context") != context or best_meta.get("context_sha256") != _hash(context)
                or best_meta.get("files") != _inventory(best_path, excluding=("refutab_checkpoint.json",))):
            raise ValueError("TAR resume best-model checkpoint provenance mismatch")


def _completed_training(directory: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    selected = directory / "selected_backbone"
    if not selected.exists():
        return None
    metadata = json.loads((selected / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("context") != context or metadata.get("context_sha256") != _hash(context)
            or metadata.get("files") != _inventory(selected, excluding=("metadata.json",))
            or metadata.get("trainer_state_sha256") != _hash(metadata["trainer_state"])):
        raise ValueError("Completed TAR training receipt context/checksum mismatch")
    return metadata


def _restore_completed_backbone(model: Any, directory: Path) -> None:
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    state = load_peft_weights(str(directory / "adapter"), device="cpu", local_files_only=True)
    incompatible = set_peft_model_state_dict(model, state, adapter_name="default")
    if incompatible.unexpected_keys:
        raise ValueError("Completed TAR backbone adapter has unexpected weight keys")


def _copy_resume(checkpoint: Path, output: Path, context: dict[str, Any]) -> Path:
    """Copy last and best checkpoints to a new attempt, changing paths only."""
    _resume_check(checkpoint, context)
    previous_output = checkpoint.parent.parent
    completed = _completed_training(previous_output, context)
    if completed is not None and previous_output.resolve() != output.resolve():
        shutil.copytree(previous_output / "selected_backbone", output / "selected_backbone")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    sources = {checkpoint.resolve()}
    if state.get("best_model_checkpoint"):
        sources.add(Path(state["best_model_checkpoint"]).resolve())
    destinations = {source: output / "hf" / source.name for source in sources}
    for source, destination in destinations.items():
        if source == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    for source, destination in destinations.items():
        if source == destination.resolve():
            continue
        path = destination / "trainer_state.json"
        checkpoint_state = json.loads(path.read_text(encoding="utf-8"))
        best = checkpoint_state.get("best_model_checkpoint")
        if best:
            best_source = Path(best).resolve()
            if best_source not in destinations:
                raise ValueError("TAR best checkpoint references an unverified checkpoint")
            checkpoint_state["best_model_checkpoint"] = str(destinations[best_source])
        _write(path, checkpoint_state)
    for source, destination in destinations.items():
        if source != destination.resolve():
            _write(destination / "refutab_checkpoint.json", {
                "context": context, "context_sha256": _hash(context),
                "files": _inventory(destination, excluding=("refutab_checkpoint.json",)),
                "copied_from": str(source),
            })
    result = destinations[checkpoint.resolve()]
    _resume_check(result, context)
    return result


@contextmanager
def _runtime_scope(text_module: Any, fine_module: Any, split_module: Any, *,
                   local_model_path: Path, output_directory: Path,
                   context: dict[str, Any], audit: dict[str, Any],
                   resume_checkpoint: Path | None):
    """Restore all process bindings on success, OOM, or any source failure."""
    import transformers
    original_model_loader = transformers.AutoModel.from_pretrained
    original_tokenizer_loader = transformers.AutoTokenizer.from_pretrained
    original_trainer = fine_module.Trainer
    original_split = split_module.split_to_val
    original_discretize = text_module.discretize_numerical
    from transformers import TrainerCallback

    def model_loader(identifier: Any, *args: Any, **kwargs: Any):
        if str(identifier) != E5_IDENTIFIER:
            raise ValueError("Unexpected model identifier in original TAR source")
        kwargs.update(local_files_only=True, use_safetensors=True)
        audit.setdefault("asset_loads", []).append("model")
        return original_model_loader(local_model_path, *args, **kwargs)

    def tokenizer_loader(identifier: Any, *args: Any, **kwargs: Any):
        if str(identifier) != E5_IDENTIFIER:
            raise ValueError("Unexpected tokenizer identifier in original TAR source")
        kwargs.update(local_files_only=True)
        audit.setdefault("asset_loads", []).append("tokenizer")
        return original_tokenizer_loader(local_model_path, *args, **kwargs)

    def observe_split(*args: Any, **kwargs: Any):
        x = kwargs.get("x", args[0] if args else None)
        y = kwargs.get("y", args[1] if len(args) > 1 else None)
        result = original_split(*args, **kwargs)
        if "supervised_split" in audit:
            raise RuntimeError("Original TAR unexpectedly requested multiple supervised splits")
        audit["supervised_split"] = {
            "input_ids": list(normalize_row_ids(x.index)),
            "fit_ids": list(normalize_row_ids(result[0].index)),
            "validation_ids": list(normalize_row_ids(result[1].index)),
            "fold_argument": kwargs.get("fold", -1), "effective_seed": 41,
            "label_counts": {str(key): int(count) for key, count in y.value_counts().items()},
            "labels_sha256": _hash([_typed_cell(value) for value in y.tolist()]),
        }
        return result

    def observe_discretize(series: pd.Series, *args: Any, **kwargs: Any):
        result = original_discretize(series, *args, **kwargs)
        n_bins = kwargs.get("n_bins", args[0] if args else 10)
        # Recover the exact native qcut edges for the audit only. The source's
        # returned labels above, without substitution, enter its split/fit.
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        _, edges = pd.qcut(numeric, q=n_bins, labels=False, duplicates="drop", retbins=True)
        audit["regression_bins"] = {
            "input_ids": list(normalize_row_ids(series.index)),
            "requested_bins": n_bins, "actual_bins": len(edges) - 1,
            "edges": edges.tolist(), "duplicates": "drop",
            "labels_sha256": _hash([_typed_cell(value) for value in result.tolist()]),
        }
        return result

    class CheckpointAudit(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            _write(checkpoint / "refutab_checkpoint.json", {
                "context": context, "context_sha256": _hash(context),
                "files": _inventory(checkpoint, excluding=("refutab_checkpoint.json",)),
            })
            return control

    class PersistentTrainer(original_trainer):
        def __init__(self, *args: Any, **kwargs: Any):
            training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
            if training_args is None:
                raise ValueError("Original TAR did not supply TrainingArguments")
            training_args.output_dir = str(output_directory / "hf")
            training_args.logging_dir = str(output_directory / "hf/logs")
            training_args.restore_callback_states_from_checkpoint = True
            callbacks = list(kwargs.get("callbacks") or ())
            callbacks.append(CheckpointAudit())
            kwargs["callbacks"] = callbacks
            audit["training_arguments"] = training_args.to_dict()
            super().__init__(*args, **kwargs)

        def train(self, *args: Any, **kwargs: Any):
            if args or "resume_from_checkpoint" in kwargs:
                raise ValueError("Original TAR unexpectedly supplied a resume argument")
            try:
                completed = _completed_training(output_directory, context)
                if completed is not None:
                    # HF may retain only its selected-best checkpoint after a
                    # completed loop. Restore the selected adapter, rather than
                    # treating that older best epoch as the interrupted last.
                    from transformers import TrainerState
                    _restore_completed_backbone(self.model.backbone, output_directory / "selected_backbone")
                    self.state = TrainerState(**completed["trainer_state"])
                    audit["training_status"] = "completed"
                    audit["restored_completed_training_without_optimizer_steps"] = True
                    return SimpleNamespace()
                result = super().train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None, **kwargs)
                audit["training_status"] = "completed"
                selected = output_directory / "selected_backbone"
                pending = output_directory / f".selected_backbone-{uuid.uuid4().hex}"
                pending.mkdir(parents=True, exist_ok=False)
                self.model.backbone.save_pretrained(pending / "adapter", safe_serialization=True, save_embedding_layers=False)
                state = asdict(self.state)
                _write(pending / "metadata.json", {
                    "context": context, "context_sha256": _hash(context),
                    "trainer_state": state, "trainer_state_sha256": _hash(state),
                    "files": _inventory(pending),
                })
                pending.rename(selected)
                return result
            except BaseException as exc:
                audit["training_status"] = "failed"
                audit["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                audit["trainer_state"] = asdict(self.state)
                _write(output_directory / "trainer_audit.json", audit)

    with patch.object(transformers.AutoModel, "from_pretrained", side_effect=model_loader), \
         patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=tokenizer_loader), \
         patch.object(fine_module, "Trainer", PersistentTrainer), \
         patch.object(split_module, "split_to_val", side_effect=observe_split), \
         patch.object(text_module, "discretize_numerical", side_effect=observe_discretize):
        yield


class TarRepresentation:
    """Selected shared TAR backbone with one unchanged sklearn PCA per field."""

    def __init__(self, wrappers: Mapping[str, Any], text_columns: Sequence[str], *,
                 train_ids: Sequence[RowId], device: str,
                 provenance: Mapping[str, Any]) -> None:
        self.text_columns = _columns(text_columns, "text_columns")
        self.train_ids = normalize_row_ids(train_ids)
        if not self.text_columns or len(self.train_ids) < 30 or tuple(wrappers) != self.text_columns:
            raise ValueError("TAR requires ordered fields, 30 fitting rows and aligned wrappers")
        self.wrappers = dict(wrappers)
        self.device = str(torch.device(device))
        self._provenance = copy.deepcopy(dict(provenance))
        models = [wrapper.model for wrapper in self.wrappers.values()]
        if any(model is not models[0] for model in models):
            raise ValueError("Historical TAR must share one tuned backbone across fields")
        self.model, self.tokenizer = models[0], next(iter(self.wrappers.values())).tokenizer
        self.model.requires_grad_(False)
        self.model.eval()
        for field, wrapper in self.wrappers.items():
            pca = wrapper.encoder
            if (wrapper.col_name != field or pca.n_components != 30 or pca.whiten
                    or pca.components_.shape != (30, 384) or pca.mean_.shape != (384,)
                    or not np.isfinite(pca.components_).all() or not np.isfinite(pca.mean_).all()):
                raise ValueError("Invalid original TAR PCA state or field alignment")
        self.pca = SimpleNamespace(
            fit_ids=self.train_ids, text_columns=self.text_columns,
            components_=np.stack([w.encoder.components_ for w in self.wrappers.values()]),
            means_=np.stack([w.encoder.mean_ for w in self.wrappers.values()]),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
            raise ValueError("TAR transform requires a DataFrame without duplicate columns")
        normalize_row_ids(frame.index)
        if set(self.text_columns) - set(frame.columns):
            raise ValueError("TAR transform is missing a declared text field")
        self.model.requires_grad_(False)
        self.model.eval()
        if len(frame) == 0:
            return np.empty((0, len(self.text_columns) * 30), dtype=np.float32)
        blocks = []
        for field in self.text_columns:
            # Preserve the source's astype(str).fillna order, including its
            # literal missing-value strings; changing this would alter TAR.
            texts = frame[field].astype(str).fillna("").tolist()
            wrapper = self.wrappers[field]
            block = wrapper.encoder.transform(wrapper.encode_texts(texts, torch.device(self.device)))
            if block.shape != (len(frame), 30) or not np.isfinite(block).all():
                raise ValueError("TAR produced invalid PCA embeddings")
            blocks.append(block)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {"format_version": 1, "kind": "historical_joint_tar",
                "text_columns": list(self.text_columns), "train_ids": list(self.train_ids),
                "pca_fit_ids": list(self.train_ids), "input_dimension": 384,
                "field_output_dimension": 30, "device": self.device,
                "prefix": "passage: {field}: {value}", "encoding_batch_size": 32,
                "max_length": 512, "missing_transform": "astype(str).fillna('')",
                "pooling": "attention-mask mean then L2 normalization",
                "inference_teacher": False, "inference_temporary_head": False,
                "provenance": copy.deepcopy(self._provenance)}

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(directory / "adapter", safe_serialization=True, save_embedding_layers=False)
        self.tokenizer.save_pretrained(directory / "tokenizer")
        arrays = {}
        for index, wrapper in enumerate(self.wrappers.values()):
            arrays[f"components_{index}"] = wrapper.encoder.components_
            arrays[f"mean_{index}"] = wrapper.encoder.mean_
            arrays[f"explained_variance_{index}"] = wrapper.encoder.explained_variance_
            arrays[f"explained_variance_ratio_{index}"] = wrapper.encoder.explained_variance_ratio_
        np.savez_compressed(directory / "pca.npz", **arrays)
        metadata = self.metadata()
        metadata["files"] = _inventory(directory)
        _write(directory / "metadata.json", metadata)

    @classmethod
    def load(cls, directory: str | Path, *, local_model_path: str | Path,
             device: str = "cpu") -> TarRepresentation:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("format_version") != 1 or metadata.get("kind") != "historical_joint_tar"
                or metadata.get("input_dimension") != 384 or metadata.get("field_output_dimension") != 30
                or metadata.get("inference_teacher") is not False or metadata.get("inference_temporary_head") is not False
                or metadata.get("train_ids") != metadata.get("pca_fit_ids")):
            raise ValueError("Invalid TAR representation format or fitting provenance")
        if metadata.get("files") != _inventory(directory, excluding=("metadata.json",)):
            raise ValueError("TAR representation checksum or file inventory mismatch")
        local, hashes = _verify_assets(local_model_path)
        provenance = metadata["provenance"]
        if (provenance.get("upstream_commit") != UPSTREAM_SHA
                or provenance.get("model_identifier") != E5_IDENTIFIER
                or provenance.get("model_revision") != E5_REVISION
                or provenance.get("model_files") != hashes):
            raise ValueError("TAR representation model/source provenance mismatch")
        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel
        base = AutoModel.from_pretrained(local, local_files_only=True, use_safetensors=True)
        model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False, local_files_only=True)
        model.to(torch.device(device))
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
        # These unchanged source wrappers implement the exact inference format
        # and PCA path. No tuned fit function or head is reconstructed here.
        text_module, _, _ = _source_modules()
        wrappers = {}
        fields = _columns(metadata["text_columns"], "text_columns")
        with np.load(directory / "pca.npz", allow_pickle=False) as arrays:
            expected = {f"{part}_{i}" for i in range(len(fields))
                        for part in ("components", "mean", "explained_variance", "explained_variance_ratio")}
            if set(arrays.files) != expected:
                raise ValueError("Invalid TAR PCA array inventory")
            for index, field in enumerate(fields):
                pca = PCA(n_components=30, random_state=42)
                pca.components_ = arrays[f"components_{index}"].copy()
                pca.mean_ = arrays[f"mean_{index}"].copy()
                pca.n_components_, pca.n_features_in_ = 30, 384
                pca.explained_variance_ = arrays[f"explained_variance_{index}"].copy()
                pca.explained_variance_ratio_ = arrays[f"explained_variance_ratio_{index}"].copy()
                if (pca.explained_variance_.shape != (30,) or pca.explained_variance_ratio_.shape != (30,)
                        or not np.isfinite(pca.explained_variance_).all()
                        or np.isinf(pca.explained_variance_ratio_).any()
                        or (pca.explained_variance_ < 0).any() or (pca.explained_variance_ratio_ < 0).any()):
                    raise ValueError("Invalid TAR PCA variance state")
                wrappers[field] = text_module.E5ColumnEncoder(model, tokenizer, pca, field)
        return cls(wrappers, fields, train_ids=metadata["train_ids"], device=device, provenance=provenance)


def fit_tar_representation(train_features: pd.DataFrame, text_columns: Sequence[str],
                           targets: pd.Series, *, task: str, train_ids: Sequence[RowId],
                           device: str, local_model_path: str | Path,
                           output_directory: str | Path,
                           training_context: Mapping[str, Any],
                           resume_checkpoint: str | Path | None = None,
                           seed: int = 42) -> TarRepresentation:
    """Fit one source TAR backbone/PCA using only caller-authorized fitting rows.

    ``targets`` must already be globally encoded class indices, or regression
    values. There is deliberately no outer-validation/test fitting argument.
    Training is GPU-only at the exact historical budget; OOM propagates.
    """
    fields = _columns(text_columns, "text_columns")
    ids = normalize_row_ids(train_ids, context="TAR fitting row IDs")
    if (not isinstance(train_features, pd.DataFrame) or train_features.columns.has_duplicates
            or not fields or set(fields) - set(train_features.columns)
            or len(ids) < 30 or normalize_row_ids(train_features.index) != ids):
        raise ValueError("TAR fitting frame, ordered fields and at least 30 row IDs must align")
    if not isinstance(targets, pd.Series) or normalize_row_ids(targets.index) != ids or targets.isna().any():
        raise ValueError("TAR targets must be a nonmissing aligned Series")
    if task not in {"regression", "binary", "multiclass"} or type(seed) is not int or seed != 42:
        raise ValueError("Historical TAR requires a supported task and fixed learner seed 42")
    values = targets.to_numpy()
    if task == "regression":
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("TAR regression targets must be finite")
        d_output = 1
    else:
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("TAR classification targets must use global integer class indices")
        classes = sorted(set(int(value) for value in values))
        if classes != list(range(len(classes))) or (task == "binary" and len(classes) != 2) or len(classes) < 2:
            raise ValueError("TAR class indices must be contiguous from zero and match the task")
        d_output = len(classes)
    selected_device = _training_device(device)
    local, hashes = _verify_assets(local_model_path)
    text_module, fine_module, split_module = _source_modules()
    if not training_context or not isinstance(training_context, Mapping):
        raise ValueError("TAR needs an explicit nonempty training context")
    context = json.loads(_json({"caller": dict(training_context), "upstream_commit": UPSTREAM_SHA,
                               "model_files": hashes, "text_columns": list(fields),
                               "train_ids": list(ids), "task": task,
                               "text_features_sha256": _hash([[_typed_cell(value) for value in row]
                                                              for row in train_features.loc[:, list(fields)].itertuples(index=False, name=None)]),
                               "targets_sha256": _hash([_typed_cell(value) for value in targets.tolist()]),
                               "training_kwargs": TAR_TRAIN_KWARGS}))
    output = Path(output_directory).resolve()
    resume = Path(resume_checkpoint).resolve() if resume_checkpoint is not None else None
    if resume:
        _resume_check(resume, context)
    if output.exists():
        prior = output / "training_context.json"
        if not resume or not prior.exists() or json.loads(prior.read_text(encoding="utf-8")) != context:
            raise FileExistsError("Refusing to reuse a TAR output directory without a matching resume context")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "training_context.json", context)
    if (output / "representation").exists():
        raise FileExistsError("A completed TAR representation already exists; load it instead of retraining")
    original_resume = resume
    if resume:
        resume = _copy_resume(resume, output, context)
    audit: dict[str, Any] = {"training_context_sha256": _hash(context),
                            "resume_checkpoint_source": str(original_resume) if original_resume else None,
                            "resume_checkpoint": str(resume) if resume else None}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        with _runtime_scope(text_module, fine_module, split_module,
                            local_model_path=local, output_directory=output,
                            context=context, audit=audit, resume_checkpoint=resume):
            wrappers = text_module.fit_text_encoders_tuned(
                x=train_features.copy(deep=True), text_features_list=list(fields),
                device=selected_device, y=targets.copy(deep=True),
                e5_train_kwargs=dict(TAR_TRAIN_KWARGS), is_cls=task != "regression",
                d_output=d_output, e5_model_name=E5_IDENTIFIER,
                pca_components=30, no_pca=False)
        split = audit.get("supervised_split", {})
        fitted = normalize_row_ids(split.get("fit_ids", ()))
        validation = normalize_row_ids(split.get("validation_ids", ()))
        if (not fitted or not validation or set(fitted) & set(validation)
                or set(fitted) | set(validation) != set(ids)
                or split.get("input_ids") != list(ids) or split.get("fold_argument") != -1):
            raise ValueError("Original TAR supervised split audit is incomplete or escaped fitting rows")
        provenance = {"upstream_commit": UPSTREAM_SHA, "model_identifier": E5_IDENTIFIER,
                      "model_revision": E5_REVISION, "model_files": hashes,
                      "training_kwargs": dict(TAR_TRAIN_KWARGS), "task": task,
                      "training_context": context, "audit": audit,
                      "selected_representation_directory": str(output / "representation"),
                      "regression_target_discretization": "unchanged source qcut up to 20 quantile bins" if task == "regression" else None}
        representation = TarRepresentation(wrappers, fields, train_ids=ids,
                                           device=str(selected_device), provenance=provenance)
        representation.save(output / "representation")
        return representation
    finally:
        _write(output / "trainer_audit.json", audit)
