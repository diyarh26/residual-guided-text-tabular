"""Option A training boundaries and teacher-free prediction bundles.

The temporary supervised representation is fitted on inner training rows.
It is not a cross-fitted embedding; held-out validation selects the gate.
No outer test row is encoded or predicted by ``fit_pipeline``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from .data import DatasetSchema, DatasetTable, normalize_row_ids, sha256_file
from .downstream import DownstreamConfig, DownstreamModel
from .features import FieldPCA, FrozenE5Config, FrozenE5Encoder, StructuredPreprocessor
from .fusion import (FusionConfig, FusionEncoder, FusionInputs, TrainConfig,
                     load_fusion, save_fusion, train_fusion)
from .splits import SplitAssignments
from .teacher import TeacherArtifacts, TeacherConfig, fit_oof_teacher, row_ids_sha256


class TextEncoder(Protocol):
    config: FrozenE5Config

    def encode(self, frame: pd.DataFrame, text_columns: tuple[str, ...]) -> np.ndarray: ...
    def metadata(self) -> dict[str, Any]: ...
    def state_sha256(self) -> str: ...
    def assert_frozen(self) -> None: ...


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _label_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool) or (type(value) is float and not np.isfinite(value)):
        raise ValueError("Class labels must be finite scalar strings, integers, floats or booleans")
    return type(value).__name__ + ":" + json.dumps(value, ensure_ascii=False, allow_nan=False)


def encode_targets(targets: pd.Series, task: str, classes: tuple = ()) -> np.ndarray:
    """Encode against an already saved training-only class order."""
    if task == "regression":
        result = np.asarray(targets, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Regression targets must be finite")
        return result
    mapping = {_label_key(value): index for index, value in enumerate(classes)}
    if len(mapping) != len(classes):
        raise ValueError("Class order contains duplicate typed labels")
    try:
        return np.asarray([mapping[_label_key(value)] for value in targets], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Target contains a class absent from the training class order") from error


def _training_classes(targets: pd.Series, task: str) -> tuple:
    if task == "regression":
        return ()
    observed = {_label_key(value): value.item() if isinstance(value, np.generic) else value for value in targets}
    classes = tuple(observed[key] for key in sorted(observed))
    if (task == "binary" and len(classes) != 2) or (task == "multiclass" and len(classes) < 3):
        raise ValueError("Training class count does not match the declared task")
    return classes


@dataclass(frozen=True)
class PipelineConfig:
    training: TrainConfig = field(default_factory=TrainConfig)
    hidden_dim: int = 64
    dropout: float = 0.1
    category_embedding_dim: int = 8
    structured_dim: int = 16
    downstream_estimators: int = 100
    downstream_threads: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.training, TrainConfig):
            raise TypeError("training must be a TrainConfig")
        for name in ("hidden_dim", "category_embedding_dim", "structured_dim", "downstream_estimators", "downstream_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")

    @classmethod
    def from_dict(cls, value: dict) -> "PipelineConfig":
        value = dict(value)
        value["training"] = TrainConfig(**value.get("training", {}))
        return cls(**value)


def load_teacher_for_table(directory: str | Path, table: DatasetTable,
                           assignments: SplitAssignments) -> TeacherArtifacts:
    """Reuse a Phase 2 run only after binding its complete manifest to this data."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest["data"]["typed_content_sha256"] != table.fingerprint():
        raise ValueError("Teacher run does not match this dataset or is incomplete")
    if manifest["split_assignments_sha256"] != assignments.fingerprint():
        raise ValueError("Teacher run split assignments differ")
    for name, digest in manifest["artifacts"]["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
            raise ValueError("Teacher run artifact checksum/path mismatch")
    result = TeacherArtifacts.load(directory / "teacher")
    _check_teacher(result, table, assignments)
    return result


def _check_teacher(teacher: TeacherArtifacts, table: DatasetTable, assignments: SplitAssignments) -> None:
    teacher.validate()
    if (teacher.config.task != table.schema.task or teacher.columns != table.schema.structured_columns
            or teacher.categorical_columns != table.schema.categorical_columns
            or teacher.train_ids != assignments.train_ids or teacher.validation_ids != assignments.validation_ids):
        raise ValueError("Teacher task, feature order or row alignment differs from this pipeline")


class ReFuTabPipeline:
    """Frozen representation plus downstream learner; no teacher or correction head."""

    def __init__(self, *, schema: DatasetSchema, config: PipelineConfig, encoder: TextEncoder,
                 pca: FieldPCA, structured: StructuredPreprocessor, fusion: FusionEncoder,
                 downstream: DownstreamModel, class_order: tuple, training: dict):
        self.schema, self.config, self.encoder = schema, config, encoder
        self.pca, self.structured, self.fusion = pca, structured, fusion
        self.downstream, self.class_order, self.training = downstream, tuple(class_order), training
        self._validate_components()

    def _validate_components(self) -> None:
        ids = normalize_row_ids(self.training["train_ids"], context="Saved pipeline fitting IDs")
        val_ids = normalize_row_ids(self.training["validation_ids"], context="Saved pipeline validation IDs")
        if not ids or not val_ids or set(ids) & set(val_ids):
            raise ValueError("Pipeline fitting/validation scopes must be nonempty and disjoint")
        if (row_ids_sha256(ids) != self.training["train_ids_sha256"]
                or row_ids_sha256(val_ids) != self.training["validation_ids_sha256"]):
            raise ValueError("Pipeline row-ID provenance hashes differ")
        if (tuple(self.pca.fit_ids) != ids or tuple(self.structured.fit_ids) != ids or tuple(self.downstream.fit_ids) != ids):
            raise ValueError("Pipeline component fitting scopes differ")
        if (tuple(self.pca.text_columns) != self.schema.text_columns
                or tuple(self.structured.numerical_columns) != self.schema.numerical_columns
                or tuple(self.structured.categorical_columns) != self.schema.categorical_columns
                or tuple(self.downstream.columns) != self.schema.structured_columns
                or tuple(self.downstream.categorical_columns) != self.schema.categorical_columns):
            raise ValueError("Pipeline component feature order differs from the schema")
        if self.downstream.config.task != self.schema.task or self.downstream.fused_dimension != 30 * len(self.schema.text_columns):
            raise ValueError("Pipeline downstream task or feature dimension differs")
        if tuple(map(_label_key, self.downstream.class_order)) != tuple(map(_label_key, self.class_order)):
            raise ValueError("Pipeline class order differs from the downstream model")
        expected = FusionConfig(task=self.schema.task, n_text_fields=len(self.schema.text_columns),
                                n_numerical=len(self.schema.numerical_columns), category_cardinalities=self.structured.cardinalities,
                                n_classes=len(self.class_order), hidden_dim=self.config.hidden_dim, dropout=self.config.dropout,
                                category_embedding_dim=self.config.category_embedding_dim, structured_dim=self.config.structured_dim)
        if asdict(expected) != asdict(self.fusion.config):
            raise ValueError("Pipeline fusion task, field/category dimensions or configuration differs")

    def _features(self, frame: pd.DataFrame, *, allow_in_sample: bool) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != self.schema.feature_columns:
            raise ValueError("Prediction requires exactly the saved feature columns in schema order")
        ids = normalize_row_ids(frame.index, context="Inference row IDs")
        if not ids:
            raise ValueError("Prediction input must not be empty")
        fit_ids = tuple(self.training["train_ids"])
        if type(ids[0]) is not type(fit_ids[0]):
            raise ValueError("Prediction row ID type differs from training")
        if not allow_in_sample and set(ids) & set(fit_ids):
            raise ValueError("Prediction overlaps training IDs; allow_in_sample is diagnostic only")
        return frame.copy(deep=True)

    def transform(self, frame: pd.DataFrame, *, allow_in_sample: bool = False) -> np.ndarray:
        frame = self._features(frame, allow_in_sample=allow_in_sample)
        self.encoder.assert_frozen()
        embedding = self.encoder.encode(frame.loc[:, list(self.schema.text_columns)], self.schema.text_columns)
        text = self.pca.transform(embedding)
        numeric, categorical = self.structured.transform(frame.loc[:, list(self.schema.structured_columns)])
        self.fusion.eval()
        device = next(self.fusion.parameters()).device
        with torch.inference_mode():
            result = self.fusion.fuse(torch.as_tensor(text, dtype=torch.float32, device=device),
                                      torch.as_tensor(numeric, dtype=torch.float32, device=device),
                                      torch.as_tensor(categorical, dtype=torch.long, device=device))
        return result.cpu().numpy().copy()

    def predict(self, frame: pd.DataFrame, *, allow_in_sample: bool = False) -> np.ndarray:
        """Return original-scale regression values or saved-order class probabilities."""
        features = self._features(frame, allow_in_sample=allow_in_sample)
        fused = self.transform(features, allow_in_sample=allow_in_sample)
        return self.downstream.predict(features.loc[:, list(self.schema.structured_columns)], fused)

    def save(self, directory: str | Path) -> Path:
        """Save a new inference bundle, excluding teacher and temporary head weights."""
        self._validate_components()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.pca.save(directory / "pca")
        self.structured.save(directory / "structured")
        save_fusion(self.fusion, directory / "fusion.pt")
        self.downstream.save(directory / "downstream")
        metadata = {"format_version": 1, "schema": self.schema.to_dict(), "config": asdict(self.config),
                    "class_order": list(self.class_order), "encoder_config": asdict(self.encoder.config),
                    "encoder_metadata": self.encoder.metadata(), "training": self.training,
                    "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scikit-learn", "lightgbm")},
                    "files_sha256": {path.relative_to(directory).as_posix(): sha256_file(path)
                                     for path in sorted(directory.rglob("*")) if path.is_file()}}
        (directory / "pipeline.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, encoder: TextEncoder | None = None,
             model_path: str | Path | None = None) -> "ReFuTabPipeline":
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported pipeline artifact format")
        expected = set(metadata["files_sha256"])
        actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path != directory / "pipeline.json"}
        if expected != actual:
            raise ValueError("Pipeline artifact inventory differs from its manifest")
        for name, digest in metadata["files_sha256"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or sha256_file(path) != digest:
                raise ValueError("Pipeline artifact checksum/path mismatch")
        if encoder is None:
            encoder_config = FrozenE5Config(**metadata["encoder_config"])
            if model_path is not None:
                encoder_config = replace(encoder_config, model_path=str(model_path))
            encoder = FrozenE5Encoder(encoder_config)
        # Relocating verified pretrained files/device is allowed; semantic settings are fixed.
        original = dict(metadata["encoder_config"])
        supplied = asdict(encoder.config)
        for key in ("model_path", "device", "batch_size"):
            original.pop(key, None)
            supplied.pop(key, None)
        if original != supplied or encoder.state_sha256() != metadata["training"]["encoder_state_sha256"]:
            raise ValueError("Encoder revision, settings or state differs from the saved bundle")
        encoder.assert_frozen()
        schema = DatasetSchema.from_dict(metadata["schema"])
        pca = FieldPCA.load(directory / "pca")
        structured = StructuredPreprocessor.load(directory / "structured")
        fusion = load_fusion(directory / "fusion.pt")
        downstream = DownstreamModel.load(directory / "downstream")
        training = metadata["training"]
        if tuple(pca.fit_ids) != tuple(training["train_ids"]) or tuple(structured.fit_ids) != tuple(training["train_ids"]):
            raise ValueError("Saved feature fitting scopes differ from the training provenance")
        if tuple(pca.text_columns) != schema.text_columns:
            raise ValueError("Saved PCA text field order differs from schema")
        return cls(schema=schema, config=PipelineConfig.from_dict(metadata["config"]), encoder=encoder,
                   pca=pca, structured=structured, fusion=fusion, downstream=downstream,
                   class_order=tuple(metadata["class_order"]), training=training)


def fit_pipeline(table: DatasetTable, assignments: SplitAssignments, encoder: TextEncoder, *,
                 config: PipelineConfig | None = None, teacher_config: TeacherConfig | None = None,
                 teacher: TeacherArtifacts | None = None, teacher_dataset_fingerprint: str | None = None,
                 checkpoint_path: str | Path | None = None, resume: bool = False,
                 training_context: dict | None = None, teacher_output: str | Path | None = None) -> ReFuTabPipeline:
    """Fit on A, select the gate on V, and fit the downstream learner on A only.

    Reusing teacher objects requires the caller's verified dataset binding;
    ``load_teacher_for_table`` validates a full Phase 2 run for that purpose.
    No test-feature or test-target view is requested in this function.
    """
    config = config or PipelineConfig()
    assignments.validate(table)
    schema = table.schema
    if not schema.text_columns or not schema.structured_columns:
        raise ValueError("Option A requires text and structured feature columns")
    train_ids, validation_ids = assignments.train_ids, assignments.validation_ids
    train = table.features(train_ids)
    validation = table.features(validation_ids)
    train_y = table.targets(train_ids)
    training_fingerprint = table.fingerprint()
    teacher_metadata: dict = {"used_for_objective": config.training.objective == "residual"}
    if config.training.objective == "residual":
        if teacher is None:
            teacher = fit_oof_teacher(train.loc[:, list(schema.structured_columns)], train_y,
                                      categorical_columns=schema.categorical_columns, train_ids=train_ids,
                                      validation_X=validation.loc[:, list(schema.structured_columns)],
                                      validation_ids=validation_ids,
                                      config=teacher_config or TeacherConfig(task=schema.task))
        elif teacher_dataset_fingerprint != training_fingerprint:
            raise ValueError("Reused teacher requires the matching verified dataset fingerprint")
        _check_teacher(teacher, table, assignments)
        if teacher_output is not None:
            teacher.save(teacher_output)
        classes = teacher.class_order
        train_offsets, validation_offsets = teacher.oof_offsets, teacher.validation_offsets
        teacher_metadata.update({"config": asdict(teacher.config),
                                 "metadata_sha256": hashlib.sha256(_canonical(teacher.metadata())).hexdigest(),
                                 "oof_offsets_sha256": hashlib.sha256(np.asarray(train_offsets).tobytes()).hexdigest(),
                                 "validation_offsets_sha256": hashlib.sha256(np.asarray(validation_offsets).tobytes()).hexdigest()})
    else:
        if teacher is not None:
            raise ValueError("The direct-target control takes no teacher artifacts")
        classes = _training_classes(train_y, schema.task)
        train_offsets, validation_offsets = None, None
    encoded_y = encode_targets(train_y, schema.task, classes)
    validation_y = encode_targets(table.targets(validation_ids), schema.task, classes)
    encoder.assert_frozen()
    encoder_hash = encoder.state_sha256()
    train_embedding = encoder.encode(train.loc[:, list(schema.text_columns)], schema.text_columns)
    validation_embedding = encoder.encode(validation.loc[:, list(schema.text_columns)], schema.text_columns)
    pca = FieldPCA(schema.text_columns, seed=config.training.seed).fit(train_embedding, train_ids)
    train_text, validation_text = pca.transform(train_embedding), pca.transform(validation_embedding)
    structured = StructuredPreprocessor(schema.numerical_columns, schema.categorical_columns).fit(
        train.loc[:, list(schema.structured_columns)], train_ids)
    train_numeric, train_categorical = structured.transform(train.loc[:, list(schema.structured_columns)])
    val_numeric, val_categorical = structured.transform(validation.loc[:, list(schema.structured_columns)])
    model_config = FusionConfig(task=schema.task, n_text_fields=len(schema.text_columns),
                                 n_numerical=len(schema.numerical_columns), category_cardinalities=structured.cardinalities,
                                 n_classes=len(classes), hidden_dim=config.hidden_dim, dropout=config.dropout,
                                 category_embedding_dim=config.category_embedding_dim, structured_dim=config.structured_dim)
    context = {"dataset_fingerprint": training_fingerprint, "schema": schema.to_dict(),
               "split_fingerprint": assignments.fingerprint(), "train_ids": list(train_ids),
               "validation_ids": list(validation_ids), "class_order": list(classes),
               "encoder_state_sha256": encoder_hash, "encoder_config": asdict(encoder.config),
               "teacher": teacher_metadata, "caller": training_context or {}}
    fitted = train_fusion(model_config, config.training,
                          train=FusionInputs(train_text, train_numeric, train_categorical, encoded_y, train_offsets),
                          validation=FusionInputs(validation_text, val_numeric, val_categorical, validation_y, validation_offsets),
                          checkpoint_path=checkpoint_path, resume=resume, training_context=context)
    fusion = FusionEncoder.from_model(fitted.model)
    fusion.eval()
    device = next(fusion.parameters()).device
    with torch.inference_mode():
        fused_train = fusion.fuse(torch.as_tensor(train_text, dtype=torch.float32, device=device),
                                  torch.as_tensor(train_numeric, dtype=torch.float32, device=device),
                                  torch.as_tensor(train_categorical, dtype=torch.long, device=device)).cpu().numpy().copy()
    downstream = DownstreamModel(DownstreamConfig(task=schema.task, seed=config.training.seed,
                                                  n_estimators=config.downstream_estimators, num_threads=config.downstream_threads))
    downstream.fit(train.loc[:, list(schema.structured_columns)], fused_train, encoded_y,
                   categorical_columns=schema.categorical_columns, class_order=classes, fit_ids=train_ids)
    encoder.assert_frozen()
    if encoder.state_sha256() != encoder_hash:
        raise RuntimeError("Frozen E5 weights changed during pipeline training")
    training = {"dataset_fingerprint": training_fingerprint, "split_fingerprint": assignments.fingerprint(),
                "train_ids": list(train_ids), "validation_ids": list(validation_ids),
                "train_ids_sha256": row_ids_sha256(train_ids), "validation_ids_sha256": row_ids_sha256(validation_ids),
                "encoder_state_sha256": encoder_hash, "teacher": teacher_metadata,
                "history": fitted.history, "best_epoch": fitted.best_epoch, "completed_epoch": fitted.completed_epoch,
                "fusion_training": fitted.training,
                "checks": {"training_only_pca": tuple(pca.fit_ids) == train_ids,
                           "training_only_structured_preprocessing": tuple(structured.fit_ids) == train_ids,
                           "frozen_e5": True, "inference_excludes_teacher_and_correction_head": True,
                           "outer_test_predictions": 0},
                "representation_scope": "supervised in-sample training embeddings; not cross-fitted",
                "caller": training_context or {}}
    return ReFuTabPipeline(schema=schema, config=config, encoder=encoder, pca=pca, structured=structured,
                           fusion=fusion, downstream=downstream, class_order=classes, training=training)
