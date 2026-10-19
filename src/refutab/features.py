"""Frozen E5 and fitting-partition-only feature transforms for Option A.

These components do not consume targets. Callers supply fitting rows explicitly;
their IDs, field order and learned states persist with checked native arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from .data import RowId, _typed_cell, normalize_row_ids, sha256_file

E5_IDENTIFIER = "intfloat/e5-small-v2"
E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
E5_FILE_HASHES = {
    "config.json": "5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f",
    "model.safetensors": "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "ae83fa6ca0333117ff12606020af925d648667ef70d92ff7f27d781ba0ca4544",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def _columns(values: Sequence[str], context: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be an ordered column sequence")
    result = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{context} must contain unique nonempty column names")
    return result


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _write_bundle(directory: str | Path, metadata: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "state.npz", **arrays)
    metadata = {**metadata, "state_sha256": sha256_file(directory / "state.npz")}
    (directory / "metadata.json").write_text(_json(metadata) + "\n", encoding="utf-8")


def _read_bundle(directory: str | Path, kind: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("kind") != kind:
        raise ValueError(f"Unsupported {kind} artifact format")
    if {path.name for path in directory.iterdir()} != {"metadata.json", "state.npz"}:
        raise ValueError(f"Unexpected {kind} artifact file inventory")
    if sha256_file(directory / "state.npz") != metadata.get("state_sha256"):
        raise ValueError(f"{kind} state checksum mismatch")
    with np.load(directory / "state.npz", allow_pickle=False) as handle:
        arrays = {name: handle[name].copy() for name in handle.files}
    return metadata, arrays


def _frame(frame: pd.DataFrame, columns: tuple[str, ...], *, exact: bool = True) -> tuple[RowId, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate feature columns are forbidden")
    if exact and tuple(frame.columns) != columns:
        raise ValueError(f"Feature columns/order must exactly match {columns}")
    if missing := set(columns) - set(frame.columns):
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return normalize_row_ids(frame.index, context="Feature row IDs")


def _fit_ids(ids: Sequence[RowId], length: int) -> tuple[RowId, ...]:
    result = normalize_row_ids(ids, context="Fitting row IDs")
    if not result or len(result) != length:
        raise ValueError("Fitting row IDs must be nonempty and match the fitting data length")
    return result


@dataclass(frozen=True)
class FrozenE5Config:
    model_path: str
    identifier: str = E5_IDENTIFIER
    revision: str = E5_REVISION
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.identifier != E5_IDENTIFIER or self.revision != E5_REVISION:
            raise ValueError("Option A requires the pinned E5-small-v2 identifier/revision")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be a nonempty local path")
        for name in ("max_length", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_length > 512:
            raise ValueError("Pinned E5 supports max_length at most 512")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be an explicit torch device string")


class FrozenE5Encoder:
    """Locally load verified pinned weights, with paired injection for unit tests.

    Encoding is stateless: each requested field keeps its own position, using
    ``passage: {field}: {value}``, masked mean pooling and L2 normalization.
    """

    def __init__(self, config: FrozenE5Config, *, model: Any = None, tokenizer: Any = None,
                 expected_hashes: Mapping[str, str] | None = None) -> None:
        self.config = config
        self._injected = model is not None or tokenizer is not None
        if (model is None) != (tokenizer is None):
            raise ValueError("Test injection requires both model and tokenizer")
        self.file_hashes: dict[str, str] = {}
        if model is None:
            hashes = dict(E5_FILE_HASHES if expected_hashes is None else expected_hashes)
            if hashes != E5_FILE_HASHES:
                raise ValueError("Expected hashes must match all six pinned E5 files")
            model_path = Path(config.model_path)
            # Transformers may consume optional tokenizer files in addition to
            # the six selected assets. Reject those rather than silently change
            # the encoding while claiming the immutable revision was verified.
            allowed = set(hashes) | {"provenance.json", "README.md"}
            if extras := {path.name for path in model_path.iterdir() if path.is_file()} - allowed:
                raise ValueError(f"Unexpected unverified E5 asset files: {sorted(extras)}")
            for name, expected in hashes.items():
                actual = sha256_file(model_path / name)
                if actual != expected:
                    raise ValueError(f"Pinned E5 file checksum mismatch: {name}")
                self.file_hashes[name] = actual
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
        self.model, self.tokenizer = model, tokenizer
        self.model.to(torch.device(config.device))
        self.model.requires_grad_(False)
        self.model.eval()
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("Frozen E5 and every submodule must remain in evaluation mode")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("Frozen E5 parameters must have requires_grad=False and no gradients")

    def state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(_json([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {"format_version": 1, "kind": "frozen_e5", "config": asdict(self.config),
                "dimension": 384, "prefix": "passage: {field}: {value}", "missing_text": "",
                "padding": "max_length", "pooling": "attention-mask mean then L2 normalization",
                "verified_files": dict(self.file_hashes), "test_injection": self._injected}

    def encode(self, frame: pd.DataFrame, text_columns: Sequence[str]) -> np.ndarray:
        columns = _columns(text_columns, "text_columns")
        if not columns:
            raise ValueError("At least one text field is required")
        _frame(frame, columns, exact=False)
        self.assert_frozen()
        result = np.empty((len(frame), len(columns), 384), dtype=np.float32)
        for field_index, column in enumerate(columns):
            values = []
            for value in frame[column].tolist():
                # Scalar tagging rejects lists/dicts, and consistently identifies
                # None/pd.NA/NaN without silently encoding the word 'nan'.
                tagged = _typed_cell(value)
                values.append(f"passage: {column}: {'' if tagged[0] == 'missing' else str(value)}")
            for start in range(0, len(values), self.config.batch_size):
                batch = self.tokenizer(values[start:start + self.config.batch_size],
                                       max_length=self.config.max_length, truncation=True,
                                       padding="max_length", return_tensors="pt")
                batch = {name: value.to(self.config.device) for name, value in batch.items()}
                with torch.inference_mode():
                    output = self.model(**batch)
                    hidden = output.last_hidden_state
                    mask = batch["attention_mask"].bool()
                    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or hidden.shape[-1] != 384:
                        raise ValueError("Pinned E5 must return (batch, tokens, 384) hidden states")
                    if not mask.any(dim=1).all():
                        raise ValueError("Every text must contain at least one attended token")
                    pooled = hidden.masked_fill(~mask[..., None], 0.0).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                    pooled = F.normalize(pooled, p=2, dim=1)
                result[start:start + len(pooled), field_index] = pooled.cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("E5 produced non-finite embeddings")
        self.assert_frozen()
        return result


class FieldPCA:
    """Independent 384-to-30 PCA per ordered text field, fitted exactly once."""

    def __init__(self, text_columns: Sequence[str], seed: int = 42) -> None:
        self.text_columns = _columns(text_columns, "text_columns")
        if not self.text_columns:
            raise ValueError("At least one text field is required")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.seed = seed
        self.fit_ids: tuple[RowId, ...] = ()
        self.components_: np.ndarray | None = None
        self.means_: np.ndarray | None = None

    def _input(self, embeddings: np.ndarray) -> np.ndarray:
        value = np.asarray(embeddings, dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != (len(self.text_columns), 384):
            raise ValueError(f"Embeddings must have shape (rows, {len(self.text_columns)}, 384)")
        if not np.isfinite(value).all():
            raise ValueError("PCA input embeddings must be finite")
        return value

    def fit(self, embeddings: np.ndarray, train_ids: Sequence[RowId]) -> FieldPCA:
        if self.components_ is not None:
            raise RuntimeError("PCA is already fitted; create a new instance for a different partition")
        values = self._input(embeddings)
        ids = _fit_ids(train_ids, len(values))
        if len(values) < 30:
            raise ValueError("PCA-30 requires at least 30 fitting rows; dimensionality is never reduced silently")
        models = [PCA(n_components=30, random_state=self.seed).fit(values[:, index, :])
                  for index in range(len(self.text_columns))]
        self.components_ = np.stack([model.components_ for model in models])
        self.means_ = np.stack([model.mean_ for model in models])
        self.fit_ids = ids
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.means_ is None:
            raise RuntimeError("PCA must be fitted before transform")
        values = self._input(embeddings)
        result = np.einsum("ntd,tkd->ntk", values - self.means_[None, :, :], self.components_)
        if not np.isfinite(result).all():
            raise ValueError("PCA produced non-finite values")
        return result.astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before saving metadata")
        return {"format_version": 1, "kind": "field_pca", "text_columns": list(self.text_columns),
                "seed": self.seed, "input_dimension": 384, "output_dimension": 30,
                "fit_ids": list(self.fit_ids), "fitting_rows": len(self.fit_ids), "whiten": False}

    def save(self, directory: str | Path) -> None:
        _write_bundle(directory, self.metadata(), {"components": self.components_, "means": self.means_})

    @classmethod
    def load(cls, directory: str | Path) -> FieldPCA:
        metadata, arrays = _read_bundle(directory, "field_pca")
        instance = cls(metadata["text_columns"], metadata["seed"])
        instance.fit_ids = _fit_ids(metadata["fit_ids"], metadata["fitting_rows"])
        if len(instance.fit_ids) < 30 or metadata.get("input_dimension") != 384 or metadata.get("output_dimension") != 30 or metadata.get("whiten") is not False:
            raise ValueError("Invalid PCA dimensions/fitting provenance")
        if set(arrays) != {"components", "means"}:
            raise ValueError("Invalid PCA state inventory")
        instance.components_, instance.means_ = arrays["components"], arrays["means"]
        if instance.components_.shape != (len(instance.text_columns), 30, 384) or instance.means_.shape != (len(instance.text_columns), 384):
            raise ValueError("Invalid PCA saved state dimensions")
        if not np.isfinite(instance.components_).all() or not np.isfinite(instance.means_).all():
            raise ValueError("PCA saved state must be finite")
        return instance


class StructuredPreprocessor:
    """Train medians/scaling and typed vocabularies; missing=0, unseen=1."""

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        self.numerical_columns = _columns(numerical_columns, "numerical_columns")
        self.categorical_columns = _columns(categorical_columns, "categorical_columns")
        if set(self.numerical_columns) & set(self.categorical_columns):
            raise ValueError("A structured column cannot have two roles")
        self.columns = self.numerical_columns + self.categorical_columns
        self.fit_ids: tuple[RowId, ...] = ()
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.vocabularies_: tuple[dict[str, int], ...] = ()

    def _numbers(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            values = frame.loc[:, list(self.numerical_columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
        except (ValueError, TypeError) as exc:
            raise ValueError("Numerical structured fields must contain numbers or missing values") from exc
        if np.isinf(values).any():
            raise ValueError("Infinite numerical structured values are forbidden")
        return values

    @staticmethod
    def _token(value: Any) -> str | None:
        tagged = _typed_cell(value)
        return None if tagged[0] == "missing" else _json(tagged)

    def fit(self, frame: pd.DataFrame, train_ids: Sequence[RowId]) -> StructuredPreprocessor:
        if self.medians_ is not None:
            raise RuntimeError("Structured preprocessing is already fitted")
        observed_ids = _frame(frame, self.columns)
        ids = _fit_ids(train_ids, len(frame))
        if ids != observed_ids:
            raise ValueError("Fitting row IDs must exactly align with the feature frame index")
        values = self._numbers(frame)
        medians = np.array([np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.0
                            for column in values.T], dtype=np.float64)
        imputed = np.where(np.isnan(values), medians[None, :], values)
        means, scales = imputed.mean(axis=0), imputed.std(axis=0)
        scales[scales == 0] = 1.0
        if not np.isfinite(medians).all() or not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("Numerical fitting statistics overflowed")
        vocabularies = []
        for column in self.categorical_columns:
            tokens = sorted({token for value in frame[column].tolist() if (token := self._token(value)) is not None})
            vocabularies.append({token: index + 2 for index, token in enumerate(tokens)})
        self.fit_ids, self.medians_, self.means_, self.scales_ = ids, medians, means, scales
        self.vocabularies_ = tuple(vocabularies)
        return self

    @property
    def cardinalities(self) -> tuple[int, ...]:
        if self.medians_ is None:
            raise RuntimeError("Structured preprocessing must be fitted first")
        return tuple(len(vocabulary) + 2 for vocabulary in self.vocabularies_)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("Structured preprocessing must be fitted before transform")
        _frame(frame, self.columns)
        values = self._numbers(frame)
        numbers = ((np.where(np.isnan(values), self.medians_[None, :], values) - self.means_) / self.scales_).astype(np.float32)
        if not np.isfinite(numbers).all():
            raise ValueError("Numerical transformation overflowed")
        categories = np.empty((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for index, (column, vocabulary) in enumerate(zip(self.categorical_columns, self.vocabularies_)):
            categories[:, index] = [0 if (token := self._token(value)) is None else vocabulary.get(token, 1)
                                    for value in frame[column].tolist()]
        return numbers, categories

    def metadata(self) -> dict[str, Any]:
        if self.medians_ is None:
            raise RuntimeError("Structured preprocessing must be fitted before saving metadata")
        return {"format_version": 1, "kind": "structured_preprocessor",
                "numerical_columns": list(self.numerical_columns), "categorical_columns": list(self.categorical_columns),
                "fit_ids": list(self.fit_ids), "fitting_rows": len(self.fit_ids),
                "missing_category": 0, "unknown_category": 1, "all_missing_numeric_fallback": 0.0,
                "vocabularies": list(self.vocabularies_), "cardinalities": list(self.cardinalities)}

    def save(self, directory: str | Path) -> None:
        _write_bundle(directory, self.metadata(), {"medians": self.medians_, "means": self.means_, "scales": self.scales_})

    @classmethod
    def load(cls, directory: str | Path) -> StructuredPreprocessor:
        metadata, arrays = _read_bundle(directory, "structured_preprocessor")
        instance = cls(metadata["numerical_columns"], metadata["categorical_columns"])
        instance.fit_ids = _fit_ids(metadata["fit_ids"], metadata["fitting_rows"])
        if metadata.get("missing_category") != 0 or metadata.get("unknown_category") != 1 or metadata.get("all_missing_numeric_fallback") != 0.0:
            raise ValueError("Invalid reserved-value conventions")
        if set(arrays) != {"medians", "means", "scales"}:
            raise ValueError("Invalid structured state inventory")
        instance.medians_, instance.means_, instance.scales_ = arrays["medians"], arrays["means"], arrays["scales"]
        for array in (instance.medians_, instance.means_, instance.scales_):
            if array.shape != (len(instance.numerical_columns),) or not np.isfinite(array).all():
                raise ValueError("Invalid numerical preprocessing state")
        if (instance.scales_ <= 0).any():
            raise ValueError("Numerical scales must be positive")
        vocabularies = metadata["vocabularies"]
        if len(vocabularies) != len(instance.categorical_columns):
            raise ValueError("Categorical vocabulary count does not match schema")
        for vocabulary in vocabularies:
            if not isinstance(vocabulary, dict) or set(vocabulary.values()) != set(range(2, len(vocabulary) + 2)) or any(type(value) is not int for value in vocabulary.values()):
                raise ValueError("Invalid categorical vocabulary indices")
            for token in vocabulary:
                decoded = json.loads(token)
                if not isinstance(decoded, list) or not decoded or decoded[0] == "missing" or _json(decoded) != token:
                    raise ValueError("Invalid typed category token")
        instance.vocabularies_ = tuple(dict(value) for value in vocabularies)
        if tuple(metadata["cardinalities"]) != instance.cardinalities:
            raise ValueError("Categorical cardinalities disagree with vocabularies")
        return instance
