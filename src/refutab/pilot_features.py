"""Immutable frozen-text caching and float32 upstream-style PCA for pilots.

The cache consumes only named text fields and row identifiers. It never saves
labels or fitted statistics. Separate complete PCA states belong to individual
fitting partitions; target-adapted encoders cannot enter the shared cache.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

from .data import RowId, _typed_cell, normalize_row_ids, sha256_file
from .features import E5_FILE_HASHES, FrozenE5Encoder, _columns, _frame


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(value) + "\n")


def _versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "transformers", "tokenizers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _encoder_identity(encoder: Any) -> dict[str, Any]:
    metadata = json.loads(_canonical(encoder.metadata()))
    metadata.pop("cache", None)  # Runtime hit counters are not model semantics.
    if isinstance(metadata.get("config"), dict):
        metadata["config"].pop("model_path", None)
    config = asdict(encoder.config) if is_dataclass(encoder.config) else dict(encoder.config)
    config.pop("model_path", None)
    state = encoder.state_sha256()
    if not isinstance(state, str) or re.fullmatch(r"[0-9a-f]{64}", state) is None:
        raise ValueError("Encoder state must have a lowercase SHA-256 fingerprint")
    return {"metadata": metadata, "config": config, "state_sha256": state}


def _runtime_binding(encoder: Any) -> dict[str, Any]:
    device = str(encoder.config.device)
    gpu = None
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(torch.device(device))
        gpu = {"name": properties.name, "capability": [properties.major, properties.minor]}
    return {"python": platform.python_version(), "platform": sys.platform,
            "machine": platform.machine(), "packages": _versions(),
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "device": device, "gpu": gpu, "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def _embeddings(values: Any, rows: int, fields: int) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.float32 or result.shape != (rows, fields, 384) or not np.isfinite(result).all():
        raise ValueError("Text encoder must return finite float32 (rows, fields, 384) embeddings")
    return result


def _source_text(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    # Pinned Joint Frozen/TAR stringify first, then fill missing values. This
    # deliberately preserves literal 'nan'/'None' rather than Option A's empty
    # missing-text policy. The caller's frame remains unchanged.
    return pd.DataFrame({column: frame[column].astype(str).fillna("") for column in columns}, index=frame.index)


class CachedFrozenEncoder:
    """Reuse immutable, text-only row embeddings across blocks and folds.

    A shared root may contain several exact environment/model bindings. Within
    one binding, each append publishes one NPZ/JSON pack by directory rename.
    Unpublished ``.pending-*`` directories are retained after interruption and
    never read as completed packs. Concurrent writers are not supported.
    """

    def __init__(self, encoder: FrozenE5Encoder, directory: str | Path, *,
                 allow_test_encoder: bool = False) -> None:
        if type(allow_test_encoder) is not bool:
            raise TypeError("allow_test_encoder must be a boolean")
        metadata = encoder.metadata()
        if not allow_test_encoder and (not isinstance(encoder, FrozenE5Encoder)
                or metadata.get("kind") != "frozen_e5" or metadata.get("test_injection") is not False
                or metadata.get("verified_files") != E5_FILE_HASHES):
            raise ValueError("Shared cache requires verified frozen E5; fixtures require allow_test_encoder=True")
        encoder.assert_frozen()
        self.encoder, self.config = encoder, encoder.config
        self.binding = {"format_version": 1, "kind": "frozen_text_cache",
                        "encoder": _encoder_identity(encoder), "environment": _runtime_binding(encoder),
                        "allow_test_encoder": allow_test_encoder}
        self.binding_sha256 = _digest(self.binding)
        self.directory = Path(directory)
        self.namespace = self.directory / ("binding-" + self.binding_sha256)
        self.namespace.mkdir(parents=True, exist_ok=True)
        binding_path = self.namespace / "binding.json"
        if not binding_path.exists():
            # A missing binding beside existing packs is corruption, not a new cache.
            if list(self.namespace.iterdir()):
                raise ValueError("Cache namespace has entries but no binding metadata")
            temporary = self.namespace / (".binding-" + uuid4().hex + ".tmp")
            _write_json(temporary, self.binding)
            temporary.replace(binding_path)
        self._rows: dict[str, np.ndarray] = {}
        self._loaded: dict[str, tuple[str, str]] = {}
        self.hits = self.misses = 0
        self._refresh()

    def _refresh(self) -> None:
        if json.loads((self.namespace / "binding.json").read_text(encoding="utf-8")) != self.binding:
            raise ValueError("Cache model/environment binding mismatch")
        entries = list(self.namespace.iterdir())
        names = {path.name for path in entries}
        if not set(self._loaded) <= names:
            raise ValueError("Previously committed cache pack is missing")
        for path in entries:
            if path.name == "binding.json":
                continue
            if path.is_dir() and re.fullmatch(r"\.pending-[0-9a-f]{32}", path.name):
                continue
            if not path.is_dir() or re.fullmatch(r"pack-[0-9a-f]{32}", path.name) is None:
                raise ValueError(f"Unexpected cache entry: {path.name}")
            if {item.name for item in path.iterdir()} != {"embeddings.npz", "metadata.json"}:
                raise ValueError("Invalid immutable cache pack inventory")
            hashes = (sha256_file(path / "metadata.json"), sha256_file(path / "embeddings.npz"))
            if path.name in self._loaded:
                if self._loaded[path.name] != hashes:
                    raise ValueError("Previously committed cache pack checksum changed")
                continue
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("format_version") != 1 or metadata.get("kind") != "frozen_text_pack" or metadata.get("binding_sha256") != self.binding_sha256 or metadata.get("binding") != self.binding:
                raise ValueError("Cache pack model/environment binding mismatch")
            if metadata.get("embeddings_sha256") != hashes[1]:
                raise ValueError("Cache pack embedding checksum mismatch")
            columns = _columns(metadata["text_columns"], "Cached text columns")
            records = metadata["rows"]
            if not columns or not isinstance(records, list) or not records or metadata.get("n_rows") != len(records):
                raise ValueError("Invalid cache pack row/field metadata")
            with np.load(path / "embeddings.npz", allow_pickle=False) as source:
                if set(source.files) != {"embeddings", "provenance_sha256"}:
                    raise ValueError("Invalid cache pack array inventory")
                provenance = {key: value for key, value in metadata.items() if key != "embeddings_sha256"}
                if not np.array_equal(source["provenance_sha256"], np.frombuffer(bytes.fromhex(_digest(provenance)), dtype=np.uint8)):
                    raise ValueError("Cache pack row/field provenance checksum mismatch")
                values = _embeddings(source["embeddings"], len(records), len(columns)).copy()
            values.flags.writeable = False
            for index, record in enumerate(records):
                if set(record) != {"row_id", "text_sha256", "key"}:
                    raise ValueError("Invalid cached row metadata inventory")
                tagged = record["row_id"]
                if not isinstance(tagged, list) or len(tagged) != 2 or tagged[0] not in ("str", "int"):
                    raise ValueError("Cache row identifiers require explicit string/integer tags")
                try:
                    row_id = tagged[1] if tagged[0] == "str" else int(tagged[1])
                    normalized = normalize_row_ids((row_id,), context="Cached row IDs")[0]
                except (ValueError, TypeError) as error:
                    raise ValueError("Invalid cached row identifier") from error
                if _typed_cell(normalized) != tagged or re.fullmatch(r"[0-9a-f]{64}", str(record["text_sha256"])) is None:
                    raise ValueError("Invalid cached row identifier/text hash")
                key = _digest({"row_id": tagged, "text_sha256": record["text_sha256"], "text_columns": list(columns)})
                if record["key"] != key or key in self._rows:
                    raise ValueError("Cache row key is invalid or duplicated across packs")
                self._rows[key] = values[index]
            self._loaded[path.name] = hashes

    def assert_frozen(self) -> None:
        self.encoder.assert_frozen()

    def state_sha256(self) -> str:
        return self.encoder.state_sha256()

    def metadata(self) -> dict[str, Any]:
        return {**self.encoder.metadata(), "cache": {
            "kind": "frozen_text_cache", "binding_sha256": self.binding_sha256,
            "directory": str(self.directory), "namespace": str(self.namespace),
            "cached_rows": len(self._rows), "packs": len(self._loaded),
            "hits": self.hits, "misses": self.misses,
            "pending_packs": sum(1 for path in self.namespace.glob(".pending-*") if path.is_dir())}}

    def encode(self, frame: pd.DataFrame, text_columns: Sequence[str]) -> np.ndarray:
        columns = _columns(text_columns, "text_columns")
        if not columns:
            raise ValueError("At least one text field is required")
        ids = _frame(frame, columns, exact=False)
        self.assert_frozen()
        if _encoder_identity(self.encoder) != self.binding["encoder"] or _runtime_binding(self.encoder) != self.binding["environment"]:
            raise ValueError("Encoder state/settings/environment changed after cache binding")
        self._refresh()
        records = []
        for row_id, row in zip(ids, frame.loc[:, list(columns)].itertuples(index=False, name=None)):
            text_hash = _digest([[column, _typed_cell(value)] for column, value in zip(columns, row)])
            record = {"row_id": _typed_cell(row_id), "text_sha256": text_hash}
            record["key"] = _digest({**record, "text_columns": list(columns)})
            records.append(record)
        missing = [index for index, record in enumerate(records) if record["key"] not in self._rows]
        self.hits += len(records) - len(missing)
        self.misses += len(missing)
        if missing:
            encoded = _embeddings(self.encoder.encode(frame.iloc[missing].loc[:, list(columns)], columns), len(missing), len(columns))
            self.assert_frozen()
            if _encoder_identity(self.encoder) != self.binding["encoder"]:
                raise ValueError("Frozen encoder changed while encoding a cache pack")
            suffix = uuid4().hex
            staging = self.namespace / (".pending-" + suffix)
            staging.mkdir(exist_ok=False)
            metadata = {"format_version": 1, "kind": "frozen_text_pack", "binding": self.binding,
                        "binding_sha256": self.binding_sha256, "text_columns": list(columns),
                        "n_rows": len(missing), "rows": [records[index] for index in missing]}
            with (staging / "embeddings.npz").open("xb") as stream:
                np.savez_compressed(stream, embeddings=encoded,
                                    provenance_sha256=np.frombuffer(bytes.fromhex(_digest(metadata)), dtype=np.uint8))
            metadata["embeddings_sha256"] = sha256_file(staging / "embeddings.npz")
            _write_json(staging / "metadata.json", metadata)
            staging.rename(self.namespace / ("pack-" + suffix))
            self._refresh()
        if not records:
            return np.empty((0, len(columns), 384), dtype=np.float32)
        return np.stack([self._rows[record["key"]] for record in records]).copy()


class FrozenRepresentation:
    """Per-field PCA-30 using unchanged sklearn float32 fit/transform math.

    Despite the name, this also accepts an already-frozen fitted TAR encoder.
    It never shares fitted PCA state between training partitions or encoders.
    """

    @classmethod
    def fit(cls, frame: pd.DataFrame, text_columns: Sequence[str], train_ids: Sequence[RowId], encoder: Any) -> "FrozenRepresentation":
        columns = _columns(text_columns, "text_columns")
        if not columns:
            raise ValueError("At least one text field is required")
        ids = _frame(frame, columns)
        requested = normalize_row_ids(train_ids, context="PCA fitting IDs")
        if not requested or ids != requested:
            raise ValueError("PCA fitting IDs must exactly match the ordered feature index")
        if len(ids) < 30:
            raise ValueError("PCA-30 requires at least 30 fitting rows")
        encoder.assert_frozen()
        identity = _encoder_identity(encoder)
        values = _embeddings(encoder.encode(_source_text(frame, columns), columns), len(frame), len(columns))
        encoder.assert_frozen()
        if _encoder_identity(encoder) != identity:
            raise ValueError("Frozen encoder changed during PCA fitting")
        instance = cls()
        instance.encoder, instance.text_columns, instance.fit_ids = encoder, columns, ids
        instance.encoder_identity = identity
        # The upstream helper supplies a separate contiguous 2-D embedding
        # matrix for each field, rather than a strided view into a 3-D block.
        instance.models = tuple(PCA(n_components=30, random_state=42).fit(np.ascontiguousarray(values[:, index, :])) for index in range(len(columns)))
        return instance

    def _check_encoder(self) -> None:
        self.encoder.assert_frozen()
        if _encoder_identity(self.encoder) != self.encoder_identity:
            raise ValueError("Representation encoder differs from its fitted state/settings")

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        _frame(frame, self.text_columns)
        self._check_encoder()
        values = _embeddings(self.encoder.encode(_source_text(frame, self.text_columns), self.text_columns), len(frame), len(self.text_columns))
        self._check_encoder()
        if not len(frame):
            return np.empty((0, len(self.text_columns) * 30), dtype=np.float32)
        result = np.concatenate([model.transform(np.ascontiguousarray(values[:, index, :])) for index, model in enumerate(self.models)], axis=1)
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("PCA representation must produce finite float32 features")
        return result

    def metadata(self) -> dict[str, Any]:
        # Components/mean determine non-whitened transform; sklearn also reads
        # explained_variance_ to discover the array namespace.
        # Unused sklearn diagnostics (e.g. variance ratios of constant fields)
        # can be NaN while the actual representation remains finite.
        attributes = ("components_", "mean_", "explained_variance_")
        versions = _versions()
        return {"format_version": 1, "kind": "pilot_frozen_representation", "text_columns": list(self.text_columns),
                "fit_ids": list(self.fit_ids), "fitting_rows": len(self.fit_ids), "seed": 42,
                "input_dimension": 384, "output_dimension_per_field": 30, "dtype": "float32", "whiten": False,
                "text_value_conversion": "pandas.Series.astype(str).fillna('')",
                "encoder": self.encoder_identity, "versions": {name: versions[name] for name in ("numpy", "scikit-learn")},
                "pca_solvers": [model._fit_svd_solver for model in self.models],
                "array_orders": {name.removesuffix("_"): ["F" if np.asarray(getattr(model, name)).flags.f_contiguous and not np.asarray(getattr(model, name)).flags.c_contiguous else "C" for model in self.models] for name in attributes}}

    def save(self, directory: str | Path) -> Path:
        self._check_encoder()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        attributes = ("components_", "mean_", "explained_variance_")
        metadata = self.metadata()
        with (directory / "state.npz").open("xb") as stream:
            np.savez_compressed(stream, **{name.removesuffix("_"): np.stack([getattr(model, name) for model in self.models]) for name in attributes},
                                provenance_sha256=np.frombuffer(bytes.fromhex(_digest(metadata)), dtype=np.uint8))
        _write_json(directory / "metadata.json", {**metadata, "state_sha256": sha256_file(directory / "state.npz")})
        return directory

    @classmethod
    def load(cls, directory: str | Path, encoder: Any) -> "FrozenRepresentation":
        directory = Path(directory)
        if {path.name for path in directory.iterdir()} != {"metadata.json", "state.npz"}:
            raise ValueError("Invalid PCA representation inventory")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("kind") != "pilot_frozen_representation" or metadata.get("state_sha256") != sha256_file(directory / "state.npz"):
            raise ValueError("PCA representation format/checksum mismatch")
        if (metadata.get("seed") != 42 or metadata.get("input_dimension") != 384 or metadata.get("output_dimension_per_field") != 30
                or metadata.get("dtype") != "float32" or metadata.get("whiten") is not False):
            raise ValueError("PCA representation dimensions/settings mismatch")
        if metadata.get("text_value_conversion") != "pandas.Series.astype(str).fillna('')":
            raise ValueError("PCA representation text/missing formatting mismatch")
        instance = cls()
        instance.text_columns = _columns(metadata["text_columns"], "Saved PCA text columns")
        instance.fit_ids = normalize_row_ids(metadata["fit_ids"], context="Saved PCA fitting IDs")
        if not instance.text_columns or len(instance.fit_ids) < 30 or len(instance.fit_ids) != metadata["fitting_rows"]:
            raise ValueError("Invalid saved PCA fitting scope/field order")
        instance.encoder, instance.encoder_identity = encoder, metadata["encoder"]
        instance._check_encoder()
        versions = _versions()
        if metadata.get("versions") != {name: versions[name] for name in ("numpy", "scikit-learn")}:
            raise ValueError("PCA representation numpy/scikit-learn versions differ")
        shapes = {"components": (len(instance.text_columns), 30, 384), "mean": (len(instance.text_columns), 384),
                  "explained_variance": (len(instance.text_columns), 30)}
        with np.load(directory / "state.npz", allow_pickle=False) as source:
            if set(source.files) != set(shapes) | {"provenance_sha256"}:
                raise ValueError("Invalid PCA representation array inventory")
            provenance = {key: value for key, value in metadata.items() if key != "state_sha256"}
            if not np.array_equal(source["provenance_sha256"], np.frombuffer(bytes.fromhex(_digest(provenance)), dtype=np.uint8)):
                raise ValueError("PCA representation fitting/field provenance checksum mismatch")
            arrays = {name: source[name].copy() for name in shapes}
        for name, shape in shapes.items():
            if arrays[name].shape != shape or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
                raise ValueError("Invalid PCA representation array shape/dtype/values")
        solvers = metadata.get("pca_solvers", [])
        if len(solvers) != len(instance.text_columns) or any(value not in ("full", "randomized", "covariance_eigh", "arpack") for value in solvers):
            raise ValueError("Invalid saved PCA solver provenance")
        orders = metadata.get("array_orders", {})
        if set(orders) != set(shapes) or any(len(value) != len(instance.text_columns) or any(order not in ("C", "F") for order in value) for value in orders.values()):
            raise ValueError("Invalid saved PCA memory-order provenance")
        models = []
        for index, solver in enumerate(solvers):
            model = PCA(n_components=30, random_state=42)
            for name, value in arrays.items():
                setattr(model, name + "_", np.array(value[index], order=orders[name][index], copy=True))
            model.n_features_in_, model.n_samples_, model.n_components_ = 384, len(instance.fit_ids), 30
            model._fit_svd_solver = solver
            models.append(model)
        instance.models = tuple(models)
        return instance
