"""LightGBM on original structured columns followed by frozen fused text blocks.

This adapter never receives a teacher, correction head, held-out target, or an
automatic validation split. The unchanged pinned categorical helpers are called
at fit time; saved category classes make prediction independent of that checkout.
No code from the historical source is copied into this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import lightgbm
from lightgbm import Booster, LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd

from .data import normalize_row_ids


_UPSTREAM_SHA = "599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
_CATEGORICAL_SHA = "d304f8c6b5cc40572ced8bbbba39c592cf8936f0a4839a61b7ceebac7866d2e6"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool):
        raise ValueError("Categories and class labels must be scalar strings, integers, floats, or booleans")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Non-finite categories and class labels are not permitted")
    return value


def _token(value: Any) -> str:
    value = _scalar(value)
    return type(value).__name__ + ":" + json.dumps(value, ensure_ascii=False, allow_nan=False)


def _ids_sha256(values: Sequence[str | int]) -> str:
    return hashlib.sha256(_json_bytes([_token(value) for value in values])).hexdigest()


@lru_cache(maxsize=1)
def _load_original_categorical() -> Any:
    """Import only the original helper file, checking the pinned source bytes."""
    source = Path(__file__).resolve().parents[2] / "third_party" / "MulTaBench" / "multabench" / "baselines" / "preprocessing" / "categorical.py"
    if not source.is_file():
        raise RuntimeError("Pinned MulTaBench categorical helpers are absent; run scripts/prepare_phase1.py --skip-model")
    normalized = source.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() != _CATEGORICAL_SHA:
        raise RuntimeError("MulTaBench categorical helper differs from the pinned historical source")
    spec = importlib.util.spec_from_file_location("_refutab_original_categorical", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned MulTaBench categorical helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class DownstreamConfig:
    task: str
    seed: int = 42
    n_estimators: int = 100
    num_threads: int = 2

    def __post_init__(self) -> None:
        if self.task not in {"regression", "binary", "multiclass"}:
            raise ValueError("task must be regression, binary, or multiclass")
        for name in ("seed", "n_estimators", "num_threads"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2 ** 32 or self.n_estimators < 1 or self.num_threads < 1:
            raise ValueError("Invalid seed, n_estimators, or num_threads")


class DownstreamModel:
    """A per-dataset/fold native LightGBM model with checked feature ordering.

    Classification targets supplied to ``fit`` are already encoded integers
    ``0..C-1`` in ``class_order``. Without an explicit class order, those encoded
    integers themselves are the saved labels. ``fused`` is an aligned, finite
    two-dimensional array, with text fields flattened in the caller's saved order.
    """

    def __init__(self, config: DownstreamConfig) -> None:
        if not isinstance(config, DownstreamConfig):
            raise TypeError("config must be a DownstreamConfig")
        self.config = config
        self.columns: tuple[str, ...] = ()
        self.categorical_columns: tuple[str, ...] = ()
        self.class_order: tuple[str | int | float | bool, ...] = ()
        self.category_classes: dict[str, tuple[str, ...]] = {}
        self.missing_token = ""
        self.fused_dimension = 0
        self.fit_ids: tuple[str | int, ...] = ()
        self._fitted = False

    @property
    def fit_ids_sha256(self) -> str:
        return _ids_sha256(self.fit_ids)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"f{index}" for index in range(len(self.columns) + self.fused_dimension))

    @property
    def feature_layout(self) -> list[dict[str, Any]]:
        return ([{"name": f"f{index}", "kind": "structured", "column": column}
                 for index, column in enumerate(self.columns)] +
                [{"name": f"f{len(self.columns) + index}", "kind": "fused", "coordinate": index}
                 for index in range(self.fused_dimension)])

    def _check_frame(self, frame: pd.DataFrame, *, fitting: bool = False) -> tuple[str | int, ...]:
        if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
            raise ValueError("Structured features must be a nonempty pandas DataFrame")
        if any(not isinstance(column, str) or not column for column in frame.columns) or frame.columns.has_duplicates:
            raise ValueError("Structured columns must have unique, nonempty string names")
        if not fitting and tuple(frame.columns) != self.columns:
            raise ValueError("Structured columns must match the saved schema and order")
        return normalize_row_ids(frame.index, context="Downstream row IDs")

    def _check_fused(self, fused: np.ndarray, count: int, *, fitting: bool = False) -> np.ndarray:
        try:
            result = np.asarray(fused, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Fused embeddings must be numeric") from exc
        if result.ndim != 2 or result.shape[0] != count or result.shape[1] < 1:
            raise ValueError("Fused embeddings must have shape (rows, positive dimension)")
        if not fitting and result.shape[1] != self.fused_dimension:
            raise ValueError("Fused embedding dimension differs from the fitted model")
        if not np.isfinite(result).all():
            raise ValueError("Fused embeddings must be finite")
        return result

    def _prepare_structured(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for column in self.columns:
            if column in self.categorical_columns:
                values = []
                for value in frame[column]:
                    if value is None or value is pd.NA or value is pd.NaT:
                        values.append(self.missing_token)
                    elif isinstance(value, (float, np.floating)) and np.isnan(value):
                        values.append(self.missing_token)
                    else:
                        values.append(_token(value))
                result[column] = values
            else:
                try:
                    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Numerical column {column!r} contains a nonnumeric value") from exc
                if np.isinf(values).any():
                    raise ValueError(f"Numerical column {column!r} contains infinity")
                result[column] = values
        return result

    def _join(self, encoded: pd.DataFrame, fused: np.ndarray) -> pd.DataFrame:
        blocks = [encoded.to_numpy(dtype=np.float64), fused]
        return pd.DataFrame(np.concatenate(blocks, axis=1), index=encoded.index, columns=self.feature_names)

    def fit(self, train_structured_df: pd.DataFrame, fused_train: np.ndarray,
            y_train: Sequence[Any], *, categorical_columns: Sequence[str],
            class_order: Sequence[Any] = (), fit_ids: Sequence[str | int]) -> DownstreamModel:
        """Fit only supplied training rows; no validation or test labels accepted."""
        if self._fitted:
            raise RuntimeError("Create a new DownstreamModel for a new fitting partition")
        index = self._check_frame(train_structured_df, fitting=True)
        ids = normalize_row_ids(fit_ids, context="Downstream fitting IDs")
        if ids != index:
            raise ValueError("Structured training index must match fit_ids in the same order")
        if isinstance(y_train, pd.Series) and normalize_row_ids(y_train.index) != ids:
            raise ValueError("Training target index must match fit_ids in the same order")
        fused = self._check_fused(fused_train, len(ids), fitting=True)
        if isinstance(categorical_columns, str):
            raise ValueError("categorical_columns must be a sequence of column names")
        categories = tuple(categorical_columns)
        if len(set(categories)) != len(categories) or not set(categories) <= set(train_structured_df.columns):
            raise ValueError("Categorical columns must be unique members of the structured schema")
        target = np.asarray(y_train)
        if target.ndim != 1 or len(target) != len(ids):
            raise ValueError("Training targets must contain one value per fitting row")
        order = tuple(_scalar(value) for value in class_order)
        if self.config.task == "regression":
            if order:
                raise ValueError("Regression does not use a class order")
            try:
                target = target.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must be finite numeric values") from exc
            if not np.isfinite(target).all():
                raise ValueError("Regression targets must be finite numeric values")
        else:
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in target):
                raise ValueError("Classification targets must be encoded integer class indices")
            target = target.astype(np.int64)
            observed = tuple(int(value) for value in np.unique(target))
            if not order:
                order = observed
            if len({_token(value) for value in order}) != len(order) or observed != tuple(range(len(order))):
                raise ValueError("Training class indices must cover the complete saved class order 0..C-1")
            if ((self.config.task == "binary" and len(order) != 2) or
                    (self.config.task == "multiclass" and len(order) < 3)):
                raise ValueError("Binary requires two classes; multiclass requires at least three")
        self.columns = tuple(train_structured_df.columns)
        self.categorical_columns = tuple(column for column in self.columns if column in categories)
        self.class_order = order
        self.fit_ids = ids
        self.fused_dimension = fused.shape[1]
        # Original helpers see scalar strings, avoiding LabelEncoder's mixed-type
        # ordering ambiguity while preserving distinct int/string/bool categories.
        original = _load_original_categorical() if self.categorical_columns else None
        self.missing_token = original.MISSING_VALUE if original is not None else "Unknown Value"
        structured = self._prepare_structured(train_structured_df)
        if original is not None:
            encoders = original.fit_categorical_encoders(structured, set(self.categorical_columns))
            self.category_classes = {column: tuple(str(value) for value in encoders[column].classes_)
                                     for column in self.categorical_columns}
            structured = original.transform_categorical_features(structured, encoders)
        features = self._join(structured, fused)
        model_class = LGBMRegressor if self.config.task == "regression" else LGBMClassifier
        model = model_class(random_state=self.config.seed, verbose=-1,
                            n_estimators=self.config.n_estimators, n_jobs=self.config.num_threads)
        # The historical model also uses 100 trees by default and no early
        # stopping. Its eval_set only logs metrics, so no eval_set is needed here.
        model.fit(features, target, categorical_feature=[self.feature_names[self.columns.index(column)]
                                                       for column in self.categorical_columns])
        if self.config.task != "regression" and tuple(model.classes_) != tuple(range(len(order))):
            raise ValueError("Fitted LightGBM classes do not match the saved order")
        self.booster_ = model.booster_
        self._fitted = True
        self._validate_model()
        return self

    def transform(self, structured: pd.DataFrame, fused: np.ndarray) -> pd.DataFrame:
        """Return the exact ordered final feature matrix, without fitting."""
        if not self._fitted:
            raise RuntimeError("Downstream model is not fitted")
        self._check_frame(structured)
        values = self._check_fused(fused, len(structured))
        encoded = self._prepare_structured(structured)
        for column in self.categorical_columns:
            mapping = {value: index for index, value in enumerate(self.category_classes[column])}
            missing_index = mapping[self.missing_token]
            encoded[column] = [mapping.get(value, missing_index) for value in encoded[column]]
        return self._join(encoded, values)

    def predict(self, structured: pd.DataFrame, fused: np.ndarray) -> np.ndarray:
        features = self.transform(structured, fused)
        prediction = np.asarray(self.booster_.predict(features, num_threads=self.config.num_threads), dtype=np.float64)
        if self.config.task == "regression":
            prediction = prediction.reshape(-1)
            if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
                raise ValueError("LightGBM emitted invalid regression predictions")
        else:
            if self.config.task == "binary":
                prediction = np.column_stack((1 - prediction, prediction))
            if (prediction.shape != (len(features), len(self.class_order)) or
                    not np.isfinite(prediction).all() or (prediction < 0).any() or
                    (prediction > 1).any() or not np.allclose(prediction.sum(axis=1), 1)):
                raise ValueError("LightGBM emitted invalid class probabilities")
        return prediction

    def _validate_model(self) -> None:
        if (self.booster_.num_feature() != len(self.feature_names) or
                tuple(self.booster_.feature_name()) != self.feature_names):
            raise ValueError("Native model features do not match saved feature order")
        objective = self.booster_.params.get("objective")
        expected = {"regression": "regression", "binary": "binary", "multiclass": "multiclass"}[self.config.task]
        if objective != expected:
            raise ValueError("Native LightGBM objective does not match the saved task")
        if self.config.task == "multiclass" and self.booster_.num_model_per_iteration() != len(self.class_order):
            raise ValueError("Native LightGBM class count does not match saved class order")

    def metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Downstream model is not fitted")
        return {"schema_version": 1, "config": asdict(self.config), "columns": list(self.columns),
                "categorical_columns": list(self.categorical_columns), "class_order": list(self.class_order),
                "category_classes": {column: list(values) for column, values in self.category_classes.items()},
                "missing_token": self.missing_token, "fused_dimension": self.fused_dimension,
                "fit_ids": list(self.fit_ids), "fit_ids_sha256": self.fit_ids_sha256,
                "feature_layout": self.feature_layout, "lightgbm_version": lightgbm.__version__,
                "categorical_source_commit": _UPSTREAM_SHA, "categorical_source_sha256": _CATEGORICAL_SHA,
                "preprocessing": "train-only typed tokens plus original categorical helpers; native numerical NaN",
                "selection_ids": []}

    def save(self, directory: str | Path) -> Path:
        """Save a new directory using native model text and JSON, never pickle."""
        metadata = self.metadata()
        self._validate_model()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.booster_.save_model(str(directory / "model.txt"))
        (directory / "metadata.json").write_bytes(_json_bytes(metadata) + b"\n")
        manifest = {"schema_version": 1, "files": {name: _sha256(directory / name)
                                                 for name in ("model.txt", "metadata.json")}}
        (directory / "manifest.json").write_bytes(_json_bytes(manifest) + b"\n")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> DownstreamModel:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != {"model.txt", "metadata.json"}:
            raise ValueError("Invalid downstream artifact manifest")
        if {path.name for path in directory.iterdir()} != {"model.txt", "metadata.json", "manifest.json"}:
            raise ValueError("Unexpected downstream artifact file inventory")
        for name, expected in manifest["files"].items():
            if not (directory / name).is_file() or _sha256(directory / name) != expected:
                raise ValueError(f"Downstream artifact checksum mismatch: {name}")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported downstream metadata version")
        model = cls(DownstreamConfig(**metadata["config"]))
        columns = metadata["columns"]
        categories = metadata["categorical_columns"]
        if (not isinstance(columns, list) or any(not isinstance(value, str) or not value for value in columns) or
                len(set(columns)) != len(columns) or not isinstance(categories, list) or
                categories != [column for column in columns if column in categories]):
            raise ValueError("Invalid saved downstream structured schema")
        model.columns = tuple(columns)
        model.categorical_columns = tuple(categories)
        model.class_order = tuple(_scalar(value) for value in metadata["class_order"])
        if len({_token(value) for value in model.class_order}) != len(model.class_order):
            raise ValueError("Duplicate saved downstream class labels")
        count = len(model.class_order)
        if ((model.config.task == "regression" and count != 0) or
                (model.config.task == "binary" and count != 2) or
                (model.config.task == "multiclass" and count < 3)):
            raise ValueError("Saved downstream task and class order disagree")
        model.fused_dimension = metadata["fused_dimension"]
        if type(model.fused_dimension) is not int or model.fused_dimension < 1:
            raise ValueError("Invalid saved fused dimension")
        model.fit_ids = normalize_row_ids(metadata["fit_ids"], context="Saved downstream fitting IDs")
        if not model.fit_ids or model.fit_ids_sha256 != metadata["fit_ids_sha256"]:
            raise ValueError("Saved downstream fitting-ID fingerprint differs")
        model.missing_token = metadata["missing_token"]
        if not isinstance(model.missing_token, str) or not model.missing_token:
            raise ValueError("Invalid saved categorical missing token")
        classes = metadata["category_classes"]
        if set(classes) != set(categories):
            raise ValueError("Saved category mappings differ from the categorical schema")
        for column, values in classes.items():
            if (not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values) or
                    len(set(values)) != len(values) or model.missing_token not in values):
                raise ValueError("Invalid saved categorical classes")
        model.category_classes = {column: tuple(values) for column, values in classes.items()}
        if metadata["feature_layout"] != model.feature_layout or metadata.get("selection_ids") != []:
            raise ValueError("Saved downstream feature layout or fitting boundary differs")
        if (metadata.get("categorical_source_commit") != _UPSTREAM_SHA or
                metadata.get("categorical_source_sha256") != _CATEGORICAL_SHA):
            raise ValueError("Saved downstream categorical source provenance differs")
        model.booster_ = Booster(model_file=str(directory / "model.txt"))
        model._validate_model()
        model._fitted = True
        return model
