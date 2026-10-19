"""Structured-only, fixed-budget CatBoost teachers with auditable OOF signals.

Callers must supply the audited structured feature view from ``refutab.data``.
This module accepts no held-out targets and performs no model selection. All
learned numerical borders, categorical statistics, and trees belong to the
individual fitting model; the conversion before CatBoost is entirely stateless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


Scalar = str | int | float | bool
Model = CatBoostClassifier | CatBoostRegressor


def _scalar(value: Any) -> Scalar:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool):
        raise ValueError(f"Expected a scalar string, integer, float, or boolean; got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite scalar values are not permitted")
    return value


def _token(value: Any) -> str:
    value = _scalar(value)
    return type(value).__name__ + ":" + json.dumps(value, ensure_ascii=False, allow_nan=False)


def row_ids_sha256(ids: Sequence[Scalar]) -> str:
    """Order-sensitive, type-preserving row-ID fingerprint."""
    encoded = json.dumps([_token(value) for value in ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_ids(ids: Sequence[Any], count: int, name: str) -> tuple[Scalar, ...]:
    values = tuple(_scalar(value) for value in ids)
    if len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} row IDs")
    if any(type(value) not in (str, int) for value in values):
        raise ValueError(f"{name} must use integer or string row IDs")
    if any(type(value) is str and not value for value in values):
        raise ValueError(f"{name} must not contain empty string row IDs")
    if len({type(value) for value in values}) > 1:
        raise ValueError(f"{name} must use one consistent row-ID type")
    if len({_token(value) for value in values}) != len(values):
        raise ValueError(f"{name} contains duplicate row IDs")
    return values


def _aligned_index(frame: pd.DataFrame | pd.Series, ids: Sequence[Scalar], name: str) -> None:
    if [_token(value) for value in frame.index] != [_token(value) for value in ids]:
        raise ValueError(f"{name} index must match supplied row IDs in the same order")


@dataclass(frozen=True)
class TeacherConfig:
    task: str
    n_splits: int = 3
    seed: int = 42
    iterations: int = 40
    depth: int = 4
    learning_rate: float = 0.1
    thread_count: int = 2
    probability_clip: float = 1e-7

    def __post_init__(self) -> None:
        if self.task not in {"regression", "binary", "multiclass"}:
            raise ValueError("task must be regression, binary, or multiclass")
        if type(self.n_splits) is not int or self.n_splits != 3:
            raise ValueError("Option A requires exactly three OOF folds")
        for name in ("seed", "iterations", "depth", "thread_count"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2 ** 32 or self.iterations < 1 or not 1 <= self.depth <= 16 or self.thread_count < 1:
            raise ValueError("Invalid seed, iterations, depth, or thread_count")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if not math.isfinite(self.probability_clip) or not 0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must be finite and in (0, .5)")


@dataclass(frozen=True)
class FitAudit:
    role: str
    fold: int | None
    fit_row_ids: tuple[Scalar, ...]
    predict_row_ids: tuple[Scalar, ...]
    fit_row_ids_sha256: str
    predict_row_ids_sha256: str
    selection_row_ids: tuple[Scalar, ...] = ()
    learned_steps: tuple[str, ...] = ("catboost_borders", "catboost_categorical_statistics", "catboost_trees")


def _audit(role: str, fold: int | None, fit_ids: Sequence[Scalar], predict_ids: Sequence[Scalar]) -> FitAudit:
    if {_token(value) for value in fit_ids} & {_token(value) for value in predict_ids}:
        raise ValueError("Teacher fit and prediction row IDs overlap")
    return FitAudit(role, fold, tuple(fit_ids), tuple(predict_ids), row_ids_sha256(fit_ids), row_ids_sha256(predict_ids))


def _prepare_frame(frame: pd.DataFrame, columns: tuple[str, ...], categorical_columns: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Teacher features must be a pandas DataFrame")
    if tuple(frame.columns) != columns or len(set(frame.columns)) != len(columns):
        raise ValueError("Teacher feature columns must exactly match the saved schema and order")
    result = frame.copy(deep=True)
    for column in columns:
        if column in categorical_columns:
            # Typed tokens distinguish missing, sentinel-looking strings, and
            # numeric/string values. CatBoost handles unseen tokens by hashing;
            # no category vocabulary is learned outside the fold model.
            result[column] = ["missing:" if pd.isna(value) else _token(value) for value in frame[column]]
        else:
            try:
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Numerical column {column!r} contains a nonnumeric value") from exc
            if np.isinf(values).any():
                raise ValueError(f"Numerical column {column!r} contains infinity; use explicit missing values")
            result[column] = values
    return result


def _targets(y: Sequence[Any], config: TeacherConfig, class_order: Sequence[Any] | None) -> tuple[np.ndarray, tuple[Scalar, ...]]:
    values = np.asarray(y, dtype=object)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("Teacher targets must be one dimensional with at least three rows")
    if config.task == "regression":
        if class_order is not None:
            raise ValueError("Regression does not accept a class order")
        try:
            targets = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression targets must be finite numeric values") from exc
        if not np.isfinite(targets).all():
            raise ValueError("Regression targets must be finite numeric values")
        return targets, ()
    labels = tuple(_scalar(value) for value in values)
    observed = {_token(value): value for value in labels}
    if class_order is None:
        order = tuple(observed[token] for token in sorted(observed))
    else:
        order = tuple(_scalar(value) for value in class_order)
    tokens = tuple(_token(value) for value in order)
    if len(set(tokens)) != len(tokens) or set(tokens) != set(observed):
        raise ValueError("class_order must contain exactly the classes observed in training, once each")
    if (config.task == "binary" and len(order) != 2) or (config.task == "multiclass" and len(order) < 3):
        raise ValueError("Binary tasks require two classes; multiclass tasks require at least three")
    mapping = {token: index for index, token in enumerate(tokens)}
    targets = np.asarray([mapping[_token(value)] for value in labels], dtype=np.int64)
    if np.bincount(targets, minlength=len(order)).min() < config.n_splits:
        raise ValueError("Each training class requires at least three rows for stratified three-fold OOF")
    return targets, order


def _fit_model(frame: pd.DataFrame, targets: np.ndarray, *, categorical_columns: tuple[str, ...], config: TeacherConfig) -> Model:
    params = dict(iterations=config.iterations, depth=config.depth, learning_rate=config.learning_rate,
                  random_seed=config.seed, thread_count=config.thread_count, task_type="CPU", nan_mode="Min",
                  verbose=False, allow_writing_files=False, use_best_model=False)
    if config.task == "regression":
        model: Model = CatBoostRegressor(loss_function="RMSE", **params)
    else:
        model = CatBoostClassifier(loss_function="Logloss" if config.task == "binary" else "MultiClass", **params)
    # Deliberately no eval_set, early_stopping_rounds, or calibration.
    model.fit(frame, targets, cat_features=list(categorical_columns))
    return model


def _predict_model(model: Model, frame: pd.DataFrame, config: TeacherConfig, class_count: int) -> np.ndarray:
    if config.task == "regression":
        predictions = np.asarray(model.predict(frame, thread_count=config.thread_count), dtype=np.float64).reshape(-1)
    else:
        raw = np.asarray(model.predict_proba(frame, thread_count=config.thread_count), dtype=np.float64)
        model_classes = [int(value) for value in model.classes_]
        if set(model_classes) != set(range(class_count)):
            raise ValueError("Fold teacher is missing a class from the saved class order")
        predictions = raw[:, [model_classes.index(index) for index in range(class_count)]]
        if (predictions < 0).any() or not np.allclose(predictions.sum(axis=1), 1.0):
            raise ValueError("Teacher emitted invalid class probabilities")
    if not np.isfinite(predictions).all():
        raise ValueError("Teacher emitted non-finite predictions")
    return predictions


def prediction_offsets(predictions: np.ndarray, config: TeacherConfig) -> np.ndarray:
    """Convert native predictions to fixed additive correction-head offsets."""
    values = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Teacher predictions must be finite")
    if config.task == "regression":
        if values.ndim != 1:
            raise ValueError("Regression predictions must be one dimensional")
        return values.copy()
    if values.ndim != 2 or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Classification predictions must contain normalized probabilities")
    if config.task == "binary":
        if values.shape[1] != 2:
            raise ValueError("Binary probabilities require two columns")
        probability = np.clip(values[:, 1], config.probability_clip, 1 - config.probability_clip)
        return np.log(probability) - np.log1p(-probability)
    if values.shape[1] < 3:
        raise ValueError("Multiclass probabilities require at least three columns")
    clipped = np.clip(values, config.probability_clip, 1 - config.probability_clip)
    return np.log(clipped / clipped.sum(axis=1, keepdims=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TeacherArtifacts:
    config: TeacherConfig
    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    class_order: tuple[Scalar, ...]
    train_ids: tuple[Scalar, ...]
    validation_ids: tuple[Scalar, ...]
    fold_ids: np.ndarray
    oof_predictions: np.ndarray
    oof_offsets: np.ndarray
    validation_predictions: np.ndarray | None
    validation_offsets: np.ndarray | None
    fold_audits: tuple[FitAudit, ...]
    validation_audit: FitAudit
    fold_models: tuple[Model, ...]
    final_model: Model
    fold_assignment: str

    def validate(self) -> None:
        """Check exactly-once alignment and excluded-row provenance after loading."""
        if not self.columns or any(type(column) is not str for column in self.columns) or len(set(self.columns)) != len(self.columns):
            raise ValueError("Saved teacher requires nonempty, unique string feature names")
        if len(set(self.categorical_columns)) != len(self.categorical_columns) or not set(self.categorical_columns).issubset(self.columns):
            raise ValueError("Saved categorical columns must be unique members of the feature schema")
        class_tokens = tuple(_token(value) for value in self.class_order)
        if len(set(class_tokens)) != len(class_tokens):
            raise ValueError("Saved class_order contains duplicate classes")
        if self.config.task == "regression" and self.class_order:
            raise ValueError("Regression teacher must not contain a class_order")
        if self.config.task == "binary" and len(self.class_order) != 2:
            raise ValueError("Binary teacher class_order must contain exactly two classes")
        if self.config.task == "multiclass" and len(self.class_order) < 3:
            raise ValueError("Multiclass teacher class_order must contain at least three classes")
        ids = _validate_ids(self.train_ids, len(self.fold_ids), "saved train_ids")
        val_ids = _validate_ids(self.validation_ids, len(self.validation_ids), "saved validation_ids")
        if self.fold_ids.ndim != 1 or self.fold_ids.dtype.kind not in "iu" or set(self.fold_ids.tolist()) != {0, 1, 2}:
            raise ValueError("Saved OOF fold assignments are invalid")
        if len(self.fold_models) != 3 or len(self.fold_audits) != 3:
            raise ValueError("Saved teacher requires exactly three fold models and audits")
        shape = (len(ids),) if self.config.task == "regression" else (len(ids), len(self.class_order))
        if self.oof_predictions.shape != shape:
            raise ValueError("Saved OOF prediction shape does not match row/class order")
        if not np.array_equal(prediction_offsets(self.oof_predictions, self.config), self.oof_offsets):
            raise ValueError("Teacher artifact offsets disagree with predictions")
        for fold, audit in enumerate(self.fold_audits):
            expected = _audit("oof", fold, [value for index, value in enumerate(ids) if self.fold_ids[index] != fold],
                              [value for index, value in enumerate(ids) if self.fold_ids[index] == fold])
            if audit != expected:
                raise ValueError("Saved OOF audit does not match excluded-row fold assignments")
        if self.validation_audit != _audit("validation", None, ids, val_ids):
            raise ValueError("Saved validation audit does not match all-training fit IDs")
        if not val_ids:
            if self.validation_predictions is not None or self.validation_offsets is not None:
                raise ValueError("Saved validation arrays require validation IDs")
        else:
            val_shape = (len(val_ids),) if self.config.task == "regression" else (len(val_ids), len(self.class_order))
            if self.validation_predictions is None or self.validation_predictions.shape != val_shape:
                raise ValueError("Saved validation prediction shape does not match row/class order")
            if not np.array_equal(prediction_offsets(self.validation_predictions, self.config), self.validation_offsets):
                raise ValueError("Saved validation offsets disagree with predictions")
        for model in (*self.fold_models, self.final_model):
            if tuple(model.feature_names_) != self.columns:
                raise ValueError("Saved model features do not match teacher column order")
            if self.config.task != "regression":
                model_classes = tuple(_scalar(value) for value in model.classes_)
                if len(model_classes) != len(self.class_order) or set(model_classes) != set(range(len(self.class_order))):
                    raise ValueError("Saved model classes do not match the encoded class range")

    def predict(self, frame: pd.DataFrame, *, allow_in_sample: bool = False) -> np.ndarray:
        """Predict native scores on new IDs; training IDs need a diagnostic override.

        These all-training-model predictions must never replace OOF training
        offsets. ``allow_in_sample`` exists solely for explicit diagnostics.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Teacher prediction features must be a pandas DataFrame")
        ids = _validate_ids(frame.index, len(frame), "prediction index")
        if ids and type(ids[0]) is not type(self.train_ids[0]):
            raise ValueError("Prediction row-ID type must match saved training row IDs")
        if not allow_in_sample and {_token(value) for value in ids} & {_token(value) for value in self.train_ids}:
            raise ValueError("Prediction IDs overlap teacher training rows; use saved OOF signals for training")
        prepared = _prepare_frame(frame, self.columns, self.categorical_columns)
        return _predict_model(self.final_model, prepared, self.config, len(self.class_order))

    def offsets(self, frame: pd.DataFrame, *, allow_in_sample: bool = False) -> np.ndarray:
        return prediction_offsets(self.predict(frame, allow_in_sample=allow_in_sample), self.config)

    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": 1, "config": asdict(self.config), "columns": list(self.columns),
            "categorical_columns": list(self.categorical_columns), "class_order": list(self.class_order),
            "train_ids": list(self.train_ids), "validation_ids": list(self.validation_ids),
            "train_ids_sha256": row_ids_sha256(self.train_ids),
            "validation_ids_sha256": row_ids_sha256(self.validation_ids),
            "fold_audits": [asdict(audit) for audit in self.fold_audits],
            "validation_audit": asdict(self.validation_audit), "fold_assignment": self.fold_assignment,
            "preprocessing": {"fit": "CatBoost only, independently inside each fitting model",
                              "numerical": "stateless numeric conversion; NaN native Min; infinity rejected",
                              "categorical": "stateless typed tokens; dedicated missing token; native unseen-token hashing"},
            "selection": "fixed iterations; no eval_set, early stopping, calibration, or hyperparameter search",
            "probabilities": "native CatBoost; encoded class indices reordered to saved class_order",
            "versions": {"catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        }

    def save(self, directory: str | Path) -> Path:
        """Write private raw artifacts to a new directory; never overwrite a run."""
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        files: dict[str, str] = {}
        for index, model in enumerate((*self.fold_models, self.final_model)):
            name = f"fold_{index}.cbm" if index < self.config.n_splits else "validation_teacher.cbm"
            model.save_model(str(directory / name), format="cbm")
            files[name] = _sha256(directory / name)
        arrays = {"fold_ids": self.fold_ids, "oof_predictions": self.oof_predictions, "oof_offsets": self.oof_offsets}
        if self.validation_predictions is not None:
            arrays["validation_predictions"] = self.validation_predictions
            arrays["validation_offsets"] = self.validation_offsets
        np.savez_compressed(directory / "predictions.npz", **arrays)
        files["predictions.npz"] = _sha256(directory / "predictions.npz")
        metadata = self.metadata()
        metadata["files_sha256"] = files
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "TeacherArtifacts":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported teacher artifact format")
        config = TeacherConfig(**metadata["config"])
        expected_files = {"predictions.npz", "validation_teacher.cbm", *(f"fold_{index}.cbm" for index in range(config.n_splits))}
        if set(metadata["files_sha256"]) != expected_files:
            raise ValueError("Teacher artifact file inventory does not match its format")
        for name, fingerprint in metadata["files_sha256"].items():
            if _sha256(directory / name) != fingerprint:
                raise ValueError(f"Teacher artifact checksum mismatch: {name}")
        models = []
        for name in [*(f"fold_{index}.cbm" for index in range(config.n_splits)), "validation_teacher.cbm"]:
            model = CatBoostRegressor() if config.task == "regression" else CatBoostClassifier()
            model.load_model(str(directory / name), format="cbm")
            models.append(model)
        def restore_audit(value: dict[str, Any]) -> FitAudit:
            value = dict(value)
            for key in ("fit_row_ids", "predict_row_ids", "selection_row_ids", "learned_steps"):
                value[key] = tuple(value[key])
            audit = FitAudit(**value)
            if row_ids_sha256(audit.fit_row_ids) != audit.fit_row_ids_sha256 or row_ids_sha256(audit.predict_row_ids) != audit.predict_row_ids_sha256:
                raise ValueError("Teacher audit row-ID fingerprint mismatch")
            return audit
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        result = cls(config=config, columns=tuple(metadata["columns"]), categorical_columns=tuple(metadata["categorical_columns"]),
                     class_order=tuple(metadata["class_order"]), train_ids=tuple(metadata["train_ids"]),
                     validation_ids=tuple(metadata["validation_ids"]), fold_ids=arrays["fold_ids"],
                     oof_predictions=arrays["oof_predictions"], oof_offsets=arrays["oof_offsets"],
                     validation_predictions=arrays.get("validation_predictions"), validation_offsets=arrays.get("validation_offsets"),
                     fold_audits=tuple(restore_audit(audit) for audit in metadata["fold_audits"]),
                     validation_audit=restore_audit(metadata["validation_audit"]), fold_models=tuple(models[:-1]),
                     final_model=models[-1], fold_assignment=metadata["fold_assignment"])
        if row_ids_sha256(result.train_ids) != metadata["train_ids_sha256"] or row_ids_sha256(result.validation_ids) != metadata["validation_ids_sha256"]:
            raise ValueError("Teacher artifact row-ID fingerprint mismatch")
        result.validate()
        return result


def fit_oof_teacher(
    X_train: pd.DataFrame,
    y_train: Sequence[Any],
    *,
    categorical_columns: Sequence[str],
    train_ids: Sequence[Scalar],
    config: TeacherConfig,
    validation_X: pd.DataFrame | None = None,
    validation_ids: Sequence[Scalar] | None = None,
    class_order: Sequence[Any] | None = None,
    fold_ids: Sequence[int] | None = None,
    fit_observer: Callable[[FitAudit], None] | None = None,
) -> TeacherArtifacts:
    """Fit three excluded-row teachers plus an A-only validation teacher.

    All sequences are positional and pandas indices must match the supplied row
    IDs exactly. Optional ``fold_ids`` are a previously persisted schedule, never
    chosen using held-out scores; each fold must preserve training class coverage.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame")
    columns = tuple(X_train.columns)
    if not columns or any(type(column) is not str for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("Teacher requires nonempty, unique string feature names")
    categorical = tuple(categorical_columns)
    if len(set(categorical)) != len(categorical) or not set(categorical).issubset(columns):
        raise ValueError("Categorical columns must be unique members of the feature schema")
    ids = _validate_ids(train_ids, len(X_train), "train_ids")
    _aligned_index(X_train, ids, "X_train")
    if isinstance(y_train, pd.Series):
        _aligned_index(y_train, ids, "y_train")
    targets, classes = _targets(y_train, config, class_order)
    if len(targets) != len(ids):
        raise ValueError("X_train and y_train row counts differ")
    prepared = _prepare_frame(X_train, columns, categorical)
    validation: pd.DataFrame | None = None
    if validation_X is None:
        if validation_ids is not None:
            raise ValueError("validation_ids require validation_X")
        val_ids: tuple[Scalar, ...] = ()
    else:
        if not isinstance(validation_X, pd.DataFrame):
            raise ValueError("validation_X must be a pandas DataFrame")
        if validation_ids is None or len(validation_X) == 0:
            raise ValueError("Nonempty validation_X requires explicit validation_ids")
        val_ids = _validate_ids(validation_ids, len(validation_X), "validation_ids")
        if ids and type(ids[0]) is not type(val_ids[0]):
            raise ValueError("Training and validation row-ID types must match")
        _aligned_index(validation_X, val_ids, "validation_X")
        _audit("validation", None, ids, val_ids)
        validation = _prepare_frame(validation_X, columns, categorical)
    schedule = np.full(len(ids), -1, dtype=np.int64)
    if fold_ids is None:
        splitter = KFold(n_splits=3, shuffle=True, random_state=config.seed) if config.task == "regression" else StratifiedKFold(n_splits=3, shuffle=True, random_state=config.seed)
        for fold, (_, predict_indices) in enumerate(splitter.split(prepared, targets)):
            schedule[predict_indices] = fold
        assignment = "KFold" if config.task == "regression" else "StratifiedKFold"
    else:
        raw_schedule = np.asarray(fold_ids)
        if raw_schedule.ndim != 1 or len(raw_schedule) != len(ids) or raw_schedule.dtype.kind not in "iu" or set(raw_schedule.tolist()) != {0, 1, 2}:
            raise ValueError("fold_ids must assign every training row to exactly one of folds 0, 1, 2")
        schedule = raw_schedule.astype(np.int64, copy=True)
        assignment = "provided_saved_schedule"
    # Validate the entire schedule before any model is fitted.
    for fold in range(3):
        mask = schedule != fold
        if mask.sum() < 2 or not (schedule == fold).any():
            raise ValueError("Each OOF fold requires prediction rows and at least two fitting rows")
        if classes and set(targets[mask].tolist()) != set(range(len(classes))):
            raise ValueError(f"Fold {fold} training partition lacks class coverage")
    shape = (len(ids),) if config.task == "regression" else (len(ids), len(classes))
    oof = np.full(shape, np.nan, dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int64)
    audits = []
    models = []
    for fold in range(3):
        fit_indices, predict_indices = np.flatnonzero(schedule != fold), np.flatnonzero(schedule == fold)
        audit = _audit("oof", fold, [ids[index] for index in fit_indices], [ids[index] for index in predict_indices])
        if fit_observer is not None:
            fit_observer(audit)
        model = _fit_model(prepared.iloc[fit_indices], targets[fit_indices], categorical_columns=categorical, config=config)
        oof[predict_indices] = _predict_model(model, prepared.iloc[predict_indices], config, len(classes))
        counts[predict_indices] += 1
        audits.append(audit)
        models.append(model)
    if not np.all(counts == 1) or not np.isfinite(oof).all():
        raise RuntimeError("OOF construction failed exactly-once aligned prediction checks")
    validation_audit = _audit("validation", None, ids, val_ids)
    if fit_observer is not None:
        fit_observer(validation_audit)
    final_model = _fit_model(prepared, targets, categorical_columns=categorical, config=config)
    val_predictions = None if validation is None else _predict_model(final_model, validation, config, len(classes))
    result = TeacherArtifacts(config=config, columns=columns, categorical_columns=categorical, class_order=classes,
                            train_ids=ids, validation_ids=val_ids, fold_ids=schedule, oof_predictions=oof,
                            oof_offsets=prediction_offsets(oof, config), validation_predictions=val_predictions,
                            validation_offsets=None if val_predictions is None else prediction_offsets(val_predictions, config),
                            fold_audits=tuple(audits), validation_audit=validation_audit, fold_models=tuple(models),
                            final_model=final_model, fold_assignment=assignment)
    result.validate()
    return result
