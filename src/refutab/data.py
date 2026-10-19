"""Explicit, non-fitting contracts for arbitrary text-tabular datasets.

Loading a table never discovers feature roles or fits a target encoder. Every
column has an explicit role, and text/missing values stay in their original
form for the component responsible for encoding them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

Task = Literal["regression", "binary", "multiclass"]
RowId = str | int


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def normalize_row_ids(values: Iterable[Any], *, context: str = "row IDs") -> tuple[RowId, ...]:
    """Require unique, non-null, single-type IDs with lossless JSON round trips.

    In particular, bool/float IDs and mixed ``1``/``"1"`` IDs are rejected.
    NumPy integer scalars are normalized to ordinary Python integers.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{context} must be a sequence of IDs, not a single string")
    result: list[RowId] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{context} must be strings or integers, not booleans")
        if isinstance(value, (int, np.integer)):
            result.append(int(value))
        elif isinstance(value, str):
            if not value:
                raise ValueError(f"{context} must not contain empty strings")
            result.append(value)
        else:
            raise ValueError(f"{context} must contain non-null strings or integers; got {value!r}")
    if result and any(type(value) is not type(result[0]) for value in result):
        raise ValueError(f"{context} must use one ID type consistently")
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contain duplicate IDs")
    return tuple(result)


def _typed_cell(value: Any) -> list[Any]:
    """Lossless scalar tags prevent object-category ``1``/``'1'`` collisions."""
    if value is None or value is pd.NA or value is pd.NaT:
        return ["missing"]
    if isinstance(value, np.generic):
        return _typed_cell(value.item())
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["missing"] if np.isnan(value) else ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, (date, datetime)):
        return ["datetime", value.isoformat()]
    if isinstance(value, timedelta):
        return ["timedelta", value.days, value.seconds, value.microseconds]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    raise ValueError(f"Dataset cells must be supported scalar values for fingerprinting; got {type(value).__name__}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSchema:
    dataset_id: str
    task: Task
    target_column: str
    text_columns: tuple[str, ...] = ()
    numerical_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    id_column: str | None = None
    excluded_columns: tuple[str, ...] = ()
    forbidden_columns: tuple[str, ...] = ()
    image_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("text_columns", "numerical_columns", "categorical_columns", "excluded_columns", "forbidden_columns", "image_columns"):
            values = getattr(self, field)
            if isinstance(values, str):
                raise ValueError(f"{field} must be an ordered sequence, not a string")
            values = tuple(values)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field} must contain nonempty column names")
            if len(values) != len(set(values)):
                raise ValueError(f"{field} contains duplicate columns")
            object.__setattr__(self, field, values)
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a nonempty string")
        if self.task not in ("regression", "binary", "multiclass"):
            raise ValueError(f"Unsupported task: {self.task!r}")
        for field in ("target_column", "id_column"):
            value = getattr(self, field)
            if value is None and field == "id_column":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty column name")
        if self.target_column == self.id_column:
            raise ValueError("The target cannot also be the row ID column")
        if self.image_columns:
            raise ValueError("Datasets whose original schema has image columns are outside the approved scope")
        features = self.feature_columns
        if not features:
            raise ValueError("At least one feature column must be declared")
        if len(features) != len(set(features)):
            raise ValueError("A feature column may have only one declared role")
        protected = {self.target_column, self.id_column, *self.excluded_columns, *self.forbidden_columns}
        if overlap := set(features) & protected:
            raise ValueError(f"Protected columns cannot enter features: {sorted(overlap)}")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.text_columns + self.numerical_columns + self.categorical_columns

    @property
    def structured_columns(self) -> tuple[str, ...]:
        return self.numerical_columns + self.categorical_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "task": self.task, "target_column": self.target_column,
            "text_columns": list(self.text_columns), "numerical_columns": list(self.numerical_columns),
            "categorical_columns": list(self.categorical_columns), "id_column": self.id_column,
            "excluded_columns": list(self.excluded_columns), "forbidden_columns": list(self.forbidden_columns),
            "image_columns": list(self.image_columns),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetSchema:
        return cls(**dict(value))


class DatasetTable:
    """Owned copy of a declared table with stable, explicit row identity.

    Without an ID column or supplied IDs, the original DataFrame index is the
    row identity. Features are always selected by saved names in schema order;
    positional column selection and automatic feature discovery are forbidden.
    """

    def __init__(self, frame: pd.DataFrame, schema: DatasetSchema,
                 row_ids: Sequence[RowId] | None = None, *,
                 source_path: str | Path | None = None, source_sha256: str | None = None) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("Dataset tables must contain at least one row")
        if frame.columns.has_duplicates:
            raise ValueError("Dataset contains duplicate column names")
        if any(not isinstance(column, str) for column in frame.columns):
            raise ValueError("All dataset column names must be strings")
        required = set(schema.feature_columns) | {schema.target_column} | set(schema.excluded_columns)
        if schema.id_column:
            required.add(schema.id_column)
        if missing := required - set(frame.columns):
            raise ValueError(f"Declared columns missing from dataset: {sorted(missing)}")
        allowed = required | set(schema.forbidden_columns)
        if undeclared := set(frame.columns) - allowed:
            raise ValueError(f"Undeclared columns require explicit exclusion or feature roles: {sorted(undeclared)}")
        inferred_ids = frame[schema.id_column].tolist() if schema.id_column else frame.index.tolist()
        ids = normalize_row_ids(inferred_ids if row_ids is None else row_ids)
        if len(ids) != len(frame):
            raise ValueError("Number of row IDs does not match the table")
        if schema.id_column and ids != normalize_row_ids(inferred_ids):
            raise ValueError("Supplied row IDs disagree with the declared ID column")
        if frame[schema.target_column].isna().any():
            raise ValueError("Target values must not be missing")
        if schema.task == "regression":
            try:
                targets = pd.to_numeric(frame[schema.target_column], errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must be numerical") from exc
            if not np.isfinite(targets).all():
                raise ValueError("Regression targets must be finite")
        self.schema = schema
        self.row_ids = ids
        self.source_index = tuple(frame.index.tolist())
        self.source_path = str(Path(source_path).resolve()) if source_path is not None else None
        self.source_sha256 = source_sha256
        self._frame = frame.copy(deep=True)
        self._frame.index = pd.Index(ids, name="refutab_row_id")

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)

    def rows(self, row_ids: Sequence[RowId] | None = None) -> pd.DataFrame:
        if row_ids is None:
            return self.frame
        ids = normalize_row_ids(row_ids, context="Requested row IDs")
        if ids and type(ids[0]) is not type(self.row_ids[0]):
            raise ValueError("Requested row ID type differs from the dataset")
        if unknown := set(ids) - set(self.row_ids):
            raise ValueError(f"Unknown row IDs requested: {sorted(unknown)[:5]}")
        return self._frame.loc[list(ids)].copy(deep=True)

    def features(self, row_ids: Sequence[RowId] | None = None) -> pd.DataFrame:
        return self.rows(row_ids).loc[:, list(self.schema.feature_columns)]

    def structured(self, row_ids: Sequence[RowId] | None = None) -> pd.DataFrame:
        return self.rows(row_ids).loc[:, list(self.schema.structured_columns)]

    def targets(self, row_ids: Sequence[RowId] | None = None) -> pd.Series:
        return self.rows(row_ids)[self.schema.target_column]

    def fingerprint(self) -> str:
        """Bind schema, row order/IDs, dtypes, columns, and contents to a digest."""
        digest = hashlib.sha256(_json_bytes({
            "format_version": 1,
            "schema": self.schema.to_dict(), "row_ids": self.row_ids,
            "columns": list(self._frame.columns), "dtypes": [str(dtype) for dtype in self._frame.dtypes],
        }))
        # pandas' non-cryptographic object hashing conflates some differently
        # typed values. Hash canonical typed cells directly instead.
        for row in self._frame.itertuples(index=False, name=None):
            digest.update(b"\n")
            digest.update(_json_bytes([_typed_cell(value) for value in row]))
        return digest.hexdigest()

    @classmethod
    def load_csv(cls, path: str | Path, schema: DatasetSchema, *,
                 expected_sha256: str | None = None, **read_csv_kwargs: Any) -> DatasetTable:
        path = Path(path)
        actual_sha256 = sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"CSV SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
        frame = pd.read_csv(path, **read_csv_kwargs)
        return cls(frame, schema, source_path=path, source_sha256=actual_sha256)


def load_csv(path: str | Path, schema: DatasetSchema, *, expected_sha256: str | None = None,
             **read_csv_kwargs: Any) -> DatasetTable:
    return DatasetTable.load_csv(path, schema, expected_sha256=expected_sha256, **read_csv_kwargs)
