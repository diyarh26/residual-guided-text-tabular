"""Persisted row-ID partitions, independent of learner fitting.

Historical benchmark assignments must be supplied explicitly. The optional
generator below is a separate local protocol and does not emulate TabSTAR's
binary holdout or multiclass singleton behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from sklearn.model_selection import train_test_split

from .data import DatasetTable, RowId, _json_bytes, _typed_cell, normalize_row_ids


def _class_tokens(values: list[Any]) -> list[str]:
    """Preserve class types in membership checks and sklearn stratification.

    Python equality conflates ``True``, ``1``, and ``1.0``; sklearn may also
    coerce mixed labels or fail while sorting them. Typed canonical scalars
    keep these classes distinct without learning a class order.
    """
    return [_json_bytes(_typed_cell(value)).decode("utf-8") for value in values]


@dataclass(frozen=True)
class SplitAssignments:
    train_ids: tuple[RowId, ...]
    validation_ids: tuple[RowId, ...]
    test_ids: tuple[RowId, ...]
    excluded_ids: tuple[RowId, ...] = ()
    seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_fingerprint: str | None = None

    def __post_init__(self) -> None:
        partitions: dict[str, tuple[RowId, ...]] = {}
        for name in ("train", "validation", "test", "excluded"):
            values = normalize_row_ids(getattr(self, f"{name}_ids"), context=f"{name} row IDs")
            if name != "excluded" and not values:
                raise ValueError(f"The {name} partition must not be empty")
            object.__setattr__(self, f"{name}_ids", values)
            partitions[name] = values
        all_ids = tuple(value for values in partitions.values() for value in values)
        try:
            normalize_row_ids(all_ids, context="Combined split row IDs")
        except ValueError as exc:
            raise ValueError(f"Split partitions overlap or use inconsistent ID types: {exc}") from exc
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        # Make nested metadata independent of caller mutation and verify JSON
        # serialization before any experiment creates artifacts.
        object.__setattr__(self, "metadata", json.loads(_json_bytes(dict(self.metadata))))
        if self.dataset_fingerprint is not None and (
            len(self.dataset_fingerprint) != 64 or any(char not in "0123456789abcdef" for char in self.dataset_fingerprint)
        ):
            raise ValueError("dataset_fingerprint must be a lowercase SHA-256 digest")

    @property
    def partitions(self) -> dict[str, tuple[RowId, ...]]:
        return {name: getattr(self, f"{name}_ids") for name in ("train", "validation", "test", "excluded")}

    def validate(self, table: DatasetTable, *, require_class_coverage: bool = True) -> None:
        assigned = set(value for values in self.partitions.values() for value in values)
        known = set(table.row_ids)
        if type(self.train_ids[0]) is not type(table.row_ids[0]):
            raise ValueError("Split row ID type differs from the dataset")
        if unknown := assigned - known:
            raise ValueError(f"Splits contain unknown row IDs: {sorted(unknown)[:5]}")
        if missing := known - assigned:
            raise ValueError(f"Splits omit row IDs; declare reduced-cap exclusions explicitly: {sorted(missing)[:5]}")
        if self.dataset_fingerprint is not None and self.dataset_fingerprint != table.fingerprint():
            raise ValueError("Split assignments belong to different dataset content or schema")
        if require_class_coverage and table.schema.task != "regression":
            training_classes = set(_class_tokens(table.targets(self.train_ids).tolist()))
            if table.schema.task == "binary" and len(training_classes) != 2:
                raise ValueError("Binary training partition must contain exactly two classes")
            if table.schema.task == "multiclass" and len(training_classes) < 3:
                raise ValueError("Multiclass training partition must contain at least three classes")
            if unseen := set(_class_tokens(table.targets(self.validation_ids).tolist())) - training_classes:
                raise ValueError(f"Validation contains classes absent from training: {sorted(unseen)}")
        # Deliberately do not discover or validate class order from test labels.

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1, **{f"{name}_ids": list(values) for name, values in self.partitions.items()},
            "seed": self.seed, "metadata": dict(self.metadata), "dataset_fingerprint": self.dataset_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitAssignments:
        value = dict(value)
        if value.pop("format_version", None) != 1:
            raise ValueError("Unsupported split-assignment format_version")
        return cls(**value)

    def fingerprint(self) -> str:
        return hashlib.sha256(_json_bytes(self.to_dict())).hexdigest()

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save full private IDs; the caller chooses an ignored artifact path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> SplitAssignments:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def generate_splits(table: DatasetTable, *, seed: int = 42,
                    validation_fraction: float = 0.1, test_fraction: float = 0.2) -> SplitAssignments:
    """Generate explicit stratified local assignments; never infer feature roles.

    Fractions refer to the complete table, so validation is adjusted after the
    test reservation. Insufficient class counts fail rather than silently
    changing to unstratified splitting. The generator is the only operation in
    this module that may use all labels, exclusively for split stratification.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    classification = table.schema.task != "regression"
    try:
        outer, test = train_test_split(
            list(table.row_ids), test_size=test_fraction, random_state=seed,
            stratify=_class_tokens(table.targets().tolist()) if classification else None,
        )
        train, validation = train_test_split(
            outer, test_size=validation_fraction / (1 - test_fraction), random_state=seed,
            stratify=_class_tokens(table.targets(outer).tolist()) if classification else None,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot construct the declared split protocol: {exc}") from exc
    assignments = SplitAssignments(
        train_ids=tuple(train), validation_ids=tuple(validation), test_ids=tuple(test), seed=seed,
        metadata={"protocol": "local_stratified_holdout_v1" if classification else "local_random_holdout_v1",
                  "validation_fraction": validation_fraction, "test_fraction": test_fraction},
        dataset_fingerprint=table.fingerprint(),
    )
    assignments.validate(table)
    return assignments
