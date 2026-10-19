"""Non-fitting split, grouping and scoring contracts for the Phase 4 pilots.

Benchmark row splits call the unchanged historical implementation. Grouped
diagnostics use benchmark-training rows only and keep complete learned
pipelines separate; these helpers never construct or combine latent vectors.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

from .baseline import audit_splits, fingerprint
from .data import DatasetTable, RowId, _json_bytes, _typed_cell, normalize_row_ids, sha256_file
from .splits import SplitAssignments, _class_tokens

HISTORICAL_SHA = "599bd6a5631c96f8aef297cc4cb6e4c197ae0dca"
INNER_GROUP_CANDIDATES = 100


def _seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def _historical_modules(root: Path):
    lock = json.loads((root / "third_party/multabench.lock.json").read_text(encoding="utf-8"))
    upstream = (root / "third_party/MulTaBench").resolve()
    if lock["commit"] != HISTORICAL_SHA or _git(upstream, "rev-parse", "HEAD") != HISTORICAL_SHA:
        raise ValueError("Historical split source differs from the immutable source pin")
    if _git(upstream, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Historical split source must remain unchanged")
    if importlib.metadata.version("tabstar") != "1.1.15":
        raise ValueError("Historical split protocol requires TabSTAR 1.1.15")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    sampling = importlib.import_module("multabench.baselines.preprocessing.sampling")
    splitting = importlib.import_module("tabstar.preprocessing.splits")
    expected_sampling = upstream / "multabench/baselines/preprocessing/sampling.py"
    expected_splitting = Path(importlib.metadata.distribution("tabstar").locate_file("tabstar/preprocessing/splits.py"))
    if Path(sampling.__file__).resolve() != expected_sampling.resolve():
        raise ValueError("Historical sampling module was not imported from the pinned checkout")
    if Path(splitting.__file__).resolve() != expected_splitting.resolve():
        raise ValueError("Historical splitting module was not imported from the pinned TabSTAR installation")
    return sampling, splitting


def benchmark_split(root: Path, table: DatasetTable, fold: int, train_examples: int = 1000,
                    *, assert_phase1: bool = True) -> SplitAssignments:
    """Reconstruct the official historical cap/outer/inner row assignments.

    Original stratification may inspect all labels solely to construct splits.
    Internal validation intentionally omits the fold argument (source seed 41).
    The Phase 1 Product Sentiment reference is checked by default for fold 0
    and cap 1000; no stored scores are used in constructing assignments.
    """
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0 or 42 + fold >= 2 ** 32:
        raise ValueError("fold must be a nonnegative integer with a valid historical seed")
    if isinstance(train_examples, bool) or not isinstance(train_examples, int) or train_examples <= 0:
        raise ValueError("train_examples must be a positive integer")
    root = Path(root).resolve()
    sampling, splitting = _historical_modules(root)
    is_classification = table.schema.task != "regression"
    x, y = sampling.subsample_dataset(table.features(), table.targets(), is_classification, train_examples, fold)
    outer_x, test_x, outer_y, _ = splitting.split_to_test(x, y, is_classification, fold, train_examples)
    fit_x, validation_x, _, _ = splitting.split_to_val(outer_x, outer_y, is_classification)
    ids = {"fit": fit_x.index.tolist(), "validation": validation_x.index.tolist(), "test": test_x.index.tolist()}
    phase1_checked = False
    if assert_phase1 and table.schema.dataset_id == "MUL_TEXT_PRODUCT_SENTIMENT" and fold == 0 and train_examples == 1000:
        reference = json.loads((root / "results/phase1_product_sentiment_frozen.json").read_text(encoding="utf-8"))
        if table.source_sha256 != reference["data"]["csv_sha256"]:
            raise ValueError("Product Sentiment CSV differs from the Phase 1 reference")
        if audit_splits(ids, x.index.tolist()) != reference["splits"]:
            raise ValueError("Historical assignments differ from Phase 1")
        if fingerprint(x.index.tolist()) != reference["data"]["sampled_ordered_ids_sha256"]:
            raise ValueError("Historical selected row order differs from Phase 1")
        phase1_checked = True
    selected = set(x.index)
    result = SplitAssignments(
        train_ids=tuple(ids["fit"]), validation_ids=tuple(ids["validation"]), test_ids=tuple(ids["test"]),
        excluded_ids=tuple(row for row in table.row_ids if row not in selected), seed=42,
        metadata={"protocol": "historical_benchmark_v1", "upstream_commit": HISTORICAL_SHA,
                  "tabstar": "1.1.15", "outer_fold": fold, "outer_seed": 42 + fold,
                  "validation_seed": 41, "train_examples": train_examples,
                  "sampled_ordered_ids_sha256": fingerprint(x.index.tolist()),
                  "phase1_assignments_verified": phase1_checked,
                  "sampling_source_sha256": sha256_file(Path(sampling.__file__)),
                  "splitting_source_sha256": sha256_file(Path(splitting.__file__))},
        dataset_fingerprint=table.fingerprint(),
    )
    result.validate(table)
    return result


def _group_columns(table: DatasetTable, group_columns: Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_columns, (str, bytes)):
        raise ValueError("group_columns must be an explicit ordered column sequence")
    columns = tuple(group_columns)
    if not columns or any(not isinstance(x, str) for x in columns) or len(set(columns)) != len(columns):
        raise ValueError("group_columns must be unique nonempty declared feature names")
    if not set(columns) <= set(table.schema.feature_columns):
        raise ValueError("Group keys may use declared features only, never targets, IDs or metadata")
    return columns


def group_fingerprints(table: DatasetTable, row_ids: Sequence[RowId],
                       group_columns: Sequence[str]) -> tuple[str, ...]:
    """One typed, feature-only digest per row, preserving supplied row order."""
    columns = _group_columns(table, group_columns)
    ids = normalize_row_ids(row_ids)
    frame = table.features(ids).loc[:, list(columns)]
    return tuple(hashlib.sha256(_json_bytes({"columns": columns,
                 "values": [_typed_cell(value) for value in row]})).hexdigest()
                 for row in frame.itertuples(index=False, name=None))


def group_fingerprint(table: DatasetTable, row_ids: Sequence[RowId], group_columns: Sequence[str]) -> str:
    """Fingerprint the ordered row-to-group schedule without reading targets."""
    ids = normalize_row_ids(row_ids)
    return hashlib.sha256(_json_bytes({"row_ids": ids, "group_columns": tuple(group_columns),
                                      "groups": group_fingerprints(table, ids, group_columns)})).hexdigest()


def group_overlap_counts(table: DatasetTable, splits: SplitAssignments,
                         group_columns: Sequence[str]) -> dict[str, int]:
    """Count shared unique groups, including known benchmark row-split limits."""
    groups = {name: set(group_fingerprints(table, ids, group_columns))
              for name, ids in splits.partitions.items() if name != "excluded"}
    return {f"{left}_{right}": len(groups[left] & groups[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}


def _encoded_labels(table: DatasetTable, ids: Sequence[RowId]) -> tuple[np.ndarray, tuple[str, ...]]:
    tokens = _class_tokens(table.targets(ids).tolist())
    classes = tuple(sorted(set(tokens)))
    required = 2 if table.schema.task == "binary" else 3
    if len(classes) < required or (table.schema.task == "binary" and len(classes) != 2):
        raise ValueError("Training labels do not cover the declared classification task")
    mapping = {value: index for index, value in enumerate(classes)}
    return np.asarray([mapping[value] for value in tokens], dtype=np.int64), classes


def _folds(labels: np.ndarray | None, groups: np.ndarray, *, seed: int, n_splits: int):
    if len(set(groups)) < n_splits:
        raise ValueError("There are fewer distinct feature groups than requested folds")
    rows = np.zeros((len(groups), 1))
    if labels is None:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(rows, groups=groups)
    for value in np.unique(labels):
        if len(set(groups[labels == value])) < n_splits:
            raise ValueError("Every class needs at least one distinct group per held-out fold")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(rows, labels, groups)


def _all_classes(labels: np.ndarray, positions: np.ndarray, classes: set[int]) -> bool:
    return set(labels[positions].tolist()) == classes


def grouped_diagnostics(table: DatasetTable, outer_splits: SplitAssignments,
                        group_columns: Sequence[str], seed: int = 42, n_splits: int = 3,
                        validation_fraction: float = 0.1) -> list[SplitAssignments]:
    """Nested A/V/D assignments entirely inside the benchmark training pool.

    Classification uses the first of at most 100 deterministic group-shuffle
    candidates whose A and V both cover the pool's classes. This is a fixed
    class-stratification constraint, never score-based selection. If infeasible,
    fail without dropping classes or breaking groups. D always contains every
    class. Original outer-test rows are explicit exclusions in every result.
    """
    _seed(seed)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    outer_splits.validate(table)
    ids = tuple(outer_splits.train_ids + outer_splits.validation_ids)
    classification = table.schema.task != "regression"
    labels = _encoded_labels(table, ids)[0] if classification else None
    classes = set(labels.tolist()) if classification else None
    groups = np.asarray(group_fingerprints(table, ids, group_columns))
    fingerprint_value = table.fingerprint()
    results = []
    for fold, (development, diagnostic) in enumerate(_folds(labels, groups, seed=seed, n_splits=n_splits)):
        if classification and not _all_classes(labels, diagnostic, classes):
            raise ValueError(f"Grouped diagnostic fold {fold} lacks a training-pool class")
        splitter = GroupShuffleSplit(n_splits=INNER_GROUP_CANDIDATES, test_size=validation_fraction, random_state=seed)
        selected = None
        for candidate, (fit_local, selection_local) in enumerate(splitter.split(np.zeros((len(development), 1)), groups=groups[development])):
            fit, selection = development[fit_local], development[selection_local]
            if not classification or (_all_classes(labels, fit, classes) and _all_classes(labels, selection, classes)):
                selected = fit, selection, candidate
                break
        if selected is None:
            raise ValueError(f"No class-complete grouped A/V split in {INNER_GROUP_CANDIDATES} prespecified candidates for fold {fold}")
        fit, selection, candidate = selected
        fit_ids, selection_ids, diagnostic_ids = (tuple(ids[int(index)] for index in positions)
                                                  for positions in (fit, selection, diagnostic))
        used = set(fit_ids + selection_ids + diagnostic_ids)
        result = SplitAssignments(
            train_ids=fit_ids, validation_ids=selection_ids, test_ids=diagnostic_ids,
            excluded_ids=tuple(row for row in table.row_ids if row not in used), seed=seed,
            metadata={"protocol": "nested_grouped_complete_pipeline_v1", "diagnostic_fold": fold,
                      "n_splits": n_splits, "group_columns": list(group_columns),
                      "group_assignment_sha256": group_fingerprint(table, ids, group_columns),
                      "outer_assignments_sha256": outer_splits.fingerprint(),
                      "validation_group_fraction": validation_fraction,
                      "inner_candidate_index": candidate, "inner_max_candidates": INNER_GROUP_CANDIDATES,
                      "inner_selection_criterion": "first candidate covering every training-pool class in A and V" if classification else "first candidate",
                      "outer_test_excluded": True, "test_partition_role": "nested_diagnostic_D_not_benchmark_test"},
            dataset_fingerprint=fingerprint_value,
        )
        result.validate(table)
        if any(group_overlap_counts(table, result, group_columns).values()):
            raise ValueError("Grouped nested A/V/D assignments overlap in feature groups")
        if not set(outer_splits.test_ids) <= set(result.excluded_ids):
            raise ValueError("Original benchmark-test rows entered a nested diagnostic")
        results.append(result)
    if sorted(row for split in results for row in split.test_ids) != sorted(ids):
        raise ValueError("Nested diagnostic folds do not cover the outer training pool exactly once")
    return results


def teacher_fold_ids(table: DatasetTable, train_ids: Sequence[RowId], group_columns: Sequence[str],
                      seed: int = 42) -> np.ndarray:
    """Return one aligned group-isolated three-fold OOF assignment per A row."""
    _seed(seed)
    ids = normalize_row_ids(train_ids)
    groups = np.asarray(group_fingerprints(table, ids, group_columns))
    labels = _encoded_labels(table, ids)[0] if table.schema.task != "regression" else None
    classes = set(labels.tolist()) if labels is not None else None
    assignments = np.full(len(ids), -1, dtype=np.int64)
    for fold, (fit, held_out) in enumerate(_folds(labels, groups, seed=seed, n_splits=3)):
        if set(groups[fit]) & set(groups[held_out]):
            raise ValueError("Teacher OOF fold shares a feature group with its fitting complement")
        if labels is not None and (not _all_classes(labels, fit, classes) or not _all_classes(labels, held_out, classes)):
            raise ValueError(f"Grouped teacher fold {fold} lacks a training class")
        if np.any(assignments[held_out] != -1):
            raise ValueError("A training row received multiple OOF fold assignments")
        assignments[held_out] = fold
    if len(assignments) == 0 or set(assignments.tolist()) != {0, 1, 2}:
        raise ValueError("Grouped teacher assignment does not cover every row exactly once")
    return assignments


def score_predictions(target_series: pd.Series, predictions: np.ndarray, task: str,
                       class_order: Sequence[Any] = ()) -> dict[str, Any]:
    """Score aligned predictions with finite explicit historical task metrics.

    Undefined R2/AUROC raises an error rather than recording a successful NaN.
    Multiclass AUROC averages OVR scores over the present saved class indices,
    matching the historical present-class fallback; log loss uses the complete
    saved class order even when a scoring partition lacks one class.
    """
    if not isinstance(target_series, pd.Series) or len(target_series) == 0 or target_series.isna().any():
        raise ValueError("Scoring targets must be a nonempty nonmissing pandas Series")
    values = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite")
    if task == "regression":
        y = np.asarray(target_series, dtype=np.float64)
        if values.shape != (len(y),) or not np.isfinite(y).all():
            raise ValueError("Regression targets/predictions must be aligned finite one-dimensional arrays")
        if len(y) < 2 or np.all(y == y[0]):
            raise ValueError("R2 is undefined for fewer than two rows or constant targets")
        mse = float(mean_squared_error(y, values))
        result = {"r2": float(r2_score(y, values, force_finite=False)), "mse": mse, "rmse": float(np.sqrt(mse))}
    elif task in {"binary", "multiclass"}:
        classes = tuple(class_order)
        tokens = _class_tokens(list(classes))
        if len(tokens) != len(set(tokens)) or (task == "binary" and len(tokens) != 2) or (task == "multiclass" and len(tokens) < 3):
            raise ValueError("class_order must uniquely cover the declared classification task")
        if values.shape != (len(target_series), len(tokens)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Classification predictions must be aligned probability columns in saved class order")
        if not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6):
            raise ValueError("Classification probability rows must sum to one")
        mapping = {value: index for index, value in enumerate(tokens)}
        target_tokens = _class_tokens(target_series.tolist())
        if set(target_tokens) - set(mapping):
            raise ValueError("Scoring target contains a class absent from the saved class order")
        y = np.asarray([mapping[value] for value in target_tokens], dtype=np.int64)
        present = np.unique(y)
        if len(present) < 2:
            raise ValueError("AUROC is undefined for fewer than two observed classes")
        if task == "binary":
            auc = float(roc_auc_score(y, values[:, 1]))
            convention = "binary: saved class index 1"
        else:
            auc = float(np.mean([roc_auc_score((y == value).astype(np.int64), values[:, value]) for value in present]))
            convention = "macro OVR: all classes" if len(present) == len(tokens) else "macro OVR: present classes only"
        result = {"auc": auc, "logloss": float(log_loss(y, values, labels=np.arange(len(tokens)))),
                  "auc_convention": convention, "present_class_indices": present.tolist(),
                  "all_classes_present": len(present) == len(tokens)}
    else:
        raise ValueError(f"Unknown scoring task: {task!r}")
    if any(isinstance(value, float) and not np.isfinite(value) for value in result.values()):
        raise ValueError("The scoring metric is undefined or nonfinite")
    return result
