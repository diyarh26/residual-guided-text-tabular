"""Fixed grouped development holdouts and nested, score-independent fit sets."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .data import DatasetTable
from .pilot_protocol import (benchmark_split, group_fingerprints,
                             group_overlap_counts, teacher_fold_ids)
from .splits import SplitAssignments, _class_tokens
from .teacher import row_ids_sha256


def fixed_development_split(table: DatasetTable, anchor: SplitAssignments,
                            group_columns, fold: int) -> SplitAssignments:
    """Restore cap-excluded training rows; reserve old test IDs unchanged.

    Only split stratification may inspect pool labels. The old test's features
    and labels are never read here. Candidate choice uses coverage, not scores.
    """
    anchor.validate(table, require_class_coverage=False)
    reserved = set(anchor.test_ids)
    ids = tuple(row for row in table.row_ids if row not in reserved)
    groups = np.asarray(group_fingerprints(table, ids, group_columns))
    labels = (np.asarray(_class_tokens(table.targets(ids).tolist()))
              if table.schema.task != "regression" else None)
    classes = set(labels) if labels is not None else None

    def complete(positions):
        return labels is None or set(labels[positions]) == classes

    outer = GroupShuffleSplit(n_splits=100, test_size=.2, random_state=42 + fold)
    for d_candidate, (development, assessment) in enumerate(outer.split(np.zeros(len(ids)), groups=groups)):
        if not complete(development) or not complete(assessment):
            continue
        inner = GroupShuffleSplit(n_splits=100, test_size=.1, random_state=142 + fold)
        for v_candidate, (a_local, v_local) in enumerate(inner.split(np.zeros(len(development)), groups=groups[development])):
            fitting, validation = development[a_local], development[v_local]
            if not complete(fitting) or not complete(validation):
                continue
            a_ids, v_ids, d_ids = (tuple(ids[int(i)] for i in p) for p in (fitting, validation, assessment))
            try:
                teacher_fold_ids(table, a_ids, group_columns)
            except ValueError:
                continue
            result = SplitAssignments(
                a_ids, v_ids, d_ids, excluded_ids=anchor.test_ids, seed=42,
                dataset_fingerprint=table.fingerprint(), metadata={
                    "protocol": "controlled_scaling_grouped_v1", "outer_fold": fold,
                    "anchor_fingerprint": anchor.fingerprint(),
                    "reserved_outer_test_ids_sha256": row_ids_sha256(anchor.test_ids),
                    "test_partition_role": "development_assessment_D_not_benchmark_test",
                    "group_columns": list(group_columns), "assessment_candidate": d_candidate,
                    "validation_candidate": v_candidate, "max_candidates": 100,
                    "assessment_seed": 42 + fold, "validation_seed": 142 + fold,
                    "assessment_group_fraction": .2, "validation_group_fraction": .1})
            result.validate(table)
            if any(group_overlap_counts(table, result, group_columns).values()):
                raise ValueError("Scaling A/V/D groups overlap")
            return result
    raise ValueError("No class-complete grouped scaling split with feasible teachers in 100x100 candidates")


def nested_fit_order(table: DatasetTable, ids, seed: int = 42) -> tuple:
    """One reproducible order, with proportional class coverage in prefixes.

    Shuffle each typed class independently using one seeded RNG. Emit one row
    per class first, then merge remaining queues by fractional within-class
    position. Ties use typed class order. Regression uses a seeded permutation.
    No V/D/old-test labels, feature statistics or model scores are consumed.
    """
    ids = tuple(ids)
    rng = np.random.default_rng(seed)
    if table.schema.task == "regression":
        return tuple(ids[int(i)] for i in rng.permutation(len(ids)))
    labels = _class_tokens(table.targets(ids).tolist())
    queues = [rng.permutation([i for i, label in enumerate(labels) if label == token]).tolist()
              for token in sorted(set(labels))]
    first = [queue[0] for queue in queues]
    rest = sorted(((j / len(queue), c, index) for c, queue in enumerate(queues)
                   for j, index in enumerate(queue[1:], start=1)))
    return tuple(ids[i] for i in first + [index for _, _, index in rest])


def nested_splits(table: DatasetTable, full: SplitAssignments, group_columns,
                  sizes=(900, 2700, "all"), seed: int = 42) -> dict[str, SplitAssignments]:
    """Validate every prefix and its grouped OOF schedule before any fitting."""
    if (not sizes or sizes[-1] != "all" or any(type(n) is not int or n <= 0 for n in sizes[:-1])
            or list(sizes[:-1]) != sorted(set(sizes[:-1]))
            or (len(sizes) > 1 and sizes[-2] >= len(full.train_ids))):
        raise ValueError("Sizes must be increasing positive counts below largest A, ending in 'all'")
    full.validate(table)
    order = nested_fit_order(table, full.train_ids, seed)
    result = {}
    for size in sizes:
        ids = order if size == "all" else order[:size]
        used = set(ids + full.validation_ids + full.test_ids)
        split = replace(full, train_ids=ids,
                        excluded_ids=tuple(row for row in table.row_ids if row not in used),
                        metadata={**full.metadata, "size": str(size), "fit_rows": len(ids),
                                  "nested_order_algorithm": "typed_class_fractional_queue_v1",
                                  "nested_order_seed": seed, "nested_order_sha256": row_ids_sha256(order)})
        split.validate(table)
        if table.schema.task != "regression" and set(_class_tokens(table.targets(ids).tolist())) != set(_class_tokens(table.targets(full.train_ids).tolist())):
            raise ValueError(f"Nested sample {size} lacks a fitting-pool class")
        teacher_fold_ids(table, ids, group_columns)
        if any(group_overlap_counts(table, split, group_columns).values()):
            raise ValueError("Nested scaling groups overlap")
        result[str(size)] = split
    return result


def build_scaling_cases(root: Path, tables: dict, config: dict) -> dict:
    """Verify historical anchors against the reviewed pilot before restoring rows."""
    from .pilot import _schema_audit

    reference = json.loads((root / config["anchor_review"]).read_text(encoding="utf-8"))
    anchors = json.loads((root / config["anchor_rows"]).read_text(encoding="utf-8"))["anchors"]
    cases = {}
    for dataset, table in tables.items():
        if table.fingerprint() != reference["dataset_fingerprints"][dataset]:
            raise ValueError("Scaling dataset differs from reviewed pilot")
        groups = config["group_columns"][dataset]
        for fold in config["outer_folds"]:
            name = f"{dataset}__fold{fold}"
            anchor = benchmark_split(root, table, fold, 1000)
            saved = anchors[name]
            portable_metadata = dict(anchor.metadata)
            # Raw source bytes differ with checkout/wheel line endings. The
            # historical modules independently verify their immutable versions.
            for key in ("sampling_source_sha256", "splitting_source_sha256"):
                portable_metadata[key] = saved["metadata"][key]
            rebound = replace(anchor, metadata=portable_metadata)
            if (rebound.fingerprint() != reference["split_fingerprints"][name]
                    or saved["fingerprint"] != rebound.fingerprint()
                    or any(saved["partitions"][k] != {"rows": len(v), "row_ids_sha256": row_ids_sha256(v)}
                           for k, v in anchor.partitions.items())):
                raise ValueError("Historical anchor differs from reviewed pilot")
            full = fixed_development_split(table, anchor, groups, fold)
            for size, split in nested_splits(table, full, groups, config["sizes"], seed=42 + fold).items():
                cases[f"{name}__size{size}"] = {
                    "dataset_id": dataset, "outer_fold": fold, "size": size,
                    "task": table.schema.task, "scope": "scaling", "split": split,
                    "group_columns": groups, "teacher_grouped": True,
                    "schema_audit": _schema_audit(table, split.train_ids),
                    "reserved_outer_test_rows": len(anchor.test_ids),
                    "methods": config["methods"]}
    return cases
