"""Historical assignments, grouped nested/OOF exclusion and finite pilot metrics."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.data import DatasetSchema, DatasetTable
from refutab.pilot_protocol import (benchmark_split, group_fingerprint, group_fingerprints,
                                    group_overlap_counts, grouped_diagnostics,
                                    score_predictions, teacher_fold_ids)
from refutab.splits import SplitAssignments


def table_and_outer(task="multiclass"):
    rows = 240
    groups = np.arange(rows) // 2
    labels = np.asarray([f"class-{group % (2 if task == 'binary' else 3)}" for group in groups])
    if task == "regression":
        labels = groups.astype(float) / 7
    schema = DatasetSchema(dataset_id="synthetic_grouped", task=task, target_column="target",
                           text_columns=("description",), numerical_columns=("amount",),
                           categorical_columns=("sector",), forbidden_columns=("fold_id", "residual"))
    frame = pd.DataFrame({"description": [f"group-{group}" for group in groups],
                          "amount": np.arange(rows, dtype=float),
                          "sector": [f"sector-{group % 4}" for group in groups], "target": labels})
    table = DatasetTable(frame, schema, row_ids=[f"row-{row:03d}" for row in range(rows)])
    ids = table.row_ids
    outer = SplitAssignments(train_ids=ids[:180], validation_ids=ids[180:210], test_ids=ids[210:],
                             dataset_fingerprint=table.fingerprint())
    outer.validate(table)
    return table, outer


class GroupedPilotProtocolTests(unittest.TestCase):
    def test_nested_complete_pipeline_groups_disjoint_and_outer_test_always_excluded(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                table, outer = table_and_outer(task)
                splits = grouped_diagnostics(table, outer, ("description",))
                self.assertEqual(len(splits), 3)
                diagnostic_ids = [row for split in splits for row in split.test_ids]
                self.assertEqual(set(diagnostic_ids), set(outer.train_ids + outer.validation_ids))
                self.assertEqual(len(diagnostic_ids), len(set(diagnostic_ids)))
                for split in splits:
                    split.validate(table)
                    self.assertTrue(set(outer.test_ids) <= set(split.excluded_ids))
                    self.assertEqual(set(group_overlap_counts(table, split, ("description",)).values()), {0})
                    self.assertTrue(0 <= split.metadata["inner_candidate_index"] < 100)
                    self.assertEqual(split.metadata["inner_max_candidates"], 100)
                    self.assertEqual(split.metadata["test_partition_role"], "nested_diagnostic_D_not_benchmark_test")
                    if task != "regression":
                        for ids in (split.train_ids, split.validation_ids, split.test_ids):
                            self.assertEqual(set(table.targets(ids)), set(table.targets(outer.train_ids)))

    def test_teacher_oof_group_exclusion_class_coverage_and_saved_order(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                table, outer = table_and_outer(task)
                ids = tuple(reversed(outer.train_ids))
                folds = teacher_fold_ids(table, ids, ("description",))
                self.assertEqual(folds.dtype, np.dtype("int64"))
                self.assertEqual(folds.shape, (len(ids),))
                self.assertEqual(set(folds), {0, 1, 2})
                groups = np.asarray(group_fingerprints(table, ids, ("description",)))
                for fold in range(3):
                    self.assertFalse(set(groups[folds == fold]) & set(groups[folds != fold]))
                    if task != "regression":
                        fit_ids = [row for row, assignment in zip(ids, folds) if assignment != fold]
                        self.assertEqual(set(table.targets(fit_ids)), set(table.targets(ids)))
                np.testing.assert_array_equal(teacher_fold_ids(table, ids, ("description",)), folds)

    def test_outer_test_content_cannot_change_nested_or_teacher_assignments(self):
        table, outer = table_and_outer()
        original = grouped_diagnostics(table, outer, ("description",))
        frame = table.frame
        frame.loc[list(outer.test_ids), "target"] = "reserved-only-label"
        frame.loc[list(outer.test_ids), "description"] = "changed reserved-only group"
        changed = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        rebound = replace(outer, dataset_fingerprint=changed.fingerprint())
        rerun = grouped_diagnostics(changed, rebound, ("description",))
        for left, right in zip(original, rerun):
            self.assertEqual(left.partitions, right.partitions)
            self.assertEqual(left.metadata["inner_candidate_index"], right.metadata["inner_candidate_index"])
            np.testing.assert_array_equal(teacher_fold_ids(table, left.train_ids, ("description",)),
                                          teacher_fold_ids(changed, right.train_ids, ("description",)))

    def test_group_hashes_use_typed_features_and_ignore_targets(self):
        table, _ = table_and_outer("regression")
        frame = table.frame
        frame["description"] = pd.Series([1, "1", None, np.nan] * 60, index=frame.index, dtype=object)
        typed = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        hashes = group_fingerprints(typed, typed.row_ids[:4], ("description",))
        self.assertNotEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[2], hashes[3])
        before = group_fingerprint(typed, typed.row_ids, ("description",))
        frame["target"] = np.arange(len(frame)) * 1000
        changed = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        self.assertEqual(group_fingerprint(changed, changed.row_ids, ("description",)), before)
        for forbidden in (("target",), ("row_id",), ("fold_id",), (), ("description", "description")):
            with self.subTest(columns=forbidden), self.assertRaises(ValueError):
                group_fingerprints(table, table.row_ids, forbidden)

    def test_overlap_counts_record_row_split_content_limits(self):
        table, outer = table_and_outer("regression")
        frame = table.frame
        for row in (outer.train_ids[0], outer.validation_ids[0], outer.test_ids[0]):
            frame.loc[row, "description"] = "same title"
        changed = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        self.assertEqual(group_overlap_counts(changed, outer, ("description",)),
                         {"train_validation": 1, "train_test": 1, "validation_test": 1})

    def test_infeasible_class_complete_inner_split_fails_after_fixed_cap(self):
        table, outer = table_and_outer()
        with self.assertRaisesRegex(ValueError, "100 prespecified candidates"):
            grouped_diagnostics(table, outer, ("description",), validation_fraction=0.001)
        frame = table.frame
        frame["description"] = frame["target"]
        impossible = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        with self.assertRaisesRegex(ValueError, "distinct group"):
            teacher_fold_ids(impossible, outer.train_ids, ("description",))

    def test_invalid_seed_group_count_and_fractions_fail(self):
        table, outer = table_and_outer()
        for args in ({"seed": True}, {"seed": 2 ** 32}, {"n_splits": 1},
                     {"validation_fraction": 0}, {"validation_fraction": 1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                grouped_diagnostics(table, outer, ("description",), **args)
        regression, regression_outer = table_and_outer("regression")
        with self.assertRaisesRegex(ValueError, "fewer distinct"):
            teacher_fold_ids(regression, regression_outer.train_ids[:2], ("description",))


class PilotMetricTests(unittest.TestCase):
    def test_regression_and_binary_metrics_and_class_order(self):
        y = pd.Series([1.0, 2.0, 4.0])
        result = score_predictions(y, np.asarray(y), "regression")
        self.assertEqual(result, {"r2": 1.0, "mse": 0.0, "rmse": 0.0})
        labels = pd.Series(["yes", "no", "yes", "no"])
        probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        result = score_predictions(labels, probabilities, "binary", ("yes", "no"))
        self.assertEqual(result["auc"], 1.0)
        typed = pd.Series([1, "1", 1, "1"], dtype=object)
        self.assertEqual(score_predictions(typed, probabilities, "binary", (1, "1"))["auc"], 1.0)

    def test_multiclass_present_class_fallback_matches_historical_convention(self):
        labels = pd.Series(["a", "c", "a", "c"])
        probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.1, 0.7],
                                    [0.6, 0.2, 0.2], [0.3, 0.1, 0.6]])
        result = score_predictions(labels, probabilities, "multiclass", ("a", "b", "c"))
        encoded = np.asarray([0, 2, 0, 2])
        expected = np.mean([roc_auc_score(encoded == value, probabilities[:, value]) for value in (0, 2)])
        self.assertEqual(result["auc"], expected)
        self.assertFalse(result["all_classes_present"])
        self.assertEqual(result["present_class_indices"], [0, 2])
        self.assertEqual(result["auc_convention"], "macro OVR: present classes only")
        self.assertTrue(np.isfinite(result["logloss"]))

    def test_undefined_missing_nonfinite_or_misaligned_scores_raise(self):
        for target, prediction in ((pd.Series([1.0]), np.asarray([1.0])),
                                   (pd.Series([1.0, 1.0]), np.asarray([1.0, 1.0])),
                                   (pd.Series([1.0, np.nan]), np.asarray([1.0, 2.0])),
                                   (pd.Series([1.0, 2.0]), np.asarray([1.0, np.inf])),
                                   (pd.Series([1.0, 2.0]), np.asarray([[1.0], [2.0]]))):
            with self.subTest(target=target.tolist()), self.assertRaises(ValueError):
                score_predictions(target, prediction, "regression")
        with self.assertRaisesRegex(ValueError, "undefined"):
            score_predictions(pd.Series(["a", "a"]), np.asarray([[0.8, 0.2], [0.7, 0.3]]), "binary", ("a", "b"))
        with self.assertRaisesRegex(ValueError, "absent"):
            score_predictions(pd.Series(["a", "new"]), np.asarray([[0.8, 0.2], [0.7, 0.3]]), "binary", ("a", "b"))
        for probabilities in (np.asarray([[0.8, 0.3], [0.2, 0.8]]), np.asarray([0.1, 0.8]),
                              np.asarray([[1.1, -0.1], [0.2, 0.8]])):
            with self.subTest(predictions=probabilities.tolist()), self.assertRaises(ValueError):
                score_predictions(pd.Series(["a", "b"]), probabilities, "binary", ("a", "b"))


@unittest.skipUnless((ROOT / "data/multabench-product-sentiment-v1/data.csv").is_file()
                     and (ROOT / "third_party/MulTaBench/.git").exists(), "historical private assets not prepared")
class HistoricalPilotProtocolTests(unittest.TestCase):
    def test_real_product_split_matches_phase1_and_grouped_nested_schedules_cover_all_classes(self):
        config = json.loads((ROOT / "configs/phase2_teacher.json").read_text(encoding="utf-8"))
        table = DatasetTable.load_csv(ROOT / config["dataset_path"], DatasetSchema.from_dict(config["schema"]),
                                      expected_sha256=config["dataset_sha256"])
        split = benchmark_split(ROOT, table, 0)
        self.assertEqual({name: len(ids) for name, ids in split.partitions.items()},
                         {"train": 900, "validation": 100, "test": 509, "excluded": 3582})
        self.assertTrue(split.metadata["phase1_assignments_verified"])
        self.assertEqual(split.metadata["validation_seed"], 41)
        diagnostic = grouped_diagnostics(table, split, ("Product_Description",))
        self.assertEqual([value.metadata["inner_candidate_index"] for value in diagnostic], [0, 0, 2])
        for value in diagnostic:
            self.assertEqual(set(group_overlap_counts(table, value, ("Product_Description",)).values()), {0})
            folds = teacher_fold_ids(table, value.train_ids, ("Product_Description",))
            self.assertEqual(set(folds), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
