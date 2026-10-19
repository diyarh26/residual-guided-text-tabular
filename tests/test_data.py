"""Dataset contracts and persisted split boundaries, without model fitting."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.data import DatasetSchema, DatasetTable, normalize_row_ids, sha256_file
from refutab.splits import SplitAssignments, generate_splits


def make_table(task="binary", rows=60):
    schema = DatasetSchema(
        dataset_id="arbitrary_demo", task=task, target_column="outcome", id_column="record",
        text_columns=("second_text", "first_text"), numerical_columns=("amount",),
        categorical_columns=("group",), excluded_columns=("partition_marker",),
        forbidden_columns=("teacher_prediction", "residual"),
    )
    labels = np.arange(rows) % (3 if task == "multiclass" else 2)
    if task == "regression":
        labels = np.linspace(-10, 10, rows)
    frame = pd.DataFrame({
        "record": [f"row-{i}" for i in range(rows)], "first_text": ["hello"] * rows,
        "second_text": [None if i == 0 else f"field {i}" for i in range(rows)],
        "amount": [np.nan if i == 0 else float(i) for i in range(rows)],
        "group": [None if i == 0 else "known" for i in range(rows)],
        "outcome": labels, "partition_marker": ["reserved"] * rows,
    }, index=np.arange(1000, 1000 + rows))
    return DatasetTable(frame, schema)


class DatasetContractTests(unittest.TestCase):
    def test_named_fields_preserve_schema_and_requested_row_order_and_missingness(self):
        table = make_table()
        self.assertEqual(table.features().columns.tolist(), ["second_text", "first_text", "amount", "group"])
        rows = ["row-4", "row-0", "row-2"]
        self.assertEqual(table.structured(rows).index.tolist(), rows)
        self.assertEqual(table.targets(rows).index.tolist(), rows)
        self.assertEqual(table.structured(rows).columns.tolist(), ["amount", "group"])
        self.assertIsNone(table.rows(rows).loc["row-0", "second_text"])
        self.assertTrue(pd.isna(table.structured(rows).loc["row-0", "amount"]))
        self.assertIsNone(table.structured(rows).loc["row-0", "group"])
        self.assertEqual(table.source_index[0], 1000)
        owned_before = table.fingerprint()
        copy = table.frame
        copy.loc["row-0", "amount"] = 1e12
        self.assertEqual(table.fingerprint(), owned_before)

    def test_protected_fields_cannot_be_features_or_implicitly_discovered(self):
        table = make_table()
        for forbidden in ("outcome", "record", "partition_marker", "teacher_prediction", "residual"):
            with self.subTest(column=forbidden), self.assertRaisesRegex(ValueError, "Protected"):
                replace(table.schema, categorical_columns=("group", forbidden))
        unreviewed = table.frame.assign(unreviewed_score=0.9)
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            DatasetTable(unreviewed, table.schema)
        reviewed_schema = replace(table.schema, excluded_columns=("partition_marker", "unreviewed_score"))
        reviewed = DatasetTable(unreviewed, reviewed_schema)
        self.assertNotIn("unreviewed_score", reviewed.features().columns)
        explicit_forbidden = DatasetTable(table.frame.assign(residual=1.0), table.schema)
        self.assertNotIn("residual", explicit_forbidden.features().columns)

    def test_original_image_schema_rejected_even_if_image_column_was_dropped(self):
        with self.assertRaisesRegex(ValueError, "original schema has image"):
            replace(make_table().schema, image_columns=("photo",))

    def test_missing_duplicate_and_ambiguous_columns_rejected(self):
        table = make_table()
        with self.assertRaisesRegex(ValueError, "missing"):
            DatasetTable(table.frame.drop(columns=["first_text"]), table.schema)
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            DatasetTable(pd.concat([table.frame, table.frame[["amount"]]], axis=1), table.schema)
        with self.assertRaisesRegex(ValueError, "only one declared role"):
            replace(table.schema, categorical_columns=("group", "amount"))
        with self.assertRaisesRegex(ValueError, "ordered sequence"):
            replace(table.schema, text_columns="first_text")

    def test_ids_are_unambiguous_json_roundtrippable_and_checked_at_lookup(self):
        for ids in ([True, False], ["a", 1], [1.0, 2.0], [None, "a"], ["a", "a"], ["", "a"], "row-0"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                normalize_row_ids(ids)
        normalized = normalize_row_ids([np.int64(10), np.int64(20)])
        self.assertEqual(json.loads(json.dumps(normalized)), [10, 20])
        self.assertIs(type(normalized[0]), int)
        table = make_table()
        with self.assertRaisesRegex(ValueError, "Unknown"):
            table.rows(["row-does-not-exist"])
        with self.assertRaisesRegex(ValueError, "type differs"):
            table.rows([1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            table.rows(["row-0", "row-0"])
        with self.assertRaisesRegex(ValueError, "disagree"):
            DatasetTable(table.frame, table.schema, row_ids=list(reversed(table.row_ids)))
        bad_ids = table.frame
        bad_ids.loc["row-1", "record"] = "row-0"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            DatasetTable(bad_ids, table.schema)

    def test_schema_csv_hash_and_fingerprint_roundtrip(self):
        table = make_table()
        restored_schema = DatasetSchema.from_dict(json.loads(json.dumps(table.schema.to_dict())))
        self.assertEqual(restored_schema, table.schema)
        self.assertEqual(DatasetTable(table.frame, restored_schema).fingerprint(), table.fingerprint())
        changed = table.frame
        changed.loc["row-0", "amount"] = 99.0
        self.assertNotEqual(DatasetTable(changed, table.schema).fingerprint(), table.fingerprint())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            table.frame.to_csv(path, index=False)
            checksum = sha256_file(path)
            loaded = DatasetTable.load_csv(path, table.schema, expected_sha256=checksum)
            self.assertEqual(loaded.source_sha256, checksum)
            self.assertEqual(loaded.row_ids, table.row_ids)
            self.assertEqual(loaded.schema, table.schema)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                DatasetTable.load_csv(path, table.schema, expected_sha256="0" * 64)

    def test_regression_requires_finite_targets(self):
        table = make_table(task="regression")
        for invalid in (np.inf, np.nan, "unparseable"):
            frame = table.frame.astype({"outcome": object})
            frame.loc["row-0", "outcome"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DatasetTable(frame, table.schema)

    def test_typed_category_content_fingerprints_distinguish_int_and_string(self):
        table = make_table()
        integers, strings = table.frame, table.frame
        integers.loc["row-0", "group"] = 1
        strings.loc["row-0", "group"] = "1"
        self.assertEqual(str(integers.group.dtype), str(strings.group.dtype))
        self.assertNotEqual(DatasetTable(integers, table.schema).fingerprint(),
                            DatasetTable(strings, table.schema).fingerprint())


class SplitContractTests(unittest.TestCase):
    def test_generated_tasks_are_deterministic_disjoint_complete_and_serializable(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                table = make_table(task=task)
                splits = generate_splits(table, seed=17)
                repeat = generate_splits(table, seed=17)
                self.assertEqual(splits.to_dict(), repeat.to_dict())
                self.assertEqual([len(splits.partitions[key]) for key in ("train", "validation", "test")], [42, 6, 12])
                all_ids = [value for ids in splits.partitions.values() for value in ids]
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(set(all_ids), set(table.row_ids))
                self.assertEqual(SplitAssignments.from_dict(splits.to_dict()).fingerprint(), splits.fingerprint())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "private" / "splits.json"
                    splits.save(path)
                    self.assertEqual(SplitAssignments.load(path).to_dict(), splits.to_dict())
                    with self.assertRaises(FileExistsError):
                        splits.save(path)

    def test_reduced_cap_requires_explicit_exclusions(self):
        table = make_table()
        splits = SplitAssignments(table.row_ids[:30], table.row_ids[30:40], table.row_ids[40:50])
        with self.assertRaisesRegex(ValueError, "omit row IDs"):
            splits.validate(table)
        complete = replace(splits, excluded_ids=table.row_ids[50:])
        complete.validate(table)
        self.assertEqual(len(complete.excluded_ids), 10)

    def test_overlap_unknown_duplicates_and_id_type_failures(self):
        table = make_table()
        with self.assertRaisesRegex(ValueError, "overlap"):
            SplitAssignments(("row-0",), ("row-1",), ("row-0",))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SplitAssignments(("row-0", "row-0"), ("row-1",), ("row-2",))
        with self.assertRaisesRegex(ValueError, "inconsistent ID types"):
            SplitAssignments((1,), ("row-1",), ("row-2",))
        complete = generate_splits(table)
        unknown = replace(complete, test_ids=complete.test_ids[:-1] + ("unknown",))
        with self.assertRaisesRegex(ValueError, "unknown"):
            unknown.validate(table)
        with self.assertRaisesRegex(ValueError, "type differs"):
            SplitAssignments((1,), (2,), (3,)).validate(table)

    def test_class_order_checks_use_training_only_and_ignore_test_labels(self):
        table = make_table()
        splits = SplitAssignments(table.row_ids[:40], table.row_ids[40:50], table.row_ids[50:])
        splits.validate(table)
        altered = table.frame
        altered.loc[list(splits.test_ids), "outcome"] = 777
        # Split validation does not establish classes from the outer test set.
        splits.validate(DatasetTable(altered, table.schema))
        altered.loc["row-40", "outcome"] = 888
        with self.assertRaisesRegex(ValueError, "Validation contains classes absent"):
            splits.validate(DatasetTable(altered, table.schema))
        only_one_class = table.frame
        only_one_class.loc[list(splits.train_ids), "outcome"] = 0
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            splits.validate(DatasetTable(only_one_class, table.schema))

    def test_mixed_typed_class_membership_distinguishes_bool_int_and_float(self):
        original = make_table()
        frame = original.frame
        frame["outcome"] = pd.Series([True if i % 2 else 1 for i in range(len(frame))],
                                     index=frame.index, dtype=object)
        table = DatasetTable(frame, original.schema)
        splits = SplitAssignments(table.row_ids[:40], table.row_ids[40:50], table.row_ids[50:])
        # True and 1 compare equal in Python, but represent two declared classes.
        splits.validate(table)
        frame.loc["row-40", "outcome"] = 1.0
        with self.assertRaisesRegex(ValueError, "Validation contains classes absent"):
            splits.validate(DatasetTable(frame, table.schema))
        frame.loc["row-40", "outcome"] = 1
        frame.loc[list(splits.test_ids), "outcome"] = "test-only-class"
        # Test labels never establish class coverage or membership.
        splits.validate(DatasetTable(frame, table.schema))

    def test_mixed_typed_binary_generator_preserves_both_strata(self):
        original = make_table()
        for first, second in ((True, 1), (1, "1")):
            with self.subTest(first=first, second=second):
                frame = original.frame
                frame["outcome"] = pd.Series([first if i % 2 else second for i in range(len(frame))],
                                             index=frame.index, dtype=object)
                table = DatasetTable(frame, original.schema)
                splits = generate_splits(table, seed=17)
                self.assertEqual(splits.to_dict(), generate_splits(table, seed=17).to_dict())
                splits.validate(table)
                for partition, expected_per_class in (("train", 21), ("validation", 3), ("test", 6)):
                    labels = table.targets(splits.partitions[partition]).tolist()
                    counts = {kind: sum(type(value) is kind for value in labels) for kind in (type(first), type(second))}
                    self.assertEqual(counts, {type(first): expected_per_class, type(second): expected_per_class})

    def test_persisted_binding_prevents_reusing_assignments_on_changed_contents(self):
        table = make_table()
        splits = generate_splits(table)
        altered = table.frame
        altered.loc["row-0", "amount"] = 123456.0
        with self.assertRaisesRegex(ValueError, "different dataset content"):
            splits.validate(DatasetTable(altered, table.schema))
        with self.assertRaisesRegex(ValueError, "format_version"):
            SplitAssignments.from_dict({**splits.to_dict(), "format_version": 99})

    def test_impossible_stratification_and_invalid_ratios_fail_explicitly(self):
        tiny = make_table(task="multiclass", rows=6)
        with self.assertRaisesRegex(ValueError, "Cannot construct"):
            generate_splits(tiny, validation_fraction=0.1, test_fraction=0.2)
        for validation, test in ((0, .1), (.9, .2), (-.1, .2)):
            with self.subTest(validation=validation, test=test), self.assertRaisesRegex(ValueError, "fractions"):
                generate_splits(make_table(), validation_fraction=validation, test_fraction=test)


if __name__ == "__main__":
    unittest.main()
