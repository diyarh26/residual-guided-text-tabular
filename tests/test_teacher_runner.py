"""Runner boundary tests with real CatBoost and small synthetic tables."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.data import DatasetSchema, DatasetTable, sha256_file
from refutab.splits import SplitAssignments
from refutab.teacher import TeacherArtifacts, fit_oof_teacher
from refutab.teacher_runner import build_assignments, run_teacher


class TeacherRunnerTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict, SplitAssignments]:
        ids = [f"row-{i:03d}" for i in range(48)]
        frame = pd.DataFrame({
            "id": ids, "first_text": [f"text {i}" for i in range(48)],
            "second_text": ["another field"] * 48,
            "number": np.arange(48, dtype=float), "category": ["a", "b", None] * 16,
            "target": np.sin(np.arange(48)) + np.arange(48) / 10,
            "fold_marker": [99] * 48,
        })
        frame.to_csv(root / "data.csv", index=False)
        schema = DatasetSchema("synthetic", "regression", "target", ("first_text", "second_text"),
                               ("number",), ("category",), "id", forbidden_columns=("fold_marker",))
        table = DatasetTable.load_csv(root / "data.csv", schema)
        splits = SplitAssignments(tuple(ids[:36]), tuple(ids[36:42]), tuple(ids[42:]),
                                  dataset_fingerprint=table.fingerprint(), metadata={"protocol": "fixed_fixture"})
        splits.save(root / "splits.json")
        config = {"schema_version": 1, "dataset_path": "data.csv", "dataset_sha256": table.source_sha256,
                  "schema": schema.to_dict(), "splits": {"protocol": "explicit", "path": "splits.json"},
                  "teacher": {"iterations": 5, "depth": 2, "thread_count": 1}, "output_root": "raw"}
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock"):
            (root / name).write_text("fixture\n", encoding="utf-8")
        return config, splits

    @staticmethod
    def git_stub(root: Path, *args: str) -> str:
        return "a" * 40 if args == ("rev-parse", "HEAD") else ""

    def test_runner_uses_only_structured_train_and_validation_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, splits = self.fixture(root)
            structured_calls, target_calls, fitted_inputs = [], [], []
            original_structured, original_targets = DatasetTable.structured, DatasetTable.targets

            def structured(table, ids=None):
                structured_calls.append(tuple(ids))
                return original_structured(table, ids)

            def targets(table, ids=None):
                target_calls.append(tuple(ids) if ids is not None else None)
                return original_targets(table, ids)

            def fit(features, labels, **kwargs):
                fitted_inputs.append((features.copy(), labels.copy(), kwargs["validation_X"].copy()))
                return fit_oof_teacher(features, labels, **kwargs)

            with patch("refutab.teacher_runner._git", self.git_stub), \
                 patch.object(DatasetTable, "structured", structured), \
                 patch.object(DatasetTable, "targets", targets), \
                 patch("refutab.teacher_runner.fit_oof_teacher", fit):
                result = run_teacher(root, Path("config.json"), run_id="checked", summary_path=Path("summary.json"))
            self.assertEqual(structured_calls, [splits.train_ids, splits.validation_ids])
            self.assertTrue(all(ids is not None and not set(ids) & set(splits.test_ids) for ids in target_calls))
            train, labels, validation = fitted_inputs[0]
            self.assertEqual(tuple(train.columns), ("number", "category"))
            self.assertEqual(tuple(validation.columns), tuple(train.columns))
            self.assertEqual(tuple(train.index), splits.train_ids)
            self.assertEqual(tuple(labels.index), splits.train_ids)
            self.assertEqual(tuple(validation.index), splits.validation_ids)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["test_predictions"], 0)
            self.assertEqual(result["test_scores"], 0)
            self.assertEqual([f["fit_rows"] for f in result["folds"]], [24, 24, 24])
            self.assertTrue(all(result["checks"].values()))
            summary = (root / "summary.json").read_text(encoding="utf-8")
            self.assertNotIn("row-000", summary)
            self.assertNotIn("text 0", summary)
            self.assertEqual(json.loads(summary), result)
            for name, digest in result["artifacts"]["files_sha256"].items():
                self.assertEqual(sha256_file(root / "raw" / "checked" / name), digest)
            saved = TeacherArtifacts.load(root / "raw" / "checked" / "teacher")
            self.assertEqual(saved.oof_predictions.shape, (36,))
            with self.assertRaises(FileExistsError):
                run_teacher(root, Path("config.json"), run_id="checked")

    def test_failure_keeps_manifest_and_existing_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.fixture(root)
            config["dataset_sha256"] = "0" * 64
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with patch("refutab.teacher_runner._git", self.git_stub), self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run_teacher(root, Path("config.json"), run_id="failed")
            manifest = json.loads((root / "raw/failed/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((root / "raw/failed/failure.txt").is_file())
            (root / "summary.json").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_teacher(root, Path("config.json"), run_id="existing-summary", summary_path=Path("summary.json"))
            self.assertEqual((root / "summary.json").read_text(encoding="utf-8"), "preserve")

    def test_local_protocol_is_explicit_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.fixture(root)
            table = DatasetTable.load_csv(root / "data.csv", DatasetSchema.from_dict(config["schema"]))
            settings = {"protocol": "local", "seed": 7, "validation_fraction": .125, "test_fraction": .25}
            first = build_assignments(root, table, settings)
            second = build_assignments(root, table, settings)
            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertEqual([len(values) for values in first.partitions.values()], [30, 6, 12, 0])

    @unittest.skipUnless((ROOT / "data/multabench-product-sentiment-v1/data.csv").exists()
                         and (ROOT / "third_party/MulTaBench/.git").exists(), "requires pinned pilot source/data")
    def test_historical_protocol_reconstructs_phase1_without_model_execution(self):
        config = json.loads((ROOT / "configs/phase2_teacher.json").read_text(encoding="utf-8"))
        table = DatasetTable.load_csv(ROOT / config["dataset_path"], DatasetSchema.from_dict(config["schema"]),
                                      expected_sha256=config["dataset_sha256"])
        splits = build_assignments(ROOT, table, config["splits"])
        self.assertEqual([len(values) for values in splits.partitions.values()], [900, 100, 509, 3582])
        self.assertEqual(splits.metadata["validation_seed"], 41)
        self.assertEqual(tuple(table.structured(splits.train_ids).columns), ("Product_Type",))


if __name__ == "__main__":
    unittest.main()
