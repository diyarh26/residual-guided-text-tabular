"""Nested development boundaries and real tiny-model continuation tests."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab import pilot as p, scaling
from refutab.data import DatasetTable
from refutab.pilot_features import CachedFrozenEncoder
from refutab.pilot_protocol import group_fingerprints, teacher_fold_ids
from refutab.scaling_protocol import fixed_development_split, nested_splits
from refutab.teacher import TeacherArtifacts
from test_pilot import Encoder, git_stub
from test_pilot_protocol import table_and_outer


AUDIT = {"status": "passed", "run_id": "fixture-prior"}


class ScalingProtocolTests(unittest.TestCase):
    def test_nested_fixed_roles_and_grouped_teachers_for_all_tasks(self):
        for task in ("regression", "binary", "multiclass"):
            table, anchor = table_and_outer(task)
            full = fixed_development_split(table, anchor, ["description"], 0)
            splits = nested_splits(table, full, ["description"], (36, 72, "all"))
            previous = set()
            for size, split in splits.items():
                self.assertTrue(previous < set(split.train_ids))
                previous = set(split.train_ids)
                self.assertEqual(split.validation_ids, full.validation_ids)
                self.assertEqual(split.test_ids, full.test_ids)
                self.assertTrue(set(anchor.test_ids) <= set(split.excluded_ids))
                schedule = teacher_fold_ids(table, split.train_ids, ["description"])
                groups = np.asarray(group_fingerprints(table, split.train_ids, ["description"]))
                for fold in range(3):
                    self.assertFalse(set(groups[schedule == fold]) & set(groups[schedule != fold]))
                self.assertEqual(split, nested_splits(table, full, ["description"], (36, 72, "all"))[size])

    def test_heldout_content_does_not_change_nested_training_order_or_teacher(self):
        table, anchor = table_and_outer()
        full = fixed_development_split(table, anchor, ["description"], 0)
        original = nested_splits(table, full, ["description"], (36, 72, "all"))
        frame = table.frame
        frame.loc[list(full.test_ids + anchor.test_ids), "target"] = "heldout-only-label"
        frame.loc[list(anchor.test_ids), "description"] = "reserved-only-text"
        changed = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        rebound = replace(full, dataset_fingerprint=changed.fingerprint())
        actual = nested_splits(changed, rebound, ["description"], (36, 72, "all"))
        for key in original:
            self.assertEqual(original[key].partitions, actual[key].partitions)
            np.testing.assert_array_equal(teacher_fold_ids(table, original[key].train_ids, ["description"]),
                                          teacher_fold_ids(changed, actual[key].train_ids, ["description"]))

    def test_outer_content_does_not_change_development_assignments(self):
        table, anchor = table_and_outer()
        original = fixed_development_split(table, anchor, ["description"], 0)
        frame = table.frame
        frame.loc[list(anchor.test_ids), "target"] = "reserved-only-label"
        frame.loc[list(anchor.test_ids), "description"] = "reserved-only-text"
        changed = DatasetTable(frame, table.schema, row_ids=table.row_ids)
        actual = fixed_development_split(changed, replace(anchor, dataset_fingerprint=changed.fingerprint()), ["description"], 0)
        self.assertEqual(original.partitions, actual.partitions)

    def test_infeasible_or_invalid_sizes_fail_without_silent_skips(self):
        table, anchor = table_and_outer()
        full = fixed_development_split(table, anchor, ["description"], 0)
        for sizes in ((72, 36, "all"), (36, 36, "all"), (900, "all"), (True, "all"), (36,)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                nested_splits(table, full, ["description"], sizes)
        with self.assertRaisesRegex(ValueError, "class|group"):
            nested_splits(table, full, ["description"], (3, "all"))

    def test_scaling_keeps_model_recipe_and_separates_old_pilot(self):
        old = json.loads((ROOT / "configs/phase4_pilot_checkpointed.json").read_text())
        new = json.loads((ROOT / "configs/phase4_scaling.json").read_text())
        for key in ("encoder", "pipeline", "teacher", "tar_execution", "methods", "outer_folds"):
            self.assertEqual(old[key], new[key])
        self.assertEqual(new["sizes"], [900, 2700, "all"])
        self.assertEqual(new["outer_test_predictions"], 0)

    def test_thin_notebook_has_audit_and_explicit_launch_guard(self):
        notebook = json.loads((ROOT / "notebooks/phase4_scaling.ipynb").read_text())
        sources = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, "scaling_notebook_cell", "exec")
                sources.append(source)
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        with self.assertRaisesRegex(RuntimeError, "LAUNCH_SCALING = True"):
            exec(sources[-1], {})
        self.assertIn('"--prior-run", str(PRIOR_RUN)', sources[-2])
        self.assertIn('"scripts/check_tar_memory.py"', sources[-2])
        self.assertIn('"scripts/run_scaling.py"', sources[-1])
        self.assertNotIn('"scripts/run_pilot.py"', sources[-1])


class ScalingRunnerTests(unittest.TestCase):
    def fixture(self, root):
        table, anchor = table_and_outer("regression")
        full = fixed_development_split(table, anchor, ["description"], 0)
        splits = nested_splits(table, full, ["description"], (36, 72, "all"))
        config = json.loads((ROOT / "configs/phase4_scaling.json").read_text())
        config.update(outer_folds=[0], sizes=[36, 72, "all"], prior_run_id=AUDIT["run_id"])
        config["pipeline"]["training"].update(epochs=1, patience=3)
        config["pipeline"]["downstream_estimators"] = 3
        config["teacher"]["iterations"] = 3
        cases = {f"{table.schema.dataset_id}__fold0__size{size}": {
            "dataset_id": table.schema.dataset_id, "outer_fold": 0, "size": size, "task": "regression",
            "scope": "scaling", "teacher_grouped": True, "group_columns": ["description"], "methods": list(p.METHODS), "split": split}
            for size, split in splits.items()}
        (root / "config.json").write_text(json.dumps(config))
        for key in ("datasets_manifest", "anchor_review", "anchor_rows"):
            path = root / config[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        for key in ("pyproject.toml", "uv.lock"):
            (root / key).write_text("fixture")
        return table, cases

    def patches(self, table, cases):
        stack = ExitStack()
        stack.enter_context(patch.object(p, "_git", git_stub))
        stack.enter_context(patch.object(p, "_load_tables", return_value=({table.schema.dataset_id: table}, {})))
        stack.enter_context(patch.object(scaling, "build_scaling_cases", return_value=cases))
        stack.enter_context(patch.object(p, "FrozenE5Encoder", Encoder))
        stack.enter_context(patch.object(p, "CachedFrozenEncoder", side_effect=lambda e, d: CachedFrozenEncoder(e, d, allow_test_encoder=True)))
        return stack

    def test_real_fits_grouped_teacher_train_scopes_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, cases = self.fixture(root)
            with self.patches(table, cases):
                result = scaling.run_scaling(root, Path("config.json"), run_id="partial", methods=["frozen", "direct", "residual"], device="cpu", prior_audit=AUDIT)
                self.assertEqual(result["completed_stages"], 9)
                self.assertFalse(result["assessment_barrier_satisfied"])
                self.assertEqual(result["completed_assessments"], 0)
                self.assertEqual(result["datasets"], {})
                directory = root / "results/raw/partial"
                self.assertFalse((directory / "assessment").exists())
                self.assertFalse((directory / "outer_evaluation").exists())
                state = json.loads((directory / "manifest.json").read_text())
                for name, case in cases.items():
                    entry = state["stages"][name + "__residual"]
                    record = json.loads((directory / entry["attempt"] / "manifest.json").read_text())
                    self.assertEqual(record["optimization"]["optimizer_steps_including_resumed_history"],
                                     int(np.ceil(len(case["split"].train_ids) / 32)))
                    teacher = TeacherArtifacts.load(directory / entry["attempt"] / "teacher")
                    self.assertEqual(teacher.train_ids, case["split"].train_ids)
                    np.testing.assert_array_equal(teacher.fold_ids, teacher_fold_ids(table, teacher.train_ids, ["description"]))
                    for relative in ("bundle/pca/metadata.json", "bundle/structured/metadata.json", "bundle/downstream/metadata.json"):
                        metadata = json.loads((directory / entry["attempt"] / relative).read_text())
                        self.assertEqual(tuple(metadata["fit_ids"]), case["split"].train_ids)
                with patch.object(p, "train_method", side_effect=AssertionError("retrained")):
                    resumed = scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)
                self.assertEqual(resumed["completed_stages"], 9)
                changed = dict(AUDIT, changed_binding=True)
                with self.assertRaisesRegex(ValueError, "provenance"):
                    scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=changed)
                first = next(iter(state["stages"].values()))
                (directory / first["attempt"] / "extra.txt").write_text("changed inventory")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)

    def test_failure_is_preserved_in_new_attempt_and_ledger_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, cases = self.fixture(root)
            with self.patches(table, cases):
                with patch.object(p, "train_method", side_effect=RuntimeError("fixture failure")), self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    scaling.run_scaling(root, Path("config.json"), run_id="failed", methods=["frozen"], device="cpu", prior_audit=AUDIT)
                directory = root / "results/raw/failed"
                result = scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)
                self.assertEqual(result["completed_stages"], 3)
                manifest = directory / "manifest.json"
                state = json.loads(manifest.read_text())
                job = next(iter(state["stages"]))
                entry = state["stages"][job]
                self.assertEqual(entry["attempt_index"], 2)
                self.assertTrue((directory / "attempts" / f"{job}__attempt1/failure.txt").exists())
                entry["status"] = "running"
                p._write(manifest, state)
                with patch.object(p, "train_method", side_effect=AssertionError("retrained after durable completion")):
                    result = scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)
                self.assertEqual(result["completed_stages"], 3)

    def test_assessment_barrier_durable_prediction_recovery_and_tamper_rejection(self):
        # Frozen copies stand in for other method fits solely to exercise the
        # complete ledger barrier. This fixture does not execute/validate TAR.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, cases = self.fixture(root)
            with self.patches(table, cases):
                scaling.run_scaling(root, Path("config.json"), run_id="barrier", methods=["frozen"], device="cpu", prior_audit=AUDIT)
                directory = root / "results/raw/barrier"
                manifest = directory / "manifest.json"
                state = json.loads(manifest.read_text())
                for name in cases:
                    origin_entry = state["stages"][name + "__frozen"]
                    for method in ("direct", "residual", "tar"):
                        job = name + "__" + method
                        target = directory / "attempts" / f"{job}__attempt1"
                        shutil.copytree(directory / origin_entry["attempt"], target)
                        record = json.loads((target / "manifest.json").read_text())
                        record.update(job=job, method=method)
                        record["context"]["job"] = job
                        p._write(target / "manifest.json", record)
                        state["stages"][job] = {**origin_entry, "attempt": target.relative_to(directory).as_posix(), "manifest_sha256": p.sha256_file(target / "manifest.json")}
                p._write(manifest, state)
                with patch.object(p, "train_method", side_effect=AssertionError("retrained")):
                    result = scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)
                self.assertTrue(result["assessment_barrier_satisfied"])
                self.assertEqual(result["completed_assessments"], 12)
                self.assertEqual(result["outer_test_predictions"], 0)
                state = json.loads(manifest.read_text())
                job = next(iter(state["assessments"]))
                state["assessments"].pop(job)
                p._write(manifest, state)
                (directory / "assessment" / job / "result.json").unlink()
                with patch.object(p, "load_method", side_effect=AssertionError("repeated D inference")):
                    recovered = scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)
                self.assertEqual(recovered["datasets"], result["datasets"])
                np.save(directory / "assessment" / job / "assessment_predictions.npy", np.zeros(12))
                with self.assertRaisesRegex(ValueError, "artifacts changed"):
                    scaling.run_scaling(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu", prior_audit=AUDIT)

    def test_gpu_and_audit_are_required_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "CUDA GPU"):
                scaling.run_scaling(root, Path("config.json"), device="cpu", prior_audit=AUDIT)
            with self.assertRaisesRegex(ValueError, "prior-pilot"):
                scaling.run_scaling(root, Path("config.json"), device="cpu", methods=["frozen"])


if __name__ == "__main__":
    unittest.main()
