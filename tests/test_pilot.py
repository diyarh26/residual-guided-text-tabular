"""Real tiny-model integration for the pilot barrier, ledgers and continuation."""

from contextlib import ExitStack
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

from refutab import pilot
from refutab.pilot_features import CachedFrozenEncoder
from refutab.synthetic import make_synthetic
import test_pipeline as fixtures


class Encoder(fixtures.FixtureEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config


def git_stub(root, *args):
    return "a" * 40 if args == ("rev-parse", "HEAD") else ""


class PilotTests(unittest.TestCase):
    def test_checkpointed_variant_keeps_all_registered_training_budgets(self):
        original = json.loads((ROOT / "configs/phase4_pilot.json").read_text(encoding="utf-8"))
        variant = json.loads((ROOT / "configs/phase4_pilot_checkpointed.json").read_text(encoding="utf-8"))
        self.assertEqual({key: value for key, value in variant.items() if key in original and key != "preregistered_date"},
                         {key: value for key, value in original.items() if key != "preregistered_date"})
        self.assertEqual(variant["tar_execution"], {"activation_checkpointing": True})

    def fixture(self, root):
        table, split = make_synthetic("regression")
        config = json.loads((ROOT / "configs/phase4_pilot.json").read_text(encoding="utf-8"))
        config["outer_folds"] = [0]
        config["diagnostics"]["enabled"] = False
        config["diagnostics"]["group_columns"] = {table.schema.dataset_id: ["description"]}
        config["pipeline"]["training"].update(epochs=2, patience=3)
        config["pipeline"]["downstream_estimators"] = 5
        config["teacher"]["iterations"] = 5
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "configs").mkdir()
        (root / config["datasets_manifest"]).write_text("{}", encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock"):
            (root / name).write_text("fixture", encoding="utf-8")
        return table, split

    def patches(self, root, table, split):
        stack = ExitStack()
        stack.enter_context(patch.object(pilot, "_git", git_stub))
        stack.enter_context(patch.object(pilot, "_load_tables", return_value=({table.schema.dataset_id: table}, {})))
        stack.enter_context(patch.object(pilot, "benchmark_split", return_value=split))
        stack.enter_context(patch.object(pilot, "_schema_audit", return_value={"scope": "fixture"}))
        stack.enter_context(patch.object(pilot, "FrozenE5Encoder", Encoder))
        stack.enter_context(patch.object(pilot, "CachedFrozenEncoder", side_effect=lambda e, d: CachedFrozenEncoder(e, d, allow_test_encoder=True)))
        return stack

    def test_partial_cpu_run_never_predicts_outer_test_and_resume_reuses_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, split = self.fixture(root)
            with self.patches(root, table, split):
                summary = pilot.run_pilot(root, Path("config.json"), run_id="partial", methods=["frozen", "direct", "residual"], device="cpu")
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["completed_stages"], 3)
                self.assertEqual(summary["planned_stages"], 4)
                self.assertEqual(summary["test_predictions"], 0)
                self.assertFalse(summary["test_barrier_satisfied"])
                self.assertFalse((root / "results/raw/partial/outer_evaluation").exists())
                self.assertNotIn("synthetic-0000", json.dumps(summary))
                with patch.object(pilot, "train_method", side_effect=AssertionError("Retrained a completed stage")):
                    continued = pilot.run_pilot(root, Path("config.json"), resume_run=Path("results/raw/partial"),
                                                methods=["frozen", "direct", "residual"], device="cpu")
                self.assertEqual(continued["completed_stages"], 3)
                self.assertEqual(continued["test_predictions"], 0)
                stage = next(iter(continued["stages"].values()))
                path = root / "results/raw/partial" / stage["attempt"] / "unexpected.txt"
                path.write_text("changed inventory", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    pilot.run_pilot(root, Path("config.json"), resume_run=Path("results/raw/partial"), methods=["frozen"], device="cpu")

    def test_gpu_is_required_for_tar_before_model_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch.object(pilot, "train_method", side_effect=AssertionError("GPU guard was bypassed")), \
                 self.assertRaisesRegex(ValueError, "CUDA GPU"):
                pilot.run_pilot(root, Path("config.json"), methods=["tar"], device="cpu")

    def test_failed_attempt_keeps_runtime_and_resume_creates_new_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, split = self.fixture(root)
            with self.patches(root, table, split):
                with patch.object(pilot, "train_method", side_effect=RuntimeError("deliberate fixture failure")), \
                     self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    pilot.run_pilot(root, Path("config.json"), run_id="failed", methods=["direct"], device="cpu")
                directory = root / "results/raw/failed"
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "failed")
                failed = next(iter(manifest["stages"].values()))
                first = directory / failed["attempt"]
                self.assertEqual(failed["attempt_index"], 1)
                self.assertGreaterEqual(failed["runtime"]["seconds"], 0)
                resumed = pilot.run_pilot(root, Path("config.json"), resume_run=directory, methods=["direct"], device="cpu")
                self.assertEqual(resumed["completed_stages"], 1)
                current = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(next(iter(current["stages"].values()))["attempt_index"], 2)
                self.assertTrue((first / "failure.txt").is_file())

    def test_ledger_interruption_recovers_completed_attempt_without_refitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, split = self.fixture(root)
            with self.patches(root, table, split):
                pilot.run_pilot(root, Path("config.json"), run_id="ledger", methods=["frozen"], device="cpu")
                path = root / "results/raw/ledger/manifest.json"
                state = json.loads(path.read_text(encoding="utf-8"))
                job = next(iter(state["stages"]))
                state["stages"][job]["status"] = "running"
                for key in ("scores", "runtime", "manifest_sha256"):
                    state["stages"][job].pop(key)
                path.write_text(json.dumps(state), encoding="utf-8")
                with patch.object(pilot, "train_method", side_effect=AssertionError("Re-fit after ledger interruption")):
                    result = pilot.run_pilot(root, Path("config.json"), resume_run=path.parent, methods=["frozen"], device="cpu")
                self.assertEqual(result["completed_stages"], 1)
                self.assertEqual(result["test_predictions"], 0)

    def test_barrier_unlock_and_durable_outer_prediction_recovery(self):
        # A copied frozen fixture stands in for the fourth completed stage only
        # to exercise orchestration. This does not validate or execute TAR.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table, split = self.fixture(root)
            with self.patches(root, table, split):
                pilot.run_pilot(root, Path("config.json"), run_id="barrier", methods=["frozen", "direct", "residual"], device="cpu")
                directory = root / "results/raw/barrier"
                path = directory / "manifest.json"
                state = json.loads(path.read_text(encoding="utf-8"))
                frozen_job = next(job for job in state["stages"] if job.endswith("__frozen"))
                tar_job = frozen_job.removesuffix("__frozen") + "__tar"
                entry = dict(state["stages"][frozen_job])
                origin = directory / entry["attempt"]
                target = directory / "attempts" / f"{tar_job}__attempt1"
                shutil.copytree(origin, target)
                record = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
                record.update(job=tar_job, method="tar")
                record["context"]["job"] = tar_job
                pilot._write(target / "manifest.json", record)
                entry.update(attempt=target.relative_to(directory).as_posix(), manifest_sha256=pilot.sha256_file(target / "manifest.json"))
                state["stages"][tar_job] = entry
                pilot._write(path, state)
                with patch.object(pilot, "train_method", side_effect=AssertionError("Refit at evaluation barrier")):
                    summary = pilot.run_pilot(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu")
                self.assertTrue(summary["test_barrier_satisfied"])
                self.assertEqual(summary["test_predictions"], 4)
                self.assertEqual(summary["status"], "completed")
                for result in summary["outer_tests"].values():
                    self.assertEqual(result["rows"], len(split.test_ids))
                    self.assertEqual(result["row_ids_sha256"], pilot.row_ids_sha256(split.test_ids))
                # Recreate interruption after durable predictions but before scores.
                state = json.loads(path.read_text(encoding="utf-8"))
                state["outer_tests"].pop(frozen_job)
                pilot._write(path, state)
                (directory / "outer_evaluation" / frozen_job / "result.json").unlink()
                with patch.object(pilot, "load_method", side_effect=AssertionError("Repeated test inference")):
                    recovered = pilot.run_pilot(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu")
                self.assertEqual(recovered["outer_tests"], summary["outer_tests"])
                np.save(directory / "outer_evaluation" / frozen_job / "test_predictions.npy", np.zeros(12))
                with self.assertRaisesRegex(ValueError, "outer-test artifacts differ"):
                    pilot.run_pilot(root, Path("config.json"), resume_run=directory, methods=["frozen"], device="cpu")


if __name__ == "__main__":
    unittest.main()
