"""Configured run and CSV prediction boundary tests with explicit text fixtures."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.data import sha256_file
from refutab.fusion_runner import run_fusion
from refutab.pipeline import ReFuTabPipeline
from refutab.synthetic import make_synthetic
import test_pipeline as pipeline_fixtures


class RunnerFixtureEncoder(pipeline_fixtures.FixtureEncoder):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def metadata(self):
        return {"kind": "frozen_e5_fixture", "dimension": 384, "prefix": "fixture", "missing_text": "",
                "pooling": "fixture", "verified_files": {}, "test_injection": True}


def git_stub(root, *args):
    return "a" * 40 if args == ("rev-parse", "HEAD") else ""


class FusionRunnerTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads((ROOT / "configs/phase3_smoke.json").read_text(encoding="utf-8"))
        config["pipeline"]["training"]["epochs"] = 1
        config["pipeline"]["downstream_estimators"] = 5
        config["teacher"]["iterations"] = 5
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for name in ("pyproject.toml", "uv.lock"):
            (root / name).write_text("fixture", encoding="utf-8")

    def test_configured_run_saves_private_bundle_and_aggregate_only_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch("refutab.fusion_runner._git", git_stub), \
                 patch("refutab.fusion_runner.FrozenE5Encoder", RunnerFixtureEncoder):
                result = run_fusion(root, Path("config.json"), run_id="synthetic", summary_path=Path("summary.json"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["test_predictions"], 0)
            self.assertEqual(result["test_scores"], 0)
            self.assertEqual([result["partitions"][name]["rows"] for name in ("train", "validation", "test")], [60, 12, 12])
            self.assertTrue(result["encoder"]["test_injection"])
            self.assertEqual(result["completed_epoch"], 1)
            text = (root / "summary.json").read_text(encoding="utf-8")
            self.assertNotIn("synthetic-0000", text)
            self.assertNotIn(str(root), text)
            for name, digest in result["artifacts"]["files_sha256"].items():
                self.assertEqual(sha256_file(root / "results/raw/synthetic" / name), digest)
            self.assertIn("bundle/downstream/manifest.json", result["artifacts"]["files_sha256"])

    def test_failure_preserves_record_and_run_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch("refutab.fusion_runner._git", return_value="dirty"), self.assertRaisesRegex(ValueError, "Commit"):
                run_fusion(root, Path("config.json"), run_id="failed")
            manifest = json.loads((root / "results/raw/failed/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((root / "results/raw/failed/failure.txt").is_file())
            with self.assertRaises(FileExistsError):
                run_fusion(root, Path("config.json"), run_id="failed")

    def test_csv_prediction_preserves_rows_class_order_and_refuses_targets_and_overwrite(self):
        table, splits, _, model = pipeline_fixtures.PipelineTests().fit()
        spec = importlib.util.spec_from_file_location("refutab_predict_cli", ROOT / "scripts/predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = table.features(splits.validation_ids)
            frame.index.name = "row_id"
            frame.to_csv(root / "input.csv")
            result = module.predict_csv(model, root / "input.csv", root / "predictions.csv")
            np.testing.assert_array_equal(result.to_numpy(), model.predict(frame))
            self.assertEqual(tuple(result.index), splits.validation_ids)
            with self.assertRaises(FileExistsError):
                module.predict_csv(model, root / "input.csv", root / "predictions.csv")
            frame["target"] = "forbidden"
            frame.to_csv(root / "bad.csv")
            with self.assertRaisesRegex(ValueError, "exactly"):
                module.predict_csv(model, root / "bad.csv", root / "new.csv")


if __name__ == "__main__":
    unittest.main()
