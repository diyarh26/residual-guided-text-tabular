"""Meaningful protocol checks using original pinned source; no network or training.

The deterministic encoder stub is used only to audit original preprocessing fit
scope. The recorded real baseline must use the actual pinned E5 model.
"""

from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/MulTaBench")]

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from refutab.baseline import audit_splits, fingerprint, run_baseline
from tabstar.preprocessing.splits import split_to_test, split_to_val
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.preprocessing.sampling import subsample_dataset
from multabench.baselines.training.metrics import calculate_metric
from multabench.datasets.objects import SupervisedTask
from multabench.e5.e5_finetune import encode_texts_with_e5


class SplitProtocolTests(unittest.TestCase):
    def test_original_reduced_training_cap_keeps_expected_test_budget(self) -> None:
        x = pd.DataFrame({"value": np.arange(5091)})
        y = pd.Series(np.arange(5091) % 4)
        sampled_x, sampled_y = subsample_dataset(x, y, True, train_examples=1000, fold=0)
        self.assertEqual(len(sampled_x), 1509)
        outer, test, outer_y, _ = split_to_test(sampled_x, sampled_y, True, fold=0, train_examples=1000)
        fit, val, _, _ = split_to_val(outer, outer_y, True)
        self.assertEqual([len(fit), len(val), len(test)], [900, 100, 509])
        audit_splits({"fit": fit.index.tolist(), "validation": val.index.tolist(), "test": test.index.tolist()}, sampled_x.index.tolist())

    def test_original_multiclass_partition_is_deterministic_and_disjoint(self) -> None:
        x = pd.DataFrame({"value": np.arange(1000)}, index=np.arange(10000, 11000))
        y = pd.Series(np.arange(1000) % 4, index=x.index)
        outer, test, outer_y, _ = split_to_test(x, y, True, fold=0, train_examples=10000)
        fit, val, _, _ = split_to_val(outer, outer_y, True)
        splits = {"fit": fit.index.tolist(), "validation": val.index.tolist(), "test": test.index.tolist()}
        report = audit_splits(splits, x.index.tolist())
        self.assertEqual([report[k]["rows"] for k in splits], [810, 90, 100])
        repeat = split_to_test(x, y, True, fold=0, train_examples=10000)
        self.assertEqual(test.index.tolist(), repeat[1].index.tolist())
        # The original internal default is fold=-1, not the outer fold.
        alternate_val = split_to_val(outer, outer_y, True, fold=0)[1]
        self.assertNotEqual(val.index.tolist(), alternate_val.index.tolist())

    def test_original_binary_adds_two_forced_test_rows(self) -> None:
        x = pd.DataFrame({"value": np.arange(1000)})
        y = pd.Series(np.arange(1000) % 2)
        train, test, _, test_y = split_to_test(x, y, True, fold=0)
        self.assertEqual(len(test), 102)
        self.assertEqual(len(train), 898)
        self.assertEqual(set(test_y), {0, 1})
        audit_splits({"train": train.index.tolist(), "test": test.index.tolist()}, x.index.tolist())

    def test_audit_rejects_overlap_and_missing_ids(self) -> None:
        with self.assertRaises(ValueError):
            audit_splits({"fit": [1, 2], "test": [2, 3]}, [1, 2, 3])
        with self.assertRaises(ValueError):
            audit_splits({"fit": [1], "test": [3]}, [1, 2, 3])
        self.assertNotEqual(fingerprint([1, 2]), fingerprint([2, 1]))


class OriginalPreprocessorTests(unittest.TestCase):
    def test_pca_30_fits_training_only_and_unknown_category_transforms(self) -> None:
        x = pd.DataFrame({"Review": [f"training text {i}" for i in range(160)], "Group": ["a", "b"] * 80, "Value": np.arange(160, dtype=float)})
        y = pd.Series(["negative", "positive"] * 80)
        seen: list[tuple[str, list[str]]] = []
        def deterministic_embeddings(texts, col_name, **kwargs):
            seen.append((col_name, list(texts)))
            return np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")).normal(size=384) for text in texts]).astype(np.float32)
        fake_model = torch.nn.Linear(1, 1).eval()
        model = LightGBM(problem_type=SupervisedTask.BINARY, device=torch.device("cpu"), pca_components=30)
        with redirect_stdout(io.StringIO()), patch("multabench.baselines.preprocessing.text_embeddings.get_vanilla_e5", return_value=(fake_model, object())), patch("multabench.baselines.preprocessing.text_embeddings.encode_texts_with_e5", side_effect=deterministic_embeddings):
            model.fit_preprocessor(x, y)
            self.assertEqual(seen, [("Review", x.Review.tolist())])
            pca = model.text_transformers["Review"].encoder
            self.assertEqual(pca.n_samples_, 160)
            self.assertEqual(pca.n_components_, 30)
            mean_before, components_before = pca.mean_.copy(), pca.components_.copy()
            heldout = pd.DataFrame({"Review": ["held-out unseen text"], "Group": ["new category"], "Value": [1e9]}, index=[9999])
            transformed, encoded_y = model.transform_preprocessor(heldout, pd.Series(["positive"], index=heldout.index))
            self.assertEqual(transformed.index.tolist(), [9999])
            self.assertEqual(transformed.shape, (1, 32))
            self.assertTrue(np.isfinite(transformed.to_numpy()).all())
            self.assertEqual(len(encoded_y), 1)
            np.testing.assert_array_equal(pca.mean_, mean_before)
            np.testing.assert_array_equal(pca.components_, components_before)
            self.assertEqual(seen[-1], ("Review", ["held-out unseen text"]))

    def test_original_e5_encoding_disables_gradients_and_dropout(self) -> None:
        observed: list[tuple[bool, bool]] = []
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
            def forward(self, input_ids, attention_mask):
                observed.append((self.training, torch.is_grad_enabled()))
                return {"last_hidden_state": self.weight[input_ids]}
        class TinyTokenizer:
            def __call__(self, texts, **kwargs):
                self.texts = texts
                return {"input_ids": torch.tensor([[0, 1, 2]] * len(texts)), "attention_mask": torch.ones((len(texts), 3), dtype=torch.long)}
        model, tokenizer = TinyModel(), TinyTokenizer()
        initial = model.weight.detach().clone()
        result = encode_texts_with_e5(["a", "b"], "Review", model, tokenizer, "cpu")
        self.assertEqual(tokenizer.texts, ["passage: Review: a", "passage: Review: b"])
        self.assertEqual(observed, [(False, False)])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        torch.testing.assert_close(model.weight, initial)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)


class OriginalMetricTests(unittest.TestCase):
    def test_multiclass_macro_ovr_and_present_class_fallback(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array([[.7, .2, .1], [.2, .5, .3], [.1, .3, .6], [.4, .5, .1], [.3, .5, .2], [.2, .1, .7]])
        expected = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        self.assertAlmostEqual(calculate_metric(y, probs, 3).score, expected)
        mask = y != 1
        subset_y, subset_p = y[mask], probs[mask]
        expected_subset = np.mean([roc_auc_score((subset_y == cls).astype(int), subset_p[:, cls]) for cls in [0, 2]])
        self.assertAlmostEqual(calculate_metric(subset_y, subset_p, 3).score, expected_subset)


class ManifestSafetyTests(unittest.TestCase):
    def test_preflight_failure_is_persisted_and_existing_run_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"output_root": "runs"}), encoding="utf-8")
            with patch("refutab.baseline._git", side_effect=ValueError("deliberate preflight failure")):
                with self.assertRaisesRegex(ValueError, "deliberate preflight failure"):
                    run_baseline(root, Path("config.json"), run_id="failed-test")
            manifest_path = root / "runs/failed-test/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIsNone(manifest["cost"]["incremental_service_charge_usd"])
            self.assertEqual(manifest["failure"]["type"], "ValueError")
            self.assertTrue((manifest_path.parent / "failure.txt").is_file())
            original = manifest_path.read_bytes()
            with self.assertRaises(FileExistsError):
                run_baseline(root, Path("config.json"), run_id="failed-test")
            self.assertEqual(manifest_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
