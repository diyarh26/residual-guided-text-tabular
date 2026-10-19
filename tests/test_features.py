"""No-network tests of frozen text and training-only feature preprocessing."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.features import (E5_FILE_HASHES, FieldPCA, FrozenE5Config,
                             FrozenE5Encoder, StructuredPreprocessor)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {"input_ids": torch.tensor([[1, 2, 999] for _ in texts]),
                "attention_mask": torch.tensor([[1, 1, 0] for _ in texts])}


class FakeE5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(1000, 384)
        self.dropout = torch.nn.Dropout(.9)
        self.forward_modes = []
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[1, 0] = 2
            self.embedding.weight[2, 1] = 4
            self.embedding.weight[999] = 1e6

    def forward(self, input_ids, attention_mask):
        self.forward_modes.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class FrozenEncoderTests(unittest.TestCase):
    def test_prefix_missing_order_masked_pooling_eval_freezing_and_batches(self):
        model, tokenizer = FakeE5(), FakeTokenizer()
        encoder = FrozenE5Encoder(FrozenE5Config("fake-test", batch_size=1, max_length=8), model=model, tokenizer=tokenizer)
        before = encoder.state_sha256()
        frame = pd.DataFrame({"later": ["alpha", None], "first": [np.nan, "beta"], "not_text": [1, 2]}, index=["a", "b"])
        result = encoder.encode(frame, ("first", "later"))
        self.assertEqual(result.shape, (2, 2, 384))
        self.assertEqual(result.dtype, np.float32)
        expected = np.zeros(384)
        expected[:2] = [1 / np.sqrt(5), 2 / np.sqrt(5)]
        np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1e-7)
        self.assertEqual([call[0][0] for call in tokenizer.calls], ["passage: first: ", "passage: first: beta", "passage: later: alpha", "passage: later: "])
        self.assertTrue(all(call[1] == {"max_length": 8, "truncation": True, "padding": "max_length", "return_tensors": "pt"} for call in tokenizer.calls))
        self.assertTrue(all(mode == (False, False, True) for mode in model.forward_modes))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(before, encoder.state_sha256())
        np.testing.assert_array_equal(encoder.encode(frame, ("first", "later")), result)
        self.assertTrue(encoder.metadata()["test_injection"])

    def test_freeze_guards_and_invalid_inputs(self):
        encoder = FrozenE5Encoder(FrozenE5Config("fake-test"), model=FakeE5(), tokenizer=FakeTokenizer())
        frame = pd.DataFrame({"text": ["example"]}, index=["only"])
        encoder.model.dropout.train()
        with self.assertRaisesRegex(RuntimeError, "evaluation"):
            encoder.encode(frame, ("text",))
        encoder.model.eval()
        next(encoder.model.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "requires_grad"):
            encoder.assert_frozen()
        next(encoder.model.parameters()).requires_grad_(False)
        for columns in ((), ("missing",), ("text", "text")):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                encoder.encode(frame, columns)
        with self.assertRaisesRegex(ValueError, "both"):
            FrozenE5Encoder(FrozenE5Config("fake"), model=FakeE5())
        with self.assertRaisesRegex(ValueError, "supported scalar"):
            encoder.encode(pd.DataFrame({"text": [["bad"]]}), ("text",))

    def test_pinned_asset_hashes_and_config_fail_before_network_loading(self):
        assets = json.loads((ROOT / "configs/phase1_assets.json").read_text())
        self.assertEqual(E5_FILE_HASHES, assets["model"]["sha256"])
        for kwargs in ({"revision": "floating"}, {"identifier": "another"}, {"batch_size": 0}, {"max_length": 513}, {"batch_size": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FrozenE5Config("local", **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "config.json").write_text("incorrect")
            with self.assertRaisesRegex(ValueError, "checksum mismatch: config.json"):
                FrozenE5Encoder(FrozenE5Config(str(path)))
            with self.assertRaisesRegex(ValueError, "all six"):
                FrozenE5Encoder(FrozenE5Config(str(path)), expected_hashes={"config.json": "0" * 64})
            (path / "added_tokens.json").write_text('{"unverified": 30000}')
            with self.assertRaisesRegex(ValueError, "Unexpected unverified E5 asset files"):
                FrozenE5Encoder(FrozenE5Config(str(path)))


class FieldPCATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = np.random.default_rng(42).normal(size=(36, 2, 384))
        cls.heldout = np.random.default_rng(71).normal(size=(5, 2, 384)) + 100
        cls.ids = tuple(f"fit-{index}" for index in range(len(cls.train)))

    def test_per_field_fit_matches_train_only_reference_and_stays_fixed_on_heldout(self):
        pca = FieldPCA(("body", "title"), seed=17).fit(self.train, self.ids)
        before = pca.components_.copy(), pca.means_.copy()
        result = pca.transform(self.heldout)
        self.assertEqual(result.shape, (5, 2, 30))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(pca.fit_ids, self.ids)
        for field in range(2):
            reference = PCA(n_components=30, random_state=17).fit(self.train[:, field])
            np.testing.assert_allclose(pca.components_[field], reference.components_)
            np.testing.assert_allclose(result[:, field], reference.transform(self.heldout[:, field]), atol=1e-4)
        pca.transform(self.heldout * -100)
        np.testing.assert_array_equal(before[0], pca.components_)
        np.testing.assert_array_equal(before[1], pca.means_)
        np.testing.assert_allclose(pca.means_, self.train.mean(axis=0))
        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            pca.fit(np.concatenate([self.train, self.heldout]), self.ids + tuple(f"val-{i}" for i in range(5)))

    def test_shape_minimum_rows_ids_finite_and_unfitted_rejections(self):
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            FieldPCA(("body", "title")).transform(self.train)
        with self.assertRaisesRegex(ValueError, "at least 30"):
            FieldPCA(("body", "title")).fit(self.train[:29], self.ids[:29])
        with self.assertRaisesRegex(ValueError, "match"):
            FieldPCA(("body", "title")).fit(self.train, self.ids[:-1])
        with self.assertRaisesRegex(ValueError, "shape"):
            FieldPCA(("body", "title")).fit(self.train[:, 0], self.ids)
        invalid = self.train.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            FieldPCA(("body", "title")).fit(invalid, self.ids)
        with self.assertRaisesRegex(ValueError, "seed"):
            FieldPCA(("body",), seed=2**32)

    def test_save_load_prediction_identity_checksum_and_no_overwrite(self):
        pca = FieldPCA(("body", "title")).fit(self.train, self.ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pca"
            pca.save(path)
            restored = FieldPCA.load(path)
            self.assertEqual(restored.metadata(), pca.metadata())
            np.testing.assert_array_equal(restored.transform(self.heldout), pca.transform(self.heldout))
            with self.assertRaises(FileExistsError):
                pca.save(path)
            with (path / "state.npz").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FieldPCA.load(path)


class StructuredFeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"amount": [1., np.nan, 5., 9.], "all_missing": [np.nan] * 4,
                                   "kind": pd.Series([1, "1", None, "known"], dtype=object).to_numpy()},
                                  index=["fit-a", "fit-b", "fit-c", "fit-d"])
        self.val = pd.DataFrame({"amount": [1000., np.nan], "all_missing": [7., np.nan], "kind": ["new", None]},
                                index=["val-a", "val-b"])
        self.prep = StructuredPreprocessor(("amount", "all_missing"), ("kind",)).fit(self.train, tuple(self.train.index))

    def test_median_scaling_all_missing_unknown_and_typed_categories(self):
        numbers, categories = self.prep.transform(self.train)
        np.testing.assert_allclose(self.prep.medians_, [5., 0.])
        np.testing.assert_allclose(self.prep.means_, [5., 0.])
        np.testing.assert_allclose(self.prep.scales_, [np.sqrt(8.), 1.])
        np.testing.assert_allclose(numbers[:, 0], [-np.sqrt(2.), 0., 0., np.sqrt(2.)], atol=1e-7)
        self.assertEqual(numbers.dtype, np.float32)
        self.assertEqual(categories.dtype, np.int64)
        self.assertNotEqual(categories[0, 0], categories[1, 0])
        self.assertEqual(categories[2, 0], 0)
        self.assertEqual(self.prep.cardinalities, (5,))
        val_num, val_cat = self.prep.transform(self.val)
        np.testing.assert_array_equal(val_cat[:, 0], [1, 0])
        np.testing.assert_array_equal(val_num[:, 1], [7., 0.])

    def test_validation_perturbation_does_not_change_any_fitted_state(self):
        before = self.prep.metadata(), self.prep.medians_.copy(), self.prep.means_.copy(), self.prep.scales_.copy()
        self.prep.transform(self.val)
        changed = self.val.copy()
        changed["amount"], changed["kind"] = [-1e7, 1e7], ["completely", "different"]
        self.prep.transform(changed)
        self.assertEqual(before[0], self.prep.metadata())
        for expected, actual in zip(before[1:], (self.prep.medians_, self.prep.means_, self.prep.scales_)):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(set(self.prep.fit_ids), set(self.train.index))
        self.assertFalse(set(self.prep.fit_ids) & set(self.val.index))
        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            self.prep.fit(self.val, tuple(self.val.index))

    def test_exact_feature_order_alignment_finite_and_unfitted_guards(self):
        for bad in (self.val[["kind", "amount", "all_missing"]], self.val.assign(label=1), self.val.drop(columns=["kind"])):
            with self.assertRaisesRegex(ValueError, "columns/order"):
                self.prep.transform(bad)
        with self.assertRaisesRegex(ValueError, "exactly align"):
            StructuredPreprocessor(("amount", "all_missing"), ("kind",)).fit(self.train, tuple(reversed(self.train.index)))
        invalid = self.val.copy()
        invalid.loc["val-a", "amount"] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            self.prep.transform(invalid)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            StructuredPreprocessor(("amount",), ()).transform(self.train[["amount"]])

    def test_numeric_only_categorical_only_and_empty_structured_shapes(self):
        for numerical, categorical in ((("amount",), ()), ((), ("kind",)), ((), ())):
            with self.subTest(numerical=numerical, categorical=categorical):
                train = self.train.loc[:, list(numerical + categorical)]
                prep = StructuredPreprocessor(numerical, categorical).fit(train, tuple(train.index))
                numbers, categories = prep.transform(train)
                self.assertEqual(numbers.shape, (4, len(numerical)))
                self.assertEqual(categories.shape, (4, len(categorical)))

    def test_save_load_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured"
            self.prep.save(path)
            restored = StructuredPreprocessor.load(path)
            self.assertEqual(restored.metadata(), self.prep.metadata())
            for actual, expected in zip(restored.transform(self.val), self.prep.transform(self.val)):
                np.testing.assert_array_equal(actual, expected)
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["cardinalities"] = [1000]
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "cardinalities"):
                StructuredPreprocessor.load(path)


if __name__ == "__main__":
    unittest.main()
