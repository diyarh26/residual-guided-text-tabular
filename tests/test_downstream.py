"""Real LightGBM checks for final feature blocks and standalone artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from refutab import downstream
from refutab.downstream import DownstreamConfig, DownstreamModel


def fixture(task: str = "multiclass"):
    count = 120
    ids = [f"train-{index:03d}" for index in np.random.default_rng(7).permutation(count)]
    frame = pd.DataFrame({"f0": np.linspace(-3, 3, count),
                          'odd["column"]': [None, "Unknown Value", 1, "1", True, 1.0] * 20}, index=ids)
    frame.loc[ids[4], "f0"] = np.nan
    fused = np.random.default_rng(12).normal(size=(count, 60))
    if task == "regression":
        target, order = fused[:, 0] * 2 + np.nan_to_num(frame["f0"].to_numpy()), ()
    elif task == "binary":
        target, order = (fused[:, 0] > 0).astype(np.int64), ("positive first", "negative second")
    else:
        target, order = np.arange(count, dtype=np.int64) % 3, ("zebra", "ant", "moose")
    validation = pd.DataFrame({"f0": [np.nan, 1e6, -1e6],
                               'odd["column"]': [None, "unseen", "Unknown Value"]},
                              index=["validation-2", "validation-0", "validation-1"])
    validation_fused = np.random.default_rng(91).normal(size=(3, 60))
    return frame, fused, np.asarray(target), order, validation, validation_fused


def fit(task: str = "multiclass") -> DownstreamModel:
    frame, fused, target, order, _, _ = fixture(task)
    return DownstreamModel(DownstreamConfig(task)).fit(frame, fused, target,
                                                      categorical_columns=('odd["column"]',),
                                                      class_order=order, fit_ids=frame.index)


class DownstreamTests(unittest.TestCase):
    def test_real_model_all_tasks_has_original_defaults_and_only_training_rows(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                frame, fused, target, order, validation, val_fused = fixture(task)
                model_class = downstream.LGBMRegressor if task == "regression" else downstream.LGBMClassifier
                original_fit = model_class.fit
                actual_fits = []

                def observed_fit(estimator, features, labels, **kwargs):
                    actual_fits.append((features.copy(), np.asarray(labels).copy(), kwargs))
                    return original_fit(estimator, features, labels, **kwargs)

                with patch.object(model_class, "fit", new=observed_fit):
                    model = fit(task)
                self.assertEqual(len(actual_fits), 1)
                features, labels, kwargs = actual_fits[0]
                self.assertEqual(tuple(features.index), tuple(frame.index))
                self.assertEqual(tuple(features.columns), model.feature_names)
                np.testing.assert_array_equal(labels, target)
                self.assertEqual(kwargs, {"categorical_feature": ["f1"]})
                self.assertEqual(model.config.n_estimators, 100)
                self.assertEqual(model.booster_.params["random_state"], 42)
                self.assertEqual(model.booster_.params["num_threads"], 2)
                self.assertEqual(model.class_order, order)
                self.assertEqual(model.fit_ids, tuple(frame.index))
                self.assertEqual(model.metadata()["selection_ids"], [])
                self.assertFalse(set(model.fit_ids) & set(validation.index))
                predictions = model.predict(validation, val_fused)
                self.assertTrue(np.isfinite(predictions).all())
                if task == "regression":
                    self.assertEqual(predictions.shape, (3,))
                else:
                    self.assertEqual(predictions.shape, (3, len(order)))
                    np.testing.assert_allclose(predictions.sum(axis=1), 1)
                np.testing.assert_array_equal(features.to_numpy(), model.transform(frame, fused).to_numpy())

    def test_original_structured_then_ordered_fused_blocks_and_collision_safe_names(self):
        frame, fused, _, _, validation, val_fused = fixture()
        model = fit()
        features = model.transform(validation, val_fused)
        self.assertEqual(features.shape, (3, 62))
        self.assertEqual(tuple(features.columns), tuple(f"f{index}" for index in range(62)))
        self.assertEqual(tuple(features.index), tuple(validation.index))
        np.testing.assert_array_equal(features["f0"].to_numpy(), validation["f0"].to_numpy())
        np.testing.assert_array_equal(features.iloc[:, 2:32].to_numpy(), val_fused[:, :30])
        np.testing.assert_array_equal(features.iloc[:, 32:].to_numpy(), val_fused[:, 30:])
        self.assertEqual(model.feature_layout[:2], [{"name": "f0", "kind": "structured", "column": "f0"},
                                                 {"name": "f1", "kind": "structured", "column": 'odd["column"]'}])
        self.assertEqual(model.feature_layout[-1], {"name": "f61", "kind": "fused", "coordinate": 59})
        self.assertEqual(fused.shape[1], model.fused_dimension)

    def test_original_helpers_fit_train_only_and_typed_categories_missing_unknown(self):
        original = downstream._load_original_categorical()
        with patch.object(original, "fit_categorical_encoders", wraps=original.fit_categorical_encoders) as fit_helper:
            with patch.object(original, "transform_categorical_features", wraps=original.transform_categorical_features) as transform_helper:
                model = fit()
        self.assertEqual(fit_helper.call_count, 1)
        self.assertEqual(transform_helper.call_count, 1)
        train = fixture()[0]
        self.assertEqual(tuple(fit_helper.call_args.args[0].index), tuple(train.index))
        probe = pd.DataFrame({"f0": [0.] * 7,
                              'odd["column"]': [1, "1", True, 1.0, None, "unseen", "Unknown Value"]},
                             index=[f"new-{i}" for i in range(7)])
        encoded = model.transform(probe, np.zeros((7, 60)))["f1"].to_numpy()
        self.assertEqual(len(set(encoded[:4])), 4)
        self.assertEqual(encoded[4], encoded[5])
        self.assertNotEqual(encoded[4], encoded[6])
        self.assertNotIn('str:"unseen"', model.category_classes['odd["column"]'])
        self.assertEqual(model.category_classes['odd["column"]'].count(model.missing_token), 1)

    def test_heldout_values_cannot_change_training_model_or_mappings(self):
        model = fit("binary")
        _, _, _, _, validation, fused = fixture("binary")
        initial_model = model.booster_.model_to_string()
        initial_metadata = model.metadata()
        common = model.predict(validation, fused)
        changed = validation.copy()
        changed["f0"] = [0, -1e100, 1e100]
        changed['odd["column"]'] = ["new-a", "new-b", "new-c"]
        model.predict(changed, fused * 99)
        self.assertEqual(model.booster_.model_to_string(), initial_model)
        self.assertEqual(model.metadata(), initial_metadata)
        np.testing.assert_array_equal(model.predict(validation, fused), common)
        # A second identical fitting call remains independent of what was
        # supplied to the first model's transform-only held-out path.
        other = fit("binary")
        self.assertEqual(model.booster_.model_to_string(), other.booster_.model_to_string())

    def test_roundtrip_all_tasks_needs_neither_teacher_head_nor_upstream_checkout(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                model = fit(task)
                _, _, _, _, validation, fused = fixture(task)
                expected = model.predict(validation, fused)
                artifact = model.save(Path(temporary) / "saved")
                self.assertEqual({path.name for path in artifact.iterdir()}, {"model.txt", "metadata.json", "manifest.json"})
                with patch.object(downstream, "_load_original_categorical", side_effect=AssertionError("upstream is unavailable")):
                    loaded = DownstreamModel.load(artifact)
                    np.testing.assert_array_equal(loaded.predict(validation, fused), expected)
                self.assertEqual(loaded.class_order, model.class_order)
                self.assertEqual(loaded.fit_ids, model.fit_ids)
                self.assertEqual(loaded.metadata(), model.metadata())
                with self.assertRaises(FileExistsError):
                    model.save(artifact)

    def test_alignment_schema_dimensions_and_invalid_values_fail_before_fit(self):
        frame, fused, target, order, _, _ = fixture()
        arguments = dict(categorical_columns=('odd["column"]',), class_order=order, fit_ids=frame.index)
        invalid_cases = [
            (frame, fused[:-1], target, arguments),
            (frame, fused, target, {**arguments, "fit_ids": tuple(reversed(frame.index))}),
            (frame, fused, pd.Series(target, index=list(reversed(frame.index))), arguments),
            (frame, fused, target, {**arguments, "categorical_columns": ("absent",)}),
            (frame, fused, target, {**arguments, "categorical_columns": ('odd["column"]',) * 2}),
            (frame, np.full_like(fused, np.nan), target, arguments),
            (frame.assign(f0=np.inf), fused, target, arguments),
        ]
        for features, block, labels, kwargs in invalid_cases:
            with self.subTest(columns=tuple(features.columns), shape=block.shape):
                with patch.object(downstream.LGBMClassifier, "fit", side_effect=AssertionError("must fail before fit")):
                    with self.assertRaises(ValueError):
                        DownstreamModel(DownstreamConfig("multiclass")).fit(features, block, labels, **kwargs)
        model = fit()
        with self.assertRaisesRegex(ValueError, "schema and order"):
            model.predict(frame.iloc[:, ::-1], fused)
        with self.assertRaisesRegex(ValueError, "dimension"):
            model.predict(frame, fused[:, :-1])
        with self.assertRaisesRegex(RuntimeError, "new DownstreamModel"):
            model.fit(frame, fused, target, **arguments)

    def test_class_indices_and_typed_class_order_are_explicit(self):
        frame, fused, target, _, _, _ = fixture()
        typed_order = (True, 1, "1")
        model = DownstreamModel(DownstreamConfig("multiclass")).fit(
            frame, fused, target, categorical_columns=('odd["column"]',), class_order=typed_order, fit_ids=frame.index)
        self.assertEqual(tuple(type(value) for value in model.class_order), (bool, int, str))
        for labels, order in ((target.astype(float), typed_order), (target + 1, typed_order),
                              (target, ("x", "x", "y")), (np.zeros(len(target), dtype=int), typed_order)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                DownstreamModel(DownstreamConfig("multiclass")).fit(
                    frame, fused, labels, categorical_columns=('odd["column"]',), class_order=order, fit_ids=frame.index)

    def test_text_only_pipeline_uses_no_fake_structured_feature(self):
        frame, fused, target, order, validation, val_fused = fixture("binary")
        model = DownstreamModel(DownstreamConfig("binary")).fit(
            frame.iloc[:, :0], fused, target, categorical_columns=(), class_order=order, fit_ids=frame.index)
        self.assertEqual(model.columns, ())
        np.testing.assert_array_equal(model.transform(frame.iloc[:, :0], fused), fused)
        self.assertEqual(model.predict(validation.iloc[:, :0], val_fused).shape, (3, 2))

    def test_corrupted_model_and_semantic_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = fit().save(Path(temporary) / "saved")
            model_path = artifact / "model.txt"
            model_path.write_bytes(model_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                DownstreamModel.load(artifact)
        with tempfile.TemporaryDirectory() as temporary:
            artifact = fit().save(Path(temporary) / "saved")
            metadata_path = artifact / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["fused_dimension"] -= 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "layout"):
                DownstreamModel.load(artifact)


if __name__ == "__main__":
    unittest.main()
