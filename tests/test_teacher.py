"""Small real-CatBoost tests for exclusion, alignment, task scores, and artifacts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
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

from refutab import teacher
from refutab.data import DatasetSchema, DatasetTable
from refutab.teacher import TeacherArtifacts, TeacherConfig, fit_oof_teacher, prediction_offsets, row_ids_sha256


def fixture(task: str = "multiclass") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, tuple[str, ...] | None]:
    count = 36
    ids = [f"row-{number:03d}" for number in np.random.default_rng(5).permutation(count)]
    frame = pd.DataFrame({"number": np.arange(count, dtype=float),
                          "category": ([None, "missing:", 1, "1", True, "other"] * 6),
                          "other": np.sin(np.arange(count))}, index=ids)
    frame.loc[ids[2], "number"] = np.nan
    if task == "regression":
        labels, order = np.arange(count, dtype=float) * 0.3 + np.cos(np.arange(count)), None
    elif task == "binary":
        labels, order = np.asarray(["yes", "no"] * 18), ("yes", "no")
    else:
        labels, order = np.asarray(["zebra", "ant", "moose"] * 12), ("moose", "zebra", "ant")
    target = pd.Series(labels, index=ids)
    validation = pd.DataFrame({"number": [np.nan, 999.0, -7.0],
                               "category": [None, "NEVER_SEEN", "missing:"],
                               "other": [0.5, 0.0, -0.5]}, index=["validation-2", "validation-0", "validation-1"])
    return frame, target, validation, order


def fit(task: str = "multiclass", **kwargs):
    frame, target, validation, order = fixture(task)
    return fit_oof_teacher(frame, target, categorical_columns=("category",), train_ids=frame.index,
                           validation_X=validation, validation_ids=validation.index,
                           class_order=order, config=TeacherConfig(task=task, iterations=8, depth=2), **kwargs)


class TeacherTests(unittest.TestCase):
    def test_real_models_cover_all_tasks_and_actual_fit_indices_match_audits(self) -> None:
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                frame, _, validation, order = fixture(task)
                observed = []
                actual_fits = []
                original = teacher._fit_model

                def observed_fit(fit_frame, targets, **kwargs):
                    actual_fits.append((tuple(fit_frame.index), targets.copy()))
                    return original(fit_frame, targets, **kwargs)

                with patch.object(teacher, "_fit_model", side_effect=observed_fit):
                    result = fit(task, fit_observer=observed.append)
                self.assertEqual(len(actual_fits), 4)
                self.assertEqual(len(observed), 4)
                self.assertEqual(tuple(result.train_ids), tuple(frame.index))
                self.assertEqual(result.class_order, () if order is None else order)
                covered = []
                for audit, (actual_ids, actual_targets) in zip(observed, actual_fits):
                    self.assertEqual(audit.fit_row_ids, actual_ids)
                    self.assertEqual(len(actual_targets), len(actual_ids))
                    self.assertFalse(set(audit.fit_row_ids) & set(audit.predict_row_ids))
                    self.assertFalse(set(audit.fit_row_ids) & set(validation.index))
                    self.assertEqual(audit.selection_row_ids, ())
                    self.assertEqual(audit.fit_row_ids_sha256, row_ids_sha256(actual_ids))
                    if audit.role == "oof":
                        covered.extend(audit.predict_row_ids)
                        expected_ids = tuple(frame.index[result.fold_ids == audit.fold])
                        self.assertEqual(audit.predict_row_ids, expected_ids)
                self.assertCountEqual(covered, frame.index.tolist())
                self.assertEqual(len(covered), len(set(covered)))
                self.assertEqual(actual_fits[-1][0], tuple(frame.index))
                self.assertEqual(result.validation_audit.predict_row_ids, tuple(validation.index))
                self.assertTrue(np.isfinite(result.oof_offsets).all())
                np.testing.assert_array_equal(result.predict(validation), result.validation_predictions)
                if task == "regression":
                    self.assertEqual(result.oof_predictions.shape, (36,))
                    np.testing.assert_array_equal(result.oof_offsets, result.oof_predictions)
                else:
                    self.assertEqual(result.oof_predictions.shape, (36, len(order)))
                    np.testing.assert_allclose(result.oof_predictions.sum(axis=1), 1)
                    if task == "binary":
                        expected = np.log(result.oof_predictions[:, 1] / result.oof_predictions[:, 0])
                        np.testing.assert_allclose(result.oof_offsets, expected)
                    else:
                        np.testing.assert_allclose(np.exp(result.oof_offsets), result.oof_predictions)
                # Every OOF slot equals that row's excluded-row model output in
                # the saved class order, even with nonmonotonic row IDs.
                for fold, model in enumerate(result.fold_models):
                    indices = np.flatnonzero(result.fold_ids == fold)
                    prepared = teacher._prepare_frame(frame.iloc[indices], result.columns, result.categorical_columns)
                    expected = model.predict(prepared) if task == "regression" else model.predict_proba(prepared)
                    np.testing.assert_array_equal(result.oof_predictions[indices], np.asarray(expected).reshape(result.oof_predictions[indices].shape))
                    self.assertEqual(model.tree_count_, 8)
                    self.assertNotIn("validation", model.get_evals_result())

    def test_own_fold_target_perturbation_does_not_change_its_oof_teacher(self) -> None:
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                frame, target, _, order = fixture(task)
                schedule = np.repeat(np.arange(3), 12)
                config = TeacherConfig(task=task, iterations=10, depth=2)
                args = dict(categorical_columns=("category",), train_ids=frame.index,
                            class_order=order, config=config, fold_ids=schedule)
                before = fit_oof_teacher(frame, target, **args)
                perturbed = target.copy()
                if task == "regression":
                    perturbed.iloc[0] += 100
                else:
                    perturbed.iloc[0], perturbed.iloc[1] = target.iloc[1], target.iloc[0]
                after = fit_oof_teacher(frame, perturbed, **args)
                np.testing.assert_array_equal(before.fold_ids, after.fold_ids)
                np.testing.assert_array_equal(before.oof_predictions[schedule == 0], after.oof_predictions[schedule == 0])
                self.assertTrue(np.any(before.oof_predictions[schedule != 0] != after.oof_predictions[schedule != 0]))

    def test_validation_values_do_not_change_any_fitted_teacher(self) -> None:
        before = fit("multiclass")
        frame, target, validation, order = fixture("multiclass")
        validation["number"] = [10000, 20000, 30000]
        validation["category"] = ["secret", "new", "unused"]
        after = fit_oof_teacher(frame, target, categorical_columns=("category",), train_ids=frame.index,
                                validation_X=validation, validation_ids=validation.index,
                                class_order=order, config=TeacherConfig(task="multiclass", iterations=8, depth=2))
        np.testing.assert_array_equal(before.oof_predictions, after.oof_predictions)
        # Compare predictions on one common external frame for the A-fitted
        # validation models, independent of which validation values were passed.
        common = fixture("multiclass")[2]
        np.testing.assert_array_equal(before.predict(common), after.predict(common))
        self.assertNotIn("validation_y", teacher.fit_oof_teacher.__annotations__)

    def test_dataset_interface_heldout_label_perturbation_leaves_teacher_identical(self) -> None:
        frame, target, validation, order = fixture("multiclass")
        synthetic_test = validation.copy()
        synthetic_test.index = ["test-0", "test-1", "test-2"]
        full = pd.concat([frame, validation, synthetic_test])
        full["label"] = [*target, "ant", "moose", "zebra", "zebra", "ant", "moose"]
        changed = full.copy()
        changed.loc[[*validation.index, *synthetic_test.index], "label"] = ["zebra", "ant", "moose", "ant", "moose", "zebra"]
        schema = DatasetSchema(dataset_id="synthetic-heldout", task="multiclass", target_column="label",
                               numerical_columns=("number", "other"), categorical_columns=("category",))
        before_table, after_table = DatasetTable(full, schema), DatasetTable(changed, schema)
        outputs = []
        for table in (before_table, after_table):
            outputs.append(fit_oof_teacher(table.structured(frame.index), table.targets(frame.index),
                                          categorical_columns=schema.categorical_columns, train_ids=frame.index,
                                          validation_X=table.structured(validation.index), validation_ids=validation.index,
                                          class_order=order, fold_ids=np.repeat(np.arange(3), 12),
                                          config=TeacherConfig(task="multiclass", iterations=8, depth=2)))
        np.testing.assert_array_equal(outputs[0].oof_predictions, outputs[1].oof_predictions)
        np.testing.assert_array_equal(outputs[0].validation_predictions, outputs[1].validation_predictions)
        np.testing.assert_array_equal(outputs[0].predict(before_table.structured(synthetic_test.index)),
                                      outputs[1].predict(after_table.structured(synthetic_test.index)))
        self.assertEqual(outputs[0].fold_audits, outputs[1].fold_audits)

    def test_saved_models_offsets_class_order_and_fingerprints_roundtrip(self) -> None:
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                result = fit(task)
                destination = Path(temporary) / "private_teacher"
                result.save(destination)
                restored = TeacherArtifacts.load(destination)
                self.assertEqual(restored.class_order, result.class_order)
                self.assertEqual(restored.train_ids, result.train_ids)
                self.assertEqual(restored.fold_audits, result.fold_audits)
                self.assertEqual(restored.config, result.config)
                for name in ("fold_ids", "oof_predictions", "oof_offsets", "validation_predictions", "validation_offsets"):
                    np.testing.assert_array_equal(getattr(restored, name), getattr(result, name))
                external = fixture(task)[2]
                np.testing.assert_array_equal(restored.predict(external), result.predict(external))
                np.testing.assert_array_equal(restored.offsets(external), result.validation_offsets)
                with self.assertRaises(FileExistsError):
                    result.save(destination)
                with (destination / "fold_0.cbm").open("ab") as handle:
                    handle.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    TeacherArtifacts.load(destination)

    def test_label_types_are_preserved_in_class_order_and_save(self) -> None:
        frame, _, validation, _ = fixture("binary")
        target = pd.Series([1, "1"] * 18, index=frame.index, dtype=object)
        result = fit_oof_teacher(frame, target, categorical_columns=("category",), train_ids=frame.index,
                                class_order=("1", 1), config=TeacherConfig(task="binary", iterations=4, depth=2))
        self.assertEqual(result.class_order, ("1", 1))
        with tempfile.TemporaryDirectory() as temporary:
            result.save(Path(temporary) / "teacher")
            restored = TeacherArtifacts.load(Path(temporary) / "teacher")
            self.assertEqual(restored.class_order, ("1", 1))
            np.testing.assert_array_equal(restored.predict(validation), result.predict(validation))

    def test_repeated_training_is_deterministic(self) -> None:
        first, second = fit("binary"), fit("binary")
        np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
        np.testing.assert_array_equal(first.oof_predictions, second.oof_predictions)
        np.testing.assert_array_equal(first.validation_predictions, second.validation_predictions)

    def test_probability_extremes_are_finite_and_use_declared_conventions(self) -> None:
        config = TeacherConfig(task="binary", probability_clip=1e-5)
        extreme = prediction_offsets(np.array([[1., 0.], [0., 1.]]), config)
        self.assertTrue(np.isfinite(extreme).all())
        self.assertLess(extreme[0], 0)
        self.assertGreater(extreme[1], 0)
        multi = prediction_offsets(np.array([[0., 1., 0.], [.2, .3, .5]]), TeacherConfig(task="multiclass"))
        self.assertTrue(np.isfinite(multi).all())
        np.testing.assert_allclose(np.exp(multi).sum(axis=1), 1)
        with self.assertRaises(ValueError):
            prediction_offsets(np.array([[.1, .1]]), config)

    def test_stateless_category_tokens_distinguish_types_missing_and_sentinels(self) -> None:
        frame = pd.DataFrame({"category": [None, np.nan, "missing:", 1, "1", True]})
        prepared = teacher._prepare_frame(frame, ("category",), ("category",))
        self.assertEqual(prepared.iloc[0, 0], prepared.iloc[1, 0])
        self.assertEqual(len(set(prepared["category"])), 5)

    def test_failures_are_deterministic_and_happen_before_fitting(self) -> None:
        frame, target, validation, order = fixture()
        base = dict(categorical_columns=("category",), train_ids=frame.index, class_order=order,
                    config=TeacherConfig(task="multiclass", iterations=4))
        cases = [
            (frame, target.iloc[::-1], base, "index"),
            (frame.iloc[::-1], target, base, "index"),
            (frame, target, {**base, "train_ids": ["duplicate"] * len(frame)}, "duplicate"),
            (frame, target, {**base, "class_order": ("zebra", "ant")}, "class_order"),
            (frame, target, {**base, "fold_ids": np.arange(len(frame)) % 3}, "class coverage"),
            (frame, target, {**base, "fold_ids": [0] * len(frame)}, "fold_ids"),
            (frame, target, {**base, "validation_X": frame.iloc[:3], "validation_ids": frame.index[:3]}, "overlap"),
            (frame, target, {**base, "validation_X": validation.iloc[:, ::-1], "validation_ids": validation.index}, "schema"),
        ]
        with patch.object(teacher, "_fit_model") as fitting:
            for features, labels, args, error in cases:
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    fit_oof_teacher(features, labels, **args)
            fitting.assert_not_called()

    def test_rare_missing_and_invalid_targets_fail_without_training(self) -> None:
        frame, _, _, _ = fixture()
        cases = [("binary", ["a"] * 34 + ["b"] * 2, "at least three"),
                 ("binary", ["a"] * 36, "two classes"),
                 ("multiclass", ["a", "b"] * 18, "at least three"),
                 ("regression", [np.nan] + list(range(35)), "finite"),
                 ("binary", [None] + ["a", "b"] * 17 + ["a"], "scalar")]
        with patch.object(teacher, "_fit_model") as fitting:
            for task, labels, error in cases:
                with self.subTest(task=task, error=error), self.assertRaisesRegex(ValueError, error):
                    fit_oof_teacher(frame, pd.Series(labels, index=frame.index), categorical_columns=("category",),
                                    train_ids=frame.index, config=TeacherConfig(task=task))
            fitting.assert_not_called()

    def test_feature_order_nonnumeric_infinity_and_insample_prediction_fail(self) -> None:
        result = fit("binary")
        frame, _, validation, _ = fixture("binary")
        with self.assertRaisesRegex(ValueError, "overlap"):
            result.predict(frame)
        self.assertEqual(result.predict(frame, allow_in_sample=True).shape, (36, 2))
        with self.assertRaisesRegex(ValueError, "schema"):
            result.predict(validation.iloc[:, ::-1])
        validation["number"] = [np.inf, 0, 1]
        with self.assertRaisesRegex(ValueError, "infinity"):
            result.predict(validation)
        validation["number"] = ["not numeric", 0, 1]
        with self.assertRaisesRegex(ValueError, "nonnumeric"):
            result.predict(validation)

    def test_configuration_is_immutable_and_rejects_unsupported_folds(self) -> None:
        config = TeacherConfig(task="regression")
        with self.assertRaises(FrozenInstanceError):
            config.iterations = 5
        for kwargs in ({"n_splits": 4}, {"n_splits": 3.0}, {"iterations": 0}, {"depth": 0}, {"thread_count": -1}, {"seed": 2 ** 32},
                       {"learning_rate": np.nan}, {"probability_clip": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TeacherConfig(task="regression", **kwargs)
        self.assertNotEqual(row_ids_sha256(["1", "2"]), row_ids_sha256(["2", "1"]))
        self.assertNotEqual(row_ids_sha256([1]), row_ids_sha256(["1"]))

    def test_artifact_validation_rejects_inconsistent_class_and_category_metadata(self) -> None:
        result = fit("binary")
        cases = [(replace(result, class_order=("yes", "yes")), "duplicate classes"),
                 (replace(result, class_order=("yes",)), "exactly two classes"),
                 (replace(result, categorical_columns=("category", "category")), "categorical columns"),
                 (replace(result, categorical_columns=("unknown",)), "categorical columns")]
        for artifact, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                artifact.validate()
        with patch.object(type(result.final_model), "classes_", property(lambda self: np.array([0, 3]))):
            with self.assertRaisesRegex(ValueError, "encoded class range"):
                result.validate()


if __name__ == "__main__":
    unittest.main()
