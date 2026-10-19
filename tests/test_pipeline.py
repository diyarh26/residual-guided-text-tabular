"""End-to-end boundary checks; deterministic text fixtures replace pretrained E5."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.data import DatasetTable
from refutab.features import FrozenE5Config
from refutab.fusion import TrainConfig
from refutab.pipeline import PipelineConfig, ReFuTabPipeline, fit_pipeline
from refutab.splits import SplitAssignments
from refutab.synthetic import make_synthetic
from refutab.teacher import TeacherConfig, fit_oof_teacher


class FixtureEncoder:
    """Explicit test substitute. Real E5 freezing/pooling has separate tests."""

    def __init__(self):
        self.config = FrozenE5Config("test-fixture", max_length=16)
        self.calls = []

    def assert_frozen(self):
        pass

    def state_sha256(self):
        return hashlib.sha256(b"deterministic-text-fixture-v1").hexdigest()

    def metadata(self):
        return {"test_injection": True, "config": asdict(self.config)}

    def encode(self, frame, text_columns):
        self.calls.append(tuple(frame.index))
        values = np.empty((len(frame), len(text_columns), 384), dtype=np.float32)
        for field, column in enumerate(text_columns):
            for row, text in enumerate(frame[column]):
                seed = int.from_bytes(hashlib.sha256(f"{column}:{text}".encode()).digest()[:8], "big")
                vector = np.random.default_rng(seed).normal(size=384)
                values[row, field] = vector / np.linalg.norm(vector)
        return values


def small_config(objective="residual", epochs=2):
    return PipelineConfig(training=TrainConfig(objective=objective, epochs=epochs, patience=epochs + 1,
                                               batch_size=20, threads=2), downstream_estimators=8)


class PipelineTests(unittest.TestCase):
    def fit(self, task="binary", objective="residual"):
        table, splits = make_synthetic(task)
        encoder = FixtureEncoder()
        model = fit_pipeline(table, splits, encoder, config=small_config(objective),
                             teacher_config=TeacherConfig(task, iterations=5, depth=2, thread_count=1))
        return table, splits, encoder, model

    def test_all_tasks_fit_only_declared_training_scopes_and_roundtrip_without_teacher_head(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                table, splits, encoder, model = self.fit(task)
                self.assertEqual(encoder.calls, [splits.train_ids, splits.validation_ids])
                self.assertEqual(tuple(model.pca.fit_ids), splits.train_ids)
                self.assertEqual(tuple(model.structured.fit_ids), splits.train_ids)
                self.assertEqual(tuple(model.downstream.fit_ids), splits.train_ids)
                self.assertFalse(hasattr(model, "teacher"))
                self.assertFalse(hasattr(model.fusion, "correction_head"))
                self.assertTrue(all(not p.requires_grad for p in model.fusion.parameters()))
                validation = table.features(splits.validation_ids)
                before = model.predict(validation)
                self.assertEqual(before.shape, (12,) if task == "regression" else (12, 2 if task == "binary" else 3))
                self.assertTrue(np.isfinite(before).all())
                path = model.save(Path(temporary) / "bundle")
                payload = torch.load(path / "fusion.pt", weights_only=True)
                self.assertFalse(any("correction_head" in key for key in payload["state_dict"]))
                self.assertFalse(any("teacher" in child.name for child in path.rglob("*")))
                with patch("refutab.pipeline.fit_oof_teacher", side_effect=AssertionError("Inference fitted teacher")), \
                     patch("refutab.downstream._load_original_categorical", side_effect=AssertionError("Inference needs source")):
                    loaded = ReFuTabPipeline.load(path, encoder=FixtureEncoder())
                    np.testing.assert_array_equal(before, loaded.predict(validation))
                with self.assertRaises(FileExistsError):
                    model.save(path)
                if task == "binary":
                    metadata_path = path / "pipeline.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["class_order"].reverse()
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "class order"):
                        ReFuTabPipeline.load(path, encoder=FixtureEncoder())

    def test_direct_control_uses_no_teacher_and_retains_same_architecture(self):
        with patch("refutab.pipeline.fit_oof_teacher", side_effect=AssertionError("Direct control used teacher")):
            table, splits, encoder, model = self.fit("binary", "direct")
        self.assertFalse(model.training["teacher"]["used_for_objective"])
        self.assertEqual(model.fusion.config.hidden_dim, 64)
        self.assertEqual(model.fusion.config.structured_dim, 16)
        self.assertEqual(model.predict(table.features(splits.validation_ids)).shape, (12, 2))

    def test_outer_test_changes_do_not_change_fitted_pipeline_or_validation_predictions(self):
        table, splits, encoder, original = self.fit()
        changed = table.frame
        changed.loc[list(splits.test_ids), "target"] = "unseen-test-label"
        changed.loc[list(splits.test_ids), "value"] = 1e8
        changed.loc[list(splits.test_ids), "description"] = "changed held-out text"
        other_table = DatasetTable(changed, table.schema)
        other_splits = SplitAssignments(splits.train_ids, splits.validation_ids, splits.test_ids,
                                        metadata=splits.metadata, dataset_fingerprint=other_table.fingerprint())
        other_encoder = FixtureEncoder()
        other = fit_pipeline(other_table, other_splits, other_encoder, config=small_config(),
                              teacher_config=TeacherConfig("binary", iterations=5, depth=2, thread_count=1))
        self.assertEqual(other_encoder.calls, [splits.train_ids, splits.validation_ids])
        np.testing.assert_array_equal(original.pca.components_, other.pca.components_)
        for key, value in original.fusion.state_dict().items():
            torch.testing.assert_close(value, other.fusion.state_dict()[key], rtol=0, atol=0)
        np.testing.assert_array_equal(original.predict(table.features(splits.validation_ids)),
                                      other.predict(other_table.features(splits.validation_ids)))

    def test_prediction_protects_schema_order_and_training_ids_and_bundle_corruption(self):
        table, splits, encoder, model = self.fit()
        with self.assertRaisesRegex(ValueError, "training IDs"):
            model.predict(table.features(splits.train_ids))
        self.assertEqual(model.predict(table.features(splits.train_ids), allow_in_sample=True).shape, (60, 2))
        validation = table.features(splits.validation_ids)
        with self.assertRaisesRegex(ValueError, "columns"):
            model.predict(validation.loc[:, list(reversed(validation.columns))])
        with tempfile.TemporaryDirectory() as temporary:
            path = model.save(Path(temporary) / "bundle")
            with (path / "fusion.pt").open("ab") as handle:
                handle.write(b"altered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ReFuTabPipeline.load(path, encoder=FixtureEncoder())

    def test_pipeline_resume_requires_identical_data_and_provenance(self):
        table, splits = make_synthetic()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "training.pt"
            config = small_config(epochs=2)
            first = fit_pipeline(table, splits, FixtureEncoder(), config=config,
                                  teacher_config=TeacherConfig("binary", iterations=5, depth=2, thread_count=1),
                                  checkpoint_path=checkpoint, training_context={"source": "fixed"})
            resumed = fit_pipeline(table, splits, FixtureEncoder(), config=config,
                                    teacher_config=TeacherConfig("binary", iterations=5, depth=2, thread_count=1),
                                    checkpoint_path=checkpoint, resume=True, training_context={"source": "fixed"})
            self.assertTrue(resumed.training["fusion_training"]["resumed"])
            np.testing.assert_array_equal(first.predict(table.features(splits.validation_ids)),
                                          resumed.predict(table.features(splits.validation_ids)))
            with self.assertRaisesRegex(ValueError, "provenance"):
                fit_pipeline(table, splits, FixtureEncoder(), config=config,
                              teacher_config=TeacherConfig("binary", iterations=5, depth=2, thread_count=1),
                              checkpoint_path=checkpoint, resume=True, training_context={"source": "changed"})

    def test_reused_teacher_needs_dataset_binding_and_exact_row_alignment(self):
        table, splits = make_synthetic()
        teacher = fit_oof_teacher(table.structured(splits.train_ids), table.targets(splits.train_ids),
                                  categorical_columns=table.schema.categorical_columns, train_ids=splits.train_ids,
                                  validation_X=table.structured(splits.validation_ids), validation_ids=splits.validation_ids,
                                  config=TeacherConfig("binary", iterations=5, thread_count=1))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            fit_pipeline(table, splits, FixtureEncoder(), config=small_config(), teacher=teacher)
        model = fit_pipeline(table, splits, FixtureEncoder(), config=small_config(), teacher=teacher,
                              teacher_dataset_fingerprint=table.fingerprint())
        self.assertTrue(model.training["teacher"]["used_for_objective"])


if __name__ == "__main__":
    unittest.main()
