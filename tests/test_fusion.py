"""Small CPU checks for gating, authorized gradients, and resumable training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.nn import functional as F

from refutab.fusion import (FusionConfig, FusionEncoder, FusionInputs, FusionModel,
                            TrainConfig, correction_loss, load_fusion, save_fusion,
                            train_fusion)


def fixture(task: str = "regression", count: int = 37, seed: int = 4) -> tuple[FusionConfig, FusionInputs]:
    generator = np.random.default_rng(seed)
    text = generator.normal(size=(count, 2, 30)).astype(np.float32)
    numerical = generator.normal(size=(count, 2)).astype(np.float32)
    categorical = generator.integers(0, 5, size=(count, 1), dtype=np.int64)
    classes = {"regression": 0, "binary": 2, "multiclass": 3}[task]
    config = FusionConfig(task, 2, 2, (5,), n_classes=classes)
    if task == "regression":
        targets = (text[:, 0, 0] + 0.2 * numerical[:, 0]).astype(np.float32)
        offsets = (0.2 * numerical[:, 0]).astype(np.float32)
    elif task == "binary":
        targets = (text[:, 0, 0] + 0.2 * numerical[:, 0] > 0).astype(np.int64)
        offsets = (0.2 * numerical[:, 0]).astype(np.float32)
    else:
        targets = np.argmax(text[:, 0, :3] + numerical[:, :1], axis=1).astype(np.int64)
        offsets = np.log(np.full((count, 3), 1 / 3, dtype=np.float32))
    return config, FusionInputs(text, numerical, categorical, targets, offsets)


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_initial_gate_is_exact_identity_and_field_order_is_preserved(self):
        config, data = fixture()
        model = FusionModel(config)
        for mode in (True, False):
            model.train(mode)
            actual = model.fuse(data.text, data.numerical, data.categorical)
            self.assertTrue(torch.equal(actual, torch.from_numpy(data.text.reshape(len(data.text), -1))))
        self.assertEqual(model.correction_head.in_features, 60)
        self.assertEqual(model.category_embeddings[0].embedding_dim, 8)
        self.assertEqual(model.structured_projection[0].out_features, 16)
        self.assertEqual(model.gate[0].out_features, 64)
        self.assertEqual(model.gate[2].p, 0.1)

    def test_shared_gate_uses_fixed_field_identity_and_bounds_scaling(self):
        config = FusionConfig("regression", 2, 0)
        model = FusionEncoder(config).eval()
        text = torch.ones(3, 2, 30)
        numerical, categorical = torch.empty(3, 0), torch.empty(3, 0, dtype=torch.int64)
        with torch.no_grad():
            model.gate[0].weight.zero_()
            model.gate[0].bias.zero_()
            model.gate[0].weight[0, -2] = 1
            model.gate[-1].weight[:, 0] = 0.5
        actual = model.fuse(text, numerical, categorical).reshape(3, 2, 30)
        self.assertTrue(torch.all(actual[:, 0] > actual[:, 1]))
        self.assertTrue(torch.all((actual > 0) & (actual < 2)))

    def test_three_residual_losses_match_explicit_reference(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                config, data = fixture(task, count=8)
                scores = torch.linspace(-0.7, 1.2, 24 if task == "multiclass" else 8)
                scores = scores.reshape(8, 3) if task == "multiclass" else scores
                target, offsets = torch.as_tensor(data.targets), torch.as_tensor(data.offsets)
                combined = scores + offsets
                expected = (F.mse_loss(combined, target.float()) if task == "regression"
                            else F.binary_cross_entropy_with_logits(combined, target.float()) if task == "binary"
                            else F.cross_entropy(combined, target.long()))
                actual = correction_loss(scores, target, offsets, task=task)
                self.assertTrue(torch.equal(actual, expected))

    def test_direct_objective_matches_zero_offset_control(self):
        for task in ("regression", "binary", "multiclass"):
            with self.subTest(task=task):
                config, data = fixture(task, count=8)
                model = FusionModel(config).eval()
                scores = model(data.text, data.numerical, data.categorical)
                direct = correction_loss(scores, data.targets, data.offsets, task=task, objective="direct")
                zero = correction_loss(scores, data.targets, torch.zeros_like(scores), task=task)
                no_offsets = correction_loss(scores, data.targets, task=task, objective="direct")
                self.assertTrue(torch.equal(direct, zero))
                self.assertTrue(torch.equal(no_offsets, zero))

    def test_only_authorized_modules_receive_gradients_and_inputs_are_detached(self):
        config, data = fixture()
        model = FusionModel(config)
        text = torch.tensor(data.text, requires_grad=True)
        numerical = torch.tensor(data.numerical, requires_grad=True)
        offsets = torch.tensor(data.offsets, requires_grad=True)
        targets = torch.tensor(data.targets, requires_grad=True)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        for _ in range(3):
            optimizer.zero_grad()
            loss = correction_loss(model(text, numerical, data.categorical), targets, offsets, task="regression")
            loss.backward()
            optimizer.step()
        for external in (text, numerical, offsets, targets):
            self.assertIsNone(external.grad)
        for prefix in ("gate.", "structured_projection.", "category_embeddings.", "correction_head."):
            self.assertTrue(any(not torch.equal(value, before[name])
                                for name, value in model.named_parameters() if name.startswith(prefix)), prefix)
        self.assertTrue(all(value.grad is not None for value in model.parameters()))

    def test_invalid_dimensions_categories_targets_and_nonfinite_fail_before_training(self):
        config, data = fixture()
        malformed = [replace(data, text=data.text[:, :, :29]),
                     replace(data, numerical=data.numerical[:, :1]),
                     replace(data, categorical=np.full_like(data.categorical, 5)),
                     replace(data, categorical=data.categorical.astype(float)),
                     replace(data, targets=data.targets[:-1]),
                     replace(data, offsets=np.full_like(data.offsets, np.nan)),
                     replace(data, text=np.full_like(data.text, np.inf))]
        for bad in malformed:
            with self.subTest(bad=str(bad)[:40]), self.assertRaises(ValueError):
                train_fusion(config, TrainConfig(epochs=1), train=bad, validation=data)
        config, binary = fixture("binary")
        for targets in (np.full(37, 2), np.full(37, 0.5), np.full(37, -1)):
            with self.assertRaises(ValueError):
                train_fusion(config, TrainConfig(epochs=1), train=replace(binary, targets=targets), validation=binary)
        with self.assertRaises(ValueError):
            train_fusion(config, TrainConfig(epochs=1), train=replace(binary, offsets=None), validation=binary)

    def test_small_training_for_all_tasks_and_objectives_selects_validation_minimum(self):
        for task in ("regression", "binary", "multiclass"):
            for objective in ("residual", "direct"):
                with self.subTest(task=task, objective=objective):
                    config, train = fixture(task)
                    _, validation = fixture(task, count=11, seed=50)
                    result = train_fusion(config, TrainConfig(objective=objective, epochs=3),
                                          train=train, validation=validation)
                    losses = [item["validation_loss"] for item in result.history]
                    self.assertEqual(result.best_epoch, losses.index(min(losses)) + 1)
                    scores = result.model(validation.text, validation.numerical, validation.categorical)
                    actual = correction_loss(scores, validation.targets, validation.offsets, task=task, objective=objective)
                    self.assertEqual(float(actual.detach()), min(losses))
                    self.assertTrue(result.training["budget_completed"])
                    self.assertTrue(np.isfinite(losses).all())

    def test_inference_roundtrip_excludes_head_teacher_and_gradients(self):
        config, train = fixture("multiclass")
        _, validation = fixture("multiclass", count=7, seed=12)
        result = train_fusion(config, TrainConfig(epochs=2), train=train, validation=validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            save_fusion(result.model, path)
            loaded = load_fusion(path)
            self.assertFalse(hasattr(loaded, "correction_head"))
            self.assertFalse(any(value.requires_grad for value in loaded.parameters()))
            self.assertFalse(loaded.training)
            expected = result.model.fuse(validation.text, validation.numerical, validation.categorical)
            actual = loaded.fuse(validation.text, validation.numerical, validation.categorical)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(result.model.encoder().fuse(validation.text, validation.numerical,
                                                                    validation.categorical), actual))
            payload = torch.load(path, weights_only=True)
            self.assertTrue(all("head" not in name and "teacher" not in name for name in payload["state_dict"]))
            with self.assertRaises(FileExistsError):
                save_fusion(result.model, path)
            payload["state_dict"]["gate.3.bias"][0] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_fusion(path)

    def test_cpu_dropout_shuffle_resume_is_exact_and_retains_last_optimizer_state(self):
        config, train = fixture("binary")
        _, validation = fixture("binary", count=13, seed=17)
        settings = TrainConfig(epochs=6, patience=10, batch_size=8)
        complete = train_fusion(config, settings, train=train, validation=validation,
                                training_context={"dataset": "synthetic", "ids": list(range(37))})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            interrupted = train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path,
                                       stop_after_epochs=2,
                                       training_context={"dataset": "synthetic", "ids": list(range(37))})
            self.assertEqual(interrupted.completed_epoch, 2)
            self.assertFalse(interrupted.training["budget_completed"])
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["completed_epoch"], 2)
            self.assertIn("optimizer_state", saved)
            self.assertIn("torch_rng", saved)
            self.assertIn("shuffle_rng", saved)
            resumed = train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path,
                                   resume=True, training_context={"dataset": "synthetic", "ids": list(range(37))})
            self.assertEqual(complete.history, resumed.history)
            self.assertEqual(complete.best_epoch, resumed.best_epoch)
            for name, value in complete.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]), name)
            second_resume = train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path,
                                         resume=True, training_context={"dataset": "synthetic", "ids": list(range(37))})
            self.assertEqual(resumed.history, second_resume.history)

    def test_resume_rejects_changed_fit_validation_offsets_context_and_budget(self):
        config, train = fixture()
        _, validation = fixture(count=9, seed=44)
        settings = TrainConfig(epochs=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path,
                         stop_after_epochs=1, training_context={"fold": 0})
            changes = [dict(train=replace(train, targets=train.targets + 1)),
                       dict(train=replace(train, offsets=train.offsets + 1)),
                       dict(validation=replace(validation, text=validation.text + 1)),
                       dict(validation=replace(validation, targets=validation.targets + 1)),
                       dict(training_context={"fold": 1}), dict(config=replace(settings, epochs=5)),
                       dict(model_config=replace(config, hidden_dim=32))]
            for change in changes:
                kwargs = dict(model_config=config, config=settings, train=train, validation=validation,
                              checkpoint_path=path, resume=True, training_context={"fold": 0})
                kwargs.update(change)
                with self.subTest(change=list(change)), self.assertRaisesRegex(ValueError, "provenance"):
                    train_fusion(**kwargs)
            with self.assertRaises(FileExistsError):
                train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path)

    def test_training_does_not_mutate_external_inputs_or_global_rng_configuration(self):
        config, data = fixture()
        snapshots = [getattr(data, field).copy() for field in ("text", "numerical", "categorical", "targets", "offsets")]
        torch.manual_seed(987)
        previous_rng = torch.get_rng_state().clone()
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        train_fusion(config, TrainConfig(epochs=1, threads=1), train=data, validation=data)
        for field, snapshot in zip(("text", "numerical", "categorical", "targets", "offsets"), snapshots):
            np.testing.assert_array_equal(getattr(data, field), snapshot)
        self.assertTrue(torch.equal(torch.get_rng_state(), previous_rng))
        self.assertEqual(torch.get_num_threads(), previous_threads)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), previous_deterministic)

    def test_validation_perturbations_do_not_enter_optimizer_updates(self):
        config, train = fixture()
        _, validation = fixture(count=9, seed=44)
        settings = TrainConfig(epochs=3, patience=10)
        changed = replace(validation, text=validation.text + 10, targets=validation.targets - 10,
                          numerical=validation.numerical - 7, offsets=validation.offsets + 5)
        with tempfile.TemporaryDirectory() as directory:
            original_path, changed_path = Path(directory) / "original.pt", Path(directory) / "changed.pt"
            original = train_fusion(config, settings, train=train, validation=validation, checkpoint_path=original_path)
            altered = train_fusion(config, settings, train=train, validation=changed, checkpoint_path=changed_path)
            original_state = torch.load(original_path, weights_only=True)["model_state"]
            changed_state = torch.load(changed_path, weights_only=True)["model_state"]
            self.assertEqual([row["train_loss"] for row in original.history],
                             [row["train_loss"] for row in altered.history])
            for name, value in original_state.items():
                self.assertTrue(torch.equal(value, changed_state[name]), name)
            self.assertNotEqual([row["validation_loss"] for row in original.history],
                                [row["validation_loss"] for row in altered.history])

    def test_training_checkpoint_detects_accidental_state_corruption(self):
        config, train = fixture()
        _, validation = fixture(count=9, seed=44)
        settings = TrainConfig(epochs=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path, stop_after_epochs=1)
            payload = torch.load(path, weights_only=True)
            payload["optimizer_state"]["state"][0]["exp_avg"][0, 0] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                train_fusion(config, settings, train=train, validation=validation, checkpoint_path=path, resume=True)

    def test_configuration_rejects_invalid_task_hyperparameters_and_dimensions(self):
        for change in ({"task": "unknown"}, {"n_text_fields": 0}, {"n_numerical": -1},
                       {"category_cardinalities": (1,)}, {"n_classes": 2}, {"dropout": 1}, {"dropout": float("nan")}):
            kwargs = dict(task="regression", n_text_fields=2, n_numerical=1)
            kwargs.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                FusionConfig(**kwargs)
        for change in ({"epochs": 0}, {"lr": float("nan")}, {"patience": 0}, {"seed": 2**32}, {"device": "mps"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TrainConfig(**change)


if __name__ == "__main__":
    unittest.main()
