"""CPU-only TAR adapter tests; Trainer/model fixtures do not establish GPU execution."""

from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from transformers import TrainerState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refutab import tar


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def save_pretrained(self, directory, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_text("fixture adapter without head")
        (directory / "adapter_config.json").write_text('{}')


class FixtureTokenizer:
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text('{}')


class FixtureArguments:
    def __init__(self, output_dir="temporary-upstream-output"):
        self.output_dir = output_dir
        self.restore_callback_states_from_checkpoint = False

    def to_dict(self):
        return dict(vars(self))


class FixtureTrainer:
    fail = False
    calls = []
    completed_step = 1

    def __init__(self, model, args, callbacks, **kwargs):
        self.model, self.args, self.callbacks = model, args, callbacks
        self.state = TrainerState()

    def train(self, resume_from_checkpoint=None):
        self.calls.append((self.args.output_dir, self.args.restore_callback_states_from_checkpoint,
                           resume_from_checkpoint))
        self.state.global_step = 1
        checkpoint = Path(self.args.output_dir) / "checkpoint-1"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.state.best_model_checkpoint = str(checkpoint)
        self.state.log_history = [{"loss": .7, "eval_loss": .8, "epoch": 1.0}]
        self.state.save_to_json(checkpoint / "trainer_state.json")
        (checkpoint / "optimizer.pt").write_bytes(b"fixture optimizer state")
        for callback in self.callbacks:
            callback.on_save(self.args, self.state, None)
        if self.fail:
            raise torch.OutOfMemoryError("explicit fixture OOM")
        # Simulate HF load-best cleanup: the final loop state can be newer
        # than the sole retained selected-best checkpoint above.
        self.state.global_step = self.completed_step
        return SimpleNamespace()


def fixture_embeddings(texts, col_name, **kwargs):
    # Stable text-dependent 384-D fixture, with varied rows supporting PCA-30.
    values = []
    for text in texts:
        digest = tar._hash([col_name, text])
        values.append(np.random.default_rng(int(digest[:8], 16)).normal(size=384))
    return np.asarray(values, dtype=np.float32)


class TarAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text, cls.fine, cls.splits = tar._source_modules()

    def setUp(self):
        FixtureTrainer.fail = False
        FixtureTrainer.calls = []
        FixtureTrainer.completed_step = 1
        self.frame = pd.DataFrame({"second": [f"beta {i}" for i in range(60)],
                                   "first": [f"alpha {i}" for i in range(60)],
                                   "number": np.arange(60)}, index=np.arange(100, 160))
        self.target = pd.Series(np.arange(60) % 2, index=self.frame.index)

    @contextmanager
    def source_fixture(self):
        original_finetune = self.fine.finetune_e5_with_lora
        def finetune_fixture(*, model_name, tokenizer, **kwargs):
            model = self.fine.AutoModel.from_pretrained(model_name)
            self.fine.AutoTokenizer.from_pretrained(model_name)
            trainer = self.fine.Trainer(model=SimpleNamespace(backbone=model),
                                        args=FixtureArguments(), callbacks=[])
            trainer.train()
            return trainer.model.backbone, tokenizer
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(tar, "_training_device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(tar, "_verify_assets", return_value=(Path("verified-fixture"), dict(tar.E5_FILE_HASHES))))
            stack.enter_context(patch.object(self.fine, "Trainer", FixtureTrainer))
            stack.enter_context(patch.object(self.fine, "finetune_e5_with_lora", side_effect=finetune_fixture))
            stack.enter_context(patch.object(self.text, "encode_texts_with_e5", side_effect=fixture_embeddings))
            stack.enter_context(patch.object(self.fine.AutoModel, "from_pretrained", return_value=FixtureModel()))
            stack.enter_context(patch.object(self.fine.AutoTokenizer, "from_pretrained", return_value=FixtureTokenizer()))
            yield
        self.assertIs(self.fine.finetune_e5_with_lora, original_finetune)

    def fit(self, output, **kwargs):
        return tar.fit_tar_representation(
            self.frame, ("first", "second"), kwargs.pop("targets", self.target),
            task=kwargs.pop("task", "binary"), train_ids=self.frame.index,
            device="cuda:0", local_model_path="verified-fixture",
            output_directory=output, training_context={"dataset": "fixture", "outer_fold": 0}, **kwargs)

    def test_historical_cli_settings_and_unchanged_function_defaults(self):
        import inspect
        from dataclasses import asdict
        from multabench.finetune.train_args import E5TrainArgs
        expected = asdict(E5TrainArgs())
        self.assertEqual({key: tar.TAR_TRAIN_KWARGS[key] for key in expected}, expected)
        signature = inspect.signature(self.fine.finetune_e5_with_lora)
        for key in ("dataloader_num_workers", "max_length", "seed"):
            self.assertEqual(tar.TAR_TRAIN_KWARGS[key], signature.parameters[key].default)

    def test_gpu_requirement_is_explicit_and_does_not_autoshrink(self):
        for device, available, count, visible in (("cpu", True, 1, "0"),
                                                  ("cuda:0", False, 0, "0"),
                                                  ("cuda:0", True, 2, "0,1"),
                                                  ("cuda:1", True, 1, "0"),
                                                  ("cuda", True, 1, "")):
            with self.subTest(device=device, count=count, visible=visible), \
                 patch.object(torch.cuda, "is_available", return_value=available), \
                 patch.object(torch.cuda, "device_count", return_value=count), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), \
                 self.assertRaisesRegex(RuntimeError, "one visible CUDA GPU"):
                tar._training_device(device)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=1), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-fixture"}):
            self.assertEqual(str(tar._training_device("cuda:0")), "cuda:0")

    def test_targets_row_alignment_class_encoding_and_seed_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            for change in ({"targets": self.target.iloc[::-1]},
                           {"targets": self.target.astype(str)},
                           {"targets": self.target + 1},
                           {"task": "binary", "targets": self.target * np.nan},
                           {"task": "multiclass", "targets": self.target * 0},
                           {"seed": 43}, {"seed": True}):
                with self.subTest(change=change.keys()), self.assertRaises(ValueError):
                    self.fit(Path(temporary) / "invalid", **change)
            self.assertFalse((Path(temporary) / "invalid").exists())

    def test_original_fit_calls_only_authorized_rows_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary, self.source_fixture():
            output = Path(temporary) / "run"
            with patch.object(self.text, "fit_text_encoders_tuned", wraps=self.text.fit_text_encoders_tuned) as observed:
                representation = self.fit(output)
            invocation = observed.call_args.kwargs
            self.assertEqual(invocation["text_features_list"], ["first", "second"])
            self.assertEqual(invocation["e5_train_kwargs"], tar.TAR_TRAIN_KWARGS)
            self.assertEqual(invocation["e5_model_name"], tar.E5_IDENTIFIER)
            self.assertFalse(invocation["no_pca"])
            self.assertEqual(invocation["pca_components"], 30)
            self.assertEqual(invocation["x"].index.tolist(), self.frame.index.tolist())
            meta = representation.metadata()
            split = meta["provenance"]["audit"]["supervised_split"]
            self.assertEqual(split["effective_seed"], 41)
            self.assertEqual(set(split["fit_ids"]) | set(split["validation_ids"]), set(self.frame.index))
            self.assertFalse(set(split["fit_ids"]) & set(split["validation_ids"]))
            # Original binary helper deliberately adds two forced held-out rows.
            self.assertEqual((len(split["fit_ids"]), len(split["validation_ids"])), (52, 8))
            self.assertEqual(meta["pca_fit_ids"], self.frame.index.tolist())
            self.assertEqual(FixtureTrainer.calls, [(str(output / "hf"), True, None)])
            heldout = self.frame.iloc[:4].copy()
            heldout.index = np.arange(500, 504)
            predicted = representation.transform(heldout)
            self.assertEqual(predicted.shape, (4, 60))
            from peft import PeftModel
            with patch.object(PeftModel, "from_pretrained", return_value=FixtureModel()):
                loaded = tar.TarRepresentation.load(output / "representation", local_model_path="verified-fixture")
            np.testing.assert_array_equal(loaded.pca.components_, representation.pca.components_)
            np.testing.assert_array_equal(loaded.pca.means_, representation.pca.means_)
            # Native NPZ preserves values/dtypes, while the original SVD view's
            # memory strides can select a different float32 BLAS reduction.
            np.testing.assert_allclose(loaded.transform(heldout), predicted, atol=1e-5, rtol=1e-5)
            self.assertEqual(loaded.pca.fit_ids, tuple(self.frame.index))
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.model.parameters()))
            self.assertFalse(loaded.model.training)
            self.assertEqual(loaded.transform(heldout.iloc[:0]).shape, (0, 60))
            with self.assertRaises(FileExistsError):
                representation.save(output / "representation")
            (output / "representation/pca.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                tar.TarRepresentation.load(output / "representation", local_model_path="verified-fixture")

    def test_regression_keeps_source_quantile_bins_and_multiclass_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary, self.source_fixture():
            for task, targets in (("regression", pd.Series(np.arange(60) * .7, index=self.frame.index)),
                                  ("multiclass", self.target + pd.Series(np.arange(60) % 3, index=self.frame.index))):
                # 60 unique regression targets produce 20 native quantile bins;
                # source split expands holdout if needed for class coverage.
                if task == "multiclass":
                    targets = pd.Series(np.arange(60) % 3, index=self.frame.index)
                if task == "regression":
                    # Original 20-bin splitter needs at least 20 holdout rows:
                    # use repeated target values, yielding six quantile bins.
                    targets = pd.Series(np.arange(60) % 3, index=self.frame.index, dtype=float)
                representation = self.fit(Path(temporary) / task, task=task, targets=targets)
                metadata = representation.metadata()["provenance"]
                if task == "regression":
                    self.assertIn("qcut", metadata["regression_target_discretization"])
                    labels = metadata["audit"]["supervised_split"]["label_counts"]
                    self.assertTrue(all(label.startswith("Quantile") for label in labels))
                    bins = metadata["audit"]["regression_bins"]
                    self.assertEqual(bins["input_ids"], self.frame.index.tolist())
                    self.assertEqual(bins["requested_bins"], 20)
                    self.assertEqual(len(bins["edges"]), bins["actual_bins"] + 1)
                else:
                    self.assertIsNone(metadata["regression_target_discretization"])

    def test_oom_restores_all_bindings_and_keeps_verified_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary, self.source_fixture():
            output = Path(temporary) / "run"
            before = (self.fine.AutoModel.from_pretrained, self.fine.AutoTokenizer.from_pretrained,
                      self.fine.Trainer, self.splits.split_to_val, self.text.discretize_numerical)
            FixtureTrainer.fail = True
            with self.assertRaisesRegex(torch.OutOfMemoryError, "fixture OOM"):
                self.fit(output)
            after = (self.fine.AutoModel.from_pretrained, self.fine.AutoTokenizer.from_pretrained,
                     self.fine.Trainer, self.splits.split_to_val, self.text.discretize_numerical)
            self.assertEqual(before, after)
            audit = json.loads((output / "trainer_audit.json").read_text())
            self.assertEqual(audit["training_status"], "failed")
            self.assertFalse((output / "representation").exists())
            checkpoint = output / "hf/checkpoint-1"
            context = json.loads((output / "training_context.json").read_text())
            tar._resume_check(checkpoint, context)
            with self.assertRaisesRegex(ValueError, "context"):
                tar._resume_check(checkpoint, {**context, "task": "other"})
            FixtureTrainer.fail = False
            representation = self.fit(output, resume_checkpoint=checkpoint)
            self.assertEqual(FixtureTrainer.calls[-1][2], str(checkpoint))
            self.assertEqual(representation.metadata()["provenance"]["audit"]["training_status"], "completed")

    def test_checkpoint_inventory_and_context_guard_before_retraining(self):
        with tempfile.TemporaryDirectory() as temporary, self.source_fixture():
            output = Path(temporary) / "run"
            FixtureTrainer.fail = True
            with self.assertRaises(torch.OutOfMemoryError):
                self.fit(output)
            checkpoint = output / "hf/checkpoint-1"
            context = json.loads((output / "training_context.json").read_text())
            (checkpoint / "optimizer.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                tar._resume_check(checkpoint, context)
            with self.assertRaises(FileExistsError):
                self.fit(output)

    def test_cross_attempt_resume_copies_checkpoint_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary, self.source_fixture():
            old = Path(temporary) / "old"
            new = Path(temporary) / "new"
            FixtureTrainer.fail = True
            with self.assertRaises(torch.OutOfMemoryError):
                self.fit(old)
            checkpoint = old / "hf/checkpoint-1"
            original_inventory = tar._inventory(old)
            FixtureTrainer.fail = False
            representation = self.fit(new, resume_checkpoint=checkpoint)
            self.assertEqual(tar._inventory(old), original_inventory)
            self.assertEqual(FixtureTrainer.calls[-1][2], str(new / "hf/checkpoint-1"))
            audit = representation.metadata()["provenance"]["audit"]
            self.assertEqual(audit["resume_checkpoint_source"], str(checkpoint))
            state = json.loads((new / "hf/checkpoint-1/trainer_state.json").read_text())
            self.assertEqual(state["best_model_checkpoint"], str(new / "hf/checkpoint-1"))

    def test_best_and_latest_checkpoint_both_copied_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / "old"
            best, last = old / "hf/checkpoint-1", old / "hf/checkpoint-2"
            context = {"fixture": True}
            for checkpoint in (best, last):
                checkpoint.mkdir(parents=True)
                tar._write(checkpoint / "trainer_state.json", {"global_step": int(checkpoint.name[-1]),
                                                               "best_model_checkpoint": str(best)})
                (checkpoint / "optimizer.pt").write_bytes(b"fixture state")
                tar._write(checkpoint / "refutab_checkpoint.json", {
                    "context": context, "context_sha256": tar._hash(context),
                    "files": tar._inventory(checkpoint),
                })
            new = Path(temporary) / "new"
            copied = tar._copy_resume(last, new, context)
            self.assertEqual(copied, new / "hf/checkpoint-2")
            tar._resume_check(copied, context)
            for checkpoint in (new / "hf/checkpoint-1", new / "hf/checkpoint-2"):
                state = json.loads((checkpoint / "trainer_state.json").read_text())
                self.assertEqual(state["best_model_checkpoint"], str(new / "hf/checkpoint-1"))
            (best / "optimizer.pt").write_bytes(b"modified best model")
            with self.assertRaisesRegex(ValueError, "best-model"):
                tar._resume_check(last, context)

    def test_completed_training_survives_interruption_before_pca_without_retraining(self):
        with tempfile.TemporaryDirectory() as temporary, self.source_fixture():
            old, new = Path(temporary) / "old", Path(temporary) / "new"
            FixtureTrainer.completed_step = 3
            with patch.object(self.text, "encode_texts_with_e5", side_effect=RuntimeError("fixture interrupted before PCA")), \
                 self.assertRaisesRegex(RuntimeError, "before PCA"):
                self.fit(old)
            self.assertTrue((old / "selected_backbone/metadata.json").is_file())
            self.assertFalse((old / "representation").exists())
            checkpoint = old / "hf/checkpoint-1"
            selected_state = json.loads((old / "selected_backbone/metadata.json").read_text())["trainer_state"]
            self.assertEqual(selected_state["global_step"], 3)
            self.assertEqual(json.loads((checkpoint / "trainer_state.json").read_text())["global_step"], 1)
            old_inventory = tar._inventory(old)
            with patch.object(tar, "_restore_completed_backbone") as restore:
                representation = self.fit(new, resume_checkpoint=checkpoint)
            self.assertEqual(len(FixtureTrainer.calls), 1)
            restore.assert_called_once()
            self.assertEqual(tar._inventory(old), old_inventory)
            self.assertTrue(representation.metadata()["provenance"]["audit"]["restored_completed_training_without_optimizer_steps"])
            self.assertEqual(representation.metadata()["provenance"]["audit"]["trainer_state"]["global_step"], 3)
            completed_context = json.loads((new / "training_context.json").read_text())
            (new / "selected_backbone/adapter/adapter_model.safetensors").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "receipt"):
                tar._completed_training(new, completed_context)

    def test_completed_backbone_native_lora_weights_restore_without_a_head(self):
        from peft import LoraConfig, get_peft_model
        from transformers import BertConfig, BertModel
        torch.manual_seed(42)
        model = get_peft_model(BertModel(BertConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, vocab_size=20)),
            LoraConfig(r=2, target_modules=["query", "value"], bias="none"))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.fill_(.123)
        expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save_pretrained(directory / "adapter", safe_serialization=True, save_embedding_layers=False)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_" in name:
                        parameter.zero_()
            tar._restore_completed_backbone(model, directory)
        for name, parameter in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        self.assertFalse(any("head" in name for name in expected))

    def test_pinned_asset_verification_rejects_modified_or_unverified_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in tar.E5_FILE_HASHES:
                (directory / filename).write_text("modified")
            with self.assertRaisesRegex(ValueError, "checksums"):
                tar._verify_assets(directory)
            (directory / "added_tokens.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "unverified"):
                tar._verify_assets(directory)

    def test_unexpected_loader_identifier_is_rejected_and_patch_restored(self):
        import transformers
        with tempfile.TemporaryDirectory() as temporary:
            loader = transformers.AutoModel.from_pretrained
            with self.assertRaisesRegex(ValueError, "Unexpected model identifier"):
                with tar._runtime_scope(self.text, self.fine, self.splits,
                                        local_model_path=Path("fixture"), output_directory=Path(temporary),
                                        context={"fixture": True}, audit={}, resume_checkpoint=None):
                    transformers.AutoModel.from_pretrained("different/model")
            self.assertEqual(transformers.AutoModel.from_pretrained, loader)


if __name__ == "__main__":
    unittest.main()
