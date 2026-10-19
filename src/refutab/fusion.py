"""Identity-initialized context gating and fixed-offset correction training.

This module receives precomputed, train-fitted PCA/structured features. It never
loads an encoder or a teacher. All external features, targets, and offsets are
detached; only the compact structured encoder, shared gate, and temporary head
are optimized. Inference artifacts deliberately omit the temporary head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FusionConfig:
    task: str
    n_text_fields: int
    n_numerical: int
    category_cardinalities: tuple[int, ...] = ()
    n_classes: int = 0
    embedding_dim: int = 30
    hidden_dim: int = 64
    dropout: float = 0.1
    category_embedding_dim: int = 8
    structured_dim: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "category_cardinalities", tuple(self.category_cardinalities))
        if self.task not in {"regression", "binary", "multiclass"}:
            raise ValueError("task must be regression, binary, or multiclass")
        for name in ("n_text_fields", "embedding_dim", "hidden_dim", "category_embedding_dim", "structured_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.n_numerical) is not int or self.n_numerical < 0:
            raise ValueError("n_numerical must be a nonnegative integer")
        if any(type(size) is not int or size < 2 for size in self.category_cardinalities):
            raise ValueError("Each categorical cardinality must reserve at least missing and unknown codes")
        if type(self.n_classes) is not int or (
            (self.task == "regression" and self.n_classes != 0)
            or (self.task == "binary" and self.n_classes != 2)
            or (self.task == "multiclass" and self.n_classes < 3)
        ):
            raise ValueError("n_classes must be 0 for regression, 2 for binary, and >=3 for multiclass")
        if not isinstance(self.dropout, (float, int)) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")


@dataclass(frozen=True)
class TrainConfig:
    objective: str = "residual"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    patience: int = 3
    device: str = "cpu"
    threads: int = 2

    def __post_init__(self) -> None:
        if self.objective not in {"residual", "direct"}:
            raise ValueError("objective must be residual or direct")
        for name in ("epochs", "batch_size", "patience", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay nonnegative, both finite")
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("device must identify a CPU or CUDA device") from exc
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU or CUDA training is supported")


@dataclass(frozen=True)
class FusionInputs:
    text: Any
    numerical: Any
    categorical: Any
    targets: Any
    offsets: Any = None


@dataclass
class FusionTrainingResult:
    model: FusionModel
    history: list[dict[str, float | int]]
    best_epoch: int
    completed_epoch: int
    training: dict[str, Any]


def _floating(value: Any, name: str, device: torch.device | str = "cpu", *, clone: bool = False) -> Tensor:
    try:
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError("Expected real numerical data")
        result = raw.detach().to(device=device, dtype=torch.float32)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{name} must be real numerical values") from exc
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result.clone().contiguous() if clone else result


def _categorical(value: Any, device: torch.device | str = "cpu", *, clone: bool = False) -> Tensor:
    try:
        raw = torch.as_tensor(value)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("categorical features must be integer codes") from exc
    if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("categorical features must be integer codes")
    result = raw.detach().to(device=device, dtype=torch.int64)
    return result.clone().contiguous() if clone else result


def _features(config: FusionConfig, text: Any, numerical: Any, categorical: Any,
              device: torch.device | str, *, clone: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    text = _floating(text, "text", device, clone=clone)
    numerical = _floating(numerical, "numerical", device, clone=clone)
    categorical = _categorical(categorical, device, clone=clone)
    if text.ndim != 3 or tuple(text.shape[1:]) != (config.n_text_fields, config.embedding_dim):
        raise ValueError("text must have shape (rows, n_text_fields, embedding_dim)")
    count = text.shape[0]
    if numerical.shape != (count, config.n_numerical):
        raise ValueError("numerical shape does not match text rows and configured numerical columns")
    if categorical.shape != (count, len(config.category_cardinalities)):
        raise ValueError("categorical shape does not match text rows and configured categorical columns")
    for column, size in enumerate(config.category_cardinalities):
        values = categorical[:, column]
        if bool(((values < 0) | (values >= size)).any()):
            raise ValueError(f"categorical column {column} has an out-of-vocabulary code")
    return text, numerical, categorical


class FusionEncoder(nn.Module):
    """Trainable structured encoder and shared gate, with no correction head."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.category_embeddings = nn.ModuleList([
            nn.Embedding(size, config.category_embedding_dim)
            for size in config.category_cardinalities
        ])
        width = config.n_numerical + len(config.category_cardinalities) * config.category_embedding_dim
        self.structured_projection = nn.Sequential(nn.Linear(width, config.structured_dim), nn.ReLU()) if width else None
        self.gate = nn.Sequential(
            nn.Linear(config.embedding_dim + config.structured_dim + config.n_text_fields, config.hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def fuse(self, text: Any, numerical: Any, categorical: Any) -> Tensor:
        device = self.gate[0].weight.device
        text, numerical, categorical = _features(self.config, text, numerical, categorical, device)
        count = text.shape[0]
        if self.structured_projection is None:
            context = text.new_zeros((count, self.config.structured_dim))
        else:
            blocks = [numerical] + [embedding(categorical[:, column])
                                    for column, embedding in enumerate(self.category_embeddings)]
            context = self.structured_projection(torch.cat(blocks, dim=1))
        identity = torch.eye(self.config.n_text_fields, device=device, dtype=text.dtype)
        gate_input = torch.cat((text, context[:, None, :].expand(-1, self.config.n_text_fields, -1),
                                identity[None, :, :].expand(count, -1, -1)), dim=2)
        fused = text * (1 + torch.tanh(self.gate(gate_input)))
        return fused.reshape(count, self.config.n_text_fields * self.config.embedding_dim)

    def forward(self, text: Any, numerical: Any, categorical: Any) -> Tensor:
        return self.fuse(text, numerical, categorical)

    @classmethod
    def from_model(cls, model: FusionEncoder) -> FusionEncoder:
        """Copy only inference weights, disable gradients, and use evaluation mode."""
        encoder = cls(model.config)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                 if not key.startswith("correction_head.")}
        encoder.load_state_dict(state, strict=True)
        return encoder.eval().requires_grad_(False)


class FusionModel(FusionEncoder):
    """Training-only extension with a modest linear correction head."""

    def __init__(self, config: FusionConfig):
        super().__init__(config)
        outputs = config.n_classes if config.task == "multiclass" else 1
        self.correction_head = nn.Linear(config.n_text_fields * config.embedding_dim, outputs)

    def forward(self, text: Any, numerical: Any, categorical: Any) -> Tensor:
        result = self.correction_head(self.fuse(text, numerical, categorical))
        return result if self.config.task == "multiclass" else result[:, 0]

    def encoder(self) -> FusionEncoder:
        return FusionEncoder.from_model(self)


def correction_loss(scores: Tensor, targets: Any, offsets: Any = None, *, task: str,
                    objective: str = "residual") -> Tensor:
    """Task loss with detached teacher scores; direct control ignores offsets."""
    if task not in {"regression", "binary", "multiclass"} or objective not in {"residual", "direct"}:
        raise ValueError("Invalid task or objective")
    if not isinstance(scores, Tensor) or not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be a finite floating tensor")
    if (task == "multiclass" and (scores.ndim != 2 or scores.shape[1] < 3)) or (
        task != "multiclass" and scores.ndim != 1
    ) or scores.shape[0] == 0:
        raise ValueError("scores have an invalid task shape")
    target = _floating(targets, "targets", scores.device)
    if target.shape != (scores.shape[0],):
        raise ValueError("targets must have one value per row")
    if task != "regression":
        classes = scores.shape[1] if task == "multiclass" else 2
        if bool(((target != target.round()) | (target < 0) | (target >= classes)).any()):
            raise ValueError("Classification targets must be encoded integers in the saved class order")
    if offsets is None:
        if objective == "residual":
            raise ValueError("Residual training requires teacher offsets")
        offset = torch.zeros_like(scores)
    else:
        offset = _floating(offsets, "offsets", scores.device)
        if offset.shape != scores.shape:
            raise ValueError("offsets and scores must have exactly matching shapes")
    combined = scores + offset if objective == "residual" else scores
    if task == "regression":
        return F.mse_loss(combined, target)
    if task == "binary":
        return F.binary_cross_entropy_with_logits(combined, target)
    return F.cross_entropy(combined, target.long())


def _inputs(config: FusionConfig, value: FusionInputs, objective: str) -> FusionInputs:
    text, numerical, categorical = _features(config, value.text, value.numerical, value.categorical, "cpu", clone=True)
    if not len(text):
        raise ValueError("Training and validation must each contain at least one row")
    targets = _floating(value.targets, "targets", clone=True)
    offsets = None if value.offsets is None else _floating(value.offsets, "offsets", clone=True)
    shape = (len(text), config.n_classes) if config.task == "multiclass" else (len(text),)
    correction_loss(torch.zeros(shape), targets, offsets, task=config.task, objective=objective)
    return FusionInputs(text, numerical, categorical, targets, offsets)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(_json({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _input_hashes(value: FusionInputs) -> dict[str, str | None]:
    return {name: None if getattr(value, name) is None else _tensor_hash(getattr(value, name))
            for name in ("text", "numerical", "categorical", "targets", "offsets")}


def _state(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_hash(state: dict[str, Tensor]) -> str:
    return hashlib.sha256(_json({name: _tensor_hash(value) for name, value in sorted(state.items())}).encode()).hexdigest()


def _payload_hash(value: Any) -> str:
    """Hash nested tensor/primitive checkpoint state without serializing code."""
    if isinstance(value, Tensor):
        encoded = "tensor:" + _tensor_hash(value)
    elif isinstance(value, dict):
        encoded = "dict:" + _json(sorted((_payload_hash(key), _payload_hash(item)) for key, item in value.items()))
    elif isinstance(value, (list, tuple)):
        encoded = type(value).__name__ + ":" + _json([_payload_hash(item) for item in value])
    elif value is None or type(value) in (bool, int, float, str):
        encoded = type(value).__name__ + ":" + _json(value)
    else:
        raise ValueError(f"Unsupported checkpoint value type: {type(value).__name__}")
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_save(value: Any, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        if not overwrite and path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_fusion(model: FusionEncoder, path: str | Path) -> None:
    """Save a new, head-free inference artifact using tensors/primitives only."""
    encoder = FusionEncoder.from_model(model)
    state = _state(encoder)
    _atomic_save({"format_version": 1, "kind": "refutab.fusion.encoder", "config": asdict(encoder.config),
                  "state_dict": state, "state_sha256": _state_hash(state)}, Path(path), overwrite=False)


def load_fusion(path: str | Path) -> FusionEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != "refutab.fusion.encoder":
        raise ValueError("Unsupported fusion encoder artifact")
    state = payload["state_dict"]
    if _state_hash(state) != payload["state_sha256"]:
        raise ValueError("Fusion encoder state checksum mismatch")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Fusion encoder weights must be finite")
    encoder = FusionEncoder(FusionConfig(**payload["config"]))
    encoder.load_state_dict(state, strict=True)
    return encoder.eval().requires_grad_(False)


def train_fusion(model_config: FusionConfig, config: TrainConfig, *, train: FusionInputs,
                 validation: FusionInputs, checkpoint_path: str | Path | None = None,
                 resume: bool = False, training_context: dict[str, Any] | None = None,
                 stop_after_epochs: int | None = None) -> FusionTrainingResult:
    """Train on fitting rows, select only by validation loss, and save each epoch.

    ``epochs`` is the fixed total budget. ``stop_after_epochs`` interrupts that
    budget at an epoch boundary for resumption; it never changes the budget.
    Resume requires exactly matching configs, input content, context, and torch
    version. CPU resume is tested against uninterrupted dropout/shuffle runs.
    CUDA is opt-in and unavailable in the pinned CPU-only environment.
    """
    train = _inputs(model_config, train, config.objective)
    validation = _inputs(model_config, validation, config.objective)
    context = json.loads(_json({} if training_context is None else training_context))
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install a separately pinned GPU environment")
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer")
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if resume and (path is None or not path.is_file()):
        raise FileNotFoundError("Resume requires an existing checkpoint_path")
    if not resume and path is not None and path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing training attempt: {path}")
    provenance = {"model_config": asdict(model_config), "training_config": asdict(config),
                  "train_inputs": _input_hashes(train), "validation_inputs": _input_hashes(validation),
                  "training_context": context, "torch_version": str(torch.__version__)}
    binding = hashlib.sha256(_json(provenance).encode()).hexdigest()
    previous_rng = torch.get_rng_state()
    previous_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        model = FusionModel(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        shuffle = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        history: list[dict[str, float | int]] = []
        completed, best_epoch, bad_epochs = 0, 0, 0
        best_loss = float("inf")
        best_state: dict[str, Tensor] = {}
        if resume:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("format_version") != 1 or saved.get("kind") != "refutab.fusion.training":
                raise ValueError("Unsupported fusion training checkpoint")
            checksum = saved.pop("payload_sha256", None)
            if checksum != _payload_hash(saved):
                raise ValueError("Fusion training checkpoint checksum mismatch")
            if saved.get("binding_sha256") != binding or _json(saved.get("provenance")) != _json(provenance):
                raise ValueError("Checkpoint provenance mismatch: config, input content, context, or environment changed")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            for state in optimizer.state.values():
                for name, value in state.items():
                    if isinstance(value, Tensor):
                        state[name] = value.to(device)
            history = saved["history"]
            completed, best_epoch, bad_epochs = saved["completed_epoch"], saved["best_epoch"], saved["bad_epochs"]
            best_loss, best_state = saved["best_loss"], saved["best_state"]
            if type(completed) is not int or not 1 <= best_epoch <= completed <= config.epochs or len(history) != completed:
                raise ValueError("Checkpoint epoch/history metadata is invalid")
            torch.set_rng_state(saved["torch_rng"])
            shuffle.set_state(saved["shuffle_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        limit = config.epochs if stop_after_epochs is None else min(config.epochs, stop_after_epochs)
        for epoch in range(completed + 1, limit + 1):
            if bad_epochs >= config.patience:
                break
            model.train()
            order = torch.randperm(len(train.text), generator=shuffle)
            total = 0.0
            for ids in order.split(config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                scores = model(train.text[ids], train.numerical[ids], train.categorical[ids])
                loss = correction_loss(scores, train.targets[ids], None if train.offsets is None else train.offsets[ids],
                                       task=model_config.task, objective=config.objective)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise ValueError("Non-finite fusion gradients")
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            model.eval()
            with torch.no_grad():
                val_scores = model(validation.text, validation.numerical, validation.categorical)
                val_loss = float(correction_loss(val_scores, validation.targets, validation.offsets,
                                                 task=model_config.task, objective=config.objective))
            if not math.isfinite(val_loss):
                raise ValueError("Non-finite validation loss")
            completed = epoch
            history.append({"epoch": epoch, "train_loss": total / len(train.text), "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch, best_state, bad_epochs = val_loss, epoch, _state(model), 0
            else:
                bad_epochs += 1
            if path is not None:
                payload = {"format_version": 1, "kind": "refutab.fusion.training", "binding_sha256": binding,
                           "provenance": provenance, "completed_epoch": completed, "best_epoch": best_epoch,
                           "bad_epochs": bad_epochs, "best_loss": best_loss, "history": history,
                           "best_state": best_state, "model_state": _state(model),
                           "optimizer_state": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                           "shuffle_rng": shuffle.get_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
                payload["payload_sha256"] = _payload_hash(payload)
                _atomic_save(payload, path, overwrite=True)
        model.load_state_dict(best_state, strict=True)
        model.eval()
        training = {**provenance, "binding_sha256": binding, "completed_epoch": completed,
                    "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history,
                    "early_stopped": bad_epochs >= config.patience, "resumed": resume,
                    "budget_completed": completed == config.epochs or bad_epochs >= config.patience,
                    "checkpoint_path": None if path is None else str(path),
                    "inference_components": ["structured_encoder", "shared_gate"],
                    "temporary_head": "linear", "selection": "minimum internal validation correction loss",
                    "exact_resume_scope": "same CPU environment; CUDA path is unverified"}
        return FusionTrainingResult(model, history, best_epoch, completed, training)
    finally:
        torch.set_rng_state(previous_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(previous_cuda_rng)
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
