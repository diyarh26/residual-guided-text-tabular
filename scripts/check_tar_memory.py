"""Synthetic full-size TAR capacity check; no benchmark rows or optimizer updates."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
from refutab.features import FrozenE5Config, FrozenE5Encoder, E5_REVISION
from refutab.tar import (_enable_activation_checkpointing, _source_modules,
                         _training_device, _verify_assets, TAR_TRAIN_KWARGS, UPSTREAM_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "artifacts/e5-small-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep existing capacity evidence; select a new output file")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    device = _training_device("cuda:0")
    local, hashes = _verify_assets(args.model_path)
    _, fine, _ = _source_modules()
    record = {"kind": "synthetic_tar_capacity_check", "status": "running", "benchmark_rows": 0,
              "optimizer_updates": 0, "batch_size": 256, "sequence_length": 512, "dtype": "float32",
              "upstream_commit": UPSTREAM_SHA, "model_revision": E5_REVISION, "model_files": hashes,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0),
              "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "scope": "Full-length synthetic forward/backward with the frozen E5 encoder also resident, as in pilot. A pass does not guarantee the entire training run."}
    start = time.perf_counter()
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    try:
        # The real pilot keeps its frozen encoder resident while TAR fits.
        frozen = FrozenE5Encoder(FrozenE5Config(model_path=str(local), device=str(device)))
        base = AutoModel.from_pretrained(local, local_files_only=True, use_safetensors=True)
        layers = list(range(base.config.num_hidden_layers))[-TAR_TRAIN_KWARGS["text_layers"]:]
        backbone = get_peft_model(base, LoraConfig(r=TAR_TRAIN_KWARGS["lora_rank"],
            lora_alpha=TAR_TRAIN_KWARGS["lora_rank"] * 2, lora_dropout=.1, bias="none",
            target_modules=fine._get_lora_target_modules(layers)))
        record["activation_checkpointing"] = _enable_activation_checkpointing(backbone)
        # Twenty outputs also cover the regression TAR quantile-bin head.
        model = fine.E5ForTuning(backbone=backbone, d_model=base.config.hidden_size, d_output=20).to(device)
        model.train()
        ids = torch.full((256, 512), 100, dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        labels = torch.arange(256, device=device) % 20
        loss = model(input_ids=ids, attention_mask=mask, labels=labels)["loss"]
        if not bool(torch.isfinite(loss)):
            raise ValueError("Synthetic loss is nonfinite")
        loss.backward()
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise ValueError("Synthetic gradients are nonfinite")
        torch.cuda.synchronize()
        frozen.assert_frozen()
        record.update(status="passed", loss=float(loss.detach().cpu()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    except BaseException as error:
        record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        record.update(seconds=time.perf_counter() - start,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: value for key, value in record.items() if key not in {"traceback", "model_files"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
