"""Resumable size-specific fitting, followed by a training-side D barrier.

The completed pilot engine supplies learner adapters and artifact primitives.
This ledger has no benchmark-test prediction path. All cases use grouped OOF
teachers; assessment lives outside immutable fitting-attempt directories.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

import numpy as np
import torch

from . import pilot as p
from .scaling_protocol import build_scaling_cases


def summarize(state: dict, cases: dict) -> dict:
    expected = state["expected_jobs"]
    ready = all(state["stages"].get(j, {}).get("status") == "completed" for j in expected)
    complete = ready and set(state["assessments"]) == set(expected)
    result = {k: state.get(k) for k in ("run_id", "status", "binding_sha256", "source", "failure", "cost", "execution_history")}
    result.update(planned_stages=len(expected),
                  completed_stages=sum(state["stages"].get(j, {}).get("status") == "completed" for j in expected),
                  assessment_barrier_satisfied=ready, completed_assessments=len(state["assessments"]),
                  outer_test_predictions=0, decision="await_complete_valid_matrix", datasets={})
    result["cases"] = {name: {"dataset_id": c["dataset_id"], "outer_fold": c["outer_fold"], "size": c["size"],
                               "fit_rows": len(c["split"].train_ids), "validation_rows": len(c["split"].validation_ids),
                               "assessment_rows": len(c["split"].test_ids), "split_fingerprint": c["split"].fingerprint()}
                       for name, c in cases.items()}
    # A partial matrix never produces a conveniently selected comparison.
    if not complete:
        return result
    for dataset in sorted({c["dataset_id"] for c in cases.values()}):
        sizes = {}
        task = next(c["task"] for c in cases.values() if c["dataset_id"] == dataset)
        metric = "r2" if task == "regression" else "auc"
        for size in state["config"]["sizes"]:
            paired, counts = [], []
            for name, case in cases.items():
                if case["dataset_id"] != dataset or case["size"] != str(size):
                    continue
                scores = {m: state["assessments"][f"{name}__{m}"]["metrics"][metric] for m in p.METHODS}
                if not np.isfinite(list(scores.values())).all():
                    raise ValueError("Nonfinite scaling assessment")
                paired.append({"outer_fold": case["outer_fold"], **scores,
                               "residual_minus_direct": scores["residual"] - scores["direct"],
                               "residual_minus_tar": scores["residual"] - scores["tar"]})
                counts.append(len(case["split"].train_ids))
            if len(paired) != len(state["config"]["outer_folds"]):
                raise ValueError("Incomplete scaling replicate coverage")
            sizes[str(size)] = {"actual_fit_rows": counts, "paired": paired,
                               "means": {key: float(np.mean([row[key] for row in paired])) for key in paired[0] if key != "outer_fold"},
                               "std_ddof1": {key: float(np.std([row[key] for row in paired], ddof=1)) if len(paired) > 1 else None for key in paired[0] if key != "outer_fold"},
                               "positive_residual_minus_direct": sum(row["residual_minus_direct"] > 0 for row in paired)}
        small, large = sizes[str(state["config"]["sizes"][0])], sizes["all"]
        screen = large["means"]["residual_minus_direct"] > 0 and large["positive_residual_minus_direct"] >= 2
        result["datasets"][dataset] = {"metric": metric, "sizes": sizes, "largest_direct_screen_passed": screen,
                                      "gap_change_from_smallest": {control: large["means"][f"residual_minus_{control}"] - small["means"][f"residual_minus_{control}"] for control in ("direct", "tar")},
                                      "interpretation": "descriptive development evidence; overlapping replicates; no automatic redesign"}
    result["decision"] = "review_task_specific_gaps_and_optimization_diagnostics"
    return result


def _verified_stage(directory: Path, entry: dict, context: dict) -> dict:
    path = (directory / entry["attempt"]).resolve()
    if not path.is_relative_to((directory / "attempts").resolve()):
        raise ValueError("Attempt path escapes run")
    record = p._verify_attempt(path, entry["manifest_sha256"])
    if (record["context"] != context or record["scores"] != entry["scores"]
            or record["runtime"] != entry["runtime"]):
        raise ValueError("Stage context changed")
    return record


def run_scaling(root: Path, config_path: Path, *, run_id=None, output_root=None,
                resume_run=None, methods=None, device=None, prior_audit: dict | None = None):
    root = root.resolve()
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    if (config.get("protocol") != "controlled_scaling_v1" or config["methods"] != list(p.METHODS)
            or config["outer_test_predictions"] != 0
            or config["tar_execution"] != {"activation_checkpointing": True}):
        raise ValueError("Scaling requires the registered protocol, four methods and checkpointed TAR")
    selected = list(p.METHODS if methods is None else methods)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(p.METHODS):
        raise ValueError("Choose unique registered methods")
    effective = json.loads(json.dumps(config))
    effective["device"] = device or config["device"]
    effective["encoder"]["device"] = effective["pipeline"]["training"]["device"] = effective["device"]
    if "tar" in selected and not effective["device"].startswith("cuda"):
        raise ValueError("TAR scaling needs a CUDA GPU")
    if not prior_audit or prior_audit.get("status") != "passed" or prior_audit.get("run_id") != config["prior_run_id"]:
        raise ValueError("A passed targeted prior-pilot artifact audit is required before scaling fits")
    source = p._git(root, "rev-parse", "HEAD")
    if p._git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Commit scaling source before recorded fitting")
    environment = p._environment(effective["device"])
    tables, _ = p._load_tables(root, config)
    cases = build_scaling_cases(root, tables, config)
    jobs = [f"{name}__{m}" for name in cases for m in p.METHODS]
    binding = {"source_commit": source, "upstream_commit": p.UPSTREAM_SHA, "effective_config": effective,
               "plan_sha256": p.sha256_file(root / config_path), "prior_audit": prior_audit,
               "asset_metadata_sha256": {key: p.sha256_file(root / config[key]) for key in ("datasets_manifest", "anchor_review", "anchor_rows")},
               "dataset_fingerprints": {k: t.fingerprint() for k, t in tables.items()},
               "split_fingerprints": {k: c["split"].fingerprint() for k, c in cases.items()},
               "environment": environment,
               "lock_sha256": {key: p.sha256_file(root / key) for key in (("environments/gpu/pyproject.toml", "environments/gpu/uv.lock") if effective["device"].startswith("cuda") else ("pyproject.toml", "uv.lock"))}}
    if resume_run is not None:
        if run_id is not None or output_root is not None:
            raise ValueError("Resume uses its existing directory; omit output-root/run-id")
        directory = (root / resume_run).resolve()
        state = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if state["binding"] != binding or state["binding_sha256"] != p._hash(binding) or state["expected_jobs"] != jobs:
            raise ValueError("Scaling continuation provenance changed")
        for name, case in cases.items():
            if p.SplitAssignments.load(directory / "splits" / f"{name}.json").fingerprint() != case["split"].fingerprint():
                raise ValueError("Saved scaling assignments changed")
    else:
        run_id = run_id or f"scaling-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must name one new directory")
        directory = (root / (output_root or config["output_root"]) / run_id).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        for name, case in cases.items():
            case["split"].save(directory / "splits" / f"{name}.json")
        state = {"format_version": 1, "run_id": run_id, "status": "started", "started_utc": p._now(),
                 "source": {"commit": source, "upstream_commit": p.UPSTREAM_SHA, "working_tree_dirty": False},
                 "binding": binding, "binding_sha256": p._hash(binding), "config": config,
                 "expected_jobs": jobs, "stages": {}, "assessments": {},
                 "hardware": {"cpu": p.platform.processor(), "logical_cpus": p.psutil.cpu_count(),
                              "ram_bytes": p.psutil.virtual_memory().total, "gpu": environment.get("cuda")},
                 "cost": {"provider_service_charge_usd": None, "basis": "Record actual provider charge after execution"}}

    if set(state["stages"]) - set(jobs) or set(state["assessments"]) - set(jobs):
        raise ValueError("Scaling ledger contains unregistered jobs")

    def context_for(job, case):
        return {"scaling_binding_sha256": state["binding_sha256"], "job": job,
                "dataset_fingerprint": tables[case["dataset_id"]].fingerprint(),
                "split_fingerprint": case["split"].fingerprint()}

    for name, case in cases.items():
        for method in p.METHODS:
            job = f"{name}__{method}"
            if state["stages"].get(job, {}).get("status") == "completed":
                _verified_stage(directory, state["stages"][job], context_for(job, case))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(effective["pipeline"]["training"]["threads"])
    state["status"] = "running"
    state.pop("failure", None)
    history = state.setdefault("execution_history", [])
    for prior in history:
        if prior["status"] == "running":
            prior["status"] = "interrupted_duration_unknown"
    execution = {"status": "running", "started_utc": p._now(), "argv": list(sys.argv), "selected_methods": selected}
    history.append(execution)
    p._write(directory / "manifest.json", state)
    start = time.perf_counter()
    memory = p._PeakRSS()
    memory.__enter__()
    try:
        encoder_config = replace(p.FrozenE5Config(**effective["encoder"]), model_path=str(root / effective["encoder"]["model_path"]))
        encoder = p.CachedFrozenEncoder(p.FrozenE5Encoder(encoder_config), directory / "frozen_cache")
        for name, case in cases.items():
            table = tables[case["dataset_id"]]
            for method in selected:
                job = f"{name}__{method}"
                prior = state["stages"].get(job)
                context = context_for(job, case)
                if prior and prior["status"] == "completed":
                    continue
                if prior:
                    path = (directory / prior["attempt"]).resolve()
                    if not path.is_relative_to((directory / "attempts").resolve()):
                        raise ValueError("Prior attempt path escapes run")
                    record = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                    if record["context"] != context:
                        raise ValueError("Prior attempt context changed")
                    if record["status"] == "completed":
                        prior.update(status="completed", manifest_sha256=p.sha256_file(path / "manifest.json"),
                                     scores=record["scores"], runtime=record["runtime"])
                        _verified_stage(directory, prior, context)
                        p._write(directory / "manifest.json", state)
                        continue
                index = prior["attempt_index"] + 1 if prior else 1
                attempt = directory / "attempts" / f"{job}__attempt{index}"
                attempt.mkdir(parents=True, exist_ok=False)
                record = {"status": "running", "job": job, "method": method, "context": context, "started_utc": p._now()}
                p._write(attempt / "manifest.json", record)
                state["stages"][job] = {"status": "running", "attempt": attempt.relative_to(directory).as_posix(), "attempt_index": index}
                p._write(directory / "manifest.json", state)
                print(f"Fitting {job}: {len(case['split'].train_ids)} A rows", flush=True)
                stage_start, stage_memory = time.perf_counter(), None
                try:
                    if effective["device"].startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    with p._PeakRSS() as stage_memory:
                        model, details = p.train_method(method, table, case, encoder, effective, attempt, context,
                                                        directory / prior["attempt"] if prior else None)
                        scores = {"train": p._score(model, table, case["split"].train_ids, attempt, "train", in_sample=True),
                                  "validation": p._score(model, table, case["split"].validation_ids, attempt, "validation")}
                        loaded = p.load_method(table, details, attempt, encoder, effective["device"])
                        np.testing.assert_allclose(model.predict(table.features(case["split"].validation_ids)),
                                                   loaded.predict(table.features(case["split"].validation_ids)), rtol=0, atol=1e-12)
                        del loaded, model
                    if method in {"direct", "residual"}:
                        steps_per_epoch = math.ceil(len(case["split"].train_ids) / effective["pipeline"]["training"]["batch_size"])
                        accounting = {"completed_epochs": details["completed_epoch"], "best_epoch": details["best_epoch"],
                                      "optimizer_steps_including_resumed_history": steps_per_epoch * details["completed_epoch"],
                                      "steps_per_epoch": steps_per_epoch}
                    elif method == "tar":
                        trainer_state = details["representation"]["provenance"]["audit"]["trainer_state"]
                        accounting = {"completed_epochs": trainer_state["epoch"],
                                      "optimizer_steps_including_resumed_history": trainer_state["global_step"],
                                      "best_model_checkpoint": trainer_state["best_model_checkpoint"]}
                    else:
                        accounting = {"optimizer_steps_including_resumed_history": 0}
                    record.update(status="completed", completed_utc=p._now(), model=details, scores=scores,
                                  optimization=accounting)
                except BaseException as error:
                    record.update(status="failed", failed_utc=p._now(), failure={"type": type(error).__name__, "message": str(error)})
                    (attempt / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    raise
                finally:
                    record["runtime"] = {"seconds": time.perf_counter() - stage_start,
                                         "sampled_peak_rss_bytes": stage_memory.peak if stage_memory else None,
                                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if effective["device"].startswith("cuda") else None}
                    record["artifact_bytes"] = sum(f.stat().st_size for f in attempt.rglob("*") if f.is_file())
                    record["files_sha256"] = p._inventory(attempt)
                    p._write(attempt / "manifest.json", record)
                    state["stages"][job].update(status=record["status"], manifest_sha256=p.sha256_file(attempt / "manifest.json"),
                                                 runtime=record["runtime"], scores=record.get("scores"), failure=record.get("failure"))
                    p._write(directory / "manifest.json", state)
                    p._write(directory / "summary.json", summarize(state, cases))
                    gc.collect()
                    if effective["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
        ready = all(state["stages"].get(j, {}).get("status") == "completed" for j in jobs)
        if ready:
            # No D prediction exists before every size/method fit is locked.
            for name, case in cases.items():
                table = tables[case["dataset_id"]]
                for method in p.METHODS:
                    job = f"{name}__{method}"
                    entry = state["stages"][job]
                    record = _verified_stage(directory, entry, context_for(job, case))
                    assessment = directory / "assessment" / job
                    assessment.mkdir(parents=True, exist_ok=True)
                    request = {"binding_sha256": state["binding_sha256"], "job": job,
                               "stage_manifest_sha256": entry["manifest_sha256"],
                               "row_ids_sha256": p.row_ids_sha256(case["split"].test_ids),
                               "rows": len(case["split"].test_ids), "class_order": record["model"]["class_order"],
                               "role": "development_D"}
                    request_path, result_path = assessment / "request.json", assessment / "result.json"
                    prediction_path = assessment / "assessment_predictions.npy"
                    if request_path.exists():
                        if json.loads(request_path.read_text(encoding="utf-8")) != request:
                            raise ValueError("Assessment request changed")
                    elif result_path.exists() or prediction_path.exists() or job in state["assessments"]:
                        raise ValueError("Assessment artifacts lack provenance request")
                    else:
                        p._write(request_path, request)
                    if result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        if (result["prediction_sha256"] != p.sha256_file(prediction_path)
                                or result["stage_manifest_sha256"] != entry["manifest_sha256"]
                                or result["row_ids_sha256"] != request["row_ids_sha256"]):
                            raise ValueError("Assessment artifacts changed")
                        actual = p._prediction_score(table, case["split"].test_ids,
                                                     np.load(prediction_path, allow_pickle=False), request["class_order"])
                        if actual != {k: v for k, v in result.items() if k not in {"prediction_sha256", "stage_manifest_sha256"}}:
                            raise ValueError("Assessment saved metrics changed")
                    else:
                        if job in state["assessments"]:
                            raise ValueError("Completed assessment result is missing")
                        if prediction_path.exists():
                            result = p._prediction_score(table, case["split"].test_ids,
                                                         np.load(prediction_path, allow_pickle=False), request["class_order"])
                        else:
                            model = p.load_method(table, record["model"], directory / entry["attempt"], encoder, effective["device"])
                            result = p._score(model, table, case["split"].test_ids, assessment, "assessment")
                            del model
                        result.update(prediction_sha256=p.sha256_file(prediction_path), stage_manifest_sha256=entry["manifest_sha256"])
                        p._write(result_path, result)
                    if job in state["assessments"] and state["assessments"][job] != result:
                        raise ValueError("Assessment ledger differs from saved result")
                    state["assessments"][job] = result
                    p._write(directory / "manifest.json", state)
                    gc.collect()
                    if effective["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
        encoder.assert_frozen()
        state["status"] = "completed" if ready else "incomplete"
        return summarize(state, cases)
    except BaseException as error:
        state.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        (directory / f"failure-{uuid4().hex[:8]}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        memory.__exit__(None, None, None)
        execution.update(status=state["status"], completed_utc=p._now(), seconds=time.perf_counter() - start, sampled_peak_rss_bytes=memory.peak)
        p._write(directory / "manifest.json", state)
        p._write(directory / "summary.json", summarize(state, cases))
        torch.set_num_threads(old_threads)
