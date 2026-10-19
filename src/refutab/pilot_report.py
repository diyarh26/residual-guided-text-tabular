"""Descriptive, task-separated pilot aggregation and the preregistered gate.

Only matching outer folds or nested complete-pipeline folds are paired.
Repeated holdouts overlap and are not independent units for significance tests.
No raw predictions, labels, model fits or cross-task score averaging occur.
"""

from __future__ import annotations

import math
from numbers import Real
import re
from typing import Any

METHODS = ("frozen", "direct", "residual", "tar")
DIAGNOSTIC_METHODS = ("direct", "residual")
OUTER_FOLDS = (0, 1, 2)
DIAGNOSTIC_FOLDS = (0, 1, 2)


def _stats(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = math.fsum(value / count for value in values) if count else None
    deviation = (math.hypot(*(value - mean for value in values)) / math.sqrt(count - 1)
                 if count >= 2 else None)
    return {"n": count, "mean": mean, "std": deviation, "std_ddof": 1}


def _paired(residual: dict[int, float], comparator: dict[int, float]) -> dict[str, Any]:
    shared = sorted(set(residual) & set(comparator))
    pairs = [{"fold": fold, "residual": residual[fold], "comparator": comparator[fold],
              "difference": residual[fold] - comparator[fold]} for fold in shared]
    return {**_stats([pair["difference"] for pair in pairs]), "pairs": pairs,
            "positive_folds": sum(pair["difference"] > 0 for pair in pairs),
            "unpaired_residual_folds": sorted(set(residual) - set(comparator)),
            "unpaired_comparator_folds": sorted(set(comparator) - set(residual))}


def _nonfinite(value: Any) -> bool:
    if isinstance(value, Real) and not isinstance(value, bool):
        return not math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_nonfinite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_nonfinite(item) for item in value)
    return False


def aggregate_pilot(summary: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a reviewed pilot summary, retaining explicit missing/failure state.

    The fixed Phase 4 design is two datasets, three outer folds, four methods,
    and three grouped diagnostics on outer fold zero for direct/residual: 36
    fitted stages and 24 outer tests. Continue unchanged is considered only
    after all are complete: positive residual-minus-TAR/direct means on BOTH
    tasks, at least two positive residual-minus-direct outer folds per task,
    and positive mean residual-minus-direct nested-D gain on BOTH tasks.
    These are descriptive decision rules, not significance claims.
    """
    if not isinstance(summary, dict) or not isinstance(summary.get("cases"), dict):
        raise ValueError("Pilot summary must contain an explicit cases mapping")
    cases = summary["cases"]
    stages = summary.get("stages", {})
    outer_tests = summary.get("outer_tests", {})
    if not isinstance(stages, dict) or not isinstance(outer_tests, dict):
        raise ValueError("Pilot stages and outer_tests must be mappings")
    issues: list[dict[str, str]] = []

    def issue(kind: str, job: str, reason: str) -> None:
        issues.append({"kind": kind, "job": job, "reason": reason})

    datasets: dict[str, dict[str, Any]] = {}
    expected: dict[str, tuple[str, str, dict[str, Any]]] = {}
    main_jobs: set[str] = set()
    design_valid = True
    for name, case in cases.items():
        if not isinstance(case, dict) or case.get("scope") not in {"main", "diagnostic"}:
            raise ValueError(f"Pilot case {name!r} has no valid explicit scope")
        dataset = case.get("dataset_id")
        fold = case.get("outer_fold")
        if not isinstance(dataset, str) or not dataset or isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"Pilot case {name!r} lacks a dataset or integer outer fold")
        record = datasets.setdefault(dataset, {"task": None, "primary_metric": None, "main": {}, "diagnostic": {}})
        task = case.get("task")
        if task is not None:
            if task not in {"regression", "binary", "multiclass"}:
                raise ValueError(f"Unknown case task: {task!r}")
            if record["task"] is not None and record["task"] != task:
                issue("task_mismatch", name, "One dataset has inconsistent declared tasks")
                design_valid = False
            record["task"] = task
            record["primary_metric"] = "r2" if task == "regression" else "auc"
        if case["scope"] == "main":
            if fold in record["main"]:
                issue("duplicate_case", name, "Two main cases declare the same dataset and outer fold")
                design_valid = False
            record["main"][fold] = name
            methods = METHODS
        else:
            diagnostic_fold = case.get("diagnostic_fold")
            if diagnostic_fold is None:
                match = re.search(r"__diagnostic(\d+)$", name)
                diagnostic_fold = int(match.group(1)) if match else None
            if isinstance(diagnostic_fold, bool) or not isinstance(diagnostic_fold, int):
                raise ValueError(f"Diagnostic case {name!r} lacks an integer diagnostic_fold")
            if diagnostic_fold in record["diagnostic"] or fold != 0:
                issue("diagnostic_design", name, "Diagnostics require unique folds on outer fold zero")
                design_valid = False
            record["diagnostic"][diagnostic_fold] = name
            methods = DIAGNOSTIC_METHODS
        for method in methods:
            job = f"{name}__{method}"
            expected[job] = (name, method, case)
            if case["scope"] == "main":
                main_jobs.add(job)
    if len(datasets) != 2:
        design_valid = False
        issue("design_coverage", "cases", "The registered pilot requires exactly two datasets")
    for dataset, record in datasets.items():
        if set(record["main"]) != set(OUTER_FOLDS) or set(record["diagnostic"]) != set(DIAGNOSTIC_FOLDS):
            design_valid = False
            issue("design_coverage", dataset, "Each dataset requires main folds 0/1/2 and grouped diagnostics 0/1/2")
    if len(expected) != 36 or len(main_jobs) != 24:
        design_valid = False
    for job in sorted(set(stages) - set(expected)):
        issue("unexpected_stage", job, "Stage does not belong to a preregistered case/method")
    for job in sorted(set(outer_tests) - main_jobs):
        issue("unexpected_outer_test", job, "Outer test does not belong to a preregistered main case/method")
    if summary.get("planned_stages") != 36:
        issue("reported_stage_count", "summary", "Reported planned_stages differs from the registered 36")

    # Compatibility with older lightweight summaries: infer a metric only when
    # scores make its task unambiguous. Never infer from dataset names.
    for dataset, record in datasets.items():
        if record["primary_metric"] is None:
            observed = set()
            for job, (name, _, case) in expected.items():
                if case["dataset_id"] != dataset:
                    continue
                candidates = [outer_tests.get(job, {})]
                stage = stages.get(job, {})
                candidates.extend(stage.get("scores", {}).values())
                for candidate in candidates:
                    metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
                    observed.update(key for key in ("auc", "r2") if key in metrics)
            if len(observed) == 1:
                record["primary_metric"] = observed.pop()
                record["task"] = "regression" if record["primary_metric"] == "r2" else "classification"
            elif len(observed) > 1:
                issue("task_mismatch", dataset, "Scores contain both classification and regression primaries")

    score_values: dict[tuple[str, str], float] = {}
    alignment: dict[tuple[str, str], tuple[Any, Any]] = {}

    def read_score(job: str, role: str, score: Any) -> float | None:
        name, _, case = expected[job]
        metric = datasets[case["dataset_id"]]["primary_metric"]
        if not isinstance(score, dict) or not isinstance(score.get("metrics"), dict):
            issue("missing_score", job, f"Missing {role} metrics")
            return None
        metrics = score["metrics"]
        value = metrics.get(metric) if metric else None
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or _nonfinite(metrics):
            issue("invalid_score", job, f"{role} primary/auxiliary metrics are missing or nonfinite")
            return None
        value = float(value)
        if (metric == "auc" and not 0 <= value <= 1) or (metric == "r2" and value > 1):
            issue("invalid_score", job, f"{role} primary metric is outside its mathematical range")
            return None
        if "in_sample" in score and score["in_sample"] != (role == "train"):
            issue("score_scope", job, f"{role} score has an inconsistent in_sample flag")
            return None
        partition = "test" if role in {"outer_test", "diagnostic"} else role
        expected_rows = case.get("partitions", {}).get(partition)
        if expected_rows is not None and score.get("rows") != expected_rows:
            issue("score_alignment", job, f"{role} row count differs from its saved case partition")
            return None
        pair = (score.get("rows"), score.get("row_ids_sha256"))
        key = (name, role)
        if key in alignment and alignment[key] != pair:
            issue("score_alignment", job, f"{role} row IDs/count differ between methods in the same case")
            return None
        alignment[key] = pair
        score_values[(job, role)] = value
        return value

    valid_stages: set[str] = set()
    failed_stages: list[dict[str, Any]] = []
    missing_stages: list[str] = []
    invalid_stages: list[str] = []
    valid_outer: set[str] = set()
    for job, (_, _, case) in expected.items():
        stage = stages.get(job)
        if not isinstance(stage, dict) or stage.get("status") != "completed":
            missing_stages.append(job)
            if isinstance(stage, dict) and stage.get("status") == "failed":
                failed_stages.append({"job": job, "failure": stage.get("failure")})
        else:
            roles = ("train", "validation", "diagnostic") if case["scope"] == "diagnostic" else ("train", "validation")
            values = [read_score(job, role, stage.get("scores", {}).get(role)) for role in roles]
            if all(value is not None for value in values):
                valid_stages.add(job)
            else:
                invalid_stages.append(job)
        if job in main_jobs and job in outer_tests:
            if read_score(job, "outer_test", outer_tests[job]) is not None:
                valid_outer.add(job)
    reported_completed = sum(isinstance(stage, dict) and stage.get("status") == "completed" for stage in stages.values())
    if summary.get("completed_stages") != reported_completed:
        issue("reported_stage_count", "summary", "Reported completed_stages differs from actual stage statuses")
    barrier = summary.get("test_barrier_satisfied") is True
    if outer_tests and not barrier:
        issue("test_barrier_violation", "outer_tests", "Outer-test scores exist before the recorded all-stage barrier")
    complete = (design_valid and len(valid_stages) == 36 and len(valid_outer) == 24 and barrier
                and summary.get("status") == "completed" and not issues)

    report_datasets = {}
    criteria_by_dataset = {}
    for dataset, record in sorted(datasets.items()):
        by_method = {method: {} for method in METHODS}
        outer_rows = []
        for fold, name in sorted(record["main"].items()):
            values = {method: score_values[(f"{name}__{method}", "outer_test")]
                      for method in METHODS if (f"{name}__{method}", "outer_test") in score_values}
            for method, value in values.items():
                by_method[method][fold] = value
            outer_rows.append({"outer_fold": fold, "case": name, "methods": values})
        paired_outer = {method: _paired(by_method["residual"], by_method[method]) for method in ("direct", "tar", "frozen")}
        diagnostic_values = {method: {} for method in DIAGNOSTIC_METHODS}
        optimism_values = {method: {} for method in DIAGNOSTIC_METHODS}
        diagnostic_rows = []
        for fold, name in sorted(record["diagnostic"].items()):
            values = {}
            for method in DIAGNOSTIC_METHODS:
                job = f"{name}__{method}"
                diagnostic = score_values.get((job, "diagnostic"))
                training = score_values.get((job, "train"))
                if diagnostic is not None:
                    diagnostic_values[method][fold] = diagnostic
                if training is not None and diagnostic is not None:
                    optimism_values[method][fold] = training - diagnostic
                if training is not None or diagnostic is not None:
                    values[method] = {"train": training, "diagnostic": diagnostic,
                                      "train_minus_diagnostic": training - diagnostic if training is not None and diagnostic is not None else None}
            diagnostic_rows.append({"diagnostic_fold": fold, "outer_fold": 0, "case": name, "methods": values})
        paired_diagnostic = _paired(diagnostic_values["residual"], diagnostic_values["direct"])
        report_datasets[dataset] = {
            "task": record["task"], "primary_metric": record["primary_metric"],
            "outer_test": {"by_fold": outer_rows,
                           "methods": {method: {**_stats(list(values.values())), "folds": sorted(values)} for method, values in by_method.items()},
                           "paired_residual_minus": paired_outer},
            "grouped_diagnostic": {"by_fold": diagnostic_rows,
                                   "methods": {method: {"diagnostic": _stats(list(diagnostic_values[method].values())),
                                                        "train_minus_diagnostic": _stats(list(optimism_values[method].values()))}
                                               for method in DIAGNOSTIC_METHODS},
                                   "paired_residual_minus_direct": paired_diagnostic},
        }
        criteria_by_dataset[dataset] = {
            "positive_mean_outer_vs_tar": paired_outer["tar"]["mean"] > 0 if paired_outer["tar"]["n"] == 3 else None,
            "positive_mean_outer_vs_direct": paired_outer["direct"]["mean"] > 0 if paired_outer["direct"]["n"] == 3 else None,
            "at_least_two_positive_outer_vs_direct": paired_outer["direct"]["positive_folds"] >= 2 if paired_outer["direct"]["n"] == 3 else None,
            "positive_mean_nested_vs_direct": paired_diagnostic["mean"] > 0 if paired_diagnostic["n"] == 3 else None,
        }
    successful = complete and all(value is True for criteria in criteria_by_dataset.values() for value in criteria.values())
    tar_jobs = {job for job, (_, method, _) in expected.items() if method == "tar"}
    missing_tar = sorted(tar_jobs - valid_stages)
    cpu_environment = str(summary.get("environment", {}).get("device", "")).startswith("cpu")
    pending_gpu = bool(missing_tar) and cpu_environment and not any(failure["job"] in tar_jobs for failure in failed_stages)
    recommendation = ("continue_unchanged" if successful else "analyze_failures_and_overfit_before_scaling" if complete
                      else "await_required_comparisons")
    return {
        "format_version": 1, "run_id": summary.get("run_id"),
        "status": "invalid" if issues else "complete" if complete else "incomplete",
        "coverage": {"complete": complete, "registered_design_covered": design_valid,
                     "expected_stages": 36, "expected_outer_tests": 24,
                     "completed_stages": reported_completed, "valid_completed_stages": len(valid_stages),
                     "valid_outer_tests": len(valid_outer), "test_barrier_satisfied": barrier,
                     "missing_stage_jobs": sorted(missing_stages), "failed_stages": failed_stages,
                     "invalid_stage_jobs": sorted(invalid_stages), "missing_outer_test_jobs": sorted(main_jobs - valid_outer),
                     "pending_gpu": pending_gpu,
                     "required_TAR": {"expected_stages": 6, "valid_completed_stages": len(tar_jobs & valid_stages),
                                      "missing_stage_jobs": missing_tar, "valid_outer_tests": len(tar_jobs & valid_outer),
                                      "status": "pending_gpu" if pending_gpu else "required_TAR" if missing_tar else "stages_complete"}},
        "datasets": report_datasets,
        "decision": {"rule": "phase4_descriptive_continue_v1", "recommendation": recommendation,
                     "eligible_for_continue_decision": complete, "criteria_by_dataset": criteria_by_dataset,
                     "reason": ("All registered descriptive criteria passed; report this recommendation before scaling."
                                if successful else "Completed comparisons require targeted failure/overfit analysis before full evaluation."
                                if complete else "Required comparisons or valid outer-test/diagnostic coverage are incomplete; no gain recommendation is supported."),
                     "automatic_redesign": False},
        "issues": issues,
        "limitations": ["Three repeated outer holdouts overlap; means and sample standard deviations are descriptive.",
                        "Two pilot datasets do not support an all-benchmark improvement or significance claim.",
                        "AUROC and R2 remain separate; no cross-task average or p-value is computed.",
                        "Nested D gains and train-minus-D gaps concern complete grouped pipelines, not cross-fitted latent embeddings."],
    }
