"""Task-separated paired aggregation and honest incomplete/negative decisions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.pilot_report import aggregate_pilot


def complete_summary():
    summary = {"format_version": 1, "run_id": "synthetic-report-only", "status": "completed",
               "environment": {"device": "cuda:0"}, "planned_stages": 36, "completed_stages": 36,
               "test_barrier_satisfied": True, "cases": {}, "stages": {}, "outer_tests": {}}

    def score(case_name, role, metric, value, rows):
        return {"rows": rows, "row_ids_sha256": f"{case_name}-{role}-rows",
                "in_sample": role == "train", "metrics": {metric: value}}

    for dataset, task, metric, offset in (("classification", "multiclass", "auc", 0.4),
                                          ("regression", "regression", "r2", 0.0)):
        for fold in range(3):
            case_name = f"{dataset}__fold{fold}"
            case = {"dataset_id": dataset, "outer_fold": fold, "scope": "main", "task": task,
                    "partitions": {"train": 30, "validation": 8, "test": 10, "excluded": 0}}
            summary["cases"][case_name] = case
            for method, base in (("frozen", 0.20), ("direct", 0.25), ("residual", 0.30), ("tar", 0.27)):
                job = f"{case_name}__{method}"
                summary["stages"][job] = {"status": "completed", "scores": {
                    "train": score(case_name, "train", metric, 0.95, 30),
                    "validation": score(case_name, "validation", metric, offset + base, 8)}}
                summary["outer_tests"][job] = score(case_name, "outer_test", metric, offset + base + fold * 0.01, 10)
        for diagnostic in range(3):
            case_name = f"{dataset}__fold0__diagnostic{diagnostic}"
            summary["cases"][case_name] = {"dataset_id": dataset, "outer_fold": 0, "scope": "diagnostic", "task": task,
                                           "diagnostic_fold": diagnostic,
                                           "partitions": {"train": 20, "validation": 5, "test": 13, "excluded": 10}}
            for method, base in (("direct", 0.22), ("residual", 0.26)):
                job = f"{case_name}__{method}"
                summary["stages"][job] = {"status": "completed", "scores": {
                    "train": score(case_name, "train", metric, 0.95, 20),
                    "validation": score(case_name, "validation", metric, offset + base, 5),
                    "diagnostic": score(case_name, "diagnostic", metric, offset + base + diagnostic * 0.01, 13)}}
    return summary


class PilotReportTests(unittest.TestCase):
    def test_complete_positive_descriptive_criteria_and_task_separation(self):
        report = aggregate_pilot(complete_summary())
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["coverage"]["valid_completed_stages"], 36)
        self.assertEqual(report["coverage"]["valid_outer_tests"], 24)
        self.assertEqual(report["decision"]["recommendation"], "continue_unchanged")
        self.assertFalse(report["decision"]["automatic_redesign"])
        self.assertEqual(report["datasets"]["classification"]["primary_metric"], "auc")
        self.assertEqual(report["datasets"]["regression"]["primary_metric"], "r2")
        self.assertNotIn("overall_mean", report)
        self.assertNotIn("p_value", json.dumps(report))
        self.assertEqual(report["issues"], [])
        json.dumps(report, allow_nan=False)

    def test_means_sample_standard_deviation_and_paired_nested_gaps(self):
        report = aggregate_pilot(complete_summary())["datasets"]["regression"]
        residual = report["outer_test"]["methods"]["residual"]
        self.assertAlmostEqual(residual["mean"], 0.31)
        self.assertAlmostEqual(residual["std"], 0.01)
        self.assertEqual(residual["std_ddof"], 1)
        paired = report["outer_test"]["paired_residual_minus"]["direct"]
        self.assertEqual(paired["positive_folds"], 3)
        self.assertAlmostEqual(paired["mean"], 0.05)
        self.assertAlmostEqual(paired["std"], 0.0)
        diagnostic = report["grouped_diagnostic"]
        self.assertAlmostEqual(diagnostic["paired_residual_minus_direct"]["mean"], 0.04)
        self.assertAlmostEqual(diagnostic["methods"]["direct"]["train_minus_diagnostic"]["mean"], 0.72)
        self.assertAlmostEqual(diagnostic["methods"]["residual"]["train_minus_diagnostic"]["mean"], 0.68)

    def test_missing_tar_cpu_reports_pending_gpu_and_blocks_recommendation(self):
        summary = complete_summary()
        summary["stages"] = {key: value for key, value in summary["stages"].items() if not key.endswith("__tar")}
        summary["outer_tests"] = {}
        summary.update(status="incomplete", completed_stages=30, test_barrier_satisfied=False,
                       environment={"device": "cpu"})
        report = aggregate_pilot(summary)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["coverage"]["valid_completed_stages"], 30)
        self.assertTrue(report["coverage"]["pending_gpu"])
        self.assertEqual(report["coverage"]["required_TAR"]["status"], "pending_gpu")
        self.assertEqual(len(report["coverage"]["required_TAR"]["missing_stage_jobs"]), 6)
        self.assertEqual(report["decision"]["recommendation"], "await_required_comparisons")
        self.assertFalse(report["decision"]["eligible_for_continue_decision"])
        self.assertIsNone(report["datasets"]["classification"]["outer_test"]["methods"]["residual"]["mean"])

    def test_pairs_use_fold_intersection_and_single_pair_has_no_std(self):
        summary = complete_summary()
        for method, folds in (("residual", (2,)), ("direct", (0,))):
            for fold in folds:
                del summary["outer_tests"][f"regression__fold{fold}__{method}"]
        report = aggregate_pilot(summary)["datasets"]["regression"]["outer_test"]["paired_residual_minus"]["direct"]
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["pairs"][0]["fold"], 1)
        self.assertEqual(report["unpaired_residual_folds"], [0])
        self.assertEqual(report["unpaired_comparator_folds"], [2])
        self.assertIsNone(report["std"])

    def test_complete_negative_or_mixed_findings_require_analysis(self):
        summary = complete_summary()
        for dataset in ("classification", "regression"):
            metric = "auc" if dataset == "classification" else "r2"
            for fold in range(3):
                direct = summary["outer_tests"][f"{dataset}__fold{fold}__direct"]["metrics"][metric]
                summary["outer_tests"][f"{dataset}__fold{fold}__residual"]["metrics"][metric] = direct - 0.02
        report = aggregate_pilot(summary)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["decision"]["recommendation"], "analyze_failures_and_overfit_before_scaling")
        self.assertFalse(report["decision"]["criteria_by_dataset"]["regression"]["positive_mean_outer_vs_direct"])

    def test_both_datasets_and_two_positive_outer_folds_are_required(self):
        summary = complete_summary()
        # Positive mean, but only one positive paired fold on the regression task.
        for fold, difference in enumerate((0.10, -0.01, -0.01)):
            direct = summary["outer_tests"][f"regression__fold{fold}__direct"]["metrics"]["r2"]
            summary["outer_tests"][f"regression__fold{fold}__residual"]["metrics"]["r2"] = direct + difference
        report = aggregate_pilot(summary)
        self.assertTrue(report["decision"]["criteria_by_dataset"]["regression"]["positive_mean_outer_vs_direct"])
        self.assertFalse(report["decision"]["criteria_by_dataset"]["regression"]["at_least_two_positive_outer_vs_direct"])
        self.assertEqual(report["decision"]["recommendation"], "analyze_failures_and_overfit_before_scaling")
        summary = complete_summary()
        summary["stages"]["regression__fold0__diagnostic0__residual"]["scores"]["diagnostic"]["metrics"]["r2"] = -1.0
        report = aggregate_pilot(summary)
        self.assertFalse(report["decision"]["criteria_by_dataset"]["regression"]["positive_mean_nested_vs_direct"])
        self.assertEqual(report["decision"]["recommendation"], "analyze_failures_and_overfit_before_scaling")

    def test_nonfinite_missing_and_alignment_failures_remain_explicit(self):
        for mutation in ("nonfinite_primary", "nonfinite_auxiliary", "missing_diagnostic", "wrong_rows", "wrong_ids"):
            summary = complete_summary()
            if mutation == "nonfinite_primary":
                summary["outer_tests"]["regression__fold0__residual"]["metrics"]["r2"] = float("nan")
            elif mutation == "nonfinite_auxiliary":
                summary["outer_tests"]["classification__fold0__residual"]["metrics"]["logloss"] = float("inf")
            elif mutation == "missing_diagnostic":
                del summary["stages"]["regression__fold0__diagnostic0__residual"]["scores"]["diagnostic"]
            elif mutation == "wrong_rows":
                summary["outer_tests"]["regression__fold0__residual"]["rows"] = 999
            else:
                summary["outer_tests"]["regression__fold0__residual"]["row_ids_sha256"] = "different rows"
            with self.subTest(mutation=mutation):
                report = aggregate_pilot(summary)
                self.assertEqual(report["status"], "invalid")
                self.assertTrue(report["issues"])
                self.assertFalse(report["coverage"]["complete"])
                self.assertEqual(report["decision"]["recommendation"], "await_required_comparisons")
                json.dumps(report, allow_nan=False)

    def test_failed_tar_is_reported_as_failure_without_claiming_only_gpu_is_missing(self):
        summary = complete_summary()
        job = "classification__fold0__tar"
        summary["stages"][job] = {"status": "failed", "failure": {"type": "RuntimeError", "message": "synthetic test failure"}}
        summary["outer_tests"] = {}
        summary.update(status="failed", completed_stages=35, test_barrier_satisfied=False)
        report = aggregate_pilot(summary)
        self.assertFalse(report["coverage"]["pending_gpu"])
        self.assertEqual(report["coverage"]["required_TAR"]["status"], "required_TAR")
        self.assertEqual(report["coverage"]["failed_stages"][0]["job"], job)
        self.assertEqual(report["coverage"]["failed_stages"][0]["failure"]["message"], "synthetic test failure")

    def test_outer_test_barrier_and_reported_counts_are_checked(self):
        summary = complete_summary()
        summary["test_barrier_satisfied"] = False
        summary["completed_stages"] = 35
        report = aggregate_pilot(summary)
        self.assertEqual(report["status"], "invalid")
        self.assertEqual({issue["kind"] for issue in report["issues"]}, {"test_barrier_violation", "reported_stage_count"})

    def test_incomplete_design_is_not_silently_treated_as_small_complete_pilot(self):
        summary = complete_summary()
        summary["cases"] = {key: value for key, value in summary["cases"].items() if value["dataset_id"] == "classification"}
        summary["stages"] = {key: value for key, value in summary["stages"].items() if key.startswith("classification")}
        summary["outer_tests"] = {key: value for key, value in summary["outer_tests"].items() if key.startswith("classification")}
        summary.update(planned_stages=18, completed_stages=18)
        report = aggregate_pilot(summary)
        self.assertFalse(report["coverage"]["registered_design_covered"])
        self.assertFalse(report["coverage"]["complete"])
        self.assertEqual(report["decision"]["recommendation"], "await_required_comparisons")

    def test_older_summaries_infer_task_from_metrics_without_dataset_name_rules(self):
        summary = complete_summary()
        for case in summary["cases"].values():
            del case["task"]
            case.pop("diagnostic_fold", None)
        report = aggregate_pilot(summary)
        self.assertEqual(report["datasets"]["classification"]["primary_metric"], "auc")
        self.assertEqual(report["datasets"]["regression"]["primary_metric"], "r2")
        self.assertEqual(report["status"], "complete")


if __name__ == "__main__":
    unittest.main()
