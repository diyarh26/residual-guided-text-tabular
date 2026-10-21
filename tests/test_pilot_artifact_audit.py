"""A complete saved-artifact fixture, with no E5 loading or fresh inference."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab import pilot as p
from refutab.data import DatasetTable
from refutab.pilot_artifact_audit import audit_completed_pilot, _same_scores
from refutab.splits import SplitAssignments
from refutab.teacher import TeacherConfig, fit_oof_teacher
from test_pilot_protocol import table_and_outer


class PilotArtifactAuditTests(unittest.TestCase):
    def fixture(self, root):
        tables, cases = {}, {}
        for task in ("regression", "multiclass"):
            original, outer = table_and_outer(task)
            table = DatasetTable(original.frame, replace(original.schema, dataset_id=task), row_ids=original.row_ids)
            tables[task] = table
            split = replace(outer, dataset_fingerprint=table.fingerprint())
            for fold in range(3):
                cases[f"{task}__fold{fold}"] = {"dataset_id": task, "scope": "main", "split": split, "methods": list(p.METHODS)}
            for fold, diagnostic in enumerate(p.grouped_diagnostics(table, split, ["description"])):
                cases[f"{task}__fold0__diagnostic{fold}"] = {"dataset_id": task, "scope": "diagnostic", "split": diagnostic, "group_columns": ["description"], "methods": ["direct", "residual"]}
        binding = {"dataset_fingerprints": {k: t.fingerprint() for k, t in tables.items()},
                   "split_fingerprints": {k: c["split"].fingerprint() for k, c in cases.items()}}
        state = {"status": "completed", "run_id": "fixture-prior", "binding": binding,
                 "binding_sha256": p._hash(binding), "config": {}, "stages": {}, "outer_tests": {}}
        directory = root / "private-run"
        directory.mkdir()
        for name, case in cases.items():
            table, split = tables[case["dataset_id"]], case["split"]
            split.save(directory / "splits" / f"{name}.json")
            classes = list(p._training_classes(table.targets(split.train_ids), table.schema.task))
            for method in case["methods"]:
                job = name + "__" + method
                attempt = directory / "attempts" / job
                attempt.mkdir(parents=True)
                record = {"status": "completed", "method": method, "context": {"pilot_binding_sha256": state["binding_sha256"], "split_fingerprint": split.fingerprint()},
                          "model": {"class_order": classes, "fit_ids_sha256": p.row_ids_sha256(split.train_ids)}, "scores": {}}
                pipeline = method in {"direct", "residual"}
                relatives = ["bundle/downstream/metadata.json", "bundle/pca/metadata.json", "bundle/structured/metadata.json"] if pipeline else ["downstream/metadata.json", "representation/metadata.json" if method == "frozen" else "tar/representation/metadata.json"]
                for relative in relatives:
                    path = attempt / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    p._write(path, {"pca_fit_ids" if relative.startswith("tar") else "fit_ids": list(split.train_ids)})
                if method == "residual":
                    teacher = fit_oof_teacher(table.structured(split.train_ids), table.targets(split.train_ids),
                                              categorical_columns=table.schema.categorical_columns, train_ids=split.train_ids,
                                              validation_X=table.structured(split.validation_ids), validation_ids=split.validation_ids,
                                              config=TeacherConfig(task=table.schema.task, iterations=1, depth=2, thread_count=1),
                                              fold_ids=p.teacher_fold_ids(table, split.train_ids, ["description"]) if case["scope"] == "diagnostic" else None)
                    teacher.save(attempt / "teacher")
                roles = {"train": split.train_ids, "validation": split.validation_ids}
                if case["scope"] == "diagnostic":
                    roles["diagnostic"] = split.test_ids
                for role, ids in roles.items():
                    prediction = np.linspace(.1, .9, len(ids)) if not classes else np.full((len(ids), len(classes)), 1 / len(classes))
                    np.save(attempt / f"{role}_predictions.npy", prediction)
                    record["scores"][role] = p._prediction_score(table, ids, prediction, classes, in_sample=role == "train")
                record["files_sha256"] = p._inventory(attempt)
                p._write(attempt / "manifest.json", record)
                state["stages"][job] = {"status": "completed", "attempt": attempt.relative_to(directory).as_posix(),
                                         "manifest_sha256": p.sha256_file(attempt / "manifest.json"), "scores": record["scores"]}
                if case["scope"] == "main":
                    output = directory / "outer_evaluation" / job
                    output.mkdir(parents=True)
                    ids = split.test_ids
                    prediction = np.linspace(.1, .9, len(ids)) if not classes else np.full((len(ids), len(classes)), 1 / len(classes))
                    np.save(output / "test_predictions.npy", prediction)
                    result = p._prediction_score(table, ids, prediction, classes)
                    result.update(prediction_sha256=p.sha256_file(output / "test_predictions.npy"), stage_manifest_sha256=state["stages"][job]["manifest_sha256"])
                    state["outer_tests"][job] = result
                    p._write(output / "result.json", result)
                    p._write(output / "request.json", {"binding_sha256": state["binding_sha256"], "stage_manifest_sha256": result["stage_manifest_sha256"], "class_order": classes})
        p._write(directory / "manifest.json", state)
        p._write(root / "review.json", {"binding_sha256": state["binding_sha256"]})
        return directory, tables, cases

    def test_complete_targeted_audit_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, tables, cases = self.fixture(root)
            config = {"anchor_review": "review.json", "prior_run_id": "fixture-prior"}
            with patch.object(p, "_load_tables", return_value=(tables, {})), patch.object(p, "build_cases", return_value=cases), \
                 patch.object(p, "load_method", side_effect=AssertionError("Audit attempted inference")), \
                 patch.object(p, "train_method", side_effect=AssertionError("Audit attempted fitting")):
                result = audit_completed_pilot(root, directory, config)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["saved_prediction_score_sets"], 108)
                self.assertEqual(result["teacher_bundles"], 12)
                self.assertEqual(result["component_fit_scopes"], 96)
                prediction = next((directory / "outer_evaluation").glob("*/test_predictions.npy"))
                prediction.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    audit_completed_pilot(root, directory, config)

    def test_reduction_roundoff_tolerated_but_alignment_or_scores_not(self):
        score = {"rows": 12, "row_ids_sha256": "a" * 64, "metrics": {"auc": .8, "all_classes_present": True}}
        self.assertTrue(_same_scores(score, {**score, "metrics": {"auc": .8 + 1e-15, "all_classes_present": True}}))
        self.assertFalse(_same_scores(score, {**score, "rows": 11}))
        self.assertFalse(_same_scores(score, {**score, "metrics": {"auc": .81, "all_classes_present": True}}))


if __name__ == "__main__":
    unittest.main()
