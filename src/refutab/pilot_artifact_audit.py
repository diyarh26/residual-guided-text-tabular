"""Read saved predictions and teacher fit scopes without fitting or inference.

This targeted prerequisite does not claim a full neural-weight loading audit.
It can run against the already mounted private Drive run in Colab on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import pilot as p
from .teacher import TeacherArtifacts


def _same_scores(actual: dict, saved: dict) -> bool:
    """Allow CPU reduction roundoff while requiring exact alignment metadata."""
    if set(actual) != set(saved) or set(actual["metrics"]) != set(saved["metrics"]):
        return False
    if {k: v for k, v in actual.items() if k != "metrics"} != {k: v for k, v in saved.items() if k != "metrics"}:
        return False
    for key, value in actual["metrics"].items():
        prior = saved["metrics"][key]
        if type(value) in (float, int) and type(prior) in (float, int):
            if not np.isclose(value, prior, rtol=0, atol=1e-12):
                return False
        elif value != prior:
            return False
    return True


def audit_completed_pilot(root: Path, directory: Path, config: dict) -> dict:
    directory = directory.resolve()
    state = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    review = json.loads((root / config["anchor_review"]).read_text(encoding="utf-8"))
    if (state["status"] != "completed" or state["run_id"] != config["prior_run_id"]
            or state["binding_sha256"] != review["binding_sha256"]
            or p._hash(state["binding"]) != state["binding_sha256"]
            or len(state["stages"]) != 36 or len(state["outer_tests"]) != 24):
        raise ValueError("Prior pilot provenance or coverage differs")
    tables, _ = p._load_tables(root, state["config"])
    if {k: t.fingerprint() for k, t in tables.items()} != state["binding"]["dataset_fingerprints"]:
        raise ValueError("Prior pilot dataset content changed")
    cases = p.build_cases(root, tables, state["config"])
    scores_checked, teachers_checked, component_scopes = 0, 0, 0
    for name, case in cases.items():
        split = p.SplitAssignments.load(directory / "splits" / f"{name}.json")
        if split.fingerprint() != state["binding"]["split_fingerprints"][name]:
            raise ValueError("Prior saved split differs from original binding")
        # Source byte metadata may differ between Windows and Linux; actual row
        # identities/order must match the pinned reconstruction for every role.
        if split.partitions != case["split"].partitions:
            raise ValueError("Prior saved row assignments differ from reconstruction")
        table = tables[case["dataset_id"]]
        for method in case["methods"]:
            job = f"{name}__{method}"
            entry = state["stages"][job]
            attempt = (directory / entry["attempt"]).resolve()
            if not attempt.is_relative_to((directory / "attempts").resolve()):
                raise ValueError("Prior attempt path escapes run")
            if entry["status"] != "completed" or p.sha256_file(attempt / "manifest.json") != entry["manifest_sha256"]:
                raise ValueError("Prior stage manifest changed")
            record = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
            if (record["context"]["pilot_binding_sha256"] != state["binding_sha256"]
                    or record["context"]["split_fingerprint"] != split.fingerprint()
                    or record["model"]["fit_ids_sha256"] != p.row_ids_sha256(split.train_ids)):
                raise ValueError("Prior stage fitting scope differs")

            def checked_json(relative):
                path = attempt / relative
                if p.sha256_file(path) != record["files_sha256"][relative]:
                    raise ValueError(f"Prior metadata checksum changed: {relative}")
                return json.loads(path.read_text(encoding="utf-8"))

            prefix = "bundle/" if method in {"direct", "residual"} else ""
            relatives = [(prefix + "downstream/metadata.json", "fit_ids")]
            if prefix:
                relatives += [("bundle/pca/metadata.json", "fit_ids"), ("bundle/structured/metadata.json", "fit_ids")]
            else:
                relatives += [("representation/metadata.json" if method == "frozen" else "tar/representation/metadata.json",
                               "fit_ids" if method == "frozen" else "pca_fit_ids")]
            for relative, key in relatives:
                if tuple(checked_json(relative)[key]) != split.train_ids:
                    raise ValueError("Prior component fitted outside A")
                component_scopes += 1
            if method == "residual":
                checked_json("teacher/metadata.json")
                teacher = TeacherArtifacts.load(attempt / "teacher")
                if teacher.train_ids != split.train_ids or teacher.validation_ids != split.validation_ids:
                    raise ValueError("Prior teacher fit/selection rows differ")
                if list(teacher.class_order) != record["model"]["class_order"]:
                    raise ValueError("Prior teacher class order differs")
                if case["scope"] == "diagnostic":
                    expected = p.teacher_fold_ids(table, split.train_ids, case["group_columns"])
                    np.testing.assert_array_equal(teacher.fold_ids, expected)
                teachers_checked += 1
            for role, saved in record["scores"].items():
                ids = {"train": split.train_ids, "validation": split.validation_ids, "diagnostic": split.test_ids}[role]
                path = attempt / f"{role}_predictions.npy"
                if p.sha256_file(path) != record["files_sha256"][path.name]:
                    raise ValueError("Prior saved predictions changed")
                actual = p._prediction_score(table, ids, np.load(path, allow_pickle=False), record["model"]["class_order"], in_sample=role == "train")
                if not _same_scores(actual, saved) or not _same_scores(actual, entry["scores"][role]):
                    raise ValueError("Prior saved prediction metrics do not reproduce")
                scores_checked += 1
            if case["scope"] == "main":
                saved = state["outer_tests"][job]
                path = directory / "outer_evaluation" / job / "test_predictions.npy"
                request = json.loads((path.parent / "request.json").read_text(encoding="utf-8"))
                if (request["binding_sha256"] != state["binding_sha256"]
                        or request["stage_manifest_sha256"] != entry["manifest_sha256"]
                        or request["class_order"] != record["model"]["class_order"]
                        or saved["prediction_sha256"] != p.sha256_file(path)):
                    raise ValueError("Prior outer prediction provenance changed")
                actual = p._prediction_score(table, split.test_ids, np.load(path, allow_pickle=False), record["model"]["class_order"])
                if not _same_scores(actual, {k: v for k, v in saved.items() if k not in {"prediction_sha256", "stage_manifest_sha256"}}):
                    raise ValueError("Prior outer saved metrics do not reproduce")
                scores_checked += 1
    if (scores_checked, teachers_checked, component_scopes) != (108, 12, 96):
        raise ValueError("Targeted prior-pilot audit coverage incomplete")
    return {"status": "passed", "run_id": state["run_id"], "binding_sha256": state["binding_sha256"],
            "manifest_sha256": p.sha256_file(directory / "manifest.json"),
            "saved_prediction_score_sets": scores_checked, "teacher_bundles": teachers_checked,
            "component_fit_scopes": component_scopes, "new_fits": 0, "new_predictions": 0,
            "scope": "saved predictions, teacher bundles and component metadata; neural weights not loaded"}
