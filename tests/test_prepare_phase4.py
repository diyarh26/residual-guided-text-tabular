"""Versioned public-data preparation: integrity, offline replay and no overwrite."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_phase4.py"
SPEC = importlib.util.spec_from_file_location("prepare_phase4", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def fixture(*, image_col=None, extra_member=None, csv_payload=None):
    csv_payload = csv_payload or b'text,number,category,target\n"two\nlines",1,a,3\nother,,b,7\n'
    metadata = json.dumps({"dataset_id": "REG_TEXT_SOURCE", "target": "target", "image_col": image_col,
                           "task_type": "reg", "num_rows": 2, "num_features": 3}).encode()
    payloads = {"data.csv": csv_payload, "metadata.json": metadata}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
        if extra_member:
            archive.writestr(extra_member, b"must not be extracted")
    archive_bytes = buffer.getvalue()
    config = {
        "directory": "data/pilot-v1", "kaggle_ref": "owner/pilot", "kaggle_version": 1,
        "url": "https://www.kaggle.com/api/v1/datasets/download/owner/pilot?datasetVersionNumber=1",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        "metadata_dataset_id": "REG_TEXT_SOURCE", "csv_columns": ["text", "number", "category", "target"],
        "schema": {"dataset_id": "REG_TEXT_PILOT", "task": "regression", "target_column": "target",
                   "text_columns": ["text"], "numerical_columns": ["number"], "categorical_columns": ["category"],
                   "forbidden_columns": ["row_id", "fold_id", "teacher_prediction", "residual"]},
        "audit": {"rows": 2},
    }
    return config, archive_bytes, payloads


class Phase4PreparationTests(unittest.TestCase):
    def test_verified_download_then_offline_replay_preserves_exact_bytes(self):
        config, archive, payloads = fixture()
        with tempfile.TemporaryDirectory(prefix="refutab-phase4-assets-") as directory:
            root = Path(directory)
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(archive)) as network:
                result = prepare.prepare_dataset(config, root)
                network.assert_called_once()
                self.assertIn("datasetVersionNumber=1", network.call_args.args[0].full_url)
            self.assertEqual(result["rows"], 2)  # multiline text is one CSV row
            destination = root / config["directory"]
            for name, payload in payloads.items():
                self.assertEqual((destination / name).read_bytes(), payload)
            with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("network accessed")):
                self.assertEqual(prepare.prepare_dataset(config, root), result)

    def test_archive_hash_failure_retains_partial_and_materializes_no_table(self):
        config, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"changed release")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    prepare.prepare_dataset(config, root)
            destination = root / config["directory"]
            self.assertEqual((destination / "archive.zip.part").read_bytes(), b"changed release")
            self.assertFalse((destination / "data.csv").exists())
            self.assertFalse((destination / "archive.zip").exists())

    def test_member_hash_failure_before_any_member_write(self):
        config, archive, _ = fixture()
        config["files"]["data.csv"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(archive)):
                with self.assertRaisesRegex(ValueError, "member content"):
                    prepare.prepare_dataset(config, root)
            self.assertFalse((root / config["directory"] / "metadata.json").exists())
            self.assertFalse((root / config["directory"] / "data.csv").exists())

    def test_existing_user_file_and_stale_partial_are_never_overwritten(self):
        config, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / config["directory"]
            destination.mkdir(parents=True)
            data = destination / "data.csv"
            data.write_bytes(b"user data")
            with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("network accessed")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    prepare.prepare_dataset(config, root)
            self.assertEqual(data.read_bytes(), b"user data")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / config["directory"]
            destination.mkdir(parents=True)
            partial = destination / "archive.zip.part"
            partial.write_bytes(b"previous interrupted bytes")
            with patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("network accessed")):
                with self.assertRaises(FileExistsError):
                    prepare.prepare_dataset(config, root)
            self.assertEqual(partial.read_bytes(), b"previous interrupted bytes")

    def test_unpinned_urls_hashes_and_outside_destination_fail_before_network(self):
        original, _, _ = fixture()
        changes = [
            {"url": "https://www.kaggle.com/api/v1/datasets/download/owner/pilot"},
            {"url": "https://www.kaggle.com/api/v1/datasets/download/owner/pilot?datasetVersionNumber=2"},
            {"url": "https://owner:token@www.kaggle.com/api/v1/datasets/download/owner/pilot?datasetVersionNumber=1"},
            {"directory": "../elsewhere"}, {"directory": "data/../../elsewhere"},
            {"archive_sha256": "missing"}, {"kaggle_version": True},
            {"files": {"../data.csv": "0" * 64}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for change in changes:
                with self.subTest(change=change), patch.object(prepare.urllib.request, "urlopen", side_effect=AssertionError("network accessed")):
                    config = {**deepcopy(original), **change}
                    with self.assertRaises(ValueError):
                        prepare.prepare_dataset(config, Path(directory))

    def test_unexpected_zip_path_is_never_extracted(self):
        config, archive, _ = fixture(extra_member="../../outside.txt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(archive)):
                with self.assertRaisesRegex(ValueError, "ZIP members"):
                    prepare.prepare_dataset(config, root)
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse((root / config["directory"] / "data.csv").exists())

    def test_original_image_metadata_and_invalid_roles_rejected(self):
        config, _, payloads = fixture(image_col="photo")
        with self.assertRaisesRegex(ValueError, "image column"):
            prepare.validate_payloads(config, payloads)
        config, _, payloads = fixture()
        config["schema"]["numerical_columns"].append("target")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare.validate_payloads(config, payloads)
        config, _, payloads = fixture()
        config["schema"]["forbidden_columns"].append("number")
        with self.assertRaisesRegex(ValueError, "Forbidden columns"):
            prepare.validate_payloads(config, payloads)

    def test_duplicate_headers_and_wrong_csv_row_count_fail(self):
        config, _, payloads = fixture(csv_payload=b"text,number,category,target\none,1,a,3\n")
        with self.assertRaisesRegex(ValueError, "row count"):
            prepare.validate_payloads(config, payloads)
        config, _, payloads = fixture(csv_payload=b"text,number,number,target\none,1,a,3\ntwo,2,b,7\n")
        with self.assertRaisesRegex(ValueError, "CSV columns/order"):
            prepare.validate_payloads(config, payloads)


if __name__ == "__main__":
    unittest.main()
