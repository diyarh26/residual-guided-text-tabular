"""Small private transfer preserves all audit inputs and refuses corruption."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab import pilot as p
from refutab.audit_transfer import package_audit, restore_audit
from refutab.pilot_artifact_audit import audit_completed_pilot
import test_pilot_artifact_audit as fixtures


class AuditTransferTests(unittest.TestCase):
    def test_transfer_reaudits_complete_fixture_without_neural_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, tables, cases = fixtures.PilotArtifactAuditTests().fixture(root)
            renamed = root / "fixture-prior"
            directory.rename(renamed)
            private_extra = renamed / "frozen_cache/unused.npy"
            private_extra.parent.mkdir()
            private_extra.write_bytes(b"must not be packaged")
            (renamed / "unneeded.pt").write_bytes(b"large checkpoint stand-in")
            output = root / "private-audit.zip"
            result = package_audit(renamed, output)
            self.assertGreater(result["files"], 100)
            with zipfile.ZipFile(output) as archive:
                self.assertFalse(any("frozen_cache" in name or name.endswith(".pt") for name in archive.namelist()))
            restored = restore_audit(output, root / "other-drive/runs")
            config = {"anchor_review": "review.json", "prior_run_id": "fixture-prior"}
            with patch.object(p, "_load_tables", return_value=(tables, {})), patch.object(p, "build_cases", return_value=cases):
                before = audit_completed_pilot(root, renamed, config)
                after = audit_completed_pilot(root, Path(restored["restored_directory"]), config)
            self.assertEqual(before, after)
            with self.assertRaises(FileExistsError):
                restore_audit(output, root / "other-drive/runs")
            with self.assertRaises(FileExistsError):
                package_audit(renamed, output)
            selected = next((renamed / "attempts").glob("*/train_predictions.npy"))
            selected.write_bytes(b"corrupt input")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                package_audit(renamed, root / "invalid.zip")
            self.assertFalse((root / "invalid.zip").exists())

    def test_missing_manifest_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "original Drive"):
                package_audit(Path(temporary) / "absent", Path(temporary) / "audit.zip")

    def test_transfer_notebook_compiles_without_ml_dependencies_or_launch(self):
        notebook = json.loads((ROOT / "notebooks/phase4_transfer_audit.ipynb").read_text())
        sources = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, "transfer_cell", "exec")
                sources.append(source)
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        combined = "\n".join(sources)
        self.assertIn('"--run-directory"', combined)
        self.assertIn("files.download", combined)
        for forbidden in ("torch", "run_scaling.py", "--launch", "pip install"):
            self.assertNotIn(forbidden, combined)

    def test_restore_rejects_corrupt_or_escaping_members_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, contents in (("../outside", b"x"), ("manifest.json", b"corrupt")):
                archive_path = root / ("traversal.zip" if ".." in relative else "corrupt.zip")
                metadata = {"format_version": 1, "run_id": "run", "purpose": "private_read_only_audit_inputs_not_full_model_backup", "files_sha256": {relative: "0" * 64}}
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("run/audit_transfer.json", json.dumps(metadata))
                    archive.writestr(f"run/{relative}", contents)
                with self.assertRaisesRegex(ValueError, "path escapes|checksum differs"):
                    restore_audit(archive_path, root / "destination")
                self.assertFalse((root / "destination").exists())


if __name__ == "__main__":
    unittest.main()
