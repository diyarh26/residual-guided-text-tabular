"""Exercise source bootstrap from a fresh clone without network or model weights."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_phase1.py"
SPEC = importlib.util.spec_from_file_location("prepare_phase1", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class SourceBootstrapTests(unittest.TestCase):
    def test_fresh_clone_and_refusal_to_overwrite_local_changes(self):
        with tempfile.TemporaryDirectory(prefix="refutab-bootstrap-test-") as directory:
            root = Path(directory)
            origin = root / "origin"
            origin.mkdir()
            subprocess.run(["git", "init", "--quiet", str(origin)], check=True)
            (origin / "source.py").write_text("version = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
            subprocess.run([
                "git", "-C", str(origin), "-c", "user.name=Bootstrap Test",
                "-c", "user.email=bootstrap-test@example.invalid", "commit", "--quiet", "-m", "fixture",
            ], check=True)
            commit = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
            project = root / "project"
            (project / "third_party").mkdir(parents=True)
            (project / "third_party/multabench.lock.json").write_text(json.dumps({"repository": str(origin), "commit": commit}), encoding="utf-8")
            with patch.object(prepare, "ROOT", project):
                prepare.prepare_upstream()
                checkout = project / "third_party/MulTaBench"
                self.assertEqual((checkout / "source.py").read_text(encoding="utf-8"), "version = 1\n")
                self.assertEqual(prepare.git("-C", str(checkout), "rev-parse", "HEAD"), commit)
                (checkout / "source.py").write_text("user_change = True\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "local changes"):
                    prepare.prepare_upstream()
                self.assertEqual((checkout / "source.py").read_text(encoding="utf-8"), "user_change = True\n")


if __name__ == "__main__":
    unittest.main()
