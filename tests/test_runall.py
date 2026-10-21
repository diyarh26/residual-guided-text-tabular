"""Fresh-Drive execution and safe, incremental GitHub publishing contracts."""

import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab import scaling
from refutab.colab_scaling import run_and_publish
from refutab.scaling_publish import GitHubResults, export_results
import test_scaling as fixtures


class RunAllTests(unittest.TestCase):
    def test_fresh_policy_fits_without_old_manifest_and_records_deferral(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = fixtures.ScalingRunnerTests()
            table, cases = fixture.fixture(root)
            config = json.loads((root / "config.json").read_text())
            config["prior_audit_policy"] = "deferred_separate_review"
            (root / "config.json").write_text(json.dumps(config))
            with fixture.patches(table, cases):
                result = scaling.run_scaling(root, Path("config.json"), run_id="fresh", methods=["frozen"], device="cpu")
            self.assertEqual(result["completed_stages"], 3)
            state = json.loads((root / "results/raw/fresh/manifest.json").read_text())
            self.assertEqual(state["binding"]["prior_audit"]["status"], "deferred")
            self.assertEqual(result["outer_test_predictions"], 0)

    def test_standalone_keeps_the_entire_registered_model_and_split_recipe(self):
        original = json.loads((ROOT / "configs/phase4_scaling.json").read_text())
        standalone = json.loads((ROOT / "configs/phase4_scaling_standalone.json").read_text())
        self.assertEqual({k: v for k, v in standalone.items() if k in original}, original)
        self.assertEqual(standalone["prior_audit_policy"], "deferred_separate_review")

    def test_notebook_run_all_starts_training_and_does_not_request_old_artifacts(self):
        notebook = json.loads((ROOT / "notebooks/ReFuTab_Run_All.ipynb").read_text())
        source = ""
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                compile(code, "runall_cell", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                source += code
        for forbidden in ("--prior-run", "PRIOR_RUN", "LAUNCH_SCALING = False", "files.upload"):
            self.assertNotIn(forbidden, source)
        for required in ("--launch", "getpass.getpass", "run_and_publish", "--resume-run", "check_tar_memory.py", "current.json"):
            self.assertIn(required, source)

    def test_scalar_export_excludes_raw_rows_predictions_messages_and_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = fixtures.ScalingRunnerTests()
            table, cases = fixture.fixture(root)
            config = json.loads((root / "config.json").read_text())
            config["prior_audit_policy"] = "deferred_separate_review"
            (root / "config.json").write_text(json.dumps(config))
            with fixture.patches(table, cases):
                scaling.run_scaling(root, Path("config.json"), run_id="export", methods=["frozen"], device="cpu")
            directory = root / "results/raw/export"
            path = directory / "manifest.json"
            state = json.loads(path.read_text())
            secret = "PRIVATE_MARKER_DO_NOT_PUBLISH"
            state["raw_text"] = state["token"] = secret
            state["failure"] = {"type": "FixtureError", "message": secret}
            for entry in state["stages"].values():
                entry["scores"]["train"]["row_ids"] = [secret]
                entry["scores"]["train"]["predictions"] = [secret]
            path.write_text(json.dumps(state))
            exports = export_results(directory, "export", "a" * 40)
            self.assertEqual(set(exports), {"summary.json", "scores.csv"})
            self.assertNotIn(secret, "".join(exports.values()))
            self.assertNotIn("row_ids", "".join(exports.values()))
            self.assertIn("FixtureError", exports["summary.json"])
            self.assertIn("validation", exports["scores.csv"])
            self.assertEqual(json.loads(exports["summary.json"])["completed_fits"], 3)

    def test_contents_api_serial_updates_deduplicates_and_checks_returned_hashes(self):
        publisher = GitHubResults("PRIVATE_TOKEN", "owner/repo", "run1", "a" * 40)
        storage, operations = {}, []
        branch = []

        def api(method, path, payload=None, **kwargs):
            operations.append((method, path, payload))
            if path.startswith("git/ref/"):
                return {"ref": "branch"} if branch else None
            if path == "git/refs":
                branch.append(payload)
                return {}
            key = path.split("?")[0]
            if method == "GET":
                return storage.get(key)
            data = base64.b64decode(payload["content"])
            digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if key in storage:
                self.assertEqual(payload["sha"], storage[key]["sha"])
            storage[key] = {"sha": digest}
            return {"content": {"sha": digest}}

        with patch.object(publisher, "_request", side_effect=api):
            publisher.publish({"summary.json": "first", "scores.csv": "csv"})
            puts = sum(op[0] == "PUT" for op in operations)
            publisher.publish({"summary.json": "first", "scores.csv": "csv"})
            self.assertEqual(sum(op[0] == "PUT" for op in operations), puts)
            publisher.publish({"summary.json": "changed", "scores.csv": "csv"})
            self.assertEqual(sum(op[0] == "PUT" for op in operations), puts + 1)
        self.assertNotIn("PRIVATE_TOKEN", repr(operations))
        with self.assertRaisesRegex(ValueError, "two reviewed"):
            publisher.publish({"checkpoint.pt": "private"})

    def test_supervisor_validates_github_before_child_and_saves_final_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = dict(root=root, directory=root / "run1", run_id="run1", source_sha="a" * 40,
                           repository="owner/repo", token="PRIVATE_TOKEN", console_log=root / "run.log", poll_seconds=.05)
            with patch("refutab.colab_scaling.GitHubResults.publish", side_effect=RuntimeError("denied")), \
                 patch("refutab.colab_scaling.subprocess.Popen", side_effect=AssertionError("started without GitHub write access")), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "denied"):
                run_and_publish([sys.executable, "-c", "print('training')"], **options)
            with patch("refutab.colab_scaling.GitHubResults.publish", return_value="url") as publish, redirect_stdout(io.StringIO()):
                url = run_and_publish([sys.executable, "-c", "print('fixture subprocess')"], **options)
            self.assertIn("results/run1", url)
            self.assertGreaterEqual(publish.call_count, 1)
            self.assertIn("fixture subprocess", (root / "run.log").read_text())
            self.assertNotIn("PRIVATE_TOKEN", (root / "run.log").read_text())
            publication = json.loads((root / "_exports/run1/publication.json").read_text())
            self.assertTrue(publication["github_saved"])

    def test_failed_child_is_published_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("refutab.colab_scaling.GitHubResults.publish", return_value="url"), redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(RuntimeError, "exit code 7"):
                run_and_publish([sys.executable, "-c", "raise SystemExit(7)"], root=root, directory=root / "run1", run_id="run1",
                                source_sha="a" * 40, repository="owner/repo", token="PRIVATE_TOKEN", console_log=root / "console.log", poll_seconds=.03)
            saved = json.loads((root / "_exports/run1/summary.json").read_text())
            self.assertEqual(saved["supervisor_status"], "failed")
            self.assertEqual(saved["completed_fits"], 0)


if __name__ == "__main__":
    unittest.main()
