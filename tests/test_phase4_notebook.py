"""Offline checks for GPU-lock isolation and the thin persistent Colab entry point."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Phase4NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / "notebooks/phase4_pilot.ipynb").read_text(encoding="utf-8"))
        cls.cells = ["".join(cell["source"]) for cell in cls.notebook["cells"] if cell["cell_type"] == "code"]

    def test_cells_compile_and_require_explicit_source_and_launch(self):
        for index, source in enumerate(self.cells):
            compile(source, f"phase4_pilot_cell_{index}", "exec")
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        settings = {}
        with redirect_stdout(io.StringIO()):
            exec(self.cells[0], settings)
        self.assertRegex(settings["PROJECT_SHA"], r"^[0-9a-f]{40}$")
        invalid = self.cells[0].replace(settings["PROJECT_SHA"], "INVALID_PROJECT_SHA", 1)
        with self.assertRaisesRegex(ValueError, "40-character"):
            exec(invalid, {})
        # The shipped launch cell must fail before any training subprocess.
        with self.assertRaisesRegex(RuntimeError, "Set LAUNCH_PILOT = True"):
            exec(self.cells[-1], {})
        source = "\n".join(self.cells)
        self.assertIn('"--detach", PROJECT_SHA', source)
        self.assertIn('"rev-parse", "HEAD"', source)
        self.assertIn('"status", "--porcelain"', source)
        self.assertIn('"--project", "environments/gpu", "--locked"', source)
        self.assertIn('"scripts/run_pilot.py"', source)
        self.assertNotIn("--allow-dirty", source)

    def test_bootstrap_log_flushes_and_preserves_subprocess_failure(self):
        # Execute only the console helper, not Colab mounts/download/setup.
        helper = next(node for source in self.cells for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "run")

        class FailingProcess:
            stdout = io.StringIO("partial training output\n")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def wait(self):
                return 7

        with tempfile.TemporaryDirectory(prefix="refutab-notebook-log-") as directory:
            log = Path(directory) / "persistent.log"
            namespace = {
                "BOOTSTRAP_LOG": log,
                "shlex": __import__("shlex"),
                "subprocess": SimpleNamespace(Popen=lambda *args, **kwargs: FailingProcess(), PIPE=-1, STDOUT=-2),
            }
            exec(compile(ast.Module(body=[helper], type_ignores=[]), "notebook_helper", "exec"), namespace)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "exit code 7"):
                namespace["run"](["fixture", "--argument"])
            saved = log.read_text(encoding="utf-8")
            self.assertIn("$ fixture --argument\n", saved)
            self.assertIn("partial training output\n", saved)
            self.assertIn("exit_code=7\n", saved)
        source = "\n".join(self.cells)
        self.assertLess(source.index('drive.mount("/content/drive")'), source.index('"uv==0.11.24"'))
        self.assertIn('BOOTSTRAP_LOG.open("x"', source)
        self.assertIn('"--output-root", str(RUNS_ROOT)', source)

    def test_gpu_preflight_rejects_cpu_before_asset_preparation(self):
        value = next(node.value for source in self.cells for node in ast.parse(source).body
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "GPU_PREFLIGHT" for target in node.targets))
        preflight = ast.literal_eval(value)
        compile(preflight, "gpu_preflight", "exec")
        torch = SimpleNamespace(__version__="2.7.1+cu126", version=SimpleNamespace(cuda="12.6"),
                                cuda=SimpleNamespace(is_available=lambda: False))
        torchvision = SimpleNamespace(__version__="0.22.1+cu126")
        with patch.dict(sys.modules, {"torch": torch, "torchvision": torchvision, "psutil": SimpleNamespace()}), patch.object(sys, "platform", "linux"), patch("platform.machine", return_value="x86_64"):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                exec(preflight, {})
        joined = "\n".join(self.cells)
        self.assertLess(joined.index('"-c", GPU_PREFLIGHT'), joined.index('"scripts/prepare_phase4.py"'))

    def test_launch_builds_distinct_new_and_resume_commands(self):
        source = self.cells[-1].replace("LAUNCH_PILOT = False", "LAUNCH_PILOT = True", 1)
        with tempfile.TemporaryDirectory(prefix="refutab-notebook-resume-") as directory:
            runs = Path(directory) / "runs"
            previous = runs / "prior-pilot"
            previous.mkdir(parents=True)
            marker = previous / "preserve.txt"
            marker.write_text("existing attempt", encoding="utf-8")
            calls = []

            def run(command, *, cwd=None):
                calls.append(command)
                return "a" * 40 if command == ["git", "rev-parse", "HEAD"] else ""

            namespace = {"Path": Path, "PROJECT_SHA": "a" * 40, "PROJECT_DIR": Path(directory) / "source",
                         "RUNS_ROOT": runs, "RUN_ID": "new-pilot", "UV_PYTHON": ["uv", "run", "python"],
                         "BOOTSTRAP_LOG": Path(directory) / "bootstrap.log", "HARDWARE_RECORD": Path(directory) / "hardware.json",
                         "run": run, "RESUME_RUN": None}
            with redirect_stdout(io.StringIO()):
                exec(source, namespace)
            self.assertEqual(calls[-1][-4:], ["--run-id", "new-pilot", "--output-root", str(runs)])
            calls.clear()
            namespace["RESUME_RUN"] = str(previous)
            with redirect_stdout(io.StringIO()):
                exec(source, namespace)
            command = calls[-1]
            self.assertEqual(command[-2:], ["--resume-run", str(previous.resolve())])
            self.assertNotIn("--run-id", command)
            self.assertNotIn("--output-root", command)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing attempt")
            namespace["RESUME_RUN"] = directory
            calls.clear()
            with self.assertRaisesRegex(ValueError, "existing private run directory"):
                exec(source, namespace)
            self.assertFalse(any("scripts/run_pilot.py" in command for command in calls))

    def test_gpu_lock_preserves_common_cpu_versions_and_pins_linux_wheels(self):
        cpu = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        gpu = tomllib.loads((ROOT / "environments/gpu/uv.lock").read_text(encoding="utf-8"))
        project = tomllib.loads((ROOT / "environments/gpu/pyproject.toml").read_text(encoding="utf-8"))
        a = {package["name"]: package for package in cpu["package"]}
        b = {package["name"]: package for package in gpu["package"]}
        differences = {name: (a[name]["version"], b[name]["version"])
                       for name in a.keys() & b.keys() if a[name]["version"] != b[name]["version"]}
        self.assertEqual(differences, {"torch": ("2.7.1+cpu", "2.7.1+cu126"), "torchvision": ("0.22.1+cpu", "0.22.1+cu126")})
        self.assertEqual(project["project"]["requires-python"], ">=3.11,<3.12")
        self.assertEqual(gpu["supported-markers"], ["platform_machine == 'x86_64' and sys_platform == 'linux'"])
        for name, version in (("torch", "2.7.1+cu126"), ("torchvision", "0.22.1+cu126")):
            self.assertEqual(b[name]["version"], version)
            self.assertEqual(b[name]["source"]["registry"], "https://download.pytorch.org/whl/cu126")
            candidates = [wheel for wheel in b[name]["wheels"] if "cp311-cp311-manylinux_2_28_x86_64.whl" in wheel["url"]]
            self.assertEqual(len(candidates), 1)
            self.assertRegex(candidates[0]["hash"], r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
