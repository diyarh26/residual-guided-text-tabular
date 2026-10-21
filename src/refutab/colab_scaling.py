"""Notebook supervision: durable console, scalar exports and GitHub checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from .scaling_publish import GitHubResults, export_results


def run_and_publish(command, *, root: Path, directory: Path, run_id: str,
                    source_sha: str, repository: str, token: str,
                    console_log: Path, poll_seconds: float = 60):
    publisher = GitHubResults(token, repository, run_id, source_sha)
    exports = directory.parent / "_exports" / run_id
    exports.mkdir(parents=True, exist_ok=True)
    console_log.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    execution_status = "preparing"

    def save_and_push(*, required=False):
        nonlocal previous
        try:
            files = export_results(directory, run_id, source_sha)
            overview = json.loads(files["summary.json"])
            overview["supervisor_status"] = execution_status
            files["summary.json"] = json.dumps(overview, indent=2, sort_keys=True, allow_nan=False) + "\n"
            for name, content in files.items():
                temporary = exports / (name + ".pending")
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(exports / name)
            if files != previous:
                publisher.publish(files)
                previous = files
                print("GitHub results saved:", publisher.url, flush=True)
            return True
        except Exception as error:
            print(f"GitHub save pending ({type(error).__name__}): {error}. Results remain on Drive.", flush=True)
            if required:
                raise
            return False

    # A real write validates repository access before expensive fitting begins.
    save_and_push(required=True)
    environment = dict(os.environ, MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_VISIBLE_DEVICES="0")
    environment.pop("GITHUB_TOKEN", None)
    environment.pop("GH_TOKEN", None)
    with console_log.open("a", encoding="utf-8", buffering=1) as log:
        log.write("Starting/resuming controlled scaling; credentials are not passed to the training process.\n")
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=os.name == "posix")
        execution_status = "running"

        def stream_output():
            with process.stdout as output:
                for line in output:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()

        reader = threading.Thread(target=stream_output, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    code = process.wait(timeout=poll_seconds)
                    execution_status = "finished" if code == 0 else "failed"
                    break
                except subprocess.TimeoutExpired:
                    save_and_push()
        except BaseException:
            execution_status = "interrupted"
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=30)
            raise
        finally:
            reader.join(timeout=10)
            published = save_and_push()
            (exports / "publication.json").write_text(json.dumps({"github_saved": published, "results_url": publisher.url}, indent=2) + "\n", encoding="utf-8")
    if code:
        raise RuntimeError(f"Training stopped with exit code {code}. Progress is saved; rerun this notebook to resume. Log: {console_log}")
    if not published:
        raise RuntimeError("Training finished, but final GitHub upload failed. Results are on Drive; rerun to retry upload.")
    return publisher.url
