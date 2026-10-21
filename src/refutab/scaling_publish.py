"""Whitelist scalar scaling results and publish them to a per-run GitHub branch.

Raw manifests, IDs, predictions, checkpoints and logs never enter the export.
The token is supplied in memory and never written to disk or command arguments.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def export_results(directory: Path, run_id: str, source_sha: str) -> dict[str, str]:
    """Produce only public scalar fields, including partial/failure coverage."""
    summary = {"run_id": run_id, "source_commit": source_sha, "status": "preparing",
               "planned_fits": 72, "completed_fits": 0, "completed_assessments": 0,
               "old_outer_test_predictions": 0, "prior_pilot_audit": "deferred_separate_review", "stages": {}}
    rows = []
    path = directory / "manifest.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["run_id"] != run_id or state["source"]["commit"] != source_sha:
            raise ValueError("Export source/run identity changed")
        summary.update(status=state["status"], planned_fits=len(state["expected_jobs"]),
                       completed_fits=sum(s["status"] == "completed" for s in state["stages"].values()),
                       completed_assessments=len(state["assessments"]),
                       dataset_fingerprints=state["binding"]["dataset_fingerprints"],
                       split_fingerprints=state["binding"]["split_fingerprints"],
                       lock_sha256=state["binding"]["lock_sha256"],
                       config_sha256=state["binding"]["plan_sha256"],
                       cost=state["cost"], environment=state["binding"]["environment"],
                       hardware=state.get("hardware"),
                       recipe={k: state["config"][k] for k in ("encoder", "pipeline", "teacher", "tar_execution", "sizes", "outer_folds")})
        if state.get("failure"):
            summary["failure_type"] = state["failure"]["type"]
        for job in state["expected_jobs"]:
            entry = state["stages"].get(job)
            if not entry:
                continue
            case, method = job.rsplit("__", 1)
            prefix, size = case.rsplit("__size", 1)
            dataset, fold = prefix.rsplit("__fold", 1)
            public_stage = {"status": entry["status"], "attempt_index": entry["attempt_index"], "runtime": entry.get("runtime")}
            if entry.get("failure"):
                public_stage["failure_type"] = entry["failure"]["type"]
            # Optimization and byte counts are safe scalars in the attempt record.
            attempt = (directory / entry["attempt"]).resolve()
            if not attempt.is_relative_to((directory / "attempts").resolve()):
                raise ValueError("Export attempt path escapes run")
            attempt_path = attempt / "manifest.json"
            if attempt_path.exists():
                record = json.loads(attempt_path.read_text(encoding="utf-8"))
                public_stage["optimization"] = record.get("optimization")
                public_stage["artifact_bytes"] = record.get("artifact_bytes")
            scores = dict(entry.get("scores") or {})
            if job in state["assessments"]:
                scores["assessment_D"] = state["assessments"][job]
            public_stage["scores"] = {}
            for role, score in scores.items():
                metrics = {k: v for k, v in score["metrics"].items() if type(v) in (int, float, bool)}
                public_stage["scores"][role] = {"rows": score["rows"], "metrics": metrics}
                for metric, value in metrics.items():
                    rows.append({"dataset": dataset, "fold": int(fold), "size": size, "method": method,
                                 "partition": role, "rows": score["rows"], "metric": metric, "value": value})
            summary["stages"][job] = public_stage
        # The existing summary's aggregation has only known scalar comparisons.
        summary_path = directory / "summary.json"
        if summary_path.exists():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved.get("binding_sha256") == state["binding_sha256"]:
                summary["comparisons"] = saved.get("datasets", {})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("dataset", "fold", "size", "method", "partition", "rows", "metric", "value"))
    writer.writeheader()
    writer.writerows(rows)
    return {"summary.json": _json(summary), "scores.csv": output.getvalue()}


class GitHubResults:
    """Serial Contents API writes to a dedicated branch, without force pushes."""

    def __init__(self, token: str, repository: str, run_id: str, source_sha: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid GitHub repository")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not re.fullmatch(r"[0-9a-f]{40}", source_sha):
            raise ValueError("Invalid run/source identity")
        self._token, self.repository, self.run_id = token, repository, run_id
        self.source_sha, self.branch = source_sha, f"results/{run_id}"
        self.url = f"https://github.com/{repository}/tree/{self.branch}/results/scaling_runs/{run_id}"

    def _request(self, method, path, payload=None, *, missing_ok=False):
        request = Request(f"https://api.github.com/repos/{self.repository}/{path}",
                          data=None if payload is None else json.dumps(payload).encode(), method=method,
                          headers={"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json",
                                   "X-GitHub-Api-Version": "2026-03-10", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 404 and missing_ok:
                return None
            # Do not echo server payloads, request headers or credentials.
            raise RuntimeError(f"GitHub results upload failed (HTTP {error.code}); check token Contents write permission and connection") from None

    def publish(self, files: dict[str, str]):
        if set(files) != {"summary.json", "scores.csv"}:
            raise ValueError("Only the two reviewed scalar exports may be uploaded")
        branch = self._request("GET", "git/ref/heads/" + self.branch, missing_ok=True)
        if branch is None:
            self._request("POST", "git/refs", {"ref": "refs/heads/" + self.branch, "sha": self.source_sha})
        for name, content in files.items():
            path = f"results/scaling_runs/{self.run_id}/{name}"
            encoded_path = quote(path, safe="/")
            previous = self._request("GET", f"contents/{encoded_path}?ref={quote(self.branch, safe='')}", missing_ok=True)
            data = content.encode("utf-8")
            git_blob_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if previous and previous["sha"] == git_blob_sha:
                continue
            payload = {"message": f"results: save {self.run_id} {name}", "branch": self.branch,
                       "content": base64.b64encode(data).decode("ascii")}
            if previous:
                payload["sha"] = previous["sha"]
            written = self._request("PUT", f"contents/{encoded_path}", payload)
            if written["content"]["sha"] != git_blob_sha:
                raise RuntimeError("GitHub upload readback hash differs")
        return self.url
