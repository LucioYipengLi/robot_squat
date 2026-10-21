"""Checkpoint discovery, immutable job manifests and isolated worker scheduling."""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def digest_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def environment_identity() -> dict:
    """Guard editable upstream revisions as well as installed package versions."""
    packages = {}
    sources = {}
    for name in (
        "torch",
        "rsl-rl",
        "tensorboard",
        "isaaclab",
        "isaaclab-rl",
        "isaaclab-assets",
        "isaaclab-tasks",
        "isaaclab-physx",
        "isaaclab-newton",
        "isaacsim",
    ):
        try:
            distribution = importlib.metadata.distribution(name)
            packages[name] = distribution.version
            direct = distribution.read_text("direct_url.json")
            url = json.loads(direct)["url"] if direct else ""
            if url.startswith("file:"):
                path = Path(url2pathname(unquote(urlparse(url).path)))
                root = subprocess.run(
                    ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if root.returncode == 0:
                    source = Path(root.stdout.strip())
                    if str(source) not in sources:
                        revision = (
                            subprocess.run(
                                ["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True,
                                timeout=30,
                                check=True,
                            )
                            .stdout.strip()
                            .decode()
                        )
                        difference = subprocess.run(
                            ["git", "-C", str(source), "diff", "--no-ext-diff", "--binary", "HEAD", "--", "*.py"],
                            capture_output=True,
                            timeout=30,
                            check=True,
                        ).stdout
                        sources[str(source)] = {
                            "revision": revision,
                            "python_changes_sha256": hashlib.sha256(difference).hexdigest(),
                        }
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages, "editable_sources": sources}


def write_json(path: Path, value: object) -> None:
    """Replace one owned artifact atomically so interruptions leave a complete record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_records(patterns: list[str]) -> list[dict]:
    """Expand paths/globs locally on Windows as well as Linux; labels may use LABEL=PATH."""
    records = []
    seen = set()
    for pattern in patterns:
        # An existing literal/glob path may itself contain '=' (common in run
        # names); only interpret LABEL=PATH when the full input is not a path.
        expanded = os.path.expanduser(pattern)
        direct_matches = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else []
        if Path(expanded).is_file() or direct_matches:
            candidate, explicit_label = expanded, ""
        else:
            explicit_label, separator, candidate = pattern.partition("=")
            if not separator:
                candidate, explicit_label = pattern, ""
            candidate = os.path.expanduser(candidate)
        matches = sorted(glob.glob(candidate)) if glob.has_magic(candidate) else [candidate]
        if not matches:
            raise ValueError(f"Checkpoint pattern matched no files: {candidate}")
        if explicit_label and len(matches) != 1:
            raise ValueError("A checkpoint label requires exactly one file")
        for match in matches:
            path = Path(match).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Checkpoint does not exist: {path}")
            if path in seen:
                continue
            seen.add(path)
            label = explicit_label or f"{path.parent.name}_{path.stem}"
            label = re.sub(r"[^\w.-]+", "_", label, flags=re.ASCII).strip("._")
            if not label:
                raise ValueError("Checkpoint label must contain ASCII letters or numbers")
            used_labels = {record["label"].casefold() for record in records}
            if label.casefold() in used_labels:
                base = label + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:8]
                label = base
                suffix = 2
                while label.casefold() in used_labels:
                    label = f"{base}_{suffix}"
                    suffix += 1
            records.append({"label": label, "path": str(path), "sha256": digest_file(path)})
    if not records:
        raise ValueError("Provide at least one checkpoint")
    return records


def source_digest() -> str:
    """Record project Python source, including the task, as part of the execution identity."""
    package = Path(__file__).resolve().parents[2]
    return canonical_digest(
        [(str(path.relative_to(package)), digest_file(path)) for path in sorted(package.rglob("*.py"))]
    )


def build_manifest(config: dict, checkpoints: list[dict], runtime: dict) -> dict:
    from .config import suite_config_digest
    from .modules import conditions

    jobs = []
    for checkpoint in checkpoints:
        for condition in conditions(config):
            for policy_output in config["policy_outputs"]:
                for seed in config["benchmark"]["seeds"]:
                    identity = {
                        "checkpoint": checkpoint,
                        **condition,
                        "policy_output": policy_output,
                        "seed": seed,
                    }
                    jobs.append({**identity, "job_id": canonical_digest(identity)[:20]})
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": suite_config_digest(config),
        "runtime": runtime,
        "source_sha256": source_digest(),
        "environment": environment_identity(),
        "checkpoints": checkpoints,
        "jobs": jobs,
    }
    manifest["identity_sha256"] = canonical_digest(
        {key: value for key, value in manifest.items() if key != "created_utc"}
    )
    return manifest


def validate_resume(manifest: dict, *, require_source=True) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation manifest version")
    expected = canonical_digest(
        {key: value for key, value in manifest.items() if key not in ("created_utc", "identity_sha256")}
    )
    if expected != manifest["identity_sha256"]:
        raise ValueError("Evaluation manifest was modified; start a new output directory")
    if require_source and manifest["source_sha256"] != source_digest():
        raise ValueError("Project source changed since this evaluation; start a new evaluation batch")
    if require_source and manifest["environment"] != environment_identity():
        raise ValueError("Evaluation dependencies or editable upstream sources changed; start a new batch")
    if not require_source:
        return
    for checkpoint in manifest["checkpoints"]:
        for key, digest_key in (
            ("path", "sha256"),
            ("agent_config", "agent_config_sha256"),
            ("env_config", "env_config_sha256"),
        ):
            if checkpoint.get(key) and digest_file(checkpoint[key]) != checkpoint[digest_key]:
                raise ValueError(f"Checkpoint input changed: {checkpoint[key]}")


def result_path(directory: Path, job: dict) -> Path:
    return directory / "cases" / job["job_id"] / "result.json"


def read_result(directory: Path, job: dict, manifest: dict | None = None) -> dict | None:
    path = result_path(directory, job)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in job:
        if value.get(key) != job[key]:
            raise ValueError(f"Result identity differs from manifest: {path}")
    if manifest is not None:
        for key in ("source_sha256", "config_sha256"):
            if value.get("metadata", {}).get(key) != manifest[key]:
                raise ValueError(f"Result provenance differs from manifest: {path}")
        if value.get("metadata", {}).get("environment") != manifest["environment"]:
            raise ValueError(f"Result dependencies differ from manifest: {path}")
    if value.get("status") != "complete":
        raise ValueError(f"Incomplete result at {path}; remove that result file before retrying")
    return value


def execute_jobs(directory: Path, manifest: dict, *, progress=None) -> tuple[list[dict], bool]:
    """Each cell gets a fresh simulator process and an independently resumable result."""
    from .progress import PROGRESS_ID_ENV, EvaluationProgress, finalize_progress, read_progress

    if progress is None:
        with EvaluationProgress(len(manifest["jobs"])) as display:
            results, complete = execute_jobs(directory, manifest, progress=display)
            display.finish(success=complete)
            return results, complete
    results = {
        job["job_id"]: result
        for job in manifest["jobs"]
        if (result := read_result(directory, job, manifest)) is not None
    }
    progress.set_cached(len(results))

    def ordered_results():
        return [results[job["job_id"]] for job in manifest["jobs"] if job["job_id"] in results]

    benchmark = manifest["config"]["benchmark"]
    num_envs = manifest["runtime"]["num_envs"]
    batches = (benchmark["episodes_per_seed"] + num_envs - 1) // num_envs
    max_steps = round(benchmark["episode_length_s"] / benchmark["control_dt"])
    for job in manifest["jobs"]:
        if job["job_id"] not in results:
            cell = result_path(directory, job).parent
            cell.mkdir(parents=True, exist_ok=True)
            attempt_id = uuid.uuid4().hex
            progress.start_job(job, episodes=benchmark["episodes_per_seed"], batches=batches, steps=max_steps)
            started = time.monotonic()

            def finalize(stage, *, cell=cell, job_id=job["job_id"], attempt_id=attempt_id, started=started, **values):
                state = finalize_progress(
                    cell / "progress.json",
                    job_id,
                    attempt_id,
                    stage,
                    time.monotonic() - started,
                    episodes_total=benchmark["episodes_per_seed"],
                    batches_total=batches,
                    batch_steps_total=max_steps,
                    **values,
                )
                progress.update(state)

            command = [
                sys.executable,
                "-m",
                "g1.rl",
                "eval",
                "--worker-manifest",
                str(directory / "manifest.json"),
                "--worker-job",
                job["job_id"],
            ]
            with (cell / "worker.log").open("w", encoding="utf-8") as log:
                process = None
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8", PROGRESS_ID_ENV: attempt_id},
                    )
                    while True:
                        progress.update(read_progress(cell / "progress.json", job["job_id"], attempt_id))
                        try:
                            code = process.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                except KeyboardInterrupt:
                    if process is not None:
                        _stop_worker(process)
                    finalize("interrupted")
                    progress.fail_job(f"续跑：--resume {directory}", interrupted=True)
                    return ordered_results(), False
                except BaseException:
                    if process is not None:
                        _stop_worker(process)
                    finalize("failed")
                    raise
            progress.update(read_progress(cell / "progress.json", job["job_id"], attempt_id))
            if code:
                finalize("failed")
                progress.fail_job(f"退出码 {code}；日志：{cell / 'worker.log'}；续跑：--resume {directory}")
                return ordered_results(), False
            try:
                result = read_result(directory, job, manifest)
                if result is None:
                    raise RuntimeError(f"Worker exited without writing {result_path(directory, job)}")
            except Exception:
                finalize("failed")
                raise
            finalize("finished", episodes_completed=benchmark["episodes_per_seed"], batches_completed=batches)
            results[job["job_id"]] = result
            progress.complete_job()
    return ordered_results(), True


def _stop_worker(process) -> None:
    """Reap an interrupted child so no simulator keeps running after the UI exits."""
    if process.poll() is not None:
        return
    with suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            process.kill()
        process.wait(timeout=5)
