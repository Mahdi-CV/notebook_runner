"""The scheduler: starts each job phase as soon as it has a slot.

Every few seconds the worker
  1. follows the running phases: copies their stage from state.json into
     jobs.db, and when a phase ends records its outcome (a validate job whose
     prepare phase passed becomes a queued GPU job);
  2. starts queued phases in priority order: prepare phases (static checks,
     annotate) up to NBR_PREPARE_SLOTS at once, GPU phases on the first fitting
     target with a free slot (authoring.capacity). A target a waiting job of
     higher priority fits is held for it, so a stream of small nightly jobs
     cannot keep an author's larger job waiting forever;
  3. merges each finished regression job's database into results/runs.db,
     so that database keeps a single writer while jobs run in parallel;
  4. writes jobs/availability.json for the web UI's Runners page.

Phases run in their own containers (docker/run-job.sh), which survive a
restart of the worker; NBR_JOB_LAUNCHER=local runs them as subprocesses
instead, for development.

Usage:
  python3 -m authoring.worker            # run until stopped
  python3 -m authoring.worker --once     # one scheduling round (tests)
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import capacity, pipeline, store

sys.path.insert(0, str(store.REPO_ROOT / "tools"))
import results_db  # noqa: E402

TICK_SECONDS = 5
SNAPSHOT_SECONDS = 15
PROBE_SECONDS = 60
PREPARE_SLOTS = int(os.environ.get("NBR_PREPARE_SLOTS") or 3)
# Finished jobs' directories are deleted after this many days (the jobs.db row stays).
RETENTION_DAYS = {"regression": 7, "validate": int(os.environ.get("NBR_JOB_RETENTION_DAYS") or 30),
                  "annotate": int(os.environ.get("NBR_JOB_RETENTION_DAYS") or 30)}
PRUNE_SECONDS = 3600
LAUNCHER = os.environ.get("NBR_JOB_LAUNCHER") or "docker"
RUN_JOB = store.REPO_ROOT / "docker" / "run-job.sh"
# Author jobs reach their credentials only through the broker (authoring/broker.py).
KEYS_FILE = store.JOBS_DIR / ".broker" / "keys.json"
BROKER_CONTAINER = "nbr-broker"  # docker/run-broker.sh
AUTHOR_KINDS = ("validate", "annotate")


# ── launching and following phases ───────────────────────────────────────────

class Local:
    """Phases as child processes (development)."""

    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}

    def start(self, job_id: str, phase: str, target: str | None, key: str | None = None) -> None:
        cmd = [sys.executable, "-m", "authoring.pipeline", phase, job_id,
               *(["--target", target] if target else [])]
        out = open(store.job_dir(job_id) / f"{phase}.stderr", "w")
        env = {**os.environ, **({"NBR_JOB_KEY": key} if key else {})}
        self.procs[job_id] = subprocess.Popen(cmd, cwd=store.REPO_ROOT, stdout=out, env=env,
                                              stderr=subprocess.STDOUT, start_new_session=True)

    def running(self, job_id: str) -> bool:
        proc = self.procs.get(job_id)
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        del self.procs[job_id]
        return False

    def stop(self, job_id: str) -> None:
        proc = self.procs.get(job_id)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)


class Docker:
    """Phases as detached containers named nbr-job-<id>, started by run-job.sh."""

    def start(self, job_id: str, phase: str, target: str | None, key: str | None = None) -> None:
        # The key goes through the environment, never argv (visible in ps).
        env = {**os.environ, **({"NBR_JOB_KEY": key} if key else {})}
        proc = subprocess.run([str(RUN_JOB), phase, job_id, *([target] if target else [])],
                              capture_output=True, text=True, timeout=120, env=env)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[-500:])

    def running(self, job_id: str) -> bool:
        proc = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                               f"nbr-job-{job_id}"], capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return False
        if proc.stdout.strip() == "true":
            return True
        subprocess.run(["docker", "rm", f"nbr-job-{job_id}"], capture_output=True, timeout=60)
        return False

    def stop(self, job_id: str) -> None:
        subprocess.run(["docker", "stop", "-t", "30", f"nbr-job-{job_id}"],
                       capture_output=True, timeout=90)


def read_state(job_id: str) -> dict:
    try:
        return json.loads((store.job_dir(job_id) / "state.json").read_text())
    except (OSError, ValueError):
        return {}


def log_tail(job_id: str, lines: int = 15) -> str:
    try:
        return "\n".join((store.job_dir(job_id) / "log.txt").read_text().splitlines()[-lines:])
    except OSError:
        return ""


# ── the scheduler ────────────────────────────────────────────────────────────

class Worker:
    def __init__(self):
        self.conn = store.connect()
        self.launcher = Local() if LAUNCHER == "local" else Docker()
        self.probes: dict[str, str] = {}
        self.last_snapshot = 0.0
        self.last_probe = 0.0
        self.last_prune = 0.0
        self.keys: dict[str, dict] = {}  # sha256(job key) -> what the broker allows it

    def tick(self) -> None:
        targets = capacity.load_targets()
        for job in store.active(self.conn):
            if job["cancel"]:
                self.cancel(job)
            elif job["status"] == "running":
                self.follow(job)
        self.schedule(targets)
        if time.time() - self.last_snapshot >= SNAPSHOT_SECONDS:
            self.snapshot(targets)
        if time.time() - self.last_prune >= PRUNE_SECONDS:
            self.prune()

    # following ---------------------------------------------------------------

    def follow(self, job: dict) -> None:
        state = read_state(job["id"])
        if state.get("stage") and state["stage"] != job["stage"] and "final" not in state:
            store.update(self.conn, job["id"], stage=state["stage"])
        if self.launcher.running(job["id"]):
            return
        (store.job_dir(job["id"]) / ".kubeconfig").unlink(missing_ok=True)
        self.revoke(job["id"])
        if "final" in state:
            final = state["final"]
            if job["kind"] == "regression":
                self.merge(job)  # before finishing: tools/ci_run.py reads runs.db once it ends
            store.finish(self.conn, job["id"], final["status"], final["stage"], final["summary"])
        elif state.get("next") == "gpu":
            store.update(self.conn, job["id"], status="queued", phase="gpu",
                         stage="Waiting for a GPU", gpus=state.get("gpus") or 1)
        else:
            store.finish(self.conn, job["id"], "error", "The job stopped unexpectedly",
                         {"message": "The job's container exited without a result.\n"
                                     + log_tail(job["id"])})

    def merge(self, job: dict) -> None:
        db = store.job_dir(job["id"]) / "runs.db"
        if not db.exists():
            return
        try:
            conn = results_db.connect()
            n = results_db.merge(conn, db)
            conn.close()
            store.log(job["id"], f"Recorded {n} result{'' if n == 1 else 's'} in the regression database")
        except Exception as exc:  # noqa: BLE001 — the JSON results are on disk regardless
            store.log(job["id"], f"warning: could not record the result in runs.db: {exc}")

    def cancel(self, job: dict) -> None:
        if job["status"] == "running":
            self.launcher.stop(job["id"])
            self.revoke(job["id"])
            if job["phase"] == "gpu" and job["kind"] in AUTHOR_KINDS:
                self.cleanup_pods(job["id"])
        store.finish(self.conn, job["id"], "cancelled", "Cancelled")

    # job keys ----------------------------------------------------------------

    def issue(self, job: dict, target: str | None, gpus: int) -> str:
        """A fresh key for one phase of an author job: the broker accepts it,
        for this job's own pod, target and GPUs, until the phase ends."""
        key = secrets.token_urlsafe(32)
        self.keys[hashlib.sha256(key.encode()).hexdigest()] = {
            "id": job["id"], "kind": job["kind"], "target": target, "gpus": gpus}
        self.save_keys()
        return key

    def revoke(self, job_id: str) -> None:
        before = len(self.keys)
        self.keys = {h: k for h, k in self.keys.items() if k["id"] != job_id}
        if len(self.keys) != before:
            self.save_keys()

    def save_keys(self) -> None:
        KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = KEYS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.keys))
        tmp.chmod(0o644)
        tmp.replace(KEYS_FILE)

    def cleanup_pods(self, job_id: str) -> None:
        """Delete a stopped job's pod (run.py's own cleanup died with its
        container) with the broker's credentials, inside its container."""
        cmd = ["-m", "authoring.broker", "cleanup", job_id]
        cmd = [sys.executable, *cmd] if LAUNCHER == "local" else \
            ["docker", "exec", BROKER_CONTAINER, "python", *cmd]
        try:
            proc = subprocess.run(cmd, cwd=store.REPO_ROOT, capture_output=True, text=True,
                                  timeout=120)
            store.log(job_id, f"Cleanup: {(proc.stdout if proc.returncode == 0 else proc.stderr).strip()[-300:]}")
        except Exception as exc:  # noqa: BLE001 — the pod's lifetime limit is the backstop
            store.log(job_id, f"warning: could not clean up the job's pod: {exc}")

    # scheduling --------------------------------------------------------------

    def schedule(self, targets: dict) -> None:
        jobs = store.active(self.conn)
        running = [j for j in jobs if j["status"] == "running"]
        used = capacity.usage(targets, [j for j in running if j["phase"] == "gpu"])
        preparing = sum(j["phase"] == "prepare" for j in running)
        held: set[str] = set()  # targets kept for a waiting job of higher priority

        for job in (j for j in jobs if j["status"] == "queued"):
            if job["phase"] == "prepare":
                if preparing < PREPARE_SLOTS:
                    self.start(job, "prepare", None)
                    preparing += 1
                continue
            try:
                req = capacity.requirements(self.notebook_path(job))
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                store.finish(self.conn, job["id"], "blocked", "Cannot run: metadata.ci is invalid",
                             {"message": str(exc)})
                continue
            # Could it ever run here, with every slot free?
            ssh_ok = job["kind"] == "regression"
            _, never = capacity.pick(req, targets, {}, job.get("only_target"), allow_ssh=ssh_ok)
            if never:
                store.finish(self.conn, job["id"], "blocked", "Cannot run: no runner fits",
                             {"message": never})
                continue
            free = {n: t for n, t in targets.items() if n not in held}
            target, _ = capacity.pick(req, free, used, job.get("only_target"), allow_ssh=ssh_ok)
            if target:
                self.start(job, "gpu", target, req.gpus)
                unit, _ = capacity.slots(targets[target])
                used[target] = used.get(target, 0) + (req.gpus if unit == "gpus" else 1)
            else:
                # Busy: hold every target this job could use, so later jobs don't take it.
                held |= {n for n, t in targets.items()
                         if not capacity.routing.unmet(req, t)
                         and (ssh_ok or t.get("type", "ssh") != "ssh")
                         and (not job.get("only_target") or n == job["only_target"])}

    def notebook_path(self, job: dict) -> Path:
        if job["kind"] == "regression":
            return store.NOTEBOOKS_DIR / job["notebook"]
        return (store.job_dir(job["id"]) / "repo" / "docs" / "notebooks"
                / job["category"] / job["notebook"])

    def start(self, job: dict, phase: str, target: str | None, gpus: int = 0) -> None:
        fields = {"status": "running", "started_at": job["started_at"] or store.now()}
        if phase == "gpu":
            fields.update(target=target, gpus=gpus, stage=f"Starting on {target}")
        else:
            fields.update(stage="Starting")
        store.update(self.conn, job["id"], **fields)
        (store.job_dir(job["id"]) / "state.json").unlink(missing_ok=True)
        if phase == "prepare" and job["source"] == "pr" \
                and not (store.job_dir(job["id"]) / "input.ipynb").exists():
            try:
                store.update(self.conn, job["id"], stage="Fetching the notebook from GitHub")
                pipeline.download(pipeline.Job(job["id"]))
            except pipeline.Stop as stop:
                store.finish(self.conn, job["id"], stop.status, str(stop).split(". ")[0][:120],
                             {"message": str(stop)})
                return
        key = self.issue(job, target, gpus) if job["kind"] in AUTHOR_KINDS else None
        try:
            self.launcher.start(job["id"], "prepare" if phase == "prepare" else "run", target, key)
            store.log(job["id"], f"Started the {phase} phase" + (f" on {target}" if target else ""))
        except Exception as exc:  # noqa: BLE001 — record it on the job, keep scheduling
            self.revoke(job["id"])
            store.finish(self.conn, job["id"], "error", "The job could not be started",
                         {"message": str(exc)})

    # availability ------------------------------------------------------------

    def snapshot(self, targets: dict) -> None:
        self.last_snapshot = time.time()
        if time.time() - self.last_probe >= PROBE_SECONDS:
            self.last_probe = time.time()
            threading.Thread(target=self.probe, args=(targets,), daemon=True).start()
        estimates = {}
        for kind in ("validate", "annotate", "regression"):
            d = store.recent_durations(self.conn, kind)
            if d:
                estimates[kind] = round(statistics.median(d) / 60, 1)
        capacity.write_snapshot(capacity.snapshot(targets, store.active(self.conn),
                                                  self.probes, estimates))

    def probe(self, targets: dict) -> None:
        for name, target in targets.items():
            self.probes[name] = capacity.probe(name, target)

    def prune(self) -> None:
        """Delete the directories of jobs that finished long ago."""
        self.last_prune = time.time()
        for kind, days in RETENTION_DAYS.items():
            rows = self.conn.execute(
                "SELECT id FROM jobs WHERE kind = ? AND finished_at IS NOT NULL "
                "AND finished_at < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)",
                (kind, f"-{days} days")).fetchall()
            for (job_id,) in rows:
                path = store.job_dir(job_id)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    def recover(self) -> None:
        """After a restart: phases still running in containers are followed
        again (keeping their keys); a local phase died with the old worker."""
        try:
            self.keys = json.loads(KEYS_FILE.read_text())
        except (OSError, ValueError):
            self.keys = {}
        live = {j["id"] for j in store.active(self.conn) if j["status"] == "running"}
        self.keys = {h: k for h, k in self.keys.items() if k["id"] in live}
        self.save_keys()
        for job in store.active(self.conn):
            if job["status"] == "running" and not self.launcher.running(job["id"]):
                self.follow(job)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Schedule authoring and regression jobs.")
    p.add_argument("--once", action="store_true", help="Run one scheduling round and exit.")
    args = p.parse_args(argv)
    worker = Worker()
    worker.recover()
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    while True:
        try:
            worker.tick()
        except Exception as exc:  # noqa: BLE001 — a bad round must not stop the scheduler
            print(f"worker: round failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.once or stopping.wait(TICK_SECONDS):
            return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
