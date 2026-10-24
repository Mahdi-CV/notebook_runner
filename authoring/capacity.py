"""Which target has room for a job, and how busy each target is.

Each runner.yaml target has a number of slots:
  k8s     a GPU budget: `max_gpus` if set, else `gpus` (the GPUs this runner
          may hold at once in that pool); a job takes the GPUs it declares
  radeon  one job: the API allows one instance per user_name. List more
          radeon targets, each with its own user_name, for more parallel runs
  ssh     `max_concurrent` jobs, default 1: runs share the host's workspace
          directory and Docker daemon

A job goes to the first target, in config order, that fits its requirements
and has room. Availability is what the web UI shows: the slots in use, the
jobs on each target and a live check of the target's API.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import store

sys.path.insert(0, str(store.REPO_ROOT / "tools"))
import routing  # noqa: E402

AVAILABILITY_FILE = "availability.json"


def slots(target: dict) -> tuple[str, int]:
    """(unit, total) for a target: unit is "gpus" or "jobs"."""
    kind = target.get("type", "ssh")
    if kind == "k8s":
        return "gpus", int(target.get("max_gpus") or target.get("gpus") or 1)
    if kind == "radeon":
        return "jobs", 1
    return "jobs", int(target.get("max_concurrent") or 1)


def usage(targets: dict, running: list[dict]) -> dict[str, int]:
    """Slots each target has in use by running jobs."""
    used = {name: 0 for name in targets}
    for job in running:
        name = job.get("target")
        if name in used:
            unit, _ = slots(targets[name])
            used[name] += int(job.get("gpus") or 1) if unit == "gpus" else 1
    return used


def pick(req: "routing.Requirements", targets: dict, used: dict[str, int],
         only: str | None = None, allow_ssh: bool = True,
         held: dict[str, str] | None = None) -> tuple[str | None, str]:
    """(target, "") for the first fitting target with room; (None, reason)
    when the fitting targets are all busy ("busy") or none fits at all.
    Author jobs pass allow_ssh=False: an SSH target gives the notebook the
    host's Docker daemon and needs AGENT_SECRET in the job. `held`: targets a
    known infrastructure issue holds a nightly notebook off (routing.holds_for)."""
    candidates = {only: targets[only]} if only in targets else ({} if only else targets)
    if not allow_ssh:
        candidates = {n: t for n, t in candidates.items() if t.get("type", "ssh") != "ssh"}
    fitting, reasons, held_off = [], [], []
    for name, target in candidates.items():
        missing = routing.unmet(req, target)
        if missing:
            reasons.append(f"{name} lacks {'; '.join(missing)}")
        elif name in (held or {}):
            held_off.append(name)
        else:
            fitting.append(name)
    if not fitting and held_off:
        return None, (f"held: known infrastructure issue on {', '.join(held_off)}: "
                      f"{held[held_off[0]]}")
    if not fitting:
        return None, "no target fits: " + (" | ".join(reasons) or "no targets configured")
    for name in fitting:
        unit, total = slots(targets[name])
        need = req.gpus if unit == "gpus" else 1
        if need > total:
            continue
        if used.get(name, 0) + need <= total:
            return name, ""
    if all((req.gpus if slots(targets[n])[0] == "gpus" else 1) > slots(targets[n])[1]
           for n in fitting):
        return None, f"needs {req.gpus} GPUs, more than any fitting target may use at once"
    return None, "busy"


def probe(name: str, target: dict) -> str:
    """A cheap live check of the target's API: "ok" or why it failed."""
    kind = target.get("type", "ssh")
    try:
        if kind == "k8s":
            import k8s_pod
            args = [*k8s_pod.Kubectl(target).base[1:], "get", "pods", "-l",
                    f"{k8s_pod.APP_LABEL}={k8s_pod.APP_NAME}", "-o", "json"]
            proc = kubectl(args)
            if proc.returncode != 0:
                return "unreachable: " + (proc.stderr.strip().splitlines() or ["kubectl failed"])[-1]
            pods = json.loads(proc.stdout).get("items", [])
            pending = sum(p.get("status", {}).get("phase") == "Pending" for p in pods)
            return "ok" + (f" ({pending} pod{'' if pending == 1 else 's'} waiting for GPUs)" if pending else "")
        if kind == "radeon":
            import radeon_pod
            st = radeon_pod.Service(target).status().get("status")
            return "ok" if st in (None, "not_found") else f"ok (instance {st})"
        return "ok"
    except Exception as exc:  # noqa: BLE001 — a probe never stops the scheduler
        return f"unreachable: {exc}"


def kubectl(args: list[str]) -> subprocess.CompletedProcess:
    """kubectl on this host, or the runner image's when the host has none."""
    if shutil.which("kubectl"):
        return subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=30)
    config = os.environ.get("KUBECONFIG") or os.environ.get("NBR_KUBECONFIG")
    if not config or not shutil.which("docker"):
        return subprocess.CompletedProcess(args, 1, "", "kubectl is not installed on this host")
    image = os.environ.get("NOTEBOOK_RUNNER_IMAGE") or "notebook-runner:latest"
    return subprocess.run(
        ["docker", "run", "--rm", "--network", "host", "-v",
         f"{os.path.expanduser(config)}:/tmp/kubeconfig:ro", "-e", "KUBECONFIG=/tmp/kubeconfig",
         "-u", "0", "--entrypoint", "kubectl", image, *args],
        capture_output=True, text=True, timeout=60)


def job_notebook(job: dict) -> Path:
    """Where a job's notebook is once its prepare phase has fetched it."""
    if job["kind"] == "regression":
        return store.NOTEBOOKS_DIR / job["notebook"]
    return (store.job_dir(job["id"]) / "repo" / "docs" / "notebooks"
            / (job.get("category") or "") / job["notebook"])


def fitting(job: dict, targets: dict) -> tuple[list[str], int]:
    """(targets a waiting job could run on, GPUs it needs), as the scheduler
    would pick them. A notebook that cannot be read counts as fitting every
    target its kind may use: this only feeds the estimates."""
    allowed = {n: t for n, t in targets.items()
               if job["kind"] == "regression" or t.get("type", "ssh") != "ssh"}
    if job.get("only_target"):
        allowed = {n: t for n, t in allowed.items() if n == job["only_target"]}
    try:
        req = requirements(job_notebook(job))
    except (OSError, ValueError, TypeError, AttributeError):
        return list(allowed), int(job.get("gpus") or 1)
    known = holds(job)
    return [n for n, t in allowed.items() if not routing.unmet(req, t) and n not in known], req.gpus


def minutes_between(start: str | None, end: str) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        return max((datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds() / 60, 0.0)
    except (TypeError, ValueError):
        return 0.0


def estimate_starts(targets: dict, running: list[dict], waiting: list[dict],
                    estimates: dict[str, float], now: str) -> dict[str, float | None]:
    """Minutes until each waiting job is likely to start, by replaying the
    queue in order. A running job ends after its kind's typical duration (one
    already past it is assumed a quarter of that from done); a waiting job
    takes the earliest room on a target it fits, never before a job ahead of
    it there (the scheduler holds a busy target for the older job). None when
    no job has finished yet to time, or no target fits."""
    if not estimates:
        return {j["id"]: None for j in waiting}

    def typical(kind: str) -> float:
        return estimates.get(kind) or max(estimates.values())

    # Per target: when its busy slots free up (minutes from now, how many),
    # and the latest start given out on it so far.
    frees: dict[str, list[tuple[float, int]]] = {n: [] for n in targets}
    floor = {n: 0.0 for n in targets}
    for job in running:
        if job.get("target") in frees:
            unit, _ = slots(targets[job["target"]])
            full = typical(job["kind"])
            left = max(full - minutes_between(job.get("started_at"), now), full / 4)
            frees[job["target"]].append((left, int(job.get("gpus") or 1) if unit == "gpus" else 1))
    starts: dict[str, float | None] = {}
    for job in waiting:
        names, gpus = fitting(job, targets)
        best = None
        for name in names:
            unit, total = slots(targets[name])
            need = gpus if unit == "gpus" else 1
            if need > total:
                continue
            for at in sorted({floor[name], *(t for t, _ in frees[name] if t > floor[name])}):
                if total - sum(n for t, n in frees[name] if t > at) >= need:
                    if best is None or at < best[0]:
                        best = (at, name, need)
                    break
        starts[job["id"]] = round(best[0], 1) if best else None
        if best:
            at, name, need = best
            frees[name].append((at + typical(job["kind"]), need))
            floor[name] = at
    return starts


def where(target: dict) -> str:
    """Where a target's runs happen, as the dashboard's Routing page used to say."""
    if target.get("type") == "k8s":
        return f"{target.get('context') or 'cluster'} / {target.get('namespace', 'notebook-runner')}"
    if target.get("type") == "radeon":
        return f"{target.get('user_name')} · pool {target.get('pod_type')}"
    return f"{target.get('user')}@{target.get('host')}"


def snapshot(targets: dict, jobs: list[dict], probes: dict[str, str],
             estimates: dict[str, float]) -> dict:
    """What the web UI shows on the Runners page."""
    now = store.now()
    running = [j for j in jobs if j["status"] == "running" and j.get("target")]
    used = usage(targets, running)
    rows = []
    for name, target in targets.items():
        unit, total = slots(target)
        rows.append({
            "name": name, "type": target.get("type", "ssh"),
            "hardware": target.get("hardware"), "gpus_per_run": target.get("gpus"),
            "gpu_memory_gb": target.get("gpu_memory_gb"), "where": where(target),
            "provides": (target.get("capabilities") or [])
                        + (["docker"] if target.get("type") in (None, "ssh") else []),
            "unit": unit, "total": total, "used": used.get(name, 0),
            "jobs": [{"id": j["id"], "kind": j["kind"], "notebook": j["notebook"],
                      "author": j.get("author"), "started_at": j.get("started_at")}
                     for j in running if j["target"] == name],
            "probe": probes.get(name, "not checked yet"),
        })
    # The GPU queue in the scheduler's order (store.active: author jobs first).
    # A job still in its prepare phase does not need a GPU yet.
    waiting = [j for j in jobs if (j["status"] == "queued" and j.get("phase") == "gpu")
               or (j["status"] == "running" and not j.get("target")
                   and j["stage"] == "Waiting for a GPU")]
    starts = estimate_starts(targets, running, waiting, estimates, now)
    nightly = [j for j in jobs if j["kind"] == "regression"]
    return {
        "updated_at": now,
        "targets": rows,
        "queue": {
            "authors_waiting": sum(j["kind"] != "regression" for j in waiting),
            "regression_waiting": sum(j["kind"] == "regression" for j in waiting),
            "running": len(running),
        },
        "waiting": [{"id": j["id"], "kind": j["kind"], "notebook": j["notebook"],
                     "author": j.get("author"), "gpus": j.get("gpus") or 1,
                     "queued_at": j.get("created_at"), "start_in": starts[j["id"]]}
                    for j in waiting],
        "nightly": {"run_id": nightly[0]["run_id"], "remaining": len(nightly)} if nightly else None,
        "estimates": estimates,
    }


def write_snapshot(data: dict) -> None:
    path = store.JOBS_DIR / AVAILABILITY_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def read_snapshot() -> dict | None:
    path = store.JOBS_DIR / AVAILABILITY_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_targets() -> dict:
    return routing.load_config()["targets"]


def holds(job: dict) -> dict[str, str]:
    """The targets runner.yaml's known_issues hold a nightly notebook off."""
    return routing.holds_for(job["notebook"], routing.load_config()) if job["kind"] == "regression" else {}


def requirements(nb_path: Path) -> "routing.Requirements":
    return routing.requirements(nb_path)
