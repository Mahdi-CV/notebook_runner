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
         only: str | None = None, allow_ssh: bool = True) -> tuple[str | None, str]:
    """(target, "") for the first fitting target with room; (None, reason)
    when the fitting targets are all busy ("busy") or none fits at all.
    Author jobs pass allow_ssh=False: an SSH target gives the notebook the
    host's Docker daemon and needs AGENT_SECRET in the job."""
    candidates = {only: targets[only]} if only in targets else ({} if only else targets)
    if not allow_ssh:
        candidates = {n: t for n, t in candidates.items() if t.get("type", "ssh") != "ssh"}
    fitting, reasons = [], []
    for name, target in candidates.items():
        missing = routing.unmet(req, target)
        if missing:
            reasons.append(f"{name} lacks {'; '.join(missing)}")
        else:
            fitting.append(name)
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


def snapshot(targets: dict, jobs: list[dict], probes: dict[str, str],
             estimates: dict[str, float]) -> dict:
    """What the web UI shows on the Runners page."""
    running = [j for j in jobs if j["status"] == "running" and j.get("target")]
    used = usage(targets, running)
    rows = []
    for name, target in targets.items():
        unit, total = slots(target)
        rows.append({
            "name": name, "type": target.get("type", "ssh"),
            "hardware": target.get("hardware"), "gpus_per_run": target.get("gpus"),
            "unit": unit, "total": total, "used": used.get(name, 0),
            "jobs": [{"id": j["id"], "kind": j["kind"], "notebook": j["notebook"],
                      "author": j.get("author"), "started_at": j.get("started_at")}
                     for j in running if j["target"] == name],
            "probe": probes.get(name, "not checked yet"),
        })
    waiting = [j for j in jobs if j["status"] == "queued"
               or (j["status"] == "running" and not j.get("target")
                   and j["stage"] == "Waiting for a GPU")]
    nightly = [j for j in jobs if j["kind"] == "regression"]
    return {
        "updated_at": store.now(),
        "targets": rows,
        "queue": {
            "authors_waiting": sum(j["kind"] != "regression" for j in waiting),
            "regression_waiting": sum(j["kind"] == "regression" for j in waiting),
            "running": len(running),
        },
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


def requirements(nb_path: Path) -> "routing.Requirements":
    return routing.requirements(nb_path)
