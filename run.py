"""
Notebook regression agent — launcher for the verify and fix agents.

Copies each notebook to the GPU node, runs a headless `claude -p` verify agent
(CLAUDE.md) and, if it fails, a fix agent (CLAUDE_FIX.md). Every outcome is
recorded in results/runs.db; the agents also write results/*.json.

Usage:
  # Single notebook
  python run.py path/to/notebook.ipynb

  # All notebooks in a directory
  python run.py --dir /path/to/notebooks/

  # Specific category only
  python run.py --dir /path/to/notebooks/ --category inference

  # Diagnose only, never auto-fix
  python run.py path/to/notebook.ipynb --mode verify

  # Interactive mode (opens claude normally — no -p flag, full conversation)
  python run.py path/to/notebook.ipynb --interactive

  # Run only the gap set (never-tested + stale per tools/status.py --gap)
  python run.py --dir /path/to/notebooks/ --gap

  # Gap set, filtered to one category
  python run.py --dir /path/to/notebooks/ --gap --category inference

  # Post this run's summary to Teams afterwards
  python run.py path/to/notebook.ipynb --notify

See `python run.py --help` for every option.
"""

import argparse
import json
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HERE        = Path(__file__).parent
TOOLS_DIR   = (HERE / "tools").resolve()
RESULTS_DIR = (HERE / "results").resolve()
FIXES_DIR   = (HERE / "fixes").resolve()   # genuine-fix notebooks retrieved by the fixer (for PRs)
EXECUTED_DIR = RESULTS_DIR / "executed"    # executed notebooks each verdict is computed from
SHIM_DIR    = (TOOLS_DIR / "http_ssh_shim").resolve()

sys.path.insert(0, str(TOOLS_DIR))
import k8s_pod
import radeon_pod
import results_db
import routing
import verdict
from write_result import write_result

# k8s runner credential from .env. Exported as KUBECONFIG so kubectl in this
# process and in the agent's own Bash calls (tools/k8s_pod.py) uses it; inside
# the CI container docker/run-ci.sh sets KUBECONFIG to the staged copy instead.
if os.getenv("NBR_KUBECONFIG") and not os.getenv("KUBECONFIG"):
    os.environ["KUBECONFIG"] = os.path.expanduser(os.environ["NBR_KUBECONFIG"])

# ── Routing ───────────────────────────────────────────────────────────────────

def _rel_key(nb: Path, base_dir: Path | None) -> str:
    """The notebook's path under the notebook root: its key in results and routes."""
    if base_dir:
        try:
            return str(nb.relative_to(base_dir))
        except ValueError:
            pass
    return nb.name


def target_env(target: dict) -> dict:
    """Environment for every subprocess that talks to `target`. A target with
    an agent_url is reached through the HMAC-signed HTTP shim: its ssh/scp come
    first on PATH, for run.py and for the agent's own Bash calls alike."""
    env = dict(os.environ)
    env.pop("AGENT_URL", None)
    if target.get("agent_url"):
        env["AGENT_URL"] = target["agent_url"]
        env["PATH"] = f"{SHIM_DIR}{os.pathsep}{env.get('PATH', '')}"
    return env


def order_by_dependencies(notebooks: list[Path], base_dir: Path | None) -> list[Path]:
    """Run each notebook after the ones in this batch it declares in
    metadata.ci.depends_on; otherwise keep the given order."""
    by_key = {_rel_key(nb, base_dir): nb for nb in notebooks}
    ordered: list[Path] = []
    seen: set[str] = set()

    def visit(key: str) -> None:
        if key in seen or key not in by_key:
            return
        seen.add(key)
        req = routing.resolve(by_key[key], {})[0]
        for dep in req.depends_on if req else []:
            visit(dep)
        ordered.append(by_key[key])

    for key in by_key:
        visit(key)
    return ordered


def collect_notebooks(directory, category):
    base = Path(directory)
    if not base.exists():
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    pattern   = f"{category}/**/*.ipynb" if category else "**/*.ipynb"
    notebooks = sorted(base.glob(pattern))
    if not notebooks:
        print(f"No notebooks found in {base}", file=sys.stderr)
        sys.exit(1)
    # Resolved absolute paths, like the other selectors, so nb.relative_to(base_dir)
    # gives the notebook's key; otherwise it falls back to the bare filename.
    return [p.resolve() for p in notebooks]


def collect_status_notebooks(selector: str, base_dir: Path,
                             category: str | None) -> list[Path]:
    """Subprocess `tools/status.py <selector>` (--gap or --failing) and resolve
    each line against base_dir.

    --gap:     never-tested or stale >7d
    --failing: latest result is fail or blocked
    """
    proc = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "status.py"), selector],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(1)

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

    result: list[Path] = []
    missing: list[str] = []
    for line in lines:
        p = (base_dir / line).resolve()
        if not p.exists():
            missing.append(line)
            continue
        result.append(p)

    if missing:
        print(f"warning: {selector} entries not found on disk (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if category:
        result = [n for n in result if n.relative_to(base_dir).parts[0] == category]

    return result


# ── Notebook transfer ─────────────────────────────────────────────────────────

def scp_notebook(local_path: Path, target: dict, support: list[Path]) -> str:
    """Copy the notebook and its support files (preflight_patch.py,
    run_notebook.sh and its settings) to an SSH target. Returns remote path."""
    remote_dir = f"{_workspace_base(target)}/tutorial_agent_runs/{local_path.stem}"
    remote_path = f"{remote_dir}/{local_path.name}"
    dest       = f"{target['user']}@{target['host']}"
    env        = target_env(target)

    print(f"  Copying {local_path.name} → {dest}:{remote_dir}/", flush=True)
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                 "-o", "ForwardX11=no", dest, f"mkdir -p {remote_dir}"],
                check=True, capture_output=True, timeout=60, env=env,
            )
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ForwardX11=no",
                 str(local_path), *map(str, support), f"{dest}:{remote_dir}/"],
                check=True, capture_output=True, timeout=60, env=env,
            )
            return remote_path
        except Exception as exc:
            if attempt < 3:
                print(f"  SCP attempt {attempt} failed ({exc}), retrying in 15s…", flush=True)
                time.sleep(15)
            else:
                raise RuntimeError(f"Failed to copy notebook after 3 attempts: {exc}") from exc


def _workspace_base(target: dict) -> str:
    """Remote base for per-notebook run dirs on an SSH target."""
    return str(target.get("workspace_base", "/home/amd")).rstrip("/")


# ── stream-json pretty printer ────────────────────────────────────────────────

# Label patterns — same logic as original agent._infer_label
_LABEL_PATTERNS = [
    (r"docker pull",            "Pulling Docker image"),
    (r"docker run.*-d",         "Starting Docker container (background)"),
    (r"docker run",             "Running in Docker"),
    (r"docker (stop|rm)",       "Stopping Docker container"),
    (r"papermill",              "Executing notebook (papermill)"),
    (r"curl.*health",           "Polling server health"),
    (r"curl",                   "HTTP request"),
    (r"pip install",            "Installing Python packages"),
    (r"resolve_docker_image",   "Resolving Docker image"),
    (r"write_result",           "Writing result JSON"),
    (r"rm -rf.*tutorial_agent", "Cleaning up run directory"),
    (r"mkdir",                  "Creating directories"),
    (r"cat.*log|tail.*log",     "Reading logs"),
    (r"python3?\s+-",           "Running Python snippet"),
    (r"python3?\s+",            "Running Python"),
    (r"ssh ",                   "SSH command"),
    (r"scp ",                   "Copying files"),
]

def _infer_label(cmd: str) -> str:
    cmd_lower = cmd.lower()
    for pattern, label in _LABEL_PATTERNS:
        if re.search(pattern, cmd_lower):
            return label
    tokens = [t for t in cmd.split() if t not in ("set", "-euo", "pipefail", "&&", ";")]
    return tokens[0][:60] if tokens else "Running command"


def _is_poll(stdout: str) -> bool:
    markers = ("not ready", "not healthy", "RUNNING", "STILL_RUNNING",
               "Pull complete", "Waiting", "Extracting")
    return any(m in stdout for m in markers) and "DONE" not in stdout


def handle_stream(proc, notebook_name: str, agent_label: str = "") -> dict:
    """
    Parse stream-json lines from claude process.
    Pretty-prints each step like the original agent.
    Returns {"cost_usd": float, "num_turns": int} from the result event.
    """
    poll_active   = False
    poll_label    = ""
    poll_start    = 0.0
    meta          = {}

    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")

        # ── Assistant turn ──────────────────────────────────────────────────
        if etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")

                if btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        if poll_active:
                            print()
                            poll_active = False
                        print(f"\n  » {text}", flush=True)

                elif btype == "tool_use":
                    name  = block.get("name", "")
                    inp   = block.get("input", {})

                    if name == "Bash":
                        cmd   = inp.get("command", "")
                        label = _infer_label(cmd)
                        if not poll_active:
                            print(f"  ▶  {label}", end="", flush=True)
                        # poll handling happens in tool_result
                    elif name == "Read":
                        path = inp.get("file_path", "")
                        fname = Path(path).name if path else "file"
                        if poll_active:
                            print()
                            poll_active = False
                        print(f"  📖  Reading {fname}", flush=True)
                    elif name == "Write":
                        path = inp.get("file_path", "")
                        fname = Path(path).name if path else "file"
                        if poll_active:
                            print()
                            poll_active = False
                        print(f"  ✏   Writing {fname}", flush=True)
                    else:
                        if poll_active:
                            print()
                            poll_active = False
                        print(f"  ·  {name}", flush=True)

        # ── Tool result ─────────────────────────────────────────────────────
        elif etype == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, list):
                    # content blocks format
                    stdout_text = " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                else:
                    stdout_text = str(content)

                if _is_poll(stdout_text):
                    if not poll_active:
                        poll_active = True
                        poll_start  = time.time()
                        poll_label  = "Waiting"
                        print(" ⏳", end="", flush=True)
                    else:
                        elapsed = int(time.time() - poll_start)
                        print(f"\r  ▶  {poll_label} ⏳ {elapsed}s elapsed", end="", flush=True)
                else:
                    if poll_active:
                        print()
                        poll_active = False
                    # Show pass/fail for the previous tool call line
                    is_err = "error" in stdout_text.lower()[:200] or "ERROR" in stdout_text[:200]
                    print(f"  {'✗' if is_err else ' ✓'}", flush=True)

        # ── Final result event ───────────────────────────────────────────────
        elif etype == "result":
            if poll_active:
                print()
                poll_active = False
            meta["cost_usd"]  = event.get("total_cost_usd", event.get("cost_usd", 0.0))
            meta["num_turns"] = event.get("num_turns", 0)
            if event.get("subtype") == "error_max_turns":
                print("\n  ✗ Hit max turns limit", flush=True)

    proc.wait()
    return meta


# ── Environment check ─────────────────────────────────────────────────────────

def run_env_check(notebook_path: Path, rel_key: str, target: dict,
                  req: routing.Requirements, hf_token: str | None,
                  skip: bool = False) -> dict | None:
    """Run tools/env_check.py for this notebook.

    Returns None if env_check passed (or was skipped, or itself errored —
    in all those cases the caller should proceed with the agent run).
    Returns a result dict if env_check BLOCKED — caller should return this
    as the notebook's outcome and skip the agent.
    """
    if skip:
        print("  [env_check] SKIPPED via --skip-env-check", flush=True)
        return None

    cmd = [
        sys.executable, str(TOOLS_DIR / "env_check.py"),
        "--notebook",       rel_key,
        "--notebook-local", str(notebook_path),
        "--gpus",           str(req.gpus),
        "--disk-gb",        str(req.disk_gb),
        "--results-dir",    str(RESULTS_DIR),
        "--write-result",
    ]
    if _is_pod(target):
        # No host to probe: GPU fit is checked statically, disk is the pod's
        # sized /workspace volume.
        cmd += ["--gpus-per-node", str(target.get("gpus", 1))]
    else:
        cmd += ["--ssh-cmd", (f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
                              f"-o ForwardX11=no {target['user']}@{target['host']}"),
                "--workspace-base", _workspace_base(target)]
    if hf_token:
        cmd += ["--hf-token", hf_token]

    started = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                                env=target_env(target))
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait(timeout=120)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        print("warning: env_check timed out — proceeding with agent run", file=sys.stderr)
        return None

    if rc == 0:
        print("  » env_check passed — proceeding to agent", flush=True)
        return None
    elif rc == 1:
        print("  » env_check blocked the run — using env_check result as outcome", flush=True)
        result = _find_latest_result(notebook_path, since=started)
        if result is not None:
            return result
        return blocked(notebook_path, "verify",
                       "env_check blocked the run but no result file was written")
    else:
        print(f"warning: env_check itself errored (exit {rc}) — proceeding with agent run",
              file=sys.stderr)
        return None


# ── Pod runners (Kubernetes, Radeon Global) ───────────────────────────────────

POD_TOOLS = {"k8s": "k8s_pod.py", "radeon": "radeon_pod.py"}
POD_WORKSPACE = "/workspace"  # same in a k8s pod and a Radeon instance


def _is_pod(target: dict | None) -> bool:
    """Targets where the agent provisions its own pod/instance per run."""
    return bool(target) and target.get("type") in POD_TOOLS


def _pod_tool(target_name: str, target: dict) -> str:
    return f"python3 {TOOLS_DIR}/{POD_TOOLS[target['type']]} --target {target_name}"


def pod_runtime(target_name: str, target: dict, notebook_path: Path,
                req: routing.Requirements, phase: str, run_id: str,
                support: list[Path]) -> tuple[str, str]:
    """Return (pod name, runtime-context block) for one agent run in a pod.

    The agent provisions the pod itself (it picks the image after reading the
    notebook); run.py only names it and deletes it afterwards.
    """
    pod  = k8s_pod.pod_name(notebook_path.stem, phase, run_id)
    tool = _pod_tool(target_name, target)
    up_cmd = (f"{tool} up --pod {pod} --gpus {req.gpus} --memory-gb {req.memory_gb} "
              + "".join(f"--env {shlex.quote(f'{k}={v}')} " for k, v in req.env.items())
              + f"--push {notebook_path} " + " ".join(f"--push {p}" for p in support))
    block = f"""RUNNER          = {target['type']}  (target {target_name}: follow CLAUDE.md "Runner: pod" — there is no SSH host)
POD_CMD         = {tool} exec --pod {pod}
K8S_UP_CMD      = {up_cmd}
NOTEBOOK_REMOTE = {POD_WORKSPACE}/{notebook_path.name}  (K8S_UP_CMD copies it in once the pod is READY)
WORKSPACE       = {POD_WORKSPACE}  (use this wherever the playbook says <WORKSPACE_BASE>/tutorial_agent_runs/<stem>)
SCHEDULE_TIMEOUT = {target.get('schedule_timeout_min', 45)} min (K8S_UP_CMD reports UNSCHEDULABLE after this)

Provision the pod once you have resolved the image (repeat until it prints READY):
  Bash: {up_cmd} --image <image:tag>

Use POD_CMD like this (HF_TOKEN and CONTAINER_ENV are already set in the pod environment):
  Bash: {tool} exec --pod {pod} '<your command here>'
"""
    if target["type"] == "radeon":
        block += """
On Radeon Global only catalog images run: K8S_UP_CMD maps the repository you
pass with --image to the target's catalog image for it (the tag is ignored) and
prints the image it launched — report that one as the resolved image. There is
no Docker in the instance.
"""
    if phase == "fix":
        block += f"""
POD_PULL_CMD    = {tool} pull --pod {pod}
Retrieve the genuine-fix notebook (CLAUDE_FIX.md Step 4c) with:
  Bash: {tool} pull --pod {pod} {POD_WORKSPACE}/{notebook_path.stem}_fixed.ipynb {FIXES_DIR}/{notebook_path.stem}.ipynb
"""
    return pod, block


def pod_down(target_name: str, target: dict, pod: str) -> None:
    """Delete the agent's pod or instance. Called from a finally, so never raises."""
    try:
        if target["type"] == "radeon":
            radeon_pod.down(target, pod)
        else:
            k8s_pod.down(k8s_pod.Kubectl(target), pod)
    except Exception as exc:  # noqa: BLE001 — cleanup must not mask the result
        print(f"warning: could not delete pod {pod} on {target_name} ({exc}); "
              f"its lifetime limit will stop it", file=sys.stderr)


# ── Core runner ───────────────────────────────────────────────────────────────

def _transport(notebook_path: Path, phase: str, target_name: str, target: dict,
               req: routing.Requirements, hf_token: str | None,
               run_id: str) -> tuple[str | None, str]:
    """Return (pod or None, runtime-context block) for one agent run: copy the
    notebook to the SSH target, or name the pod the agent will provision.
    Raises RuntimeError if the copy fails."""
    if _is_pod(target):
        pod, block = pod_runtime(target_name, target, notebook_path, req, phase, run_id,
                                 support_files(notebook_path, req))
    else:
        pod, block = None, _ssh_runtime(
            target, hf_token, scp_notebook(notebook_path, target, support_files(notebook_path, req)))
    name = _output_name(notebook_path, phase)
    where = (f"{POD_WORKSPACE}/{name}" if pod else
             f"{_run_dir(target, notebook_path)}/{name}  (inside the container: /workspace/{name})")
    return pod, block + f"""
OUTPUT_NOTEBOOK = {where}
  papermill must write the executed notebook here. Do not delete it or the run
  dir: run.py collects it, decides pass/fail/blocked from it, and cleans up.
"""


def _output_name(notebook_path: Path, phase: str) -> str:
    return f"{notebook_path.stem}_{'fixed_out' if phase == 'fix' else 'out'}.ipynb"


def _run_dir(target: dict, notebook_path: Path) -> str:
    return f"{_workspace_base(target)}/tutorial_agent_runs/{notebook_path.stem}"


def collect_output(notebook_path: Path, phase: str, target_name: str, target: dict,
                   pod: str | None) -> Path | None:
    """Copy the executed notebook back, then remove the SSH run dir (a pod is
    deleted by the caller). Returns the local copy, or None if there is none."""
    name  = _output_name(notebook_path, phase)
    local = EXECUTED_DIR / (f"{notebook_path.stem}_{phase}_"
                            f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}.ipynb")
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if pod:
            subprocess.run([*shlex.split(_pod_tool(target_name, target)), "pull", "--pod", pod,
                            f"{POD_WORKSPACE}/{name}", str(local)],
                           capture_output=True, timeout=1900, check=False)
        else:
            dest, run_dir = f"{target['user']}@{target['host']}", _run_dir(target, notebook_path)
            env = target_env(target)
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", f"{dest}:{run_dir}/{name}",
                            str(local)], capture_output=True, timeout=300, check=False, env=env)
            # Stop the agent's containers (a timeboxed run may leave them up), then
            # remove the run dir; container-written files are root-owned, so via a container.
            stem = notebook_path.stem
            cleanup = (f"docker rm -f {stem} {stem}_srv >/dev/null 2>&1; "
                       f"docker run --rm -v {run_dir}:/workspace alpine sh -c "
                       f"'rm -rf /workspace/* /workspace/.[!.]*' 2>/dev/null; rm -rf {run_dir}")
            subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", dest, cleanup],
                           capture_output=True, timeout=120, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        print(f"warning: could not collect {name} ({exc})", file=sys.stderr)
    return local if local.is_file() and local.stat().st_size else None


def finalize(notebook_path: Path, phase: str, agent_result: dict,
             executed: Path | None) -> dict:
    """Set the verdict on this attempt's result JSON, writing one if the agent
    did not. The agent's own status, if it gave one, is kept as agent_status."""
    v = verdict.verdict(executed)
    if agent_result.get("timebox_minutes"):
        v["timebox_minutes"] = agent_result["timebox_minutes"]
        if v["status"] == "blocked":
            v["reason"] = f"stopped at its {agent_result['timebox_minutes']}-minute timebox; {v['reason']}"
    path = agent_result.get("result_path")
    if path:
        payload = json.loads(Path(path).read_text())
    else:
        path = str(write_result(
            str(notebook_path), None,
            agent_result.get("message") or "The agent wrote no diagnosis.",
            results_dir=str(RESULTS_DIR),
            agent="claude_code_fix" if phase == "fix" else "claude_code_verify"))
        payload = json.loads(Path(path).read_text())
    claimed = payload.get("status")
    if claimed not in (None, "unverified"):
        payload["agent_status"] = claimed
    payload["status"] = v["status"]
    payload["verdict"] = {**v, "agent_disagrees": claimed not in (None, "unverified", v["status"])}
    if v["status"] != "pass":
        payload["message"] = v["reason"]
    if executed:
        payload["executed_notebook"] = str(executed)
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"  » verdict: {v['status'].upper()} — {v['reason']}", flush=True)
    return {**payload, "result_path": path, "cost_usd": agent_result.get("cost_usd"),
            "num_turns": agent_result.get("num_turns")}


def blocked(notebook_path: Path, phase: str, reason: str) -> dict:
    """Record an attempt that never got to run the notebook."""
    return finalize(notebook_path, phase, {"message": reason}, None) | {"message": reason}


def support_files(notebook_path: Path, req: routing.Requirements) -> list[Path]:
    """Files every run needs next to the notebook: preflight_patch.py,
    run_notebook.sh, its settings (run_notebook.env, from metadata.ci) and the
    notebook's requirements file, if it declares one."""
    stage = HERE / "logs" / ".stage" / notebook_path.stem
    stage.mkdir(parents=True, exist_ok=True)
    files = [TOOLS_DIR / "preflight_patch.py", TOOLS_DIR / "run_notebook.sh",
             stage / "run_notebook.env"]
    requirements = ""
    if isinstance(req.requirements, str):  # a file next to the notebook
        source = notebook_path.parent / req.requirements
        if source.is_file():
            files.append(source)
            requirements = source.name
    elif req.requirements:  # a list of pip specs
        (stage / "ci_requirements.txt").write_text("\n".join(req.requirements) + "\n")
        files.append(stage / "ci_requirements.txt")
        requirements = "ci_requirements.txt"
    params = " ".join(f"-p {shlex.quote(str(k))} {shlex.quote(str(v))}"
                      for k, v in req.parameters.items())
    (stage / "run_notebook.env").write_text(
        f"SYSTEM_PACKAGES={shlex.quote(' '.join(req.system_packages))}\n"
        f"REQUIREMENTS={shlex.quote(requirements)}\n"
        f"TIMEOUT_MINUTES={_papermill_minutes(req)}\n"
        f"PAPERMILL_ARGS=({params})\n"
        + "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in req.env.items()))
    return files


def _requirements_ctx(target: dict, req: routing.Requirements) -> str:
    """Runtime-context lines for what the notebook declares in metadata.ci."""
    env = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in req.env.items())
    lines = [
        f"GPU_HARDWARE    = {target.get('hardware', 'unknown')}",
        f"MEMORY_LIMIT    = {req.memory_gb}g  (use for docker run --memory)",
        f"CONTAINER_ENV   = {env or '(none)'}  (add to docker run)",
        ("RUN_NOTEBOOK    = bash run_notebook.sh <input.ipynb> <output.ipynb>  (run it in the "
         "notebook's run dir; it installs the notebook's declared system packages and "
         "requirements into a sandbox venv and runs papermill with its CI parameters and "
         "timeout)"),
        (f"TIMEBOX_MINUTES = {req.timeout_minutes}  (the runner stops you after this; "
         f"run_notebook.sh gives papermill {_papermill_minutes(req)} of them)"),
    ]
    if req.image:
        lines.append(f"IMAGE_REPO      = {req.image}  (the notebook declares this repo; "
                     f"resolve its tag with resolve_docker_image.py)")
    return "\n".join(lines)


def _papermill_minutes(req: routing.Requirements) -> int:
    """Papermill's share of the timebox, leaving the agent time to diagnose."""
    return max(req.timeout_minutes - 10, req.timeout_minutes // 2)


def run_notebook(notebook_path: Path, rel_key: str, target_name: str, target: dict,
                 req: routing.Requirements, interactive: bool = False,
                 skip_env_check: bool = False, run_id: str = "") -> dict:
    hf_token = os.getenv("HF_TOKEN")

    # 0. Environment check — fail fast on unwinnable cases
    env_blocked = run_env_check(notebook_path, rel_key, target, req, hf_token,
                                skip=skip_env_check)
    if env_blocked is not None:
        return env_blocked

    try:
        pod, transport_ctx = _transport(notebook_path, "verify", target_name, target, req,
                                        hf_token, run_id)
    except RuntimeError as e:
        return blocked(notebook_path, "verify", str(e))

    prompt = "\n".join([
        f"Run the regression test for this notebook: {notebook_path}",
        "",
        ("Follow the playbook in CLAUDE.md exactly. Start by reading the notebook "
         "with the Read tool, then proceed through all phases. When done, call "
         "write_result.py then finish. Do not retry content failures — report them accurately."),
    ])

    runtime_ctx = f"""## Runtime Context for this Run

{transport_ctx}
{_requirements_ctx(target, req)}
NOTEBOOK_LOCAL  = {notebook_path}
TOOLS_DIR       = {TOOLS_DIR}
RESULTS_DIR     = {RESULTS_DIR}

resolve_docker_image:
  Bash: python3 {TOOLS_DIR}/resolve_docker_image.py <repo> {target.get('hardware', 'mi300x')}

write_result (call this when done):
  Bash: python3 {TOOLS_DIR}/write_result.py \\
    --notebook "{notebook_path}" \\
    --summary "..." \\
    --issues '[...]' \\
    --fixes  '[...]' \\
    --results-dir "{RESULTS_DIR}"
"""

    try:
        if interactive:
            # No -p: the user converses with claude; pick up whatever result it wrote.
            print(f"\n  Opening interactive Claude session for {notebook_path.name}")
            print("  Runtime context is in the system prompt. Type your instructions.\n")
            started = time.time()
            subprocess.run(["claude", "--append-system-prompt", runtime_ctx],
                           cwd=str(HERE), check=False, env=target_env(target))
            result = (_find_latest_result(notebook_path, since=started)
                      or {"message": "Interactive session ended without a result"})
        else:
            result = _run_agent("VERIFY", notebook_path, prompt, runtime_ctx, target,
                                req.timeout_minutes)
        executed = collect_output(notebook_path, "verify", target_name, target, pod)
    finally:
        if pod:
            pod_down(target_name, target, pod)
    return finalize(notebook_path, "verify", result, executed)


def _ssh_runtime(target: dict, hf_token: str | None, remote_path: str) -> str:
    """Runtime-context lines for an SSH target."""
    ssh_flags  = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ForwardX11=no"
    ssh_target = f"{target['user']}@{target['host']}"
    ssh_cmd    = f"ssh {ssh_flags} {ssh_target}"
    base       = _workspace_base(target)
    return f"""SSH_CMD     = {ssh_cmd}
SSH_HF_CMD  = ssh {ssh_flags} {ssh_target} (prefix commands that need HF_TOKEN with: export HF_TOKEN='{hf_token}';)
NOTEBOOK_REMOTE = {remote_path}  (already on the server — do NOT copy again)
WORKSPACE_BASE  = {base}  (per-notebook run dir is {base}/tutorial_agent_runs/<stem>)

Use SSH_CMD like this:
  Bash: {ssh_cmd} '<your command here>'

For commands that need HF_TOKEN:
  Bash: {ssh_cmd} 'export HF_TOKEN='{hf_token}'; <your command here>'
"""


def _run_agent(label: str, notebook_path: Path, prompt: str, system_prompt: str,
               target: dict, timebox_minutes: int) -> dict:
    """Run one headless agent and return the result JSON it wrote, plus cost.

    A one-shot `claude -p` has no scheduler to resume it, so deny the
    wakeup/cron tools and make the agent poll in-turn (CLAUDE.md). The agent
    and everything it started locally are stopped at the timebox.
    """
    print(f"\n  Starting {label.lower()} agent (headless)…", flush=True)
    started = time.time()
    proc = subprocess.Popen(
        [
            "claude",
            "-p", prompt,
            "--append-system-prompt", system_prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "200",
            "--dangerously-skip-permissions",
            "--disallowedTools", "ScheduleWakeup,CronCreate,CronList,CronDelete",
        ],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,   # show claude errors live instead of swallowing them
        text=True,
        cwd=str(HERE),
        env=target_env(target),
        start_new_session=True,  # its own process group, so the timebox stops all of it
    )
    stopped = threading.Event()

    def stop() -> None:
        stopped.set()
        print(f"\n  ✗ {timebox_minutes}-minute timebox reached — stopping the agent", flush=True)
        for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                pass

    timer = threading.Timer(timebox_minutes * 60, stop)
    timer.daemon = True
    timer.start()
    try:
        meta = handle_stream(proc, notebook_path.name, agent_label=label)
    finally:
        timer.cancel()

    result = _find_latest_result(notebook_path, since=started) or {
        "message": f"{label.capitalize()} agent finished but did not write a result file",
    }
    result["cost_usd"]  = meta.get("cost_usd")
    result["num_turns"] = meta.get("num_turns")
    if stopped.is_set():
        result["timebox_minutes"] = timebox_minutes
    return result


def run_fixer(notebook_path: Path, target_name: str, target: dict,
              req: routing.Requirements, verification_result_path: Path,
              run_id: str = "") -> dict:
    """Run the fixer agent on a notebook using a prior verification result."""

    FIXES_DIR.mkdir(parents=True, exist_ok=True)

    # Check for auto_fixable issues — skip if none
    if not _has_auto_fixable_issues(verification_result_path):
        print("  No auto_fixable issues — skipping fixer agent", flush=True)
        result = _find_latest_result(notebook_path)
        return result or blocked(notebook_path, "fix", "No verification result to fix from")

    # The verify run's dir or pod is gone, so copy again / use a fresh pod.
    try:
        pod, transport_ctx = _transport(notebook_path, "fix", target_name, target, req,
                                        os.getenv("HF_TOKEN"), run_id)
    except RuntimeError as e:
        return blocked(notebook_path, "fix", str(e))

    prompt = "\n".join([
        f"Fix the issues identified by the verification agent for: {notebook_path}",
        "",
        f"The verification result is at: {verification_result_path}",
        "",
        "Follow the CLAUDE_FIX.md playbook provided in the system prompt below.",
        "Start by reading the verification result JSON and the notebook,",
        "then proceed through all steps. Only fix issues marked auto_fixable.",
        "Write the final result via write_result.py, then finish.",
    ])

    runtime_ctx = f"""## Runtime Context for this Fix Run

{transport_ctx}
{_requirements_ctx(target, req)}
NOTEBOOK_LOCAL  = {notebook_path}
TOOLS_DIR       = {TOOLS_DIR}
RESULTS_DIR     = {RESULTS_DIR}
FIXES_DIR       = {FIXES_DIR}  (retrieve <nb>_fixed.ipynb here as <stem>.ipynb — see CLAUDE_FIX.md Step 4c)
VERIFICATION_RESULT = {verification_result_path}

resolve_docker_image:
  Bash: python3 {TOOLS_DIR}/resolve_docker_image.py <repo> {target.get('hardware', 'mi300x')}

write_result (call this when done):
  Bash: python3 {TOOLS_DIR}/write_result.py \\
    --notebook "{notebook_path}" \\
    --summary "..." \\
    --issues '<all issues>' \\
    --fixes  '<fixes array>' \\
    --agent "claude_code_fix" \\
    --verification-result "{verification_result_path}" \\
    --results-dir "{RESULTS_DIR}"
"""
    system_prompt = (HERE / "CLAUDE_FIX.md").read_text() + "\n\n---\n\n" + runtime_ctx

    try:
        result = _run_agent("FIX", notebook_path, prompt, system_prompt, target,
                            req.timeout_minutes)
        executed = collect_output(notebook_path, "fix", target_name, target, pod)
    finally:
        if pod:
            pod_down(target_name, target, pod)
    return finalize(notebook_path, "fix", result, executed)


def _find_latest_result(notebook_path: Path, since: float | None = None) -> dict | None:
    """Find the most recent result JSON written by write_result.py for this notebook.

    Pass `since` (a time.time() taken before the agent started) to ignore results
    from earlier runs — otherwise an agent that exits without writing anything
    silently inherits the previous run's verdict.
    """
    result_path = _find_latest_result_path(notebook_path)
    if result_path is not None and since is not None and result_path.stat().st_mtime < since:
        return None
    if result_path is None:
        return None
    with open(result_path) as f:
        result = json.load(f)
    # Carried along so results_db can link the row to its JSON file.
    result["result_path"] = str(result_path)
    return result


def _find_latest_result_path(notebook_path: Path) -> Path | None:
    """Return the Path to the most recent result JSON for this notebook, or None."""
    stem    = notebook_path.stem
    matches = sorted(RESULTS_DIR.glob(f"{stem}_*.json"), reverse=True)
    return matches[0] if matches else None


def _has_auto_fixable_issues(result_path: Path) -> bool:
    """Check whether a result JSON contains any auto_fixable issues."""
    with open(result_path) as f:
        data = json.load(f)
    return any(
        iss.get("fixability") == "auto_fixable"
        for iss in data.get("issues", [])
    )


# ── Results database ──────────────────────────────────────────────────────────

def _db_call(conn, fn, *args):
    """Run a results_db write, never letting a DB problem fail the pass.

    The agent's JSON result is already on disk; `tools/results_db.py import`
    can backfill anything missed here.
    """
    if conn is None:
        return None
    try:
        return fn(conn, *args)
    except sqlite3.Error as exc:
        print(f"warning: results DB write failed ({exc}) — JSON result is unaffected",
              file=sys.stderr)
        return None


def _notify_teams(run_id: str) -> None:
    """Post this run's summary card. Never fatal — same as ci_run.py Phase E."""
    subprocess.run(
        [sys.executable, str(TOOLS_DIR / "report_teams.py"),
         "--run-id", run_id, "--always-post"],
        check=False,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Notebook regression agent — Claude Code edition")
    p.add_argument("notebook", nargs="*")
    p.add_argument("--target",      default=None,
                   help="Only consider this runner.yaml target (e.g. a k8s pool). "
                        "Notebooks it cannot satisfy are reported, not run.")
    p.add_argument("--dir",         help="Run all notebooks under this directory")
    p.add_argument("--category",    choices=["inference", "fine_tune", "pretrain", "gpu_dev_optimize"])
    p.add_argument("--interactive", "-i", action="store_true")
    p.add_argument("--gap", action="store_true",
                   help="Run only notebooks reported by tools/status.py --gap "
                        "(never-tested or stale >7d). Requires --dir. Composes with --category.")
    p.add_argument("--failing", action="store_true",
                   help="Run only notebooks reported by tools/status.py --failing "
                        "(latest result is a hard fail). Requires --dir. Composes with --category.")
    p.add_argument("--skip-env-check", action="store_true",
                   help="Bypass env_check.py entirely. Use only when env_check "
                        "is known to be broken or for development debugging.")
    p.add_argument("--mode", choices=["verify", "fix", "full"], default="full",
                   help="verify=diagnosis only, fix=apply fixes from prior result, "
                        "full=verify then fix (default)")
    p.add_argument("--verification-result", default=None,
                   help="Path to verification result JSON (required for --mode fix)")
    p.add_argument("--run-id", default=None,
                   help="Group this invocation's results under an existing run id "
                        "(ci_run.py uses this so its batches form one run). "
                        "Default: a fresh UTC timestamp.")
    p.add_argument("--notify", action="store_true",
                   help="After the run, post a summary of it to Teams "
                        "(needs TEAMS_WEBHOOK_URL; skipped with a note if unset).")
    return p.parse_args()


def main():
    args     = parse_args()
    config   = routing.load_config()
    targets  = config["targets"]
    base_dir = Path(args.dir).resolve() if args.dir else None

    if args.gap and args.failing:
        print("Error: --gap and --failing are mutually exclusive", file=sys.stderr)
        sys.exit(2)
    if (args.gap or args.failing) and args.notebook:
        print("Error: --gap/--failing cannot be combined with positional notebook paths",
              file=sys.stderr)
        sys.exit(2)
    if (args.gap or args.failing) and base_dir is None:
        print("Error: --gap/--failing requires --dir to resolve notebook keys to local files",
              file=sys.stderr)
        sys.exit(2)
    if not targets:
        print(f"Error: no targets configured in {routing.CONFIG_PATH} "
              f"(copy runner.example.yaml)", file=sys.stderr)
        sys.exit(2)
    if args.target and args.target not in targets:
        print(f"Error: --target {args.target} is not a target in {routing.CONFIG_PATH}",
              file=sys.stderr)
        sys.exit(2)

    if args.gap:
        notebooks = collect_status_notebooks("--gap", base_dir, args.category)
        if not notebooks:
            if args.category:
                print(f"\nNo gap notebooks in category '{args.category}'.")
            else:
                print("\nGap is empty — every testable notebook has a recent result. Nothing to do.")
            sys.exit(0)
    elif args.failing:
        notebooks = collect_status_notebooks("--failing", base_dir, args.category)
        if not notebooks:
            if args.category:
                print(f"\nNo failing notebooks in category '{args.category}'.")
            else:
                print("\nNo failing notebooks — everything passes. Nothing to do.")
            sys.exit(0)
    elif args.notebook:
        notebooks = [Path(n).resolve() for n in args.notebook]
        if base_dir is None and notebooks:
            candidate = notebooks[0].parent
            while candidate != candidate.parent:
                if any((candidate / cat).is_dir() for cat in
                       ("inference", "fine_tune", "pretrain", "gpu_dev_optimize")):
                    base_dir = candidate
                    break
                candidate = candidate.parent
    elif args.dir:
        notebooks = collect_notebooks(args.dir, args.category)
    else:
        print("Error: provide a notebook path or --dir", file=sys.stderr)
        sys.exit(1)
    notebooks = order_by_dependencies(notebooks, base_dir)

    # Validate --mode fix requires --verification-result (for single notebooks)
    if args.mode == "fix" and not args.verification_result and len(notebooks) == 1:
        # For single notebook fix mode, require explicit result path
        result_path = _find_latest_result_path(notebooks[0])
        if result_path is None:
            print("Error: --mode fix requires --verification-result or a prior result in results/",
                  file=sys.stderr)
            sys.exit(1)

    mode_label = args.mode
    if args.interactive:
        mode_label = "interactive"

    print("\nNotebook regression agent  [Claude Code]")
    print(f"  Notebooks : {len(notebooks)}")
    print(f"  Mode      : {mode_label}")
    print(f"  Targets   : {args.target or ', '.join(targets)}")
    print(f"  Results   : {RESULTS_DIR}")

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    try:
        db = results_db.connect()
    except sqlite3.Error as exc:
        print(f"warning: could not open results DB {results_db.DEFAULT_DB} ({exc}) — "
              f"continuing with JSON results only", file=sys.stderr)
        db = None
    _db_call(db, results_db.start_run, run_id, mode_label,
             args.target or "routed", sys.argv[1:])
    print(f"  Run ID    : {run_id}  (db: {results_db.DEFAULT_DB})\n")

    results = []
    skipped = []
    gc_done: set[str] = set()

    for nb in notebooks:
        print(f"{'='*60}")
        print(f"Testing: {nb.name}")
        print(f"{'='*60}")

        rel_key = _rel_key(nb, base_dir)
        req, target_name, target, reason = routing.resolve(nb, targets, args.target)
        if req:
            req.timeout_minutes = req.timeout_minutes or config["timebox_minutes"]
        if target is None:
            print(f"  NO TARGET — {reason}\n")
            skipped.append({"notebook": str(nb), "reason": reason})
            _db_call(db, results_db.record_result, run_id, rel_key, str(nb), "skip",
                     {"status": "skip", "message": reason}, None)
            continue

        if target.get("type") == "k8s":
            print(f"  Target     : {target_name} (k8s {target.get('context')}/"
                  f"{target.get('namespace')}, {target.get('hardware')})")
            if target_name not in gc_done:
                # Clear pods a killed earlier run left behind before adding more.
                try:
                    k8s_pod.gc(k8s_pod.Kubectl(target))
                except Exception as exc:  # noqa: BLE001 — hygiene, not a precondition
                    print(f"warning: k8s gc failed ({exc})", file=sys.stderr)
                gc_done.add(target_name)
        elif target.get("type") == "radeon":
            print(f"  Target     : {target_name} (Radeon Global as {target.get('user_name')}, "
                  f"{target.get('hardware')})")
        else:
            print(f"  Target     : {target_name} ({target.get('user')}@{target.get('host')}, "
                  f"{target.get('hardware')})")

        result = None
        verify_cost = 0.0
        verify_turns = 0

        # ── Phase: Verify ────────────────────────────────────────────────
        if args.mode in ("verify", "full"):
            print("\n  ── VERIFY ──")
            try:
                result = run_notebook(nb, rel_key, target_name, target, req,
                                      interactive=args.interactive,
                                      skip_env_check=args.skip_env_check, run_id=run_id)
            except Exception as exc:  # noqa: BLE001 — record the crash, keep the batch going
                result = blocked(nb, "verify", f"Runner crashed: {exc}")

            _db_call(db, results_db.record_result, run_id, rel_key, str(nb),
                     "verify", result, target_name)

            verify_cost  = result.get("cost_usd") or 0.0
            verify_turns = result.get("num_turns") or 0

            v_status = result.get("status", "fail")
            v_icon   = "✓" if v_status == "pass" else "✗"
            print(f"  {v_icon} VERIFY: {v_status.upper()}", flush=True)

        # ── Phase: Fix ───────────────────────────────────────────────────
        if args.mode == "fix" or (args.mode == "full" and result
                                   and result.get("status") != "pass"):
            # Determine verification result path
            if args.mode == "fix" and args.verification_result:
                vr_path = Path(args.verification_result)
            else:
                vr_path = _find_latest_result_path(nb)

            if vr_path and vr_path.exists():
                print("\n  ── FIX ──")
                try:
                    fix_result = run_fixer(nb, target_name, target, req, vr_path, run_id=run_id)
                    # Recorded before cost accumulation: each DB row carries
                    # its own phase's cost, so SUM(cost_usd) never double-counts.
                    _db_call(db, results_db.record_result, run_id, rel_key,
                             str(nb), "fix", fix_result, target_name)
                    # Accumulate costs from both agents
                    fix_cost  = fix_result.get("cost_usd") or 0.0
                    fix_turns = fix_result.get("num_turns") or 0
                    fix_result["cost_usd"]  = verify_cost + fix_cost
                    fix_result["num_turns"] = verify_turns + fix_turns
                    result = fix_result

                    f_status = result.get("status", "fail")
                    f_icon   = "✓" if f_status == "pass" else "✗"
                    print(f"  {f_icon} FIX: {f_status.upper()}", flush=True)
                except Exception as exc:  # noqa: BLE001 — record the crash, keep the batch going
                    print(f"  ✗ Fixer crashed: {exc}", flush=True)
                    _db_call(db, results_db.record_result, run_id, rel_key, str(nb), "fix",
                             blocked(nb, "fix", f"Fixer crashed: {exc}"), target_name)
            elif args.mode == "fix":
                print(f"  ✗ No verification result found for {nb.name}", flush=True)
                result = blocked(nb, "fix", "No verification result available for fixer")
                _db_call(db, results_db.record_result, run_id, rel_key,
                         str(nb), "fix", result, target_name)

        if result is None:
            result = blocked(nb, args.mode, "No result produced")

        results.append({"notebook": str(nb), **result})
        status = result.get("status", "fail")
        cost   = result.get("cost_usd")
        turns  = result.get("num_turns")
        cost_str  = f"  cost=${cost:.4f}" if cost else ""
        turns_str = f"  turns={turns}"    if turns else ""
        icon   = "✓" if status == "pass" else "✗"
        print(f"\n  {icon} {status.upper()}{cost_str}{turns_str}\n", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = [r for r in results if r.get("status") == "pass"]
    failed = [r for r in results if r.get("status") == "fail"]
    blocked_runs = [r for r in results if r.get("status") not in ("pass", "fail")]
    total_cost = sum(r.get("cost_usd") or 0 for r in results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed  : {len(passed)}/{len(results)}")
    print(f"  Failed  : {len(failed)}/{len(results)}")
    if blocked_runs:
        print(f"  Blocked : {len(blocked_runs)}/{len(results)}")
    if skipped:
        print(f"  No target : {len(skipped)}")
    if total_cost:
        print(f"  Total cost : ${total_cost:.4f}")
    if failed or blocked_runs:
        print("\nNot passing:")
        for r in failed + blocked_runs:
            print(f"  ✗ {Path(r['notebook']).name} ({r.get('status')})")
            if r.get("message"):
                print(f"    {r['message']}")

    _db_call(db, results_db.finish_run, run_id)
    if db is not None:
        db.close()
        if args.notify:
            _notify_teams(run_id)
    elif args.notify:
        print("[teams] skipped — the card is built from the results DB, which "
              "could not be opened", file=sys.stderr)

    sys.exit(0 if not (failed or blocked_runs) else 1)


if __name__ == "__main__":
    main()
