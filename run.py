"""
Notebook regression agent — Claude Code edition.

Identical CLI to the original run.py. Swaps agent.run() for claude -p.

Usage:
  # Single notebook
  python run.py path/to/notebook.ipynb

  # All notebooks in a directory
  python run.py --dir /path/to/notebooks/

  # Specific category only
  python run.py --dir /path/to/notebooks/ --category inference

  # Interactive mode (opens claude normally — no -p flag, full conversation)
  python run.py path/to/notebook.ipynb --interactive

  # Run only the gap set (never-tested + stale per tools/status.py --gap)
  python run.py --dir /path/to/notebooks/ --gap

  # Gap set, filtered to one category
  python run.py --dir /path/to/notebooks/ --gap --category inference

Options:
  --dir         Run all notebooks under this directory
  --category    Filter notebooks by subdirectory name
  --manifest    Path to manifest YAML (default: manifest.yaml next to this file)
  --log-level   DEBUG | INFO (default: INFO)
  --interactive Open claude interactively instead of headless
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

HERE        = Path(__file__).parent
TOOLS_DIR   = (HERE / "tools").resolve()
RESULTS_DIR = (HERE / "results").resolve()
FIXES_DIR   = (HERE / "fixes").resolve()   # genuine-fix notebooks retrieved by the fixer (for PRs)

# Remote workspace base for per-notebook run dirs. Defaults to /home/amd for the
# tw* servers (where amd is the login user); override with WORKSPACE_BASE on nodes
# that use a different home (e.g. /home/gh-runner on the self-hosted CI box).
WORKSPACE_BASE = os.getenv("WORKSPACE_BASE", "/home/amd").rstrip("/")

# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_notebook_config(nb_path, base_dir, manifest, cli_host, cli_user):
    if base_dir:
        try:
            rel_key = str(nb_path.relative_to(base_dir))
        except ValueError:
            rel_key = nb_path.name
    else:
        rel_key = nb_path.name

    servers   = manifest.get("servers", {})
    notebooks = manifest.get("notebooks", {})
    entry     = notebooks.get(rel_key) or notebooks.get(nb_path.name) or {}

    if entry.get("skip"):
        return None

    host = cli_host or os.getenv("GPU_HOST")
    user = cli_user or os.getenv("GPU_USER")

    server_hardware = None
    server_name = entry.get("server")
    if server_name and server_name in servers:
        srv = servers[server_name]
        host = srv.get("host", host)
        user = srv.get("user", user)
        server_hardware = srv.get("hardware")

    return {"host": host, "user": user,
            "server_hardware": server_hardware, "manifest_entry": entry,
            "rel_key": rel_key}


def filter_by_max_gpus(notebooks, base_dir, manifest, max_gpus):
    """Split notebooks into (kept, dropped) by manifest gpus_required <= max_gpus.

    dropped is a list of (path, gpus_required) so the caller can report exactly
    what was skipped — never silently truncate the run. Notebooks absent from the
    manifest (or with no gpus_required) default to 1, i.e. runnable on any node.
    """
    if max_gpus is None:
        return list(notebooks), []
    nbs = manifest.get("notebooks", {})
    kept, dropped = [], []
    for nb in notebooks:
        if base_dir:
            try:
                rel_key = str(nb.resolve().relative_to(Path(base_dir).resolve()))
            except ValueError:
                rel_key = nb.name
        else:
            rel_key = nb.name
        entry = nbs.get(rel_key) or nbs.get(nb.name) or {}
        req = entry.get("gpus_required", 1)
        if req is None:
            req = 1
        if req > max_gpus:
            dropped.append((nb, req))
        else:
            kept.append(nb)
    return kept, dropped


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
    # Return resolved absolute paths so nb.relative_to(base_dir.resolve()) works
    # downstream — matches the positional/gap/failing selectors, which already
    # resolve. Without this, manifest lookups (skip/expected_result/server/gpus)
    # silently miss on --dir runs because rel_key falls back to the bare name.
    return [p.resolve() for p in notebooks]


def collect_gap_notebooks(base_dir: Path, category: str | None) -> list[Path]:
    """Subprocess tools/status.py --gap and resolve each line against base_dir."""
    proc = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "status.py"), "--gap"],
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
        print("warning: gap entries not found on disk (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if category:
        result = [n for n in result if n.relative_to(base_dir).parts[0] == category]

    return result


def collect_failing_notebooks(base_dir: Path, category: str | None) -> list[Path]:
    """Subprocess tools/status.py --failing and resolve each line against base_dir.

    Mirrors collect_gap_notebooks(), but targets notebooks whose latest result
    is a hard failure (or a partial that does not count as pass).
    """
    proc = subprocess.run(
        [sys.executable, str(TOOLS_DIR / "status.py"), "--failing"],
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
        print("warning: failing entries not found on disk (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if category:
        result = [n for n in result if n.relative_to(base_dir).parts[0] == category]

    return result


# ── Notebook transfer (same as original agent) ────────────────────────────────

def scp_notebook(local_path: Path, host: str, user: str) -> str:
    """Copy notebook and preflight_patch.py to remote server. Returns remote path."""
    stem       = local_path.stem
    # Fixed workspace base (WORKSPACE_BASE, not /home/{user}) so it matches the
    # literal <base>/tutorial_agent_runs/<stem> paths the agent uses from the
    # runtime context, regardless of the SSH user. Defaults to /home/amd for the
    # tw* servers; the self-hosted CI box sets WORKSPACE_BASE=/home/gh-runner.
    remote_dir = f"{WORKSPACE_BASE}/tutorial_agent_runs/{stem}"
    remote_path = f"{remote_dir}/{local_path.name}"
    target     = f"{user}@{host}"
    preflight  = TOOLS_DIR / "preflight_patch.py"

    print(f"  Copying {local_path.name} → {target}:{remote_dir}/", flush=True)
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
                 "-o", "ForwardX11=no", target, f"mkdir -p {remote_dir}"],
                check=True, capture_output=True, timeout=60,
            )
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ForwardX11=no",
                 str(local_path), str(preflight), f"{target}:{remote_dir}/"],
                check=True, capture_output=True, timeout=60,
            )
            return remote_path
        except Exception as exc:
            if attempt < 3:
                print(f"  SCP attempt {attempt} failed ({exc}), retrying in 15s…", flush=True)
                time.sleep(15)
            else:
                raise RuntimeError(f"Failed to copy notebook after 3 attempts: {exc}") from exc


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
                        print(f" ⏳", end="", flush=True)
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
            meta["cost_usd"]  = event.get("cost_usd", 0.0)
            meta["num_turns"] = event.get("num_turns", 0)
            if event.get("subtype") == "error_max_turns":
                print(f"\n  ✗ Hit max turns limit", flush=True)

    proc.wait()
    return meta


# ── Environment check ─────────────────────────────────────────────────────────

def run_env_check(
    notebook_path: Path,
    rel_key: str,
    host: str,
    user: str,
    hf_token: str | None,
    manifest_path: Path,
    skip: bool = False,
) -> dict | None:
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
        "--ssh-cmd",        (f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
                             f"-o ForwardX11=no {user}@{host}"),
        "--manifest",       str(manifest_path),
        "--results-dir",    str(RESULTS_DIR),
        "--write-result",
    ]
    if hf_token:
        cmd += ["--hf-token", hf_token]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)
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
        result = _find_latest_result(notebook_path)
        if result is not None:
            return result
        return {"status": "fail",
                "message": "env_check blocked the run but no result file was written",
                "cost_usd": None, "num_turns": 0}
    else:
        print(f"warning: env_check itself errored (exit {rc}) — proceeding with agent run",
              file=sys.stderr)
        return None


# ── Core runner ───────────────────────────────────────────────────────────────

def run_notebook(notebook_path: Path, host: str, user: str,
                 hf_token: str | None, manifest_entry: dict,
                 server_hardware: str | None, interactive: bool,
                 rel_key: str = "", manifest_path: Path | None = None,
                 skip_env_check: bool = False) -> dict:

    # 0. Environment check — fail fast on unwinnable cases
    env_blocked = run_env_check(
        notebook_path = notebook_path,
        rel_key       = rel_key,
        host          = host,
        user          = user,
        hf_token      = hf_token,
        manifest_path = manifest_path or HERE / "manifest.yaml",
        skip          = skip_env_check,
    )
    if env_blocked is not None:
        return env_blocked

    # 1. Copy notebook to remote server (infrastructure — not the agent's job)
    try:
        remote_path = scp_notebook(notebook_path, host, user)
    except RuntimeError as e:
        return {"status": "fail", "message": str(e), "cost_usd": None}

    # 2. Build the task prompt (same information as original context.py)
    ssh_flags  = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ForwardX11=no"
    ssh_target = f"{user}@{host}"
    ssh_cmd    = f"ssh {ssh_flags} {ssh_target}"
    hf_export  = f"export HF_TOKEN='{hf_token}';" if hf_token else ""
    ssh_hf_cmd = f"ssh {ssh_flags} {ssh_target} '{hf_export}"

    prompt_lines = [
        f"Run the regression test for this notebook: {notebook_path}",
        "",
        "Follow the playbook in CLAUDE.md exactly. Start by reading the notebook "
        "with the Read tool, then proceed through all phases. When done, call "
        "write_result.py then finish. Do not retry content failures — report them accurately.",
    ]
    if manifest_entry:
        prompt_lines += ["", "## Manifest overrides"]
        docker_overrides = manifest_entry.get("docker_overrides")
        if docker_overrides:
            prompt_lines.append("Docker image overrides (regex → replacement):")
            for pat, rep in docker_overrides.items():
                prompt_lines.append(f"  - `{pat}` → `{rep}`")
        expected = manifest_entry.get("expected_result")
        if expected == "partial":
            prompt_lines.append(
                "expected_result=partial: a 'partial' outcome counts as passing. "
                "Do not retry indefinitely if only a secondary check fails."
            )
        notes = manifest_entry.get("notes")
        if notes:
            prompt_lines.append(f"Known issues / notes: {notes.strip()}")

    prompt = "\n".join(prompt_lines)

    # 3. Runtime context injected into system prompt — replaces SYSTEM_TEMPLATE.format()
    runtime_ctx = f"""## Runtime Context for this Run

SSH_CMD     = {ssh_cmd}
SSH_HF_CMD  = ssh {ssh_flags} {ssh_target} (prefix commands that need HF_TOKEN with: export HF_TOKEN='{hf_token}';)
GPU_HARDWARE   = {server_hardware or 'unknown'}
NOTEBOOK_LOCAL  = {notebook_path}
NOTEBOOK_REMOTE = {remote_path}  (already on the server — do NOT copy again)
WORKSPACE_BASE  = {WORKSPACE_BASE}  (per-notebook run dir is {WORKSPACE_BASE}/tutorial_agent_runs/<stem>)
TOOLS_DIR       = {TOOLS_DIR}
RESULTS_DIR     = {RESULTS_DIR}

Use SSH_CMD like this:
  Bash: {ssh_cmd} '<your command here>'

For commands that need HF_TOKEN:
  Bash: {ssh_cmd} 'export HF_TOKEN='{hf_token}'; <your command here>'

resolve_docker_image:
  Bash: python3 {TOOLS_DIR}/resolve_docker_image.py <repo> {server_hardware or 'mi300x'}

write_result (call this when done):
  Bash: python3 {TOOLS_DIR}/write_result.py \\
    --notebook "{notebook_path}" \\
    --status pass|fail|partial \\
    --summary "..." \\
    --issues '[...]' \\
    --fixes  '[...]' \\
    --results-dir "{RESULTS_DIR}"
"""

    # 4. Interactive mode: drop -p flag, let user converse with claude normally
    if interactive:
        print(f"\n  Opening interactive Claude session for {notebook_path.name}")
        print(f"  Runtime context is in the system prompt. Type your instructions.\n")
        subprocess.run(
            ["claude", "--append-system-prompt", runtime_ctx],
            cwd=str(HERE)
        )
        # In interactive mode, find the result file manually after session ends
        result = _find_latest_result(notebook_path)
        return result or {"status": "unknown", "message": "Interactive session ended"}

    # 5. Headless mode: fully autonomous
    print(f"\n  Starting Claude Code agent (headless)…", flush=True)
    proc = subprocess.Popen(
        [
            "claude",
            "-p", prompt,
            "--append-system-prompt", runtime_ctx,
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "200",
            "--dangerously-skip-permissions",
        ],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,   # show claude errors live instead of swallowing them
        text=True,
        cwd=str(HERE),
    )

    meta = handle_stream(proc, notebook_path.name, agent_label="VERIFY")

    # 6. Read back the result JSON written by write_result.py
    result = _find_latest_result(notebook_path)
    if result:
        result["cost_usd"]  = meta.get("cost_usd")
        result["num_turns"] = meta.get("num_turns")
        return result

    return {
        "status":    "fail",
        "message":   "Agent finished but did not write a result file",
        "cost_usd":  meta.get("cost_usd"),
        "num_turns": meta.get("num_turns"),
    }


def run_fixer(notebook_path: Path, host: str, user: str,
              hf_token: str | None, manifest_entry: dict,
              server_hardware: str | None,
              verification_result_path: Path,
              rel_key: str = "") -> dict:
    """Run the fixer agent on a notebook using a prior verification result."""

    FIXES_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Check for auto_fixable issues — skip if none
    if not _has_auto_fixable_issues(verification_result_path):
        print("  No auto_fixable issues — skipping fixer agent", flush=True)
        result = _find_latest_result(notebook_path)
        return result or {"status": "fail", "message": "No auto_fixable issues"}

    # 1. Copy notebook to remote server (verification cleaned up)
    try:
        remote_path = scp_notebook(notebook_path, host, user)
    except RuntimeError as e:
        return {"status": "fail", "message": str(e), "cost_usd": None}

    # 2. Build the fixer prompt
    prompt_lines = [
        f"Fix the issues identified by the verification agent for: {notebook_path}",
        "",
        f"The verification result is at: {verification_result_path}",
        "",
        "Follow the CLAUDE_FIX.md playbook provided in the system prompt below.",
        "Start by reading the verification result JSON and the notebook,",
        "then proceed through all steps. Only fix issues marked auto_fixable.",
        "Write the final result via write_result.py, then finish.",
    ]
    if manifest_entry:
        expected = manifest_entry.get("expected_result")
        if expected == "partial":
            prompt_lines.append(
                "\nexpected_result=partial: a 'partial' outcome counts as passing."
            )
        notes = manifest_entry.get("notes")
        if notes:
            prompt_lines.append(f"\nKnown issues / notes: {notes.strip()}")

    prompt = "\n".join(prompt_lines)

    # 3. Build runtime context (same as verification + VERIFICATION_RESULT)
    ssh_flags  = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ForwardX11=no"
    ssh_target = f"{user}@{host}"
    ssh_cmd    = f"ssh {ssh_flags} {ssh_target}"

    runtime_ctx = f"""## Runtime Context for this Fix Run

SSH_CMD     = {ssh_cmd}
SSH_HF_CMD  = ssh {ssh_flags} {ssh_target} (prefix commands that need HF_TOKEN with: export HF_TOKEN='{hf_token}';)
GPU_HARDWARE   = {server_hardware or 'unknown'}
NOTEBOOK_LOCAL  = {notebook_path}
NOTEBOOK_REMOTE = {remote_path}  (already on the server — do NOT copy again)
WORKSPACE_BASE  = {WORKSPACE_BASE}  (per-notebook run dir is {WORKSPACE_BASE}/tutorial_agent_runs/<stem>)
TOOLS_DIR       = {TOOLS_DIR}
RESULTS_DIR     = {RESULTS_DIR}
FIXES_DIR       = {FIXES_DIR}  (retrieve <nb>_fixed.ipynb here as <stem>.ipynb — see CLAUDE_FIX.md Step 4d)
VERIFICATION_RESULT = {verification_result_path}

Use SSH_CMD like this:
  Bash: {ssh_cmd} '<your command here>'

For commands that need HF_TOKEN:
  Bash: {ssh_cmd} 'export HF_TOKEN='{hf_token}'; <your command here>'

resolve_docker_image:
  Bash: python3 {TOOLS_DIR}/resolve_docker_image.py <repo> {server_hardware or 'mi300x'}

write_result (call this when done):
  Bash: python3 {TOOLS_DIR}/write_result.py \\
    --notebook "{notebook_path}" \\
    --status pass|fail \\
    --summary "..." \\
    --issues '<all issues>' \\
    --fixes  '<fixes array>' \\
    --agent "claude_code_fix" \\
    --verification-result "{verification_result_path}" \\
    --results-dir "{RESULTS_DIR}"
"""

    # 4. Read CLAUDE_FIX.md and append to system prompt
    fixer_playbook_path = HERE / "CLAUDE_FIX.md"
    with open(fixer_playbook_path) as f:
        fixer_playbook = f.read()

    system_prompt = fixer_playbook + "\n\n---\n\n" + runtime_ctx

    # 5. Spawn headless fixer agent
    print(f"\n  Starting fixer agent (headless)…", flush=True)
    proc = subprocess.Popen(
        [
            "claude",
            "-p", prompt,
            "--append-system-prompt", system_prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--max-turns", "200",
            "--dangerously-skip-permissions",
        ],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        cwd=str(HERE),
    )

    meta = handle_stream(proc, notebook_path.name, agent_label="FIX")

    # 6. Read back the result JSON
    result = _find_latest_result(notebook_path)
    if result:
        result["cost_usd"]  = meta.get("cost_usd")
        result["num_turns"] = meta.get("num_turns")
        return result

    return {
        "status":    "fail",
        "message":   "Fixer agent finished but did not write a result file",
        "cost_usd":  meta.get("cost_usd"),
        "num_turns": meta.get("num_turns"),
    }


def _find_latest_result(notebook_path: Path) -> dict | None:
    """Find the most recent result JSON written by write_result.py for this notebook."""
    result_path = _find_latest_result_path(notebook_path)
    if result_path is None:
        return None
    with open(result_path) as f:
        return json.load(f)


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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Notebook regression agent — Claude Code edition")
    p.add_argument("notebook", nargs="*")
    p.add_argument("--host",        default=None)
    p.add_argument("--user",        default=None)
    p.add_argument("--dir",         help="Run all notebooks under this directory")
    p.add_argument("--manifest",    default=str(HERE / "manifest.yaml"))
    p.add_argument("--category",    choices=["inference", "fine_tune", "pretrain", "gpu_dev_optimize"])
    p.add_argument("--max-gpus",    type=int, default=None,
                   help="Only run notebooks whose manifest gpus_required is <= this. "
                        "Notebooks needing more GPUs are skipped (reported, not failed). "
                        "Notebooks with no gpus_required in the manifest default to 1.")
    p.add_argument("--log-level",   default="INFO", choices=["DEBUG", "INFO"])
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
    return p.parse_args()


def main():
    args     = parse_args()
    manifest = load_manifest(Path(args.manifest))
    base_dir = Path(args.dir).resolve() if args.dir else None

    if args.gap and args.failing:
        print("Error: --gap and --failing are mutually exclusive", file=sys.stderr)
        sys.exit(2)
    if (args.gap or args.failing) and args.notebook:
        print("Error: --gap/--failing cannot be combined with positional notebook paths",
              file=sys.stderr)
        sys.exit(2)
    if (args.gap or args.failing) and base_dir is None:
        print("Error: --gap/--failing requires --dir to resolve manifest-relative paths to local files",
              file=sys.stderr)
        sys.exit(2)

    if args.gap:
        notebooks = collect_gap_notebooks(base_dir, args.category)
        if not notebooks:
            if args.category:
                print(f"\nNo gap notebooks in category '{args.category}'.")
            else:
                print("\nGap is empty — every testable notebook has a recent result. Nothing to do.")
            sys.exit(0)
    elif args.failing:
        notebooks = collect_failing_notebooks(base_dir, args.category)
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

    # Filter by GPU capacity of the target node (e.g. --max-gpus 1 for a 1-GPU box).
    if args.max_gpus is not None:
        notebooks, dropped = filter_by_max_gpus(notebooks, base_dir, manifest, args.max_gpus)
        if dropped:
            print(f"Skipping {len(dropped)} notebook(s) needing > {args.max_gpus} GPU(s):")
            for nb, req in sorted(dropped, key=lambda d: d[0].name):
                print(f"  - {nb.name} (needs {req})")
            print()
        if not notebooks:
            print(f"No notebooks require <= {args.max_gpus} GPU(s). Nothing to do.")
            sys.exit(0)

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

    print(f"\nNotebook regression agent  [Claude Code]")
    print(f"  Notebooks : {len(notebooks)}")
    print(f"  Mode      : {mode_label}")
    print(f"  Results   : {RESULTS_DIR}\n")

    results = []
    skipped = []

    for nb in notebooks:
        print(f"{'='*60}")
        print(f"Testing: {nb.name}")
        print(f"{'='*60}")

        cfg = resolve_notebook_config(nb, base_dir, manifest, args.host, args.user)
        if cfg is None:
            entry  = (manifest.get("notebooks", {})
                      .get(str(nb.relative_to(base_dir)) if base_dir else nb.name, {}))
            reason = entry.get("skip_reason", "no reason given")
            print(f"  SKIP — {reason}\n")
            skipped.append({"notebook": str(nb), "reason": reason})
            continue

        host = cfg["host"]
        user = cfg["user"]
        if not host or not user:
            print("Error: no host/user. Set GPU_HOST/GPU_USER in .env or pass --host/--user",
                  file=sys.stderr)
            sys.exit(1)

        print(f"  GPU server : {user}@{host}")

        result = None
        verify_cost = 0.0
        verify_turns = 0

        # ── Phase: Verify ────────────────────────────────────────────────
        if args.mode in ("verify", "full"):
            print(f"\n  ── VERIFY ──")
            try:
                result = run_notebook(
                    notebook_path   = nb,
                    host            = host,
                    user            = user,
                    hf_token        = os.getenv("HF_TOKEN"),
                    manifest_entry  = cfg["manifest_entry"],
                    server_hardware = cfg["server_hardware"],
                    interactive     = args.interactive,
                    rel_key         = cfg["rel_key"],
                    manifest_path   = Path(args.manifest),
                    skip_env_check  = args.skip_env_check,
                )
            except Exception as exc:
                result = {"status": "fail", "message": f"Runner crashed: {exc}"}

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
                print(f"\n  ── FIX ──")
                try:
                    fix_result = run_fixer(
                        notebook_path   = nb,
                        host            = host,
                        user            = user,
                        hf_token        = os.getenv("HF_TOKEN"),
                        manifest_entry  = cfg["manifest_entry"],
                        server_hardware = cfg["server_hardware"],
                        verification_result_path = vr_path,
                        rel_key         = cfg["rel_key"],
                    )
                    # Accumulate costs from both agents
                    fix_cost  = fix_result.get("cost_usd") or 0.0
                    fix_turns = fix_result.get("num_turns") or 0
                    fix_result["cost_usd"]  = verify_cost + fix_cost
                    fix_result["num_turns"] = verify_turns + fix_turns
                    result = fix_result

                    f_status = result.get("status", "fail")
                    f_icon   = "✓" if f_status == "pass" else "✗"
                    print(f"  {f_icon} FIX: {f_status.upper()}", flush=True)
                except Exception as exc:
                    print(f"  ✗ Fixer crashed: {exc}", flush=True)
            elif args.mode == "fix":
                print(f"  ✗ No verification result found for {nb.name}", flush=True)
                result = {"status": "fail",
                          "message": "No verification result available for fixer"}

        if result is None:
            result = {"status": "fail", "message": "No result produced"}

        results.append({"notebook": str(nb), **result})
        status = result.get("status", "fail")
        cost   = result.get("cost_usd")
        turns  = result.get("num_turns")
        cost_str  = f"  cost=${cost:.4f}" if cost else ""
        turns_str = f"  turns={turns}"    if turns else ""
        icon   = "✓" if status == "pass" else ("~" if status == "partial" else "✗")
        print(f"\n  {icon} {status.upper()}{cost_str}{turns_str}\n", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    def _is_pass(r):
        nb_key = Path(r["notebook"]).name
        entry  = next(
            (v for k, v in manifest.get("notebooks", {}).items()
             if Path(k).name == nb_key),
            {}
        ) or {}
        expected = entry.get("expected_result", "pass")
        return r.get("status") == "pass" or (expected == "partial" and r.get("status") == "partial")

    passed = [r for r in results if _is_pass(r)]
    failed = [r for r in results if not _is_pass(r)]
    total_cost = sum(r.get("cost_usd") or 0 for r in results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed  : {len(passed)}/{len(results)}")
    print(f"  Failed  : {len(failed)}/{len(results)}")
    if skipped:
        print(f"  Skipped : {len(skipped)}")
    if total_cost:
        print(f"  Total cost : ${total_cost:.4f}")
    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  ✗ {Path(r['notebook']).name}")
            if r.get("message"):
                print(f"    {r['message']}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
