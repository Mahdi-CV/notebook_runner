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
TOOLS_DIR   = (HERE.parent.parent / "tools").resolve()
RESULTS_DIR = (HERE / "results").resolve()

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
            "server_hardware": server_hardware, "manifest_entry": entry}


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
    return notebooks


# ── Notebook transfer (same as original agent) ────────────────────────────────

def scp_notebook(local_path: Path, host: str, user: str) -> str:
    """Copy notebook to remote server. Returns remote path."""
    stem       = local_path.stem
    remote_dir = f"/home/{user}/tutorial_agent_runs/{stem}"
    remote_path = f"{remote_dir}/{local_path.name}"
    target     = f"{user}@{host}"

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
                 str(local_path), f"{target}:{remote_path}"],
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


def handle_stream(proc, notebook_name: str) -> dict:
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


# ── Core runner ───────────────────────────────────────────────────────────────

def run_notebook(notebook_path: Path, host: str, user: str,
                 hf_token: str | None, manifest_entry: dict,
                 server_hardware: str | None, interactive: bool) -> dict:

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

    meta = handle_stream(proc, notebook_path.name)

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


def _find_latest_result(notebook_path: Path) -> dict | None:
    """Find the most recent result JSON written by write_result.py for this notebook."""
    stem    = notebook_path.stem
    matches = sorted(RESULTS_DIR.glob(f"{stem}_*.json"), reverse=True)
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Notebook regression agent — Claude Code edition")
    p.add_argument("notebook", nargs="*")
    p.add_argument("--host",        default=None)
    p.add_argument("--user",        default=None)
    p.add_argument("--dir",         help="Run all notebooks under this directory")
    p.add_argument("--manifest",    default=str(HERE / "manifest.yaml"))
    p.add_argument("--category",    choices=["inference", "fine_tune", "pretrain", "gpu_dev_optimize"])
    p.add_argument("--log-level",   default="INFO", choices=["DEBUG", "INFO"])
    p.add_argument("--interactive", "-i", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    manifest = load_manifest(Path(args.manifest))
    base_dir = Path(args.dir).resolve() if args.dir else None

    if args.notebook:
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

    print(f"\nNotebook regression agent  [Claude Code]")
    print(f"  Notebooks : {len(notebooks)}")
    print(f"  Mode      : {'interactive' if args.interactive else 'headless (autonomous)'}")
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

        try:
            result = run_notebook(
                notebook_path   = nb,
                host            = host,
                user            = user,
                hf_token        = os.getenv("HF_TOKEN"),
                manifest_entry  = cfg["manifest_entry"],
                server_hardware = cfg["server_hardware"],
                interactive     = args.interactive,
            )
        except Exception as exc:
            result = {"status": "fail", "message": f"Runner crashed: {exc}"}

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
