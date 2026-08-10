"""
Environment validation for a notebook before the agent runs.

Runs three cheap checks against the local notebook and remote GPU server:
HF token gated-model access, GPU count, disk space. On a blocking failure
with --write-result, invokes tools/write_result.py so the dashboard surfaces
the failure immediately instead of leaving the notebook never-tested.

This is launcher-level environment validation. It never modifies notebook
content and is invisible to the agent.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "manifest.yaml"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
WRITE_RESULT_SCRIPT = REPO_ROOT / "tools" / "write_result.py"

GATED_MODEL_RE = re.compile(
    r"\b(meta-llama|mistralai|google|microsoft|deepseek-ai)/[A-Za-z0-9._-]+"
)
HTTP_TIMEOUT = 10
SSH_TIMEOUT = 15
USER_AGENT = "notebook-runner-env-check/0.1"
MIN_DISK_GB = 100


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        warn(f"manifest not found at {path}")
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("notebooks", {}) or {}


def extract_model_ids(notebook_path: Path) -> list[str]:
    with open(notebook_path) as f:
        nb = json.load(f)
    found: list[str] = []
    seen: set[str] = set()
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for match in GATED_MODEL_RE.finditer(src):
            mid = match.group(0)
            if mid not in seen:
                seen.add(mid)
                found.append(mid)
    return found


def hf_check_model(model_id: str, token: str) -> tuple[int | None, str]:
    """Return (http_status_or_None, message).

    Uses HEAD on the file-resolve endpoint, which enforces gated-repo
    download permissions — unlike the metadata API which returns 200 even
    for models the token cannot download.
    """
    req = urllib.request.Request(
        f"https://huggingface.co/{model_id}/resolve/main/config.json",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        },
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, "ok"
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except (urllib.error.URLError, TimeoutError) as e:
        return None, f"network error: {e}"


def ssh_run(ssh_cmd: str, remote_cmd: str) -> tuple[int, str, str]:
    """Run a command via ssh. Returns (returncode, stdout, stderr)."""
    full = f"{ssh_cmd} {remote_cmd!r}"
    proc = subprocess.run(
        full, shell=True, capture_output=True, text=True, timeout=SSH_TIMEOUT
    )
    return proc.returncode, proc.stdout, proc.stderr


class Check:
    def __init__(self, name: str):
        self.name = name
        self.status: str = "pending"   # ok | fail | skipped | warning
        self.reason: str = ""
        self.issue: dict | None = None  # populated on fail

    def ok(self) -> None:
        self.status = "ok"

    def fail(self, reason: str, issue: dict) -> None:
        self.status = "fail"
        self.reason = reason
        self.issue = issue

    def skip(self, reason: str) -> None:
        self.status = "skipped"
        self.reason = reason

    def warn(self, reason: str) -> None:
        self.status = "warning"
        self.reason = reason

    def print_result(self) -> None:
        if self.status == "ok":
            print("  ✓ ok")
        elif self.status == "fail":
            print(f"  ✗ FAIL: {self.reason}")
        elif self.status == "skipped":
            print(f"  ! SKIPPED: {self.reason}")
        elif self.status == "warning":
            print(f"  ! WARNING: {self.reason} (not blocking)")
        else:
            print(f"  ? {self.status}: {self.reason}")


def run_check_hf(notebook_local: Path, hf_token: str | None) -> Check:
    chk = Check("HF token gated-model access")
    if not hf_token:
        chk.skip("HF_TOKEN not set")
        warn("HF_TOKEN env var not set; skipping HF gated-model check")
        return chk

    models = extract_model_ids(notebook_local)
    if not models:
        print("    no gated models referenced")
        chk.ok()
        return chk

    blocking_reason: str | None = None
    blocking_issue: dict | None = None
    nonblocking_notes: list[str] = []

    for mid in models:
        status, msg = hf_check_model(mid, hf_token)
        if status == 200:
            print(f"    {mid}: ok")
        elif status == 401:
            reason = f"HF_TOKEN is invalid or expired (HTTP 401 for {mid})"
            blocking_reason = reason
            blocking_issue = {
                "cell_index": None,
                "error_type": "missing_dependency",
                "description": reason,
                "proposed_fix": "Set HF_TOKEN to a valid HuggingFace user access token. Generate one at https://huggingface.co/settings/tokens",
            }
            print(f"    {mid}: HTTP 401 (token invalid or expired)")
            break
        elif status == 403:
            reason = f"HF_TOKEN lacks download access to gated model {mid} (HTTP 403)"
            blocking_reason = reason
            blocking_issue = {
                "cell_index": None,
                "error_type": "missing_dependency",
                "description": reason,
                "proposed_fix": f"Request access at https://huggingface.co/{mid} and wait for approval, then re-run.",
            }
            print(f"    {mid}: HTTP 403 (no download access — gated model)")
            break
        elif status == 404:
            note = f"{mid}: HTTP 404 (config.json not found — model ID may be a regex false positive or the model may have been renamed/removed)"
            nonblocking_notes.append(note)
            print(f"    {note}")
        elif status == 429:
            note = f"{mid}: HTTP 429 (rate-limited)"
            nonblocking_notes.append(note)
            warn(f"HF API rate limited while checking {mid}; not blocking")
            print(f"    {note}")
        else:
            note = f"{mid}: unexpected response status={status} msg={msg}"
            nonblocking_notes.append(note)
            warn(note)
            print(f"    {note}")

    if blocking_reason and blocking_issue:
        chk.fail(blocking_reason, blocking_issue)
    else:
        chk.ok()
    return chk


def run_check_gpu(ssh_cmd: str, gpus_required: int) -> Check:
    chk = Check("GPU count")
    remote = "ls /dev/dri/renderD* 2>/dev/null | wc -l"
    try:
        rc, out, serr = ssh_run(ssh_cmd, remote)
    except subprocess.TimeoutExpired:
        chk.warn("SSH timed out")
        warn("SSH timed out during GPU count check")
        return chk
    except Exception as e:
        chk.warn(f"SSH error: {e}")
        warn(f"SSH error during GPU count check: {e}")
        return chk

    if rc != 0:
        chk.warn(f"SSH exit {rc}: {serr.strip()[:120]}")
        warn(f"SSH failed (exit {rc}) during GPU count check; not blocking")
        return chk

    try:
        count = int(out.strip())
    except ValueError:
        chk.warn(f"could not parse GPU count from output: {out.strip()[:80]!r}")
        warn(f"could not parse GPU count from output: {out.strip()[:80]!r}")
        return chk

    print(f"    found {count} GPU(s); notebook requires {gpus_required}")
    if count < gpus_required:
        reason = f"only {count} GPU(s) available, notebook requires {gpus_required}"
        chk.fail(reason, {
            "cell_index": None,
            "error_type": "missing_dependency",
            "description": reason,
            "proposed_fix": f"Run this notebook on a server with at least {gpus_required} GPU(s), or update the manifest gpus_required field.",
        })
    else:
        chk.ok()
    return chk


def run_check_disk(ssh_cmd: str) -> Check:
    chk = Check("Disk space")
    remote = "df -BG --output=avail /home/amd | tail -1 | tr -dc '0-9'"
    try:
        rc, out, serr = ssh_run(ssh_cmd, remote)
    except subprocess.TimeoutExpired:
        chk.warn("SSH timed out")
        warn("SSH timed out during disk space check")
        return chk
    except Exception as e:
        chk.warn(f"SSH error: {e}")
        warn(f"SSH error during disk space check: {e}")
        return chk

    if rc != 0:
        chk.warn(f"SSH exit {rc}: {serr.strip()[:120]}")
        warn(f"SSH failed (exit {rc}) during disk space check; not blocking")
        return chk

    try:
        avail_gb = int(out.strip())
    except ValueError:
        chk.warn(f"could not parse disk avail from output: {out.strip()[:80]!r}")
        warn(f"could not parse disk avail from output: {out.strip()[:80]!r}")
        return chk

    print(f"    {avail_gb} GB free on /home/amd; require >= {MIN_DISK_GB} GB")
    if avail_gb < MIN_DISK_GB:
        reason = f"only {avail_gb} GB free on /home/amd, need >= {MIN_DISK_GB} GB"
        chk.fail(reason, {
            "cell_index": None,
            "error_type": "missing_dependency",
            "description": reason,
            "proposed_fix": f"Free disk space on the server until /home/amd has at least {MIN_DISK_GB} GB available.",
        })
    else:
        chk.ok()
    return chk


def write_failure_result(
    notebook_local: Path,
    issues: list[dict],
    first_reason: str,
    results_dir: Path,
) -> None:
    summary = f"Blocked by environment check: {first_reason}. The agent did not run."
    cmd = [
        sys.executable,
        str(WRITE_RESULT_SCRIPT),
        "--notebook", str(notebook_local),
        "--status", "fail",
        "--summary", summary,
        "--issues", json.dumps(issues),
        "--fixes", "[]",
        "--results-dir", str(results_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err(f"write_result.py failed (exit {proc.returncode}): {proc.stderr.strip()}")
    else:
        out_path = proc.stdout.strip()
        print(f"  wrote result: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Environment check for a notebook before the agent runs.")
    p.add_argument("--notebook",       required=True, help="Notebook path as it appears in the manifest (e.g. inference/foo.ipynb)")
    p.add_argument("--notebook-local", required=True, help="Absolute path to the local .ipynb file to scan")
    p.add_argument("--ssh-cmd",        default=None,  help="Full SSH prefix (e.g. 'ssh -o ... user@host')")
    p.add_argument("--manifest",       default=str(DEFAULT_MANIFEST), help="Path to manifest.yaml")
    p.add_argument("--results-dir",    default=str(DEFAULT_RESULTS_DIR), help="Where to write result JSON on failure")
    p.add_argument("--write-result",   action="store_true", help="Write a result JSON on blocking failure")
    p.add_argument("--hf-token",       default=None, help="HuggingFace token (default: $HF_TOKEN)")
    args = p.parse_args()

    notebook_local = Path(args.notebook_local)
    if not notebook_local.exists():
        err(f"notebook file does not exist: {notebook_local}")
        return 2

    try:
        with open(notebook_local) as f:
            json.load(f)
    except json.JSONDecodeError as e:
        err(f"notebook is not valid JSON: {e}")
        return 2
    except OSError as e:
        err(f"could not read notebook: {e}")
        return 2

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    entry = manifest.get(args.notebook)
    if entry is None:
        warn(f"notebook '{args.notebook}' not in manifest; defaulting gpus_required=1")
        entry = {}
    gpus_required = int(entry.get("gpus_required", 1) or 1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    results_dir = Path(args.results_dir)

    checks: list[Check] = []

    print("CHECK 1/3: HF token gated-model access...")
    chk1 = run_check_hf(notebook_local, hf_token)
    chk1.print_result()
    checks.append(chk1)

    print("CHECK 2/3: GPU count...")
    if not args.ssh_cmd:
        chk2 = Check("GPU count")
        chk2.skip("--ssh-cmd not provided")
        warn("--ssh-cmd not provided; skipping GPU and disk checks")
        chk2.print_result()
    else:
        chk2 = run_check_gpu(args.ssh_cmd, gpus_required)
        chk2.print_result()
    checks.append(chk2)

    print("CHECK 3/3: Disk space...")
    if not args.ssh_cmd:
        chk3 = Check("Disk space")
        chk3.skip("--ssh-cmd not provided")
        chk3.print_result()
    else:
        chk3 = run_check_disk(args.ssh_cmd)
        chk3.print_result()
    checks.append(chk3)

    blocking = [c for c in checks if c.status == "fail"]
    if blocking:
        first = blocking[0]
        print(f"ENV CHECK FAILED: {first.reason}")
        if args.write_result:
            issues = [c.issue for c in blocking if c.issue]
            write_failure_result(notebook_local, issues, first.reason, results_dir)
        return 1

    print("ENV CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
