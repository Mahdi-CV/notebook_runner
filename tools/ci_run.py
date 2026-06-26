"""
CI orchestrator for the notebook regression agent.

One command that runs an automated regression pass and updates the dashboard:

  Phase A — Freshness : run the gap set (never-tested + stale >7d) in --mode full
                        (verify, then auto-fix anything that fails).
  Phase B — Failures  : re-aggregate, then run notebooks that are STILL failing
                        in --mode full. Deduped against Phase A so nothing that
                        already ran this pass runs twice.
  Phase C — Dashboard : regenerate STATUS.md from the fresh results.
  Phase D — Commit    : auto-commit STATUS.md + results/ + manifest.yaml
                        (skip with --no-commit).

Everything is tee'd to logs/ci_<timestamp>.log.

Usage:
  python3 tools/ci_run.py                       # full pass over notebooks/
  python3 tools/ci_run.py --category inference  # one category only
  python3 tools/ci_run.py --no-commit           # update files, don't commit
  python3 tools/ci_run.py --skip-gap            # only attack current failures
  python3 tools/ci_run.py --dry-run             # print the plan, run nothing

Scheduling (deferred — pick one once the baseline looks right):
  - cron on this host (GPU access is via SSH from here):
      # nightly at 02:17 local
      17 2 * * *  cd /path/to/notebook_runner && \
                  ./venv/bin/python tools/ci_run.py >> logs/cron.log 2>&1
  - systemd timer: an equivalent OnCalendar=*-*-* 02:17 unit calling the same line.
  - GitHub Actions: only viable with a self-hosted runner that has SSH access to
    the GPU server plus HF_TOKEN/GPU_HOST/GPU_USER secrets — the GPU box is private.
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RUN_PY = REPO_ROOT / "run.py"
STATUS_PY = HERE / "status.py"
LOGS_DIR = REPO_ROOT / "logs"


class Tee:
    """Write to stdout and a logfile at once."""

    def __init__(self, logfile):
        self.logfile = logfile

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.logfile.write(text)
        self.logfile.flush()

    def line(self, text: str = "") -> None:
        self.write(text + "\n")


def selector_paths(selector: str, base_dir: Path, category: str | None) -> list[Path]:
    """Return resolved local paths from `status.py <selector>` (--gap or --failing)."""
    proc = subprocess.run(
        [sys.executable, str(STATUS_PY), selector],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"status.py {selector} failed (exit {proc.returncode})")

    paths: list[Path] = []
    for line in proc.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if category and rel.split("/", 1)[0] != category:
            continue
        p = (base_dir / rel).resolve()
        if p.exists():
            paths.append(p)
        else:
            sys.stderr.write(f"warning: {selector} entry not on disk, skipped: {rel}\n")
    return paths


def stream(cmd: list[str], tee: Tee) -> int:
    """Run a command, streaming combined output to the tee. Returns exit code."""
    tee.line(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(REPO_ROOT),
    )
    for line in proc.stdout:
        tee.write(line)
    proc.wait()
    return proc.returncode


def run_batch(paths: list[Path], base_dir: Path, tee: Tee, dry_run: bool) -> None:
    cmd = [sys.executable, str(RUN_PY), "--dir", str(base_dir), "--mode", "full",
           *[str(p) for p in paths]]
    if dry_run:
        tee.line(f"[dry-run] would run: {' '.join(cmd)}")
        return
    rc = stream(cmd, tee)
    # run.py exits 1 when any notebook fails — expected, don't abort the pipeline.
    tee.line(f"[batch finished, run.py exit={rc}]")


def git_commit(tee: Tee, ts: str, dry_run: bool) -> None:
    paths = ["STATUS.md", "results/", "manifest.yaml"]
    if dry_run:
        tee.line(f"[dry-run] would: git add {' '.join(paths)} && git commit")
        return
    subprocess.run(["git", "add", *paths], cwd=str(REPO_ROOT), check=False)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=str(REPO_ROOT), check=False
    )
    if staged.returncode == 0:
        tee.line("[commit] nothing changed — skipping commit")
        return
    msg = f"ci: automated regression pass {ts}"
    rc = subprocess.run(
        ["git", "commit", "-m", msg], cwd=str(REPO_ROOT), check=False,
        capture_output=True, text=True,
    )
    tee.write(rc.stdout + rc.stderr)
    tee.line("[commit] done" if rc.returncode == 0 else "[commit] FAILED")


def main() -> int:
    p = argparse.ArgumentParser(description="Automated notebook regression CI pass.")
    p.add_argument("--dir", default=str(REPO_ROOT / "notebooks"),
                   help="Notebook root (default: ./notebooks).")
    p.add_argument("--category",
                   choices=["inference", "fine_tune", "pretrain", "gpu_dev_optimize"])
    p.add_argument("--skip-gap", action="store_true",
                   help="Skip Phase A (freshness); only re-attempt current failures.")
    p.add_argument("--no-commit", action="store_true",
                   help="Update STATUS.md/results but do not git commit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and the batches that would run; execute nothing.")
    args = p.parse_args()

    base_dir = Path(args.dir).resolve()
    if not base_dir.exists():
        print(f"Error: notebook dir not found: {base_dir}", file=sys.stderr)
        return 2

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    logpath = LOGS_DIR / f"ci_{ts}.log"

    with open(logpath, "w") as logfile:
        tee = Tee(logfile)
        tee.line("=" * 64)
        tee.line(f"CI regression pass  {ts}  (log: {logpath.name})")
        tee.line("=" * 64)

        # ── Phase A — Freshness (gap set) ───────────────────────────────────
        gap_stems: set[str] = set()
        if args.skip_gap:
            tee.line("\n── Phase A — Freshness: SKIPPED (--skip-gap) ──")
        else:
            gap_paths = selector_paths("--gap", base_dir, args.category)
            gap_stems = {p.stem for p in gap_paths}
            tee.line(f"\n── Phase A — Freshness: {len(gap_paths)} gap notebook(s) ──")
            for p in gap_paths:
                tee.line(f"    {p.name}")
            if gap_paths:
                run_batch(gap_paths, base_dir, tee, args.dry_run)

        # ── Phase B — Failures (deduped against Phase A) ────────────────────
        failing_paths = selector_paths("--failing", base_dir, args.category)
        failing_paths = [p for p in failing_paths if p.stem not in gap_stems]
        tee.line(f"\n── Phase B — Failures: {len(failing_paths)} still-failing "
                 f"notebook(s) not already run this pass ──")
        for p in failing_paths:
            tee.line(f"    {p.name}")
        if failing_paths:
            run_batch(failing_paths, base_dir, tee, args.dry_run)

        # ── Phase C — Dashboard ─────────────────────────────────────────────
        tee.line("\n── Phase C — Regenerating STATUS.md ──")
        if args.dry_run:
            tee.line("[dry-run] would run: status.py --write")
        else:
            stream([sys.executable, str(STATUS_PY), "--write"], tee)

        # ── Phase D — Commit ────────────────────────────────────────────────
        tee.line("\n── Phase D — Commit ──")
        if args.no_commit:
            tee.line("[commit] skipped (--no-commit)")
        else:
            git_commit(tee, ts, args.dry_run)

        tee.line(f"\nCI pass complete. Log: {logpath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
