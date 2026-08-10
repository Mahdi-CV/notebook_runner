#!/usr/bin/env bash
#
# Weekly regression trigger for the notebook agent (Goal 1).
#
# Runs tools/ci_run.py --push under flock so a new scheduled fire never overlaps
# a still-running pass (fine-tune notebooks can run for hours). All output is
# tee'd to logs/cron_<timestamp>.log.
#
# Install (Linux cron):
#   crontab -e
#   17 3 * * 0  /mnt/c/Users/karchaka/notebook_runner/tools/ci_cron.sh
#
# WSL note: cron only fires while WSL is running. If the host may be asleep at
# the scheduled time, drive this from Windows Task Scheduler instead:
#   wsl.exe bash -lc '/mnt/c/Users/karchaka/notebook_runner/tools/ci_cron.sh'

set -uo pipefail

# Resolve the repo root from this script's own location — no hardcoded cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/venv/bin/python"
LOCKFILE="$REPO_ROOT/logs/ci_cron.lock"
TS="$(date +%Y%m%dT%H%M%S)"
LOGFILE="$REPO_ROOT/logs/cron_${TS}.log"

mkdir -p "$REPO_ROOT/logs"

if [[ ! -x "$PYTHON" ]]; then
  echo "ci_cron: venv python not found at $PYTHON" >&2
  exit 1
fi

# flock -n: if a previous pass still holds the lock, skip this fire (don't queue).
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "ci_cron: previous pass still running (lock held) — skipping $TS" \
    | tee -a "$REPO_ROOT/logs/cron_skipped.log"
  exit 0
fi

# stdin from /dev/null so nothing can block waiting for input under cron.
"$PYTHON" tools/ci_run.py --push < /dev/null 2>&1 | tee "$LOGFILE"
rc="${PIPESTATUS[0]}"

echo "ci_cron: pass $TS finished (ci_run exit=$rc, log=$LOGFILE)"
exit "$rc"
