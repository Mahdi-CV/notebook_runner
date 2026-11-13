#!/usr/bin/env bash
#
# Copy this checkout's code to the dashboard host, without sudo.
#
# Only code is copied: the apps, tools and docs. The host's own state (.env,
# runner.yaml, jobs/, results/, logs/, checkouts/, the venvs) is never touched,
# and files that exist only on the host are left alone. Templates take effect
# at once: both apps reload them from disk. Python changes need --restart.
#
# One-time host setup (as root), so the deploy user can write the code and
# have the services restarted without sudo:
#   cd /data/notebook_runner
#   chown -R <user>:<group> authoring dashboard docker tools run.py pyproject.toml uv.lock \
#     README.md CLAUDE.md CLAUDE_FIX.md runner.example.yaml .env.example
#   install -m 644 docker/notebook-runner-restart.{path,service} /etc/systemd/system/
#   systemctl daemon-reload && systemctl enable --now notebook-runner-restart.path
# --restart then just writes docker/.restart-request, and the path unit
# restarts the dashboard, the worker and the authoring web app as root.
#
# Usage:
#   tools/deploy.sh             # copy the code
#   tools/deploy.sh --dry-run   # list what would change
#   tools/deploy.sh --restart   # copy, then restart the dashboard, worker and authoring app
# Environment:
#   NBR_DASHBOARD_HOST  ssh target (default: atlvaimciapp01)
#   NBR_DASHBOARD_ROOT  pipeline root on that host (default: /data/notebook_runner)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${NBR_DASHBOARD_HOST:-atlvaimciapp01}"
ROOT="${NBR_DASHBOARD_ROOT:-/data/notebook_runner}"

dry_run=() restart=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=(--dry-run) ;;
    --restart) restart=1 ;;
    *) echo "usage: $0 [--dry-run] [--restart]" >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"
# World-readable on the host: the dashboard runs as nobody and the authoring
# container as uid 1001, and both read this code. -p, or new files get the
# host user's umask (027 there) instead of these modes. --inplace: the root
# of the tree stays root's, so rsync cannot put a temporary file next to the
# top-level files (README.md, run.py, ...), only overwrite them.
rsync -rltp --inplace --itemize-changes --chmod=Dgo+rx,Fgo+r "${dry_run[@]}" \
  --exclude __pycache__ --exclude '*.pyc' \
  authoring dashboard docker tools run.py pyproject.toml uv.lock \
  README.md CLAUDE.md CLAUDE_FIX.md runner.example.yaml .env.example \
  "$HOST:$ROOT/"

if (( restart )) && (( ${#dry_run[@]} == 0 )); then
  # Ask notebook-runner-restart.path, then wait for its one-shot service to run.
  ssh "$HOST" /bin/sh -s <<EOF
show() { systemctl show -p "\$1" --value notebook-runner-restart.service; }
systemctl is-active -q notebook-runner-restart.path \
  || { echo "deploy: notebook-runner-restart.path is not enabled on $HOST (see tools/deploy.sh)" >&2; exit 1; }
before=\$(show InactiveEnterTimestampMonotonic)
date > "$ROOT/docker/.restart-request"
for _ in \$(seq 120); do
  sleep 1
  [ "\$(show InactiveEnterTimestampMonotonic)" != "\$before" ] && break
done
result=\$(show Result)
[ "\$result" = success ] && echo restarted || { echo "deploy: restart \$result (journalctl -u notebook-runner-restart)" >&2; exit 1; }
EOF
fi
