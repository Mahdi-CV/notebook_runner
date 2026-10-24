#!/usr/bin/env bash
#
# Push result files from an ad-hoc run on this machine to the dashboard host
# and import them into its results/runs.db.
#
# The daily pass on the dashboard host writes runs.db directly; this is for
# runs started anywhere else. Only result files the host doesn't already have
# are copied, and the import skips files it has recorded before, so running it
# repeatedly is safe.
#
# Usage:
#   tools/push_results.sh
# Environment:
#   NBR_DASHBOARD_HOST  ssh target (default: root@xcovdashdbd01)
#   NBR_DASHBOARD_ROOT  pipeline root on that host (default: /data/notebook_runner)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${NBR_DASHBOARD_HOST:-root@xcovdashdbd01}"
ROOT="${NBR_DASHBOARD_ROOT:-/data/notebook_runner}"

# No -o/-g/-p: files must arrive owned by uid 1001 (the daily pass's uid), not this machine's uid.
rsync -rt --ignore-existing --no-owner --no-group \
  --chown=1001:1001 --chmod=D755,F644 \
  --include='*.json' --exclude='*' \
  "$REPO_ROOT/results/" "$HOST:$ROOT/results/"

# The import runs as root and may leave a root-owned journal or new database,
# so hand the results directory and database back to uid 1001 afterwards.
ssh "$HOST" "cd '$ROOT' && .venv-dashboard/bin/python tools/results_db.py import \
  && chown 1001:1001 \"\$(readlink -f results)\" results/runs.db*"
