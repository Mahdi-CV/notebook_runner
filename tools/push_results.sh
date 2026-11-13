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
#   NBR_DASHBOARD_HOST  ssh target (default: atlvaimciapp01; the user needs sudo there,
#                       since the host takes no root logins)
#   NBR_DASHBOARD_ROOT  pipeline root on that host (default: /data/notebook_runner)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${NBR_DASHBOARD_HOST:-atlvaimciapp01}"
ROOT="${NBR_DASHBOARD_ROOT:-/data/notebook_runner}"

# No -o/-g: the files arrive as root (sudo rsync) and are handed to the owner of
# results/ below, the uid the daily pass writes as (1001 under Docker, the
# service account under rootless podman).
rsync -rt --ignore-existing --no-owner --no-group --chmod=D755,F644 \
  --include='*.json' --exclude='*' --rsync-path='sudo rsync' \
  "$REPO_ROOT/results/" "$HOST:$ROOT/results/"

# The import runs as root and may leave a root-owned journal or new database,
# so hand the results back to their owner afterwards.
# The script goes on stdin: the remote login shell may not be bash.
ssh "$HOST" sudo bash -s <<EOF
set -e
cd '$ROOT'
.venv-dashboard/bin/python tools/results_db.py import
owner=\$(stat -c %u:%g "\$(readlink -f results)")
chown "\$owner" results/*.json results/runs.db*
EOF
