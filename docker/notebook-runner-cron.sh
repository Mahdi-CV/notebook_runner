#!/usr/bin/env bash
#
# Daily unattended trigger for the CI pass. Wraps docker/run-ci.sh with the
# flags for an unattended run: diagnose-only (no auto-fix), no git commit/push,
# Teams notification on. Extra arguments pass through to ci_run.py, e.g.
# `--server workshop-k8s`.
#
# Schedule it from crontab, e.g.:
#   0 6 * * * /path/to/notebook_runner/docker/notebook-runner-cron.sh >> /path/to/notebook_runner/logs/cron_daily.log 2>&1

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec docker/run-ci.sh --verify-only --no-commit "$@"
