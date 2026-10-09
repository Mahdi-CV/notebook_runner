#!/usr/bin/env bash
#
# Daily unattended trigger for the notebook regression CI pass on
# xcovdashdbd01. Wraps docker/run-ci.sh with the flags appropriate for an
# unattended run: diagnose-only (no auto-fix), no git commit/push, Teams
# notification stays on (that's the point of running this daily).
#
# Installed via crontab, not this repo's own scheduling — see the header
# comment this script's crontab entry carries for the installation record.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec docker/run-ci.sh --verify-only --no-commit
