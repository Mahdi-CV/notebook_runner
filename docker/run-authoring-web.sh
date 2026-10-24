#!/usr/bin/env bash
#
# (Re)start the authoring web app (authoring/app.py) as a long-running
# container from the runner image, as the image's uid 1001: the same user the
# job containers and the nightly pass write jobs/ as. It holds no credentials;
# authoring/worker.py does the work.
#
# It listens on 127.0.0.1 only: the regression dashboard serves it under
# /authoring/ on its own port (NBR_AUTHORING_UPSTREAM on the dashboard's unit).
# NBR_AUTHORING_BIND=0.0.0.0 NBR_AUTHORING_PREFIX= serves it on its own port instead.
#
# Usage:
#   docker/run-authoring-web.sh            # port NBR_AUTHORING_PORT (default 8081)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE="${NOTEBOOK_RUNNER_IMAGE:-notebook-runner:latest}"
PORT="${NBR_AUTHORING_PORT:-8081}"
NAME=notebook-authoring

mkdir -p "$REPO_ROOT/jobs"
chown 1001:1001 "$REPO_ROOT/jobs" 2>/dev/null || true

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp:rw,size=256m \
  --security-opt no-new-privileges \
  --cpus 1 --memory 1g \
  -p "${NBR_AUTHORING_BIND:-127.0.0.1}:$PORT:8081" \
  -e NBR_JOBS_DIR=/work/jobs \
  -e NBR_DASHBOARD_URL="${NBR_DASHBOARD_URL:-/}" \
  -e NBR_AUTHORING_PREFIX="${NBR_AUTHORING_PREFIX-/authoring}" \
  -e NBR_MAX_UPLOAD_MB="${NBR_MAX_UPLOAD_MB:-10}" \
  -v "$REPO_ROOT/authoring:/work/authoring:ro" \
  -v "$REPO_ROOT/tools:/work/tools:ro" \
  -v "$REPO_ROOT/dashboard/templates:/work/dashboard/templates:ro" \
  -v "$REPO_ROOT/runner.yaml:/work/runner.yaml:ro" \
  -v "$REPO_ROOT/jobs:/work/jobs:rw" \
  "$IMAGE" \
  uvicorn authoring.app:app --host 0.0.0.0 --port 8081 >/dev/null
echo "notebook-authoring: listening on ${NBR_AUTHORING_BIND:-127.0.0.1}:$PORT (prefix ${NBR_AUTHORING_PREFIX-/authoring})"
