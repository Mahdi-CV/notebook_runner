#!/usr/bin/env bash
#
# Start one phase of an authoring or regression job in its own container,
# detached, named nbr-job-<job id>. authoring/worker.py calls this and follows
# the container until it exits; a job reports through files in its own
# directory (authoring/pipeline.py), the only part of jobs/ it can write.
#
# Usage:
#   docker/run-job.sh prepare JOB_ID          # static checks, or annotate
#   docker/run-job.sh run JOB_ID TARGET       # a run on a GPU target
#
# Like docker/run-ci.sh, the container is read-only, runs as the image's
# non-root user, and gets only the secrets its phase needs as -e variables:
#   prepare   the LLM gateway (an annotate job runs the agent); no GitHub,
#             Hugging Face or cluster credentials, because the agent reads
#             author content. Pull requests are fetched by the worker.
#   run       the gateway, HF_TOKEN (NBR_AUTHOR_HF_TOKEN instead for author
#             jobs, if set: give authors a read-only token), the cluster and
#             Radeon credentials, AGENT_SECRET.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PHASE="${1:?usage: run-job.sh prepare|run JOB_ID [TARGET]}"
JOB_ID="${2:?usage: run-job.sh prepare|run JOB_ID [TARGET]}"
TARGET="${3:-}"
[[ "$JOB_ID" =~ ^[a-z][0-9a-f]+$ ]] || { echo "run-job: bad job id $JOB_ID" >&2; exit 2; }
IMAGE="${NOTEBOOK_RUNNER_IMAGE:-notebook-runner:latest}"
JOBS_DIR="${NBR_JOBS_DIR:-$REPO_ROOT/jobs}"
JOB_DIR="$JOBS_DIR/$JOB_ID"
CHECKOUT="$REPO_ROOT/checkouts/notebooks"
NB_SUBDIR="${NBR_NOTEBOOKS_SUBDIR:-docs/notebooks}"
NBR_AGENT_UID=1001

[[ -d "$JOB_DIR" ]] || { echo "run-job: no job directory $JOB_DIR" >&2; exit 2; }
[[ -d "$CHECKOUT/.git" ]] || { echo "run-job: no gpuaidev checkout at $CHECKOUT (run docker/run-ci.sh once)" >&2; exit 2; }
KIND="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kind"])' "$JOB_DIR/request.json")"

set -a
# shellcheck disable=SC1091  # path is runtime-resolved
source "$REPO_ROOT/.env"
set +a

# The container writes as uid 1001: its job directory, and Radeon's per-identity state.
mkdir -p "$REPO_ROOT/logs/.radeon"
chown -R "$NBR_AGENT_UID:$NBR_AGENT_UID" "$JOB_DIR" "$REPO_ROOT/logs/.radeon" 2>/dev/null || true

ENV_ARGS=(-e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_CUSTOM_HEADERS
          -e ANTHROPIC_MODEL -e ANTHROPIC_DEFAULT_OPUS_MODEL -e ANTHROPIC_DEFAULT_SONNET_MODEL
          -e ANTHROPIC_DEFAULT_HAIKU_MODEL -e CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
          -e NBR_JOBS_DIR=/work/jobs -e NBR_CHECKOUT=/work/checkouts/notebooks)
MOUNTS=(-v "$REPO_ROOT/run.py:/work/run.py:ro"
        -v "$REPO_ROOT/tools:/work/tools:ro"
        -v "$REPO_ROOT/authoring:/work/authoring:ro"
        -v "$REPO_ROOT/CLAUDE.md:/work/CLAUDE.md:ro"
        -v "$REPO_ROOT/CLAUDE_FIX.md:/work/CLAUDE_FIX.md:ro"
        -v "$REPO_ROOT/runner.yaml:/work/runner.yaml:ro"
        -v "$CHECKOUT:/work/checkouts/notebooks:ro"
        -v "$CHECKOUT/$NB_SUBDIR:/work/notebooks:ro"
        -v "$JOB_DIR:/work/jobs/$JOB_ID:rw")
LIMITS=(--cpus 2 --memory 2g)

if [[ "$PHASE" == "run" ]]; then
  [[ -n "$TARGET" ]] || { echo "run-job: run needs a TARGET" >&2; exit 2; }
  LIMITS=(--cpus 4 --memory 4g)
  if [[ "$KIND" != "regression" && -n "${NBR_AUTHOR_HF_TOKEN:-}" ]]; then
    HF_TOKEN="$NBR_AUTHOR_HF_TOKEN"
  fi
  ENV_ARGS+=(-e HF_TOKEN -e AGENT_SECRET -e RADEON_CLOUD_API_TOKEN)
  MOUNTS+=(-v "$REPO_ROOT/logs/.radeon:/work/logs/.radeon:rw")
  if [[ "$KIND" == "regression" ]]; then
    # The pass's JSON results and the fixer's notebooks go where run-ci.sh puts them.
    MOUNTS+=(-v "$REPO_ROOT/results:/work/results:rw" -v "$REPO_ROOT/fixes:/work/fixes:rw")
  fi
  if [[ -n "${NBR_KUBECONFIG:-}" ]]; then
    KUBE_STAGE="$JOB_DIR/.kubeconfig"
    install -m 600 "$NBR_KUBECONFIG" "$KUBE_STAGE"
    chown "$NBR_AGENT_UID:$NBR_AGENT_UID" "$KUBE_STAGE" 2>/dev/null || chmod 644 "$KUBE_STAGE"
    ENV_ARGS+=(-e KUBECONFIG="/work/jobs/$JOB_ID/.kubeconfig")
  fi
  CMD=(python -m authoring.pipeline run "$JOB_ID" --target "$TARGET")
else
  CMD=(python -m authoring.pipeline prepare "$JOB_ID")
fi

docker run -d \
  --name "nbr-job-$JOB_ID" \
  --label "notebook-runner/job=$JOB_ID" \
  --read-only \
  --tmpfs /tmp:rw,size=512m \
  --tmpfs /home/agent:rw,exec,size=1g,uid=$NBR_AGENT_UID,gid=$NBR_AGENT_UID \
  --security-opt no-new-privileges \
  "${LIMITS[@]}" \
  "${ENV_ARGS[@]}" \
  "${MOUNTS[@]}" \
  "$IMAGE" \
  "${CMD[@]}" >/dev/null
