#!/usr/bin/env bash
#
# (Re)start the credential broker (authoring/broker.py): the one container
# that holds the secrets authoring jobs use, so that the jobs themselves hold
# none. It joins two networks:
#   nbr-jobs  internal (no internet): the author job containers, which reach
#             the broker as http://nbr-broker:8790 and nothing else
#   bridge    the broker's own way out to the LLM gateway, the cluster,
#             Radeon Global, Hugging Face and Docker Hub
# It publishes no port: the worker deletes a cancelled job's pod with
# `docker exec nbr-broker python -m authoring.broker cleanup <job id>`.
#
# Usage:
#   docker/run-broker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE="${NOTEBOOK_RUNNER_IMAGE:-notebook-runner:latest}"
NAME=nbr-broker
# shellcheck source=docker/engine.sh
source "$SCRIPT_DIR/engine.sh"

set -a
# shellcheck disable=SC1091  # path is runtime-resolved
source "$REPO_ROOT/.env"
set +a

"$CLI" network inspect nbr-jobs >/dev/null 2>&1 || "$CLI" network create --internal nbr-jobs >/dev/null
# Rootless podman cannot add a network to a running container (its default
# network is not a bridge), so there the broker starts on both.
NETWORKS=()
[[ "$CLI" == podman ]] && NETWORKS=(--network podman --network nbr-jobs)

mkdir -p "$REPO_ROOT/jobs/.broker" "$REPO_ROOT/logs/.radeon" "$REPO_ROOT/logs/.broker"
agent_owns "$REPO_ROOT/logs/.radeon" 2>/dev/null || true

KUBE_ARGS=()
if [[ -n "${NBR_KUBECONFIG:-}" ]]; then
  KUBE_STAGE="$REPO_ROOT/logs/.broker/kubeconfig"
  install -m 600 "$NBR_KUBECONFIG" "$KUBE_STAGE"
  agent_owns "$KUBE_STAGE" 2>/dev/null || chmod 644 "$KUBE_STAGE"
  KUBE_ARGS=(-v "$KUBE_STAGE:/etc/notebook-runner/kubeconfig:ro"
             -e KUBECONFIG=/etc/notebook-runner/kubeconfig)
fi

"$CLI" rm -f "$NAME" >/dev/null 2>&1 || true
"$CLI" run -d \
  --name "$NAME" \
  "${RESTART[@]}" \
  ${USERNS[@]+"${USERNS[@]}"} \
  ${NETWORKS[@]+"${NETWORKS[@]}"} \
  --read-only \
  --tmpfs /tmp:rw,size=4g \
  --tmpfs /home/agent:rw,size=64m,$AGENT_TMPFS \
  --security-opt no-new-privileges \
  --cpus 2 --memory 6g \
  -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_CUSTOM_HEADERS \
  -e HF_TOKEN -e NBR_AUTHOR_HF_TOKEN -e NBR_AUTHOR_HF_SECRET \
  -e RADEON_CLOUD_API_TOKEN \
  -e NBR_BROKER_KEYS=/work/keys/keys.json \
  ${KUBE_ARGS[@]+"${KUBE_ARGS[@]}"} \
  -v "$REPO_ROOT/authoring:/work/authoring:ro" \
  -v "$REPO_ROOT/tools:/work/tools:ro" \
  -v "$REPO_ROOT/runner.yaml:/work/runner.yaml:ro" \
  -v "$REPO_ROOT/jobs/.broker:/work/keys:ro" \
  -v "$REPO_ROOT/logs/.radeon:/work/logs/.radeon:rw" \
  "$IMAGE" \
  uvicorn authoring.broker:app --host 0.0.0.0 --port 8790 >/dev/null
[[ "$CLI" == podman ]] || "$CLI" network connect nbr-jobs "$NAME"
echo "nbr-broker: running (jobs reach it at http://nbr-broker:8790 on nbr-jobs)"
