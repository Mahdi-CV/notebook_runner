#!/usr/bin/env bash
#
# Launch one CI pass inside the notebook-runner container.
#
# This is the container-native replacement for tools/ci_cron.sh. It keeps the
# same two guarantees that wrapper had — no overlapping passes, everything
# logged — and adds the isolation the bare-metal version could not give:
# the agent runs as a non-root user in a throwaway filesystem.
#
# Usage:
#   docker/run-ci.sh                    # full pass, commit + push
#   docker/run-ci.sh --dry-run          # print the plan, run nothing
#   docker/run-ci.sh --category inference --no-commit
#
# Any arguments are passed straight through to tools/ci_run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

IMAGE="${NOTEBOOK_RUNNER_IMAGE:-notebook-runner:latest}"
TS="$(date +%Y%m%dT%H%M%S)"
LOGFILE="$REPO_ROOT/logs/cron_${TS}.log"
LOCKFILE="$REPO_ROOT/logs/ci_cron.lock"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results" "$REPO_ROOT/fixes"

# --- preflight ------------------------------------------------------------
# Fail with a specific message now rather than a confusing one 40 minutes in.

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "run-ci: missing .env (copy .env.example and fill it in)" >&2
  exit 1
fi

# Claude credentials.
#
# On the AMD network the agent authenticates through the enterprise gateway
# (llm-api.amd.com), NOT with a first-party Anthropic key: ANTHROPIC_BASE_URL
# points at the gateway and ANTHROPIC_CUSTOM_HEADERS carries the APIM
# subscription header. ANTHROPIC_API_KEY is a placeholder the gateway ignores,
# but the CLI still requires it to be non-empty.
#
# These live in .env (gitignored). run.py calls load_dotenv() for its own use,
# but the Claude CLI reads them from the *process environment*, so this wrapper
# must export them into the container itself — sourcing .env here is what makes
# `docker run -e ANTHROPIC_BASE_URL` below actually carry a value.
#
# `set -a` exports everything .env defines; the subshell-free source is
# deliberate so the vars are visible to the docker run at the bottom.
set -a
# shellcheck disable=SC1091  # path is runtime-resolved
source "$REPO_ROOT/.env"
set +a

# Note `claude auth status` reports loggedIn:true whenever a key is merely
# present — it does not validate it. Only a real request proves credentials
# work, which is what docker/README.md's smoke test does.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "run-ci: ANTHROPIC_API_KEY is not set (use 'dummy' with the gateway)." >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_BASE_URL:-}" || -z "${ANTHROPIC_CUSTOM_HEADERS:-}" ]]; then
  echo "run-ci: ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS are unset." >&2
  echo "        Without them the CLI talks to api.anthropic.com and gets 401." >&2
  echo "        Set them in .env — see docker/README.md." >&2
  exit 1
fi

# HF_TOKEN gates every notebook that pulls a gated model (meta-llama,
# mistralai, ...). Warn loudly rather than exit: plenty of notebooks don't need
# it, and refusing to run at all would be worse than running a reduced set.
#
# The warning matters because of what an empty token actually causes. In an
# observed run the agent, finding HF_TOKEN empty, went looking for it — reading
# .env and dumping `env` — instead of recording "credential missing" and moving
# on. Surfacing it here keeps that from being discovered mid-pass.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "run-ci: warning — HF_TOKEN is empty; gated-model notebooks will fail" >&2
  echo "        their env check and be reported as failures." >&2
fi

# The gateway only resolves on the AMD network. Failing here is far clearer
# than a pass that dies on its first API call 30 seconds in.
gw_host="$(sed -E 's|https?://||; s|/.*||' <<<"$ANTHROPIC_BASE_URL")"
if ! getent hosts "$gw_host" >/dev/null 2>&1; then
  echo "run-ci: cannot resolve $gw_host — this host is not on the AMD network." >&2
  echo "        Run the orchestrator from an AMD-network host (e.g. xcovdashdbd01)." >&2
  exit 1
fi

# HTTP transport shim, not real SSH.
#
# AMD's egress silently drops SSH (and refuses TLS outright) to the GPU
# node's IP range — see docs/PLAN-http-agent-shim.md. tools/http_ssh_shim/
# replaces ssh/scp with drop-ins that speak HMAC-signed HTTP instead, and
# run.py prepends that directory to PATH whenever AGENT_URL/AGENT_SECRET are
# set (which .env sourcing above already exported into this shell).
#
# No key file, no known_hosts, no staging/chown dance the way a real SSH
# key needed: the secret is a plain string forwarded via -e, same as every
# other credential in this script.
if [[ -z "${AGENT_URL:-}" || -z "${AGENT_SECRET:-}" ]]; then
  echo "run-ci: AGENT_URL / AGENT_SECRET are unset." >&2
  echo "        The GPU node is not reachable over real SSH from this network" >&2
  echo "        (AMD's egress silently drops it) — set both in .env to use the" >&2
  echo "        HTTP transport shim. See docker/README.md and .env.example." >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "run-ci: image $IMAGE not found. Build it with:" >&2
  echo "        docker build -f docker/Dockerfile -t $IMAGE ." >&2
  exit 1
fi

# --- no-overlap guard -----------------------------------------------------
# A fine-tune pass can run for hours. flock -n means a new scheduled fire is
# skipped, never queued — two passes driving the same GPU node would fight
# over $WORKSPACE_BASE/tutorial_agent_runs and produce garbage verdicts.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "run-ci: previous pass still running — skipping $TS" \
    | tee -a "$REPO_ROOT/logs/cron_skipped.log"
  exit 0
fi

echo "run-ci: starting pass $TS (image=$IMAGE, log=$LOGFILE)"

# --- run ------------------------------------------------------------------
# Mounts, and why each one:
#   results/, fixes/, logs/  read-write  — the pass's actual output
#   STATUS.md, manifest.yaml read-write  — Phase C/D rewrite these
#   notebooks/               read-only   — inputs; the agent must never edit
#                                          them (patching happens on the GPU
#                                          node, on a copy)
#   (.env is deliberately NOT mounted — see below)
#   tools/ (includes http_ssh_shim/) read-only — run.py prepends this to
#                                          PATH itself when AGENT_URL/
#                                          AGENT_SECRET are set; no separate
#                                          mount or key staging needed
#   .git                     read-write  — Phase D commits
#
# --read-only on the root filesystem with tmpfs for the writable spots: the
# agent runs model-authored shell commands, and this bounds what a bad one can
# scribble on. /home/agent is a tmpfs because the CLI writes session state there.
#
# .env is NOT mounted, deliberately. This wrapper already sources it and passes
# the individual variables the runtime needs via -e, so the file itself buys
# nothing inside the container — and it holds the APIM key, the GPU host, and
# the Teams webhook. That matters because the agent runs with
# --dangerously-skip-permissions: in an observed run it went looking for a
# missing HF_TOKEN by reading .env and dumping `env`. The read was refused by
# an interactive permission rule that does not exist in the container, so the
# only reliable control is to not mount the file.
docker run --rm \
  --name "notebook-runner-${TS}" \
  --read-only \
  --tmpfs /tmp:rw,size=512m \
  --tmpfs /home/agent:rw,exec,size=1g,uid=1001,gid=1001 \
  --security-opt no-new-privileges \
  --cpus 4 --memory 4g \
  -e ANTHROPIC_API_KEY \
  -e ANTHROPIC_BASE_URL \
  -e ANTHROPIC_CUSTOM_HEADERS \
  -e ANTHROPIC_MODEL \
  -e ANTHROPIC_DEFAULT_OPUS_MODEL \
  -e ANTHROPIC_DEFAULT_SONNET_MODEL \
  -e ANTHROPIC_DEFAULT_HAIKU_MODEL \
  -e CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  -e TEAMS_WEBHOOK_URL \
  -e AGENT_URL \
  -e AGENT_SECRET \
  -e GIT_AUTHOR_NAME="notebook-runner" \
  -e GIT_AUTHOR_EMAIL="notebook-runner@local" \
  -e GIT_COMMITTER_NAME="notebook-runner" \
  -e GIT_COMMITTER_EMAIL="notebook-runner@local" \
  -v "$REPO_ROOT/run.py:/work/run.py:ro" \
  -v "$REPO_ROOT/tools:/work/tools:ro" \
  -v "$REPO_ROOT/CLAUDE.md:/work/CLAUDE.md:ro" \
  -v "$REPO_ROOT/CLAUDE_FIX.md:/work/CLAUDE_FIX.md:ro" \
  -v "$REPO_ROOT/notebooks:/work/notebooks:ro" \
  -v "$REPO_ROOT/results:/work/results:rw" \
  -v "$REPO_ROOT/fixes:/work/fixes:rw" \
  -v "$REPO_ROOT/logs:/work/logs:rw" \
  -v "$REPO_ROOT/STATUS.md:/work/STATUS.md:rw" \
  -v "$REPO_ROOT/manifest.yaml:/work/manifest.yaml:rw" \
  -v "$REPO_ROOT/.git:/work/.git:rw" \
  "$IMAGE" \
  python tools/ci_run.py "$@" 2>&1 | tee "$LOGFILE"

rc="${PIPESTATUS[0]}"
echo "run-ci: pass $TS finished (exit=$rc, log=$LOGFILE)"
exit "$rc"
