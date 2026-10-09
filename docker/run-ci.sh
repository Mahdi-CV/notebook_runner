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

# Dedicated application key, not a human's personal key.
#
# This key is mounted into a container that runs model-authored shell commands,
# so it should grant exactly one thing: access to the GPU node. Using a
# personal key would hand that blast radius every host that key can reach, and
# make revocation mean "rotate my whole identity" instead of "delete one line
# from one authorized_keys".
#
# Passphrase-less by necessity — an unattended cron pass has no tty to unlock
# a key. That makes file custody the only protection: keep it 0600 and treat
# it as a credential, not a convenience.
: "${GPU_SSH_KEY:=$HOME/.ssh/id_nbrunner_ed25519}"
if [[ ! -f "$GPU_SSH_KEY" ]]; then
  echo "run-ci: GPU ssh key not found at $GPU_SSH_KEY" >&2
  echo "        Create one with:" >&2
  echo "          ssh-keygen -t ed25519 -f $GPU_SSH_KEY -N '' -C notebook-runner-ci" >&2
  echo "        then append the .pub to the GPU node's ~/.ssh/authorized_keys." >&2
  echo "        Override the path with GPU_SSH_KEY=." >&2
  exit 1
fi

# A group/world-readable private key is a finding, and ssh itself will refuse
# it — fail here with an actionable message instead of mid-pass.
key_perms="$(stat -c '%a' "$GPU_SSH_KEY")"
if [[ "$key_perms" != "600" && "$key_perms" != "400" ]]; then
  echo "run-ci: $GPU_SSH_KEY has mode $key_perms — must be 600 or 400." >&2
  echo "        Fix with: chmod 600 $GPU_SSH_KEY" >&2
  exit 1
fi

# Stage the key under the agent's uid.
#
# Bind-mounting the key directly does not work: a correct 0600 key is owned by
# the invoking user, the container runs as uid 1001, and a bind mount preserves
# host ownership — so ssh inside the container reports
#   Load key "/home/agent/.ssh/id_gpu": Permission denied
# and falls through to "Permission denied (publickey)", which looks like a
# rejected key rather than an unreadable one.
#
# Copy to a 0600 uid-1001 staging file instead. It lives under logs/ (already
# gitignored) and is removed on exit, so the key is not left lying around in a
# second location.
KEY_STAGE="$REPO_ROOT/logs/.id_gpu_stage"
install -m 600 "$GPU_SSH_KEY" "$KEY_STAGE"
if ! chown 1001:1001 "$KEY_STAGE" 2>/dev/null; then
  # Unprivileged host user: fall back to world-readable *inside the staging
  # copy only*. Acceptable because logs/ is not world-traversable and the file
  # is deleted on exit — but warn, because it is a real (brief) exposure.
  chmod 644 "$KEY_STAGE"
  echo "run-ci: warning — cannot chown the staged key to uid 1001 (needs root)." >&2
  echo "        Staged copy is mode 644 for the container's lifetime." >&2
fi
trap 'rm -f "$KEY_STAGE"' EXIT

# known_hosts is mounted read-only and StrictHostKeyChecking is on, so the GPU
# host must already be trusted on this machine. Generating it here (rather than
# disabling the check in the container) keeps the trust decision on the host,
# where a human made it.
KNOWN_HOSTS="$REPO_ROOT/logs/.known_hosts_gpu"
if [[ ! -s "$KNOWN_HOSTS" ]]; then
  echo "run-ci: building known_hosts for the GPU node…"
  ssh-keyscan -H 134.199.202.42 > "$KNOWN_HOSTS" 2>/dev/null || {
    echo "run-ci: ssh-keyscan failed — is the GPU node reachable?" >&2
    exit 1
  }
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "run-ci: image $IMAGE not found. Build it with:" >&2
  echo "        docker build -f docker/Dockerfile -t $IMAGE ." >&2
  exit 1
fi

# --- no-overlap guard -----------------------------------------------------
# A fine-tune pass can run for hours. flock -n means a new scheduled fire is
# skipped, never queued — two passes driving the same GPU node would fight
# over /home/amd/tutorial_agent_runs and produce garbage verdicts.
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
#   ssh key, known_hosts     read-only   — credentials
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
  -v "$KEY_STAGE:/home/agent/.ssh/id_gpu:ro" \
  -v "$KNOWN_HOSTS:/home/agent/.ssh/known_hosts:ro" \
  "$IMAGE" \
  python tools/ci_run.py "$@" 2>&1 | tee "$LOGFILE"

rc="${PIPESTATUS[0]}"
echo "run-ci: pass $TS finished (exit=$rc, log=$LOGFILE)"
exit "$rc"
