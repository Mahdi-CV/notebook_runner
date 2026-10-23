#!/usr/bin/env bash
#
# Launch one CI pass inside the notebook-runner container.
#
# Passes never overlap (flock) and everything is logged; the agent runs as a
# non-root user in a read-only container. Container passes never git-commit:
# results go to results/ (and the dashboard), not to the repository.
#
# Usage:
#   docker/run-ci.sh                    # full pass (verify, then auto-fix)
#   docker/run-ci.sh --dry-run          # print the plan, run nothing
#   docker/run-ci.sh --category inference --verify-only
#
# Any arguments are passed straight through to tools/ci_run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

IMAGE="${NOTEBOOK_RUNNER_IMAGE:-notebook-runner:latest}"
TS="$(date +%Y%m%dT%H%M%S)"
LOCKFILE="$REPO_ROOT/logs/ci_run.lock"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results" "$REPO_ROOT/fixes"

# --- preflight ------------------------------------------------------------
# Fail with a specific message now rather than a confusing one 40 minutes in.

if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "run-ci: missing .env (copy .env.example and fill it in)" >&2
  exit 1
fi

# Export everything in .env: the Claude CLI reads the gateway credentials from
# its environment, so the `docker run -e` forwards below must carry values.
set -a
# shellcheck disable=SC1091  # path is runtime-resolved
source "$REPO_ROOT/.env"
set +a

# A full disk kills a pass before ci_run.py can post its summary card, so check
# the repo and Docker's storage up front and alert Teams instead.
MIN_FREE_GB="${NBR_MIN_FREE_GB:-2}"
low_disk=""
for path in "$REPO_ROOT" "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"; do
  [[ -e "$path" ]] || continue
  free_gb=$(df -P -BG "$path" | awk 'NR==2 {gsub("G","",$4); print $4}')
  if (( free_gb < MIN_FREE_GB )); then
    low_disk+="${path} has ${free_gb}G free ($(df -P "$path" | awk 'NR==2 {print $6}') filesystem). "
  fi
done
if [[ -n "$low_disk" ]]; then
  echo "run-ci: disk space below ${MIN_FREE_GB}G — not starting pass $TS: $low_disk" >&2
  python3 "$REPO_ROOT/tools/teams_alert.py" \
    "Notebook regression pass skipped: disk full" \
    "${low_disk}The pass needs at least ${MIN_FREE_GB}G free and was not started. Free space on the host, then the next scheduled pass will run normally." \
    || true
  exit 1
fi

# --- write-mount ownership guard ------------------------------------------
# The container writes these paths as uid 1001. If any is owned by another uid
# the agent cannot record its results, so don't start a pass that can't succeed.
NBR_AGENT_UID=1001
bad_owner=""
for p in results fixes logs STATUS.md; do
  [[ -e "$REPO_ROOT/$p" ]] || continue
  # -L: results/ may be a symlink (to NFS); check the directory it points to
  owner=$(stat -L -c %u "$REPO_ROOT/$p")
  if [[ "$owner" != "$NBR_AGENT_UID" ]]; then
    bad_owner+="${p} is owned by uid ${owner} (must be ${NBR_AGENT_UID}). "
  fi
done
if [[ -n "$bad_owner" ]]; then
  echo "run-ci: agent cannot write its outputs — not starting pass $TS: $bad_owner" >&2
  echo "        Fix as root: chown ${NBR_AGENT_UID}:${NBR_AGENT_UID} <path>" >&2
  python3 "$REPO_ROOT/tools/teams_alert.py" \
    "Notebook regression pass skipped: outputs not writable" \
    "${bad_owner}The agent runs as uid ${NBR_AGENT_UID}, so it could not have written its results and the pass was not started. Fix as root with: chown ${NBR_AGENT_UID}:${NBR_AGENT_UID} <path>" \
    || true
  exit 1
fi

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
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "run-ci: warning — HF_TOKEN is empty; gated-model notebooks will fail" >&2
  echo "        their env check and be reported as failures." >&2
fi

# The gateway only resolves on the AMD network. Failing here is far clearer
# than a pass that dies on its first API call 30 seconds in.
gw_host="$(sed -E 's|https?://||; s|/.*||' <<<"$ANTHROPIC_BASE_URL")"
if ! getent hosts "$gw_host" >/dev/null 2>&1; then
  echo "run-ci: cannot resolve $gw_host — this host is not on the AMD network." >&2
  echo "        Run the orchestrator from a host on the AMD network." >&2
  exit 1
fi

# Where notebooks may run: this deployment's targets (see runner.example.yaml).
if [[ ! -f "$REPO_ROOT/runner.yaml" ]]; then
  echo "run-ci: missing runner.yaml (copy runner.example.yaml and describe your targets)" >&2
  exit 1
fi
# Targets with an agent_url are reached through tools/http_ssh_shim/, signed with AGENT_SECRET.
if grep -q '^[^#]*agent_url:' "$REPO_ROOT/runner.yaml" && [[ -z "${AGENT_SECRET:-}" ]]; then
  echo "run-ci: runner.yaml has agent_url targets but AGENT_SECRET is unset in .env." >&2
  exit 1
fi

# The notebooks under test are the upstream repository's tip, checked out
# under checkouts/ and exposed as notebooks/. A real notebooks/ directory
# (a hand-copied tree) would shadow it, so refuse rather than test stale copies.
NB_REPO="${NBR_NOTEBOOKS_REPO:-https://github.com/ROCm/gpuaidev.git}"
NB_REF="${NBR_NOTEBOOKS_REF:-main}"
NB_SUBDIR="${NBR_NOTEBOOKS_SUBDIR:-docs/notebooks}"
CHECKOUT="$REPO_ROOT/checkouts/notebooks"
if [[ -e "$REPO_ROOT/notebooks" && ! -L "$REPO_ROOT/notebooks" ]]; then
  echo "run-ci: notebooks/ is a directory; move it aside so it can point at the" >&2
  echo "        upstream checkout ($NB_REPO, $NB_SUBDIR)." >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "run-ci: image $IMAGE not found. Build it with:" >&2
  echo "        docker build -f docker/Dockerfile -t $IMAGE ." >&2
  exit 1
fi

# Kubernetes runner (optional). NBR_KUBECONFIG is the namespace-scoped
# ServiceAccount kubeconfig from `tools/k8s_pod.py kubeconfig` — never an admin
# one: the agent can read whatever credential the container holds.
# Staged, not bind-mounted: a 0600 file owned by the host user is unreadable
# to the container's uid 1001.
KUBE_ARGS=()
if [[ -n "${NBR_KUBECONFIG:-}" ]]; then
  if [[ ! -f "$NBR_KUBECONFIG" ]]; then
    echo "run-ci: NBR_KUBECONFIG=$NBR_KUBECONFIG does not exist." >&2
    exit 1
  fi
  KUBE_STAGE="$REPO_ROOT/logs/.kubeconfig_stage"
  install -m 600 "$NBR_KUBECONFIG" "$KUBE_STAGE"
  if ! chown 1001:1001 "$KUBE_STAGE" 2>/dev/null; then
    # Unprivileged host user: world-readable inside logs/ for the pass only.
    chmod 644 "$KUBE_STAGE"
    echo "run-ci: warning — cannot chown the staged kubeconfig to uid 1001 (needs root)." >&2
    echo "        Staged copy is mode 644 for the container's lifetime." >&2
  fi
  trap 'rm -f "$KUBE_STAGE"' EXIT
  KUBE_ARGS=(-v "$KUBE_STAGE:/etc/notebook-runner/kubeconfig:ro"
             -e KUBECONFIG=/etc/notebook-runner/kubeconfig)

  # Prove the API answers from this host, with the image's kubectl, before a
  # pass depends on it.
  if ! docker run --rm "${KUBE_ARGS[@]}" --entrypoint kubectl "$IMAGE" \
       version --request-timeout=10s >/dev/null 2>&1; then
    echo "run-ci: the Kubernetes API in $NBR_KUBECONFIG is unreachable from this host." >&2
    echo "        Check: docker run --rm -v ...:/etc/notebook-runner/kubeconfig:ro \\" >&2
    echo "               -e KUBECONFIG=/etc/notebook-runner/kubeconfig --entrypoint kubectl $IMAGE version" >&2
    exit 1
  fi
fi

# --- no-overlap guard -----------------------------------------------------
# A fine-tune pass can run for hours. flock -n means a new scheduled fire is
# skipped, never queued — two passes driving the same GPU node would fight
# over $WORKSPACE_BASE/tutorial_agent_runs and produce garbage verdicts.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "run-ci: previous pass still running — skipping $TS"
  exit 0
fi

# Under the lock: a running pass must never see its notebooks change.
if [[ -d "$CHECKOUT/.git" ]]; then
  if ! { git -C "$CHECKOUT" fetch -q --depth 1 origin "$NB_REF" \
         && git -C "$CHECKOUT" reset -q --hard FETCH_HEAD; }; then
    echo "run-ci: warning — could not update $NB_REPO@$NB_REF; testing the last checkout" >&2
  fi
elif ! git clone -q --depth 1 --branch "$NB_REF" "$NB_REPO" "$CHECKOUT"; then
  echo "run-ci: could not clone $NB_REPO@$NB_REF — not starting pass $TS" >&2
  exit 1
fi
ln -sfn "checkouts/notebooks/$NB_SUBDIR" "$REPO_ROOT/notebooks"

echo "run-ci: starting pass $TS (image=$IMAGE, notebooks=$NB_REPO@$(git -C "$CHECKOUT" rev-parse --short HEAD))"

# Queue mode (NBR_QUEUE=1 in .env): the pass hands its notebooks to the
# authoring worker (authoring/worker.py), which runs them in parallel on free
# runner slots after any author jobs, instead of running them here one by one.
QUEUE_ARGS=()
if [[ "${NBR_QUEUE:-}" == "1" ]]; then
  mkdir -p "$REPO_ROOT/jobs"
  QUEUE_ARGS=(-e NBR_QUEUE=1 -e NBR_JOBS_DIR=/work/jobs
              -v "$REPO_ROOT/jobs:/work/jobs:rw"
              -v "$REPO_ROOT/authoring:/work/authoring:ro")
fi

# --- run ------------------------------------------------------------------
# Mounts, and why each one:
#   results/, fixes/, logs/  read-write  — the pass's actual output
#   STATUS.md                read-write  — Phase C rewrites it
#   runner.yaml              read-only   — the targets notebooks may run on
#   notebooks/               read-only   — the upstream checkout; the agent must
#                                          never edit it (patching happens on
#                                          the GPU node, on a copy)
#   kubeconfig (optional)    read-only   — staged NBR_KUBECONFIG, k8s runner
#   tools/ (incl. http_ssh_shim/) read-only
#
# --read-only on the root filesystem with tmpfs for the writable spots: the
# agent runs model-authored shell commands, and this bounds what a bad one can
# scribble on. /home/agent is a tmpfs because the CLI writes session state there.
#
# .env is not mounted: the agent runs with --dangerously-skip-permissions and
# could read every secret in it. Only the variables it needs are passed with -e.
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
  -e HF_TOKEN \
  -e AGENT_SECRET \
  -e RADEON_CLOUD_API_TOKEN \
  ${KUBE_ARGS[@]+"${KUBE_ARGS[@]}"} \
  ${QUEUE_ARGS[@]+"${QUEUE_ARGS[@]}"} \
  -v "$REPO_ROOT/run.py:/work/run.py:ro" \
  -v "$REPO_ROOT/tools:/work/tools:ro" \
  -v "$REPO_ROOT/CLAUDE.md:/work/CLAUDE.md:ro" \
  -v "$REPO_ROOT/CLAUDE_FIX.md:/work/CLAUDE_FIX.md:ro" \
  -v "$CHECKOUT/$NB_SUBDIR:/work/notebooks:ro" \
  -v "$REPO_ROOT/results:/work/results:rw" \
  -v "$REPO_ROOT/fixes:/work/fixes:rw" \
  -v "$REPO_ROOT/logs:/work/logs:rw" \
  -v "$REPO_ROOT/STATUS.md:/work/STATUS.md:rw" \
  -v "$REPO_ROOT/runner.yaml:/work/runner.yaml:ro" \
  "$IMAGE" \
  python tools/ci_run.py --no-commit "$@" && rc=0 || rc=$?

# ci_run.py writes the pass log (logs/ci_<UTC timestamp>.log) itself.
echo "run-ci: pass $TS finished (exit=$rc)"
exit "$rc"
