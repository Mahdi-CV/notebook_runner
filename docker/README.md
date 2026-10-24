# Containerized CI for the notebook regression agent

The image runs the **orchestrator**, not the notebooks. Inside it: Python,
`kubectl` and the Claude Code CLI. `run.py` spawns the CLI, the CLI
drives the GPU node, and all ROCm/papermill work happens there in AMD's own
images. This container never needs a GPU, `/dev/kfd`, or ROCm.

## Build

```bash
docker build -f docker/Dockerfile -t notebook-runner:latest .
```

## Run

```bash
docker/run-ci.sh                 # full pass (verify, then auto-fix)
docker/run-ci.sh --dry-run       # print the plan, run nothing
docker/run-ci.sh --category inference --verify-only
docker/notebook-runner-cron.sh   # the daily unattended pass (verify-only)
```

Arguments pass through to `tools/ci_run.py`. Container passes never git-commit;
results land in `results/` and the dashboard.

---

## Claude credentials

**On the AMD network the agent does not use a first-party Anthropic API key.**
It authenticates through the AMD enterprise gateway, configured entirely by
environment variables:

| Variable | Role |
|---|---|
| `ANTHROPIC_BASE_URL` | Gateway endpoint (`https://llm-api.amd.com/...`). Without it the CLI calls `api.anthropic.com` and gets **401**. |
| `ANTHROPIC_CUSTOM_HEADERS` | The APIM subscription header (`Ocp-Apim-...`). This is the real secret. |
| `ANTHROPIC_API_KEY` | Must be non-empty or the CLI won't start. The gateway ignores the value — `dummy` is correct. |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` | Gateway-side model names; the public IDs are not valid here. |

These live in `.env` (gitignored, `chmod 600`). `run-ci.sh` sources it and
forwards each var with a bare `-e NAME`, so nothing is baked into the image.

> **Quote every value in `.env`.** `run-ci.sh` `source`s the file, and bash is
> less forgiving than `python-dotenv`: an unquoted
> `Ocp-Apim-Subscription-Key: <key>` parses as a command, and
> `Claude-Sonnet-5[1m]` as a glob.

> **`claude auth status` is not a validity check.** It reports
> `"loggedIn": true` whenever a key is merely *present*. Use the smoke test.

### Smoke test

```bash
set -a; source .env; set +a
docker run --rm \
  -e ANTHROPIC_API_KEY -e ANTHROPIC_BASE_URL -e ANTHROPIC_CUSTOM_HEADERS \
  -e ANTHROPIC_MODEL -e ANTHROPIC_DEFAULT_HAIKU_MODEL \
  notebook-runner:latest \
  claude -p 'reply with exactly: CONTAINER_OK' \
    --dangerously-skip-permissions --output-format json
```

Expect `"result": "CONTAINER_OK"` and `"is_error": false`. A
`"api_error_status": 401` means the gateway variables did not reach the
container.

Off the AMD network, a real `ANTHROPIC_API_KEY=sk-ant-...` works with the two
gateway variables unset, but `run-ci.sh` requires them, so that check would
need relaxing.

---

## Two hard runtime constraints

**1. The agent must not run as root.** `run.py` passes
`--dangerously-skip-permissions`, and the CLI refuses it under uid 0, so the
image runs as user `agent` (uid 1001).

**2. Nothing the runtime needs may live under `/home/agent`.** `run-ci.sh`
mounts a tmpfs there so the CLI has a writable HOME under a read-only rootfs,
and that mount **shadows whatever the build put there**. That's why the
Dockerfile relocates the CLI to `/opt/claude`; left in `~/.local`, the image
fails at runtime with `exec claude failed: No such file or directory`.

---

## Targets and notebooks

`runner.yaml` (from `runner.example.yaml`, mounted read-only) lists the GPU
hosts and k8s pools; each notebook runs on the first one that meets the
requirements it declares (`python3 tools/routing.py` prints every route). An
SSH target reachable only over HTTP(S) gets an `agent_url`, and `run.py` sends
its `ssh`/`scp` through the HMAC-signed shim in `tools/http_ssh_shim/`
(`AGENT_SECRET` in `.env`); k8s targets need `NBR_KUBECONFIG` (see
`tools/k8s_pod.py`).

Each pass first updates `checkouts/notebooks` to the tip of
`NBR_NOTEBOOKS_REPO`@`NBR_NOTEBOOKS_REF` (default ROCm/gpuaidev@main) and
points `notebooks` at its `NBR_NOTEBOOKS_SUBDIR`.
