# Containerized CI for the notebook regression agent

The image runs the **orchestrator**, not the notebooks. Inside it: Python,
`git`, `ssh`, and the Claude Code CLI. `run.py` spawns the CLI, the CLI SSHes to
the GPU node, and all ROCm/papermill work happens there in AMD's own images.
This container never needs a GPU, `/dev/kfd`, or ROCm.

## Build

```bash
docker build -f docker/Dockerfile -t notebook-runner:latest .
```

## Run

```bash
docker/run-ci.sh                 # full 4-phase pass, commit + push
docker/run-ci.sh --dry-run       # print the plan, run nothing
docker/run-ci.sh --category inference --no-commit
```

Arguments pass through to `tools/ci_run.py`.

---

## Claude credentials — what the CLI actually needs

**On the AMD network the agent does not use a first-party Anthropic API key.**
It authenticates through the AMD enterprise gateway, which is configured
entirely by environment variables:

| Variable | Role |
|---|---|
| `ANTHROPIC_BASE_URL` | Gateway endpoint (`https://llm-api.amd.com/...`). Without it the CLI calls `api.anthropic.com` and gets **401**. |
| `ANTHROPIC_CUSTOM_HEADERS` | The APIM subscription header (`Ocp-Apim-...`). This is the real secret. |
| `ANTHROPIC_API_KEY` | Must be non-empty or the CLI won't start. The gateway ignores the value — `dummy` is correct. |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` | Gateway-side model names; the public IDs are not valid here. |

These live in `.env` (gitignored, `chmod 600`). `run-ci.sh` sources it and
forwards each var with a bare `-e NAME`, so nothing is baked into the image.
There is no `claude auth login` and no `~/.claude/.credentials.json` — an
interactive OAuth login is the wrong tool for a headless CI box.

> **Quote every value in `.env`.** `run-ci.sh` `source`s the file so the CLI
> inherits the vars, and bash is less forgiving than `python-dotenv`: an
> unquoted `ANTHROPIC_CUSTOM_HEADERS=Ocp-Apim-Subscription-Key: <key>` parses
> as a command plus argument and dies with `<key>: command not found`, while
> `Claude-Sonnet-5[1m]` is treated as a glob. Both readers accept quotes.

> **`claude auth status` is not a validity check.** It reports
> `"loggedIn": true` whenever a key is merely *present*. With
> `ANTHROPIC_API_KEY=dummy` and no gateway URL it still says logged-in, then
> every real request returns 401. Trust the smoke test below, not `auth status`.

### Smoke test (the only check that proves credentials work)

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

### Alternative: a first-party key

If you run this outside the AMD network with a real Anthropic key, set
`ANTHROPIC_API_KEY=sk-ant-...` and **unset** `ANTHROPIC_BASE_URL` /
`ANTHROPIC_CUSTOM_HEADERS`. `run-ci.sh` currently *requires* the gateway
variables; relax that check for this case.

A subscription-based `claude setup-token` long-lived token
(`ANTHROPIC_AUTH_TOKEN`) also works, but it is tied to a human's Claude
subscription — a poor fit for shared CI and not what this image assumes.

---

## Two hard runtime constraints

**1. The agent must not run as root.** `run.py` passes
`--dangerously-skip-permissions`, and the CLI hard-exits under uid 0:
`"--dangerously-skip-permissions cannot be used with root/sudo privileges"`.
The image therefore creates and runs as user `agent` (uid 1001). There is an
`IS_SANDBOX=1` bypass; running as a normal user is the better answer.

**2. Nothing the runtime needs may live under `/home/agent`.** `run-ci.sh`
mounts a tmpfs there so the CLI has a writable HOME under a read-only rootfs,
and that mount **shadows whatever the build put there**. This is why the
Dockerfile relocates the CLI to `/opt/claude` and the SSH config to
`/etc/ssh/ssh_config.d/`. Leaving the CLI in `~/.local` builds a perfectly
healthy-looking image that dies at runtime with:

```
[FATAL tini (7)] exec claude failed: No such file or directory
```

---

## SSH key

The pipeline uses a **dedicated application keypair**, not a personal key:
`~/.ssh/id_nbrunner_ed25519` (override with `GPU_SSH_KEY` in `.env`). It is
mounted into a container that executes model-authored shell commands, so it
should unlock exactly one host. Revoking it means deleting one line from the
GPU node's `authorized_keys` — not rotating a human identity.

Create and install:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_nbrunner_ed25519 -N '' -C notebook-runner-ci
ssh-copy-id -i ~/.ssh/id_nbrunner_ed25519.pub root@134.199.202.42
```

Passphrase-less is deliberate — an unattended cron pass has no tty to unlock a
key — which makes file custody the only protection. Keep it `0600`.

> **The key is staged, not mounted directly.** A correct `0600` key is owned by
> you; the container runs as uid 1001; bind mounts preserve host ownership. So
> mounting the original yields
> `Load key "/home/agent/.ssh/id_gpu": Permission denied`, which then surfaces
> as `Permission denied (publickey)` — indistinguishable from a key the server
> rejected. `run-ci.sh` copies it to a uid-1001 `0600` staging file under
> `logs/` and deletes it on exit.

## Network topology (read before deploying)

Measured reachability:

| From | Gateway (`llm-api.amd.com`) | GPU node (`134.199.202.42:22`) |
|---|---|---|
| Your workstation | reachable | reachable |
| `xcovdashdbd01` | reachable (10.214.150.4) | **blocked** |
| GPU node (`eddie`) | **no DNS** | — |

The orchestrator needs **both**. Today only the workstation has both, so
`xcovdashdbd01` cannot yet run unattended passes — see `HANDOFF.md` for the
options (firewall exception vs. SSH tunnel vs. running on a host with both).
