# Notebook regression agent — orchestrator image.
#
# This container does NOT run notebooks. It runs run.py, which spawns the
# Claude Code CLI, which SSHes to a GPU node (ssh eddie) and drives Docker
# there. No GPU, no /dev/kfd, no ROCm in here — just Python, git, ssh, and
# the claude binary.
#
# Build:  docker build -f docker/Dockerfile -t notebook-runner:latest .
# Run:    see docker/run-ci.sh

FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------------------
# System deps
#   openssh-client : run.py shells out to ssh/scp for every GPU command
#   git            : Phase D commits STATUS.md; reporter needs it for PR diffs
#   ripgrep        : the agent's Grep tool prefers rg and is much slower without
#   ca-certificates: TLS to api.anthropic.com and github.com
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
        ripgrep \
        tini \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Non-root user — NOT optional.
#
# run.py spawns `claude --dangerously-skip-permissions`, and the CLI hard-exits
# under uid 0:
#   "--dangerously-skip-permissions cannot be used with root/sudo privileges"
# There is an IS_SANDBOX=1 escape hatch, but running the agent as a normal user
# is the honest fix: it keeps the blast radius of a rogue Bash call inside the
# container's own $HOME instead of its root filesystem.
#
# uid 1001 is arbitrary but fixed, so a bind-mounted results/ dir has stable
# ownership on the host across rebuilds.
# ---------------------------------------------------------------------------
ARG AGENT_UID=1001
ARG AGENT_GID=1001
RUN groupadd -g "${AGENT_GID}" agent \
    && useradd -m -u "${AGENT_UID}" -g "${AGENT_GID}" -s /bin/bash agent

# ---------------------------------------------------------------------------
# Python deps, installed via uv into a venv the agent user owns.
#
# uv.lock is the source of truth — `uv sync --frozen` fails loudly if the lock
# and pyproject.toml disagree, so a CI image can never silently drift onto a
# different pyyaml than the one that was tested.
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock /build/
RUN cd /build \
    && uv sync --frozen --no-dev --no-install-project \
    && chown -R "${AGENT_UID}:${AGENT_GID}" /opt/venv

ENV PATH="/opt/venv/bin:/home/agent/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# Claude Code CLI.
#
# Pinned by version, not "latest" — the agent's behaviour is part of what this
# pipeline measures, so an unannounced CLI bump must not silently change the
# verdicts. Bump CLAUDE_VERSION deliberately and re-baseline.
#
# Installed as the agent user (the installer writes to ~/.local and a
# root-owned install is unreadable to the user that runs it), then relocated
# to /opt/claude.
#
# The relocation is load-bearing, not tidiness: run-ci.sh mounts a tmpfs over
# /home/agent so the CLI has a writable HOME under a read-only rootfs. That
# mount SHADOWS anything baked into /home/agent at build time — leaving the
# binary in ~/.local produces "exec claude failed: No such file or directory"
# at runtime while the image itself looks perfectly healthy.
# ---------------------------------------------------------------------------
ARG CLAUDE_VERSION=2.1.226
USER agent
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- "${CLAUDE_VERSION}" \
    && claude --version

USER root
RUN mv /home/agent/.local/share/claude /opt/claude \
    && ln -sf "/opt/claude/versions/${CLAUDE_VERSION}" /usr/local/bin/claude \
    && chown -R "${AGENT_UID}:${AGENT_GID}" /opt/claude \
    && rm -rf /home/agent/.local
USER agent
RUN claude --version

# ---------------------------------------------------------------------------
# SSH client config.
#
# Installed to /etc/ssh/ssh_config.d/ (pulled in by the stock Include on line
# 19 of /etc/ssh/ssh_config) rather than ~/.ssh/config — same tmpfs-shadowing
# reason as the CLI above: /home/agent is replaced at runtime, so a per-user
# config baked here would silently vanish and every ssh would fall back to
# defaults (no BatchMode → a hung pass instead of a clean failure).
#
# The GPU host key arrives at runtime via a mounted known_hosts (see
# run-ci.sh) rather than StrictHostKeyChecking=no, so a MITM on the GPU
# network can't silently become "the GPU server".
# ---------------------------------------------------------------------------
USER root
COPY docker/ssh_config /etc/ssh/ssh_config.d/10-notebook-runner.conf
RUN chmod 644 /etc/ssh/ssh_config.d/10-notebook-runner.conf
USER agent

WORKDIR /work

# tini reaps the ssh and claude children that run.py leaves behind when a run
# is killed mid-flight; without a real init, a timed-out pass leaks zombies.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "tools/ci_run.py", "--push"]
