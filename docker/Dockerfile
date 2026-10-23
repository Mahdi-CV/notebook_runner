# Notebook regression agent — orchestrator image.
#
# This container does NOT run notebooks. It runs run.py, which spawns the
# Claude Code CLI, which drives the GPU node (over the HTTP shim in
# tools/http_ssh_shim/, or in a k8s pod via tools/k8s_pod.py). No GPU, no
# /dev/kfd, no ROCm in here — just Python, kubectl, and the claude binary.
#
# Build:  docker build -f docker/Dockerfile -t notebook-runner:latest .
# Run:    see docker/run-ci.sh

FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------------------
# System deps
#   ripgrep        : the agent's Grep tool prefers rg and is much slower without
#   ca-certificates: TLS to the Claude API gateway and github.com
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ripgrep \
        tini \
    && rm -rf /var/lib/apt/lists/*

# kubectl for the k8s runner (tools/k8s_pod.py). Pinned to the cluster's minor
# version (kubectl supports ±1 minor) and checked against the published sha256.
ARG KUBECTL_VERSION=v1.34.1
RUN curl -fsSLo /usr/local/bin/kubectl \
        "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && echo "$(curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl.sha256")  /usr/local/bin/kubectl" \
        | sha256sum -c - \
    && chmod 755 /usr/local/bin/kubectl

# Non-root user: `claude --dangerously-skip-permissions` refuses to run as
# uid 0. The uid is fixed so bind-mounted output dirs keep stable ownership.
ARG AGENT_UID=1001
ARG AGENT_GID=1001
RUN groupadd -g "${AGENT_GID}" agent \
    && useradd -m -u "${AGENT_UID}" -g "${AGENT_GID}" -s /bin/bash agent

# Python deps from uv.lock (`--frozen` fails if it disagrees with pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock /build/
RUN cd /build \
    && uv sync --frozen --no-dev --no-install-project \
    && chown -R "${AGENT_UID}:${AGENT_GID}" /opt/venv

ENV PATH="/opt/venv/bin:/home/agent/.local/bin:${PATH}"

# Claude Code CLI, pinned: CLI changes can change verdicts, so bump deliberately.
# Installed as the agent user, then moved to /opt/claude because run-ci.sh
# mounts a tmpfs over /home/agent, which would hide anything left in ~/.local.
ARG CLAUDE_VERSION=2.1.308
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

WORKDIR /work

# tini reaps the children run.py leaves behind when a run is killed mid-flight.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "tools/ci_run.py", "--no-commit"]
