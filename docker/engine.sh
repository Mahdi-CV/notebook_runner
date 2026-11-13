# The container engine, for the scripts in docker/ (sourced, after REPO_ROOT is set).
#
# docker (the default) runs as root, and the containers write as uid 1001 on
# the host. podman (NBR_CONTAINER_CLI=podman, in the environment or .env) runs
# rootless as the service account: the containers still run as uid 1001, the
# image's agent user, but everything they write belongs to that account on the
# host, so nothing needs root or a chown.
#
# Sets:
#   CLI          the engine command
#   USERNS       docker run flags mapping the service account to uid 1001
#   RESTART      the restart policy of long-running containers (podman's
#                podman-restart.service brings back only "always" at boot)
#   AGENT_UID    the host uid the containers write as
#   AGENT_TMPFS  --tmpfs options giving the mount to uid 1001 (podman has no uid=)
# and agent_owns PATH..., which hands paths to that uid (a no-op under podman).

CLI="${NBR_CONTAINER_CLI:-$(sed -n "s/^NBR_CONTAINER_CLI=[\"']\{0,1\}\([a-z]*\).*/\1/p" "$REPO_ROOT/.env" 2>/dev/null)}"
CLI="${CLI:-docker}"
if [[ "$CLI" == podman ]]; then
  USERNS=(--userns=keep-id:uid=1001,gid=1001)
  RESTART=(--restart always)
  AGENT_UID="$(id -u)"
  AGENT_TMPFS=U
else
  USERNS=()
  RESTART=(--restart unless-stopped)
  AGENT_UID=1001
  AGENT_TMPFS=uid=1001,gid=1001
fi

agent_owns() {
  [[ "$CLI" == podman ]] && return 0
  chown "$AGENT_UID:$AGENT_UID" "$@"
}
