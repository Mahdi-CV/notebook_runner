#!/usr/bin/env bash
#
# Execute a notebook with papermill in a sandbox environment. run.py pushes
# this next to the notebook (with run_notebook.env, written from the notebook's
# metadata.ci) to every target, so a notebook runs the same way everywhere:
#
#   1. install the declared system packages (apt), if any are missing;
#   2. create a virtual environment on top of the image's Python with uv, which
#      needs neither python3-venv nor a pip the system lets it use; the image
#      Python's packages (its ROCm PyTorch) stay importable;
#   3. install the declared requirements, papermill and ipykernel into it;
#   4. register it as the papermill kernel with its bin/ first on PATH, so the
#      notebook's own `!pip` / `%pip` cells install into it too;
#   5. run papermill under the run's timeout.
#
# Usage: bash run_notebook.sh INPUT.ipynb OUTPUT.ipynb
# Settings (run_notebook.env, all optional): SYSTEM_PACKAGES, REQUIREMENTS
# (a file next to the notebook), TIMEOUT_MINUTES, PAPERMILL_ARGS (array).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INPUT=$1
OUTPUT=$2
SYSTEM_PACKAGES=""
REQUIREMENTS=""
TIMEOUT_MINUTES=50
PAPERMILL_ARGS=()
# shellcheck disable=SC1091
[[ -f run_notebook.env ]] && source run_notebook.env
VENV="$PWD/.nbr-venv"

log() { echo "[run_notebook] $*"; }

SUDO=""
[[ $(id -u) -ne 0 ]] && command -v sudo >/dev/null && SUDO="sudo -n"

missing=()
for pkg in $SYSTEM_PACKAGES; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if ((${#missing[@]})); then
  log "installing system packages: ${missing[*]}"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}"
fi

if ! command -v uv >/dev/null; then
  log "installing uv"
  # Images whose system Python is externally managed (PEP 668) refuse a plain
  # pip install; uv itself is the only thing installed outside the venv.
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q --user uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating $VENV on top of $(command -v python3)"
  uv venv -q --seed --system-site-packages --python "$(command -v python3)" "$VENV"
  # --system-site-packages only exposes the base interpreter's packages. When the
  # image's python3 is itself a venv or conda env (e.g. /opt/venv in
  # rocm/comfyui), its own site-packages hold the ROCm PyTorch, so add them too,
  # with site.addsitedir so their .pth files (eggs, editable installs) apply.
  python3 -c 'import site; print("\n".join(site.getsitepackages()))' | "$VENV/bin/python" -c '
import os, sys, sysconfig
extra = [p for p in sys.stdin.read().split() if os.path.isdir(p) and p not in sys.path]
if extra:
    with open(sysconfig.get_path("purelib") + "/_nbr_image_site.pth", "w") as f:
        f.write(f"import site; [site.addsitedir(p) for p in {extra!r}]\n")
'
fi
deps=(papermill ipykernel)
[[ -n "$REQUIREMENTS" ]] && deps+=(-r "$REQUIREMENTS")
log "installing ${deps[*]}"
uv pip install -q --python "$VENV/bin/python" "${deps[@]}"

"$VENV/bin/python" -m ipykernel install --prefix "$VENV" --name nbr \
  --display-name "Python 3 (notebook sandbox)" \
  --env PATH "$VENV/bin:$PATH" --env VIRTUAL_ENV "$VENV" >/dev/null

log "running papermill (timeout ${TIMEOUT_MINUTES}m)"
exec timeout "${TIMEOUT_MINUTES}m" "$VENV/bin/python" -m papermill "$INPUT" "$OUTPUT" \
  --kernel nbr --execution-timeout 3000 "${PAPERMILL_ARGS[@]}"
