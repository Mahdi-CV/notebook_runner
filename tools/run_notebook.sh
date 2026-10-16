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
# (a file next to the notebook), TIMEOUT_MINUTES, PAPERMILL_ARGS (array),
# ASSETS (array: files pushed with the notebook that it reads from ../assets),
# SANDBOX (0: install into and run with the image's Python, for notebooks whose
# tools start their own processes with it, such as Ray workers).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INPUT=$1
OUTPUT=$2
SYSTEM_PACKAGES=""
REQUIREMENTS=""
TIMEOUT_MINUTES=50
PAPERMILL_ARGS=()
ASSETS=()
SANDBOX=1
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

if [[ $SANDBOX == 0 ]]; then
  PY=$(command -v python3)
  log "no sandbox: using $PY"
  export PIP_BREAK_SYSTEM_PACKAGES=1
elif ! command -v uv >/dev/null; then
  log "installing uv"
  # Images whose system Python is externally managed (PEP 668) refuse a plain
  # pip install; uv itself is the only thing installed outside the venv.
  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q --user uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi

[[ $SANDBOX == 0 ]] || PY="$VENV/bin/python"
if [[ $SANDBOX != 0 && ! -x "$VENV/bin/python" ]]; then
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
        f.write(f"import site; list(map(site.addsitedir, {extra!r}))\n")
'
fi
deps=(papermill ipykernel)
[[ -n "$REQUIREMENTS" ]] && deps+=(-r "$REQUIREMENTS")
# Hold the image's PyTorch stack: a requirement that cannot use it should fail
# here, not quietly replace the ROCm build with a PyPI (CUDA) one.
CONSTRAINTS="$PWD/.nbr-image-constraints.txt"
"$PY" - > "$CONSTRAINTS" <<'PY'
from importlib.metadata import PackageNotFoundError, version
for name in ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm"):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        pass
PY
log "installing ${deps[*]} (holding: $(tr '\n' ' ' < "$CONSTRAINTS"))"
# pip, not uv: uv does not count the image packages the venv inherits as
# installed, so anything that depends on torch would pull a second one.
"$PY" -m pip install -q --disable-pip-version-check -c "$CONSTRAINTS" "${deps[@]}"

# PYTHONPATH: the image's own console scripts (torchrun, accelerate, deepspeed)
# run the image Python, which would not see what the notebook installs here.
if [[ $SANDBOX == 0 ]]; then
  "$PY" -m ipykernel install --user --name nbr --display-name "Python 3 (image)" >/dev/null
else
  SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  "$VENV/bin/python" -m ipykernel install --prefix "$VENV" --name nbr \
    --display-name "Python 3 (notebook sandbox)" \
    --env PATH "$VENV/bin:$PATH" --env VIRTUAL_ENV "$VENV" \
    --env PYTHONPATH "$SITE${PYTHONPATH:+:$PYTHONPATH}" >/dev/null
fi

# In the repository, notebooks read shared files from ../assets: run the kernel
# one level down so the pushed copies are found the same way.
if ((${#ASSETS[@]})); then
  mkdir -p assets notebook
  cp "${ASSETS[@]}" assets/
  PAPERMILL_ARGS+=(--cwd notebook)
fi

log "running papermill (timeout ${TIMEOUT_MINUTES}m)"
exec timeout "${TIMEOUT_MINUTES}m" "$PY" -m papermill "$INPUT" "$OUTPUT" \
  --kernel nbr --execution-timeout 3000 "${PAPERMILL_ARGS[@]}"
