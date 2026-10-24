"""
Decide where a notebook runs: match what it needs to the targets this runner has.

A notebook declares its needs in its own metadata (`metadata.ci`, documented
upstream in ROCm/gpuaidev .github/CI_METADATA.md); the targets are this
deployment's private runner.yaml (see runner.example.yaml). Neither knows about
the other: a notebook never names a machine, and a target never names a
notebook. The one exception is holds: notebooks held off targets where a
known infrastructure problem stops them, with the reason, so the nightly pass
does not spend a run rediscovering it. People set them on the dashboard; the
authoring service keeps them in jobs/holds.json (authoring/holds.py).

Usage:
  python3 tools/routing.py [--dir notebooks/] [--target NAME]   # print every route
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "runner.yaml"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Product -> architecture -> families. Requirements and targets may use any of
# the three; a target matches when the names overlap.
PRODUCT_ARCH = {
    "mi300x": "gfx942", "mi308x": "gfx942", "mi325x": "gfx942",
    "mi350x": "gfx950", "mi355x": "gfx950",
    "mi250x": "gfx90a", "mi210": "gfx90a",
    "mi100": "gfx908",
    "rx7900xtx": "gfx1100", "w7900": "gfx1100",
    "rx9070xt": "gfx1201",
    "radeon8060s": "gfx1151",
}
ARCH_FAMILIES = {
    "gfx942": ("instinct", "cdna3"),
    "gfx950": ("instinct", "cdna4"),
    "gfx90a": ("instinct", "cdna2"),
    "gfx908": ("instinct", "cdna1"),
    "gfx1100": ("radeon", "rdna3"),
    "gfx1201": ("radeon", "rdna4"),
    "gfx1151": ("radeon", "rdna35"),
}
KNOWN_HARDWARE = (set(PRODUCT_ARCH) | set(ARCH_FAMILIES)
                  | {f for fams in ARCH_FAMILIES.values() for f in fams})

DEFAULTS = {"hardware": ["instinct"], "gpus": 1, "memory_gb": 64, "disk_gb": 100}
DEFAULT_TIMEBOX_MINUTES = 60  # runner.yaml timebox_minutes overrides

# A code cell that starts containers itself needs a Docker daemon on the target.
DOCKER_RE = re.compile(r"\bdocker\s+(run|exec|pull|compose|build|start)\b")


@dataclass
class Requirements:
    hardware: list[str]
    gpus: int
    memory_gb: int
    disk_gb: int
    gpu_memory_gb: int | None = None
    capabilities: set[str] = field(default_factory=set)
    env: dict[str, str] = field(default_factory=dict)
    image: str | None = None
    parameters: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    timeout_minutes: int | None = None
    requirements: list[str] | str | None = None  # pip specs, or a file next to the notebook
    system_packages: list[str] = field(default_factory=list)
    sandbox: bool = True  # False: run in the image's Python, not a sandbox venv


def hardware_names(hardware: str) -> set[str]:
    """Every name a piece of hardware answers to: product, arch, families."""
    hw = hardware.lower()
    arch = PRODUCT_ARCH.get(hw, hw)
    return {hw, arch, *ARCH_FAMILIES.get(arch, ())}


def requirements(nb_path: Path) -> Requirements:
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    ci = {**DEFAULTS, **(nb.get("metadata", {}).get("ci") or {})}
    capabilities = set(ci.get("capabilities") or [])
    if any(c.get("cell_type") == "code" and DOCKER_RE.search("".join(c.get("source", "")))
           for c in nb.get("cells", [])):
        capabilities.add("docker")
    return Requirements(
        hardware=[h.lower() for h in ci["hardware"]], gpus=int(ci["gpus"]),
        memory_gb=int(ci["memory_gb"]), disk_gb=int(ci["disk_gb"]),
        gpu_memory_gb=ci.get("gpu_memory_gb"), capabilities=capabilities,
        env=dict(ci.get("env") or {}), image=ci.get("image"),
        parameters=dict(ci.get("parameters") or {}),
        depends_on=list(ci.get("depends_on") or []),
        timeout_minutes=int(ci["timeout_minutes"]) if ci.get("timeout_minutes") else None,
        requirements=ci.get("requirements") or None,
        system_packages=list(ci.get("system_packages") or []),
        sandbox=bool(ci.get("sandbox", True)),
    )


def load_config(path: Path = CONFIG_PATH) -> dict:
    """runner.yaml as {"targets": {name: target}, "timebox_minutes": N}, with
    the dashboard's holds (holds_path) as "holds": {notebook key: {reason,
    targets, since, by, at}}; no targets if runner.yaml is missing."""
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    try:
        holds = json.loads(holds_path(path).read_text())
    except (OSError, ValueError):
        holds = {}
    return {"targets": data.get("targets") or {},
            "timebox_minutes": int(data.get("timebox_minutes") or DEFAULT_TIMEBOX_MINUTES),
            "holds": {k: e for k, e in holds.items() if isinstance(e, dict) and e.get("reason")}}


def holds_path(config_path: Path = CONFIG_PATH) -> Path:
    """jobs/holds.json beside runner.yaml (or in NBR_JOBS_DIR)."""
    jobs = os.environ.get("NBR_JOBS_DIR")
    return (Path(jobs) if jobs else Path(config_path).parent / "jobs") / "holds.json"


def holds_for(key: str, config: dict) -> dict[str, str]:
    """{target name: reason} for the targets a hold keeps this notebook (its key
    under the notebook root) off. A hold without `targets` covers every target."""
    entry = (config.get("holds") or {}).get(key)
    if not entry:
        return {}
    return {name: str(entry["reason"]) for name in (entry.get("targets") or config["targets"])}


def target_capabilities(target: dict) -> set[str]:
    caps = set(target.get("capabilities") or [])
    if target.get("type", "ssh") == "ssh":
        caps.add("docker")  # SSH targets run the notebook's containers on the host
    return caps


def unmet(req: Requirements, target: dict) -> list[str]:
    """What `target` lacks for `req`; empty when it fits."""
    missing = []
    if not hardware_names(str(target.get("hardware", ""))) & set(req.hardware):
        missing.append(f"hardware {'/'.join(req.hardware)}")
    if int(target.get("gpus", 1)) < req.gpus:
        missing.append(f"{req.gpus} GPUs")
    if req.gpu_memory_gb and int(target.get("gpu_memory_gb", 0)) < req.gpu_memory_gb:
        missing.append(f"{req.gpu_memory_gb} GB per GPU")
    lacking = req.capabilities - target_capabilities(target)
    if lacking:
        missing.append(", ".join(sorted(lacking)))
    return missing


def route(req: Requirements, targets: dict, only: str | None = None,
          held: dict[str, str] | None = None) -> tuple[str | None, dict | None, str]:
    """Return (name, target, "") for the first target in config order that fits
    and the notebook is not held off (`held`, from holds_for), or (None, None,
    reason) when none does: "held: ..." when only held targets fit."""
    unknown = [h for h in req.hardware if h not in KNOWN_HARDWARE]
    if unknown:
        return None, None, f"unknown hardware {', '.join(unknown)} in metadata.ci"
    candidates = {only: targets[only]} if only in targets else ({} if only else targets)
    if not candidates:
        return None, None, (f"no target named {only}" if only else "no targets configured")
    reasons, held_off = [], []
    for name, target in candidates.items():
        missing = unmet(req, target)
        if not missing and name in (held or {}):
            held_off.append(name)
        elif not missing:
            return name, target, ""
        else:
            reasons.append(f"{name} lacks {'; '.join(missing)}")
    if held_off:
        return None, None, (f"held: known infrastructure issue on {', '.join(held_off)}: "
                            f"{held[held_off[0]]}")
    return None, None, "no target fits: " + " | ".join(reasons)


def misses(req: Requirements, targets: dict, only: str | None = None
           ) -> list[tuple[str, list[str]]]:
    """[(name, what it lacks)] for each candidate target that does not fit, in
    config order: the parts of route's "no target fits" reason, for listing
    one target per line."""
    candidates = {only: targets[only]} if only in targets else ({} if only else targets)
    return [(name, missing) for name, target in candidates.items()
            if (missing := unmet(req, target))]


def resolve(nb_path: Path, targets: dict, only: str | None = None,
            held: dict[str, str] | None = None
            ) -> tuple[Requirements | None, str | None, dict | None, str]:
    """(requirements, target name, target, reason) for one notebook file. A
    notebook whose metadata.ci cannot be read gets no target, with the reason."""
    try:
        req = requirements(nb_path)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return None, None, None, f"invalid notebook or metadata.ci: {e}"
    return (req, *route(req, targets, only, held))


def notebook_keys(notebooks_dir: Path = NOTEBOOKS_DIR) -> list[str]:
    """Every notebook, as its path under the notebook root (the result key)."""
    if not notebooks_dir.is_dir():
        return []
    return sorted(str(p.relative_to(notebooks_dir))
                  for p in notebooks_dir.rglob("*.ipynb")
                  if ".ipynb_checkpoints" not in p.parts)


def routes(notebooks_dir: Path = NOTEBOOKS_DIR, config: dict | None = None,
           only: str | None = None) -> dict[str, tuple[str | None, str]]:
    """{key: (target name or None, reason)} for every notebook. With no targets
    configured (e.g. a dashboard without runner.yaml) every reason is empty."""
    config = config or load_config()
    targets = config["targets"]
    out = {}
    for key in notebook_keys(notebooks_dir):
        if not targets:
            out[key] = (None, "")
            continue
        _, name, _, reason = resolve(notebooks_dir / key, targets, only, holds_for(key, config))
        out[key] = (name, reason)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Print where each notebook would run.")
    p.add_argument("--dir", default=str(NOTEBOOKS_DIR))
    p.add_argument("--target", default=None, help="Only consider this target.")
    args = p.parse_args()
    if not load_config()["targets"]:
        print(f"no targets configured in {CONFIG_PATH} (copy runner.example.yaml)")
        return 1
    for key, (name, reason) in routes(Path(args.dir), only=args.target).items():
        print(f"{key:70} {name or '— ' + reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
