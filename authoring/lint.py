"""Contribution-guideline checks that need no GPU.

These follow the ROCm AI Developer Hub tutorial contribution guidelines and
the annotate-notebook-for-ci skill: what a reviewer would otherwise check by
eye. Each finding is

  {"check": ..., "severity": "error" | "warning" | "info",
   "cell": index or None, "message": ..., "fix": ...}

An error is something the notebook must change before it can be published
(or before CI can run it); a warning needs the author's judgment; info is
context for the reviewer.

Usage:
  python3 -m authoring.lint NOTEBOOK.ipynb     # prints the findings as JSON
"""

import json
import re
import sys
from pathlib import Path

PRODUCT_NAMES = {
    "mi300x": "MI300X", "mi308x": "MI308X", "mi325x": "MI325X", "mi350x": "MI350X",
    "mi355x": "MI355X", "mi250x": "MI250X", "mi210": "MI210", "mi100": "MI100",
    "rx7900xtx": "7900 XTX", "w7900": "W7900", "rx9070xt": "9070 XT",
    "radeon8060s": "8060S",
}

PERF_CLAIM = re.compile(
    r"\d+(?:\.\d+)?\s*(?:x|×|times)\s+(?:faster|slower|speed)"
    r"|\d+(?:\.\d+)?\s*(?:x|×|times)\s+(?:higher|lower|better|more)\s+"
    r"(?:throughput|latency|performance|bandwidth|tokens)"
    r"|speed-?ups?\b|outperform\w*"
    r"|\d+(?:\.\d+)?\s*(?:%|percent)\s+(?:faster|slower|higher|lower|better|more throughput|less latency)"
    r"|\d[\d,.]*\s*(?:tokens?|tok)\s*(?:/|per\s+)s(?:ec(?:ond)?)?\b"
    r"|(?:faster|slower)\s+than\s+(?:the\s+)?(?:nvidia|h100|a100|h200|b200|cuda)",
    re.I)
HISTORY_NOTE = re.compile(
    r"earlier versions? of this (?:tutorial|notebook)|this tutorial (?:originally|previously) used",
    re.I)
HF_TOKEN_LITERAL = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
GIT_CLONE = re.compile(r"git\s+clone\b([^\n]*)")
MOVING_BRANCH = re.compile(r"(?:--branch|-b)[\s=]+(main|master|dev|develop|nightly)\b")
PINNED_CLONE = re.compile(r"(?:--branch|-b)[\s=]+\S+")
# AMD organisations whose copies of these projects are forks of an upstream
# that supports ROCm itself.
AMD_FORK = re.compile(
    r"github\.com[/:](?:ROCm|AMD-AIG-AIM|AMD-AGI|amd)/"
    r"(Megatron-LM|vllm|sglang|pytorch|torchtitan|transformers|flash-attention|triton|"
    r"DeepSpeed|llama\.cpp|xformers|bitsandbytes|unsloth|apex|TransformerEngine)\b", re.I)
SERVER_START = re.compile(
    r"(?:vllm\s+serve|sglang\.launch_server|ollama\s+serve|text-generation-launcher|"
    r"python3?\s+main\.py\s+--listen|jupyter[- ]lab)\b")
GPUS_REQUIRED = re.compile(r"GPUs?\s+required\W*?(\d+)", re.I)
LONG_CELL_LINES = 60


def source(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def tags(cell: dict) -> list[str]:
    return cell.get("metadata", {}).get("tags", []) or []


def finding(check, severity, cell, message, fix) -> dict:
    return {"check": check, "severity": severity, "cell": cell, "message": message, "fix": fix}


def lint(nb: dict) -> list[dict]:
    cells = nb.get("cells", [])
    ci = (nb.get("metadata") or {}).get("ci") or {}
    out: list[dict] = []
    out += _metadata(ci)
    out += _structure(cells, ci)
    out += _markdown_text(cells)
    out += _code(cells)
    out += _readability(cells)
    return out


def _metadata(ci: dict) -> list[dict]:
    out = []
    if not ci:
        out.append(finding(
            "ci-metadata", "error", None,
            "The notebook has no metadata.ci, so CI cannot tell which hardware to run it on.",
            'Add "ci": {"hardware": [...], "gpus": N} to the notebook metadata '
            "(see .github/CI_METADATA.md), or run the annotate tool."))
        return out
    for key in ("hardware", "gpus"):
        if key not in ci:
            out.append(finding("ci-metadata", "error", None,
                               f"metadata.ci has no `{key}`, which is required.",
                               f"Declare ci.{key} (see .github/CI_METADATA.md)."))
    return out


def _structure(cells: list[dict], ci: dict) -> list[dict]:
    out = []
    markdown = [(i, source(c)) for i, c in enumerate(cells) if c.get("cell_type") == "markdown"]
    if not markdown or not re.match(r"\s*#\s+\S", markdown[0][1]):
        out.append(finding(
            "introduction", "warning", markdown[0][0] if markdown else None,
            "The notebook does not open with a title and introduction.",
            "Start with a markdown cell: a `# Title`, then the goal, the tools used and "
            "what the reader will learn."))
    prereq = next(((i, s) for i, s in markdown if re.search(r"^#+\s*Prerequisites", s, re.I | re.M)),
                  None)
    if prereq is None:
        out.append(finding(
            "prerequisites", "error", None,
            "There is no Prerequisites section.",
            "Add a `## Prerequisites` section listing the AMD GPU model, GPUs required, "
            "ROCm version, Docker image, packages, models and tokens the tutorial needs."))
        return out
    idx, text = prereq
    # The Prerequisites section runs until the next heading of the same or higher level.
    section = text[text.lower().find("prerequisites"):]
    for i, s in markdown:
        if i > idx and re.match(r"\s*#{1,2}\s", s):
            break
        if i > idx:
            section += "\n" + s
    names = [PRODUCT_NAMES.get(h, h) for h in ci.get("hardware", []) if isinstance(h, str)]
    if names and not any(n.lower() in section.lower() for n in names):
        out.append(finding(
            "hardware", "warning", idx,
            f"The Prerequisites don't name the GPU the notebook declares ({', '.join(names)}).",
            "Name the AMD GPU model(s) the tutorial was tested on in the Prerequisites."))
    m = GPUS_REQUIRED.search(section)
    gpus = ci.get("gpus")
    if m is None:
        out.append(finding(
            "gpus-required", "warning", idx,
            "The Prerequisites don't state how many GPUs are required.",
            f"Add a line such as `* **GPUs required**: {gpus or 1}` to the Hardware section. "
            "If the default settings use more, say which settings the minimum needs."))
    elif isinstance(gpus, int) and int(m.group(1)) != gpus:
        out.append(finding(
            "gpus-required", "error", idx,
            f"The Prerequisites say {m.group(1)} GPUs are required, but metadata.ci.gpus is "
            f"{gpus}. CI must test at the documented minimum.",
            "Make the GPUs-required line and ci.gpus agree."))
    return out


def _markdown_text(cells: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "markdown":
            continue
        s = source(c)
        for m in PERF_CLAIM.finditer(s):
            snippet = s[max(0, m.start() - 60): m.end() + 40].replace("\n", " ").strip()
            out.append(finding(
                "performance-claim", "warning", i,
                f"Possible performance claim: “…{snippet}…”",
                "Tutorials must not state speedups, throughput, latency or comparisons. "
                "Explain what the benchmark measures and let readers read their own output."))
            break
        if HISTORY_NOTE.search(s):
            out.append(finding(
                "history-note", "warning", i,
                "The text describes what earlier versions of the tutorial did.",
                "Describe the current setup only."))
    return out


def _code(cells: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(cells):
        s = source(c)
        if HF_TOKEN_LITERAL.search(s):
            out.append(finding(
                "secret", "error", i, "The cell contains what looks like a Hugging Face token.",
                "Remove the token; read it from the HF_TOKEN environment variable instead."))
        if c.get("cell_type") != "code":
            continue
        for m in GIT_CLONE.finditer(s):
            args = m.group(1)
            if MOVING_BRANCH.search(args):
                out.append(finding(
                    "unpinned-clone", "warning", i,
                    f"`git clone{args[:80]}` follows a moving branch.",
                    "Pin the clone to a release tag (`--branch vX.Y.Z`) so upstream changes "
                    "cannot break the tutorial unnoticed."))
            elif not PINNED_CLONE.search(args) and not re.search(r"git\s+checkout\s+\S+", s):
                out.append(finding(
                    "unpinned-clone", "warning", i,
                    f"`git clone{args[:80]}` is not pinned to a release.",
                    "Clone a release tag (`--branch vX.Y.Z`), or check out a dated commit."))
        for m in AMD_FORK.finditer(s):
            out.append(finding(
                "amd-fork", "warning", i,
                f"The cell uses an AMD copy of {m.group(1)}.",
                "Use the upstream project if it supports ROCm; keep the fork only if the "
                "tutorial needs something upstream does not have, and say why."))
        if "ci-skip" not in tags(c):
            for line in s.splitlines():
                if SERVER_START.search(line) and not re.search(r"&\s*$|ensure_server|nohup|-d\b", line) \
                        and line.lstrip().startswith(("!", "%")) :
                    out.append(finding(
                        "foreground-server", "error", i,
                        "The cell starts a server in the foreground, so the notebook stops there.",
                        "Start it in the background with the ensure_server helper "
                        "(.github/CI_METADATA.md, Servers) and stop it with stop_server."))
                    break
            if re.search(r"^\s*!?\s*(?:sudo\s+)?docker\s+(?:run|pull|exec|compose)\b", s, re.M):
                out.append(finding(
                    "docker-in-cell", "warning", i,
                    "A code cell runs docker. CI runs notebooks inside a container and pods "
                    "cannot start nested containers.",
                    "Run the server as a background process in the notebook's own image and "
                    "declare that image in ci.image; keep the docker run command in markdown."))
        if "ci-skip" in tags(c):
            out.append(finding("ci-skip", "info", i, "CI skips this cell (tagged ci-skip).",
                               "Make sure it really needs a person or a browser."))
    return out


def _readability(cells: list[dict]) -> list[dict]:
    out = []
    prev_code = None
    for i, c in enumerate(cells):
        kind = c.get("cell_type")
        if kind == "code":
            lines = source(c).count("\n") + 1
            if lines > LONG_CELL_LINES and "hide-input" not in tags(c):
                out.append(finding(
                    "long-cell", "warning", i, f"The code cell is {lines} lines long.",
                    "Split it into smaller steps, each introduced by a line of markdown."))
            if prev_code is not None and source(c).strip() and source(cells[prev_code]).strip():
                out.append(finding(
                    "no-lead-in", "warning", i,
                    f"Code cell {i} follows code cell {prev_code} with no explanation between.",
                    "Add a one-line markdown cell saying what this cell does and why."))
            prev_code = i
        elif kind == "markdown" and source(c).strip():
            prev_code = None
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    nb = json.loads(Path(argv[0]).read_text())
    print(json.dumps(lint(nb), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
