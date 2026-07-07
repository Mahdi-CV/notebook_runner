"""
Deterministic pre-flight patcher for notebook regression testing.

Applies exactly 4 allowed patches — no more, no less:
  1. notebook_login() → login(token=os.environ["HF_TOKEN"])
  2. input() calls → static string "test"
  3. Gradio .launch() → skip the cell
  4. Audio playback (IPython.display.Audio, sounddevice) → skip the cell

Usage:
  python3 preflight_patch.py --input nb.ipynb --output nb_patched.ipynb
  # prints JSON report to stdout
"""

import argparse
import json
import re
import sys
from pathlib import Path

SKIP_CELL = "# SKIPPED by preflight_patch: {reason}\npass"

HF_LOGIN_REPLACEMENT = (
    'import os\n'
    'from huggingface_hub import login\n'
    'login(token=os.environ["HF_TOKEN"])'
)

_NOTEBOOK_LOGIN_RE = re.compile(
    r'notebook_login\s*\(', re.MULTILINE
)
_NOTEBOOK_LOGIN_IMPORT_RE = re.compile(
    r'from\s+huggingface_hub\s+import\s+[^#\n]*\bnotebook_login\b'
)

_INPUT_CALL_RE = re.compile(
    r'\binput\s*\([^)]*\)'
)

_GRADIO_LAUNCH_RE = re.compile(
    r'^[^#\n]*\.\s*launch\s*\(', re.MULTILINE
)
_GRADIO_IMPORT_RE = re.compile(
    r'\bgradio\b|import\s+gradio'
)

_AUDIO_RE = re.compile(
    r'IPython\.display\.Audio|Audio\s*\(|sounddevice\.play|sd\.play'
)
_AUDIO_IMPORT_RE = re.compile(
    r'import\s+sounddevice|from\s+IPython\.display\s+import\s+[^#\n]*\bAudio\b'
)


def _is_trivial_cell(src: str, pattern_re: re.Pattern) -> bool:
    """Check if the cell contains only the pattern match and nothing else meaningful."""
    lines = [ln.strip() for ln in src.strip().splitlines()
             if ln.strip() and not ln.strip().startswith('#')]
    if not lines:
        return True
    non_matching = [ln for ln in lines
                    if not pattern_re.search(ln)
                    and not ln.startswith('import ')
                    and not ln.startswith('from ')]
    return len(non_matching) == 0


def patch_notebook_login(src: str) -> tuple[str | None, str | None]:
    if not _NOTEBOOK_LOGIN_RE.search(src) and not _NOTEBOOK_LOGIN_IMPORT_RE.search(src):
        return None, None

    lines = src.splitlines()
    patched_lines = []
    login_import_added = False

    for line in lines:
        stripped = line.strip()
        # Handle import lines containing notebook_login
        if _NOTEBOOK_LOGIN_IMPORT_RE.match(stripped):
            # Extract other imports from the same line
            # e.g. "from huggingface_hub import notebook_login, HfApi" → keep HfApi
            match = re.match(r'from\s+huggingface_hub\s+import\s+(.+)', stripped)
            if match:
                imports = [s.strip() for s in match.group(1).split(',')]
                remaining = [i for i in imports if i != 'notebook_login']
                if remaining:
                    patched_lines.append(f"from huggingface_hub import {', '.join(remaining)}")
            if not login_import_added:
                patched_lines.append("import os")
                patched_lines.append("from huggingface_hub import login")
                login_import_added = True
        # Handle notebook_login() call lines
        elif _NOTEBOOK_LOGIN_RE.search(stripped):
            if not login_import_added:
                patched_lines.append("import os")
                patched_lines.append("from huggingface_hub import login")
                login_import_added = True
            patched_lines.append('login(token=os.environ["HF_TOKEN"])')
        else:
            patched_lines.append(line)

    return '\n'.join(patched_lines), "Replaced notebook_login() with login(token=os.environ['HF_TOKEN'])"


def patch_input_calls(src: str) -> tuple[str | None, str | None]:
    if not _INPUT_CALL_RE.search(src):
        return None, None

    if _is_trivial_cell(src, _INPUT_CALL_RE):
        return SKIP_CELL.format(reason="cell is only input() calls"), "Skipped cell containing only input() calls"

    patched = _INPUT_CALL_RE.sub('"test"', src)
    count = len(_INPUT_CALL_RE.findall(src))
    return patched, f"Replaced {count} input() call(s) with static string 'test'"


def patch_gradio_launch(src: str) -> tuple[str | None, str | None]:
    if not _GRADIO_LAUNCH_RE.search(src):
        return None, None

    if _is_trivial_cell(src, _GRADIO_LAUNCH_RE):
        return SKIP_CELL.format(reason="Gradio .launch()"), "Skipped cell containing only .launch()"

    lines = src.splitlines()
    patched_lines = []
    for line in lines:
        if _GRADIO_LAUNCH_RE.match(line) or (line.strip() and re.search(r'\.\s*launch\s*\(', line) and not line.strip().startswith('#')):
            patched_lines.append(f"# SKIPPED: {line}")
        else:
            patched_lines.append(line)
    return '\n'.join(patched_lines), "Commented out .launch() call"


def patch_audio_playback(src: str) -> tuple[str | None, str | None]:
    if not _AUDIO_RE.search(src) and not _AUDIO_IMPORT_RE.search(src):
        return None, None

    if _is_trivial_cell(src, _AUDIO_RE):
        return SKIP_CELL.format(reason="audio playback"), "Skipped cell containing only audio playback"

    lines = src.splitlines()
    patched_lines = []
    for line in lines:
        if _AUDIO_RE.search(line) and not line.strip().startswith('#'):
            patched_lines.append(f"# SKIPPED: {line}")
        else:
            patched_lines.append(line)
    return '\n'.join(patched_lines), "Commented out audio playback"


PATCHERS = [
    ("notebook_login", patch_notebook_login),
    ("input_call", patch_input_calls),
    ("gradio_launch", patch_gradio_launch),
    ("audio_playback", patch_audio_playback),
]


def patch_notebook(nb: dict) -> dict:
    """Apply all pre-flight patches. Returns report dict."""
    patches_applied = []
    cells_skipped = []

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        src = "".join(cell.get("source", []))
        if not src.strip():
            continue

        for patch_type, patcher in PATCHERS:
            new_src, description = patcher(src)
            if new_src is not None:
                cell["source"] = [new_src]
                entry = {"cell_index": i, "type": patch_type, "description": description}
                if new_src.startswith("# SKIPPED by preflight_patch:"):
                    cells_skipped.append(entry)
                else:
                    patches_applied.append(entry)
                src = new_src
                break

    return {
        "patches_applied": patches_applied,
        "cells_skipped": cells_skipped,
    }


def main():
    p = argparse.ArgumentParser(description="Deterministic pre-flight notebook patcher")
    p.add_argument("--input", required=True, help="Input notebook path")
    p.add_argument("--output", required=True, help="Output (patched) notebook path")
    args = p.parse_args()

    with open(args.input) as f:
        nb = json.load(f)

    report = patch_notebook(nb)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
