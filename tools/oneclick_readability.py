"""Check that the cell-readability rules still hold after the one-click render.

The notebooks.amd.com one-click launch does not serve a tutorial as written: it
*removes* setup and marked cells before the reader sees it (jupyterServer's
services/strip_setup.py is the renderer). Dropping a markdown or setup cell that
sat between two code cells pushes those code cells together — so a notebook that
reads fine in the repo can violate Rule 1 (no two code cells back to back with
no one-line markdown lead-in) once rendered.

This is the mechanical check the style fan-out runs locally: it renders the
notebook through the *real* one-click transform and scans the kept cells, in the
order the reader clicks them, for code cells left adjacent with no markdown
between. It is a local developer aid — it is deliberately NOT wired into CI.

Usage:
  python3 tools/oneclick_readability.py NOTEBOOK [NOTEBOOK ...]
  python3 tools/oneclick_readability.py --strip-setup PATH NOTEBOOK
  python3 tools/oneclick_readability.py --json NOTEBOOK

The one-click renderer lives in the jupyterServer repo. Point at it with
--strip-setup or $STRIP_SETUP; the default is ~/projects/jupyterServer/
services/strip_setup.py.

Exit status: 1 if any notebook has a post-render adjacency, 0 if all are clean,
2 if the renderer could not be found or run (the check could not be performed).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_STRIP_SETUP = Path(
    os.environ.get("STRIP_SETUP", "~/projects/jupyterServer/services/strip_setup.py")
).expanduser()


def first_line(cell) -> str:
    """The first non-empty source line of a cell, for pointing a human at it."""
    src = cell.get("source", "")
    text = "".join(src) if isinstance(src, list) else src
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "(empty cell)"


def render_oneclick(notebook: Path, strip_setup: Path) -> list[dict]:
    """Run the one-click transform and return the cells it keeps, in order."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{notebook.stem}_stripped.ipynb"
        proc = subprocess.run(
            [sys.executable, str(strip_setup), str(notebook), str(out), "--no-lychee"],
            capture_output=True, text=True,
        )
        # strip_setup returns 2 for anchor/lychee nits but still writes the file;
        # only a missing output means the render genuinely failed.
        if not out.is_file():
            raise RuntimeError(
                f"renderer did not produce output (exit {proc.returncode}):\n"
                f"{proc.stderr.strip() or proc.stdout.strip()}")
        return json.loads(out.read_text(encoding="utf-8")).get("cells", [])


def adjacencies(cells: list[dict]) -> list[dict]:
    """Rule 1 violations: a code cell whose previous kept cell is also code."""
    out = []
    prev = None  # (render_index, cell) of the last cell seen
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            if prev is not None and prev[1].get("cell_type") == "code":
                out.append({
                    "after_index": prev[0], "code_index": idx,
                    "after": first_line(prev[1]), "code": first_line(cell),
                })
        prev = (idx, cell)
    return out


def check(notebook: Path, strip_setup: Path) -> dict:
    kept = render_oneclick(notebook, strip_setup)
    return {"notebook": str(notebook), "kept_cells": len(kept),
            "violations": adjacencies(kept)}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebooks", nargs="+", type=Path)
    p.add_argument("--strip-setup", type=Path, default=DEFAULT_STRIP_SETUP,
                   help="Path to jupyterServer's strip_setup.py (default: %(default)s).")
    p.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = p.parse_args()

    if not args.strip_setup.is_file():
        print(f"one-click renderer not found: {args.strip_setup}\n"
              f"check out jupyterServer, or point --strip-setup / $STRIP_SETUP at it.",
              file=sys.stderr)
        return 2

    reports, had_error = [], False
    for nb in args.notebooks:
        try:
            reports.append(check(nb, args.strip_setup))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"{nb}: could not render: {exc}", file=sys.stderr)
            had_error = True

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            v = r["violations"]
            status = "clean" if not v else f"{len(v)} adjacency(ies)"
            print(f"[{status:18}] {r['notebook']}  ({r['kept_cells']} cells after render)")
            for a in v:
                print(f"{'':20}! code cell {a['code_index']} follows code cell "
                      f"{a['after_index']} with no markdown between")
                print(f"{'':22}{a['after_index']}: {a['after']}")
                print(f"{'':22}{a['code_index']}: {a['code']}")

    total = sum(len(r["violations"]) for r in reports)
    print(f"\n{len(reports)} notebook(s) rendered, "
          f"{total} post-render adjacency(ies).", file=sys.stderr)
    if had_error:
        return 2
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
