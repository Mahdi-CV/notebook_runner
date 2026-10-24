"""
Decide pass / fail / blocked from the notebook papermill executed.

The agent runs the notebook and explains what broke; it does not decide the
outcome. This reads the executed notebook itself:

  blocked  no executed notebook, nothing ran, or execution stopped without an
           error (killed, or the agent never finished the run)
  fail     a cell raised, or ran into the per-cell --execution-timeout (recorded
           as a KeyboardInterrupt in that cell, or, by some papermill versions,
           as a "completed" cell that ran for the whole limit)
  pass     every code cell ran without an error and papermill exited cleanly

Papermill's per-cell status is not used for this: after a timeout it marks the
cells it never ran "completed". Execution counts, error outputs, cell durations
and the exit status run_notebook.sh records (metadata.nbr) are.

Usage:
  python3 tools/verdict.py executed.ipynb     # prints the verdict as JSON
"""

import json
import sys
from pathlib import Path

SKIP_MARKER = "# SKIPPED by preflight_patch"
# Cells papermill adds to the executed notebook; not part of the original.
INSERTED_TAGS = {"injected-parameters", "papermill-error-cell-tag"}
# run_notebook.sh's papermill --execution-timeout, for notebooks that predate
# its metadata.nbr record.
DEFAULT_EXECUTION_TIMEOUT = 3000


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def verdict(executed: Path | None) -> dict:
    """{status, reason, cells: {executed, failed, skipped, not_run}, failed_cell, error}.
    failed_cell is the index in the original notebook (papermill's inserted
    cells are not counted)."""
    def result(status, reason, counts=None, failed_cell=None, error=None):
        return {"status": status, "reason": reason,
                "cells": counts or {"executed": 0, "failed": 0, "skipped": 0, "not_run": 0},
                "failed_cell": failed_cell, "error": error}

    if executed is None or not Path(executed).is_file():
        return result("blocked", "papermill produced no executed notebook")
    try:
        nb = json.loads(Path(executed).read_text(encoding="utf-8"))
        cells = nb["cells"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return result("blocked", f"executed notebook is unreadable: {e}")

    counts = {"executed": 0, "failed": 0, "skipped": 0, "not_run": 0}
    first_error = None
    timed_out = None  # (cell, seconds) for a cell that ran for the whole per-cell limit
    run = nb.get("metadata", {}).get("nbr") or {}
    limit = run.get("execution_timeout") or DEFAULT_EXECUTION_TIMEOUT
    exit_status = run.get("papermill_exit")
    original_index = -1
    for cell in cells:
        tags = set(cell.get("metadata", {}).get("tags") or [])
        if tags & INSERTED_TAGS:
            continue
        original_index += 1
        if cell.get("cell_type") != "code" or not _source(cell).strip():
            continue
        if _source(cell).startswith(SKIP_MARKER):
            counts["skipped"] += 1
            continue
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        if errors:
            counts["failed"] += 1
            if first_error is None:
                e = errors[0]
                first_error = (original_index,
                               f"{e.get('ename', 'Error')}: {e.get('evalue', '')}".strip(": "))
        elif cell.get("execution_count") is not None:
            counts["executed"] += 1
            duration = (cell.get("metadata", {}).get("papermill") or {}).get("duration")
            if timed_out is None and duration is not None and duration >= limit - 1:
                timed_out = (original_index, duration)
        else:
            counts["not_run"] += 1

    if first_error:
        index, error = first_error
        if error.startswith("KeyboardInterrupt"):
            error += " (the cell hit the per-cell execution timeout or was interrupted)"
        return result("fail", f"cell {index} raised {error}", counts, index, error)
    if timed_out:
        index, seconds = timed_out
        error = f"cell ran for {seconds:.0f}s and hit the {limit}s per-cell execution timeout"
        return result("fail", f"cell {index}: {error}", counts, index, error)
    if counts["executed"] == 0:
        return result("blocked", "no code cell executed", counts)
    if exit_status == 124:
        return result("blocked", "papermill was stopped at the run's time limit", counts)
    if exit_status:
        return result("blocked", f"papermill exited with status {exit_status} without a "
                                 f"cell error", counts)
    if counts["not_run"]:
        n = counts["not_run"]
        return result("blocked", f"execution stopped without an error; {n} code "
                                 f"cell{'' if n == 1 else 's'} never ran", counts)
    n = counts["executed"]
    return result("pass", f"{n} code cell{'' if n == 1 else 's'} ran without an error", counts)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    print(json.dumps(verdict(Path(sys.argv[1])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
