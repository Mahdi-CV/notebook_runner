"""
Write a structured notebook regression result JSON to results/.

Usage:
  python3 write_result.py \\
    --notebook path/to/notebook.ipynb \\
    --status pass|fail|partial \\
    --summary "One paragraph summary" \\
    [--issues '[{"cell_index":3,"error_type":"version_incompatibility",...}]'] \\
    [--fixes  '[{"cell_index":3,"fix_description":"...","patch":"...","validated":true}]'] \\
    [--results-dir /path/to/results/]

Prints the path of the written file to stdout.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def write_result(
    notebook_path: str,
    status: str,
    summary: str,
    issues: list | None = None,
    fixes: list | None = None,
    results_dir: str | None = None,
) -> Path:
    """
    Write result JSON and return the path it was written to.

    status:  "pass" | "fail" | "partial"
    issues:  list of {cell_index, error_type, description, proposed_fix}
    fixes:   list of {cell_index, fix_description, patch, validated}
    """
    out_dir = Path(results_dir) if results_dir else Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    nb_name = Path(notebook_path).stem
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_file = out_dir / f"{nb_name}_{timestamp}.json"

    payload = {
        "notebook": notebook_path,
        "status": status,
        "summary": summary,
        "issues": issues or [],
        "fixes": fixes or [],
        "timestamp": datetime.utcnow().isoformat(),
        "agent": "claude_code",
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    return out_file


def main():
    p = argparse.ArgumentParser(description="Write notebook regression result JSON")
    p.add_argument("--notebook",    required=True, help="Path to the notebook that was tested")
    p.add_argument("--status",      required=True, choices=["pass", "fail", "partial"])
    p.add_argument("--summary",     required=True, help="One paragraph summary of what happened")
    p.add_argument("--issues",      default="[]",  help="JSON array of issue objects")
    p.add_argument("--fixes",       default="[]",  help="JSON array of fix objects")
    p.add_argument("--results-dir", default=None,  help="Override output directory")
    args = p.parse_args()

    try:
        issues = json.loads(args.issues)
        fixes  = json.loads(args.fixes)
    except json.JSONDecodeError as e:
        print(f"ERROR: could not parse --issues or --fixes JSON: {e}", file=sys.stderr)
        sys.exit(1)

    out_file = write_result(
        notebook_path=args.notebook,
        status=args.status,
        summary=args.summary,
        issues=issues,
        fixes=fixes,
        results_dir=args.results_dir,
    )

    print(str(out_file))


if __name__ == "__main__":
    main()
