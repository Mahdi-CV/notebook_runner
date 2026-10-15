"""
Aggregate per-notebook regression results into a status dashboard.

Reads results/*.json for every notebook in notebooks/ and prints a markdown
summary + table. Notebooks no configured target can run are listed apart (see
tools/routing.py). Supports --gap (work-order list), --json
(machine-readable), --write (also write STATUS.md), --results-dir PATH.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
STATUS_MD_PATH = REPO_ROOT / "STATUS.md"

TS_FILENAME_RE = re.compile(r"^(?P<stem>.+)_(?P<ts>\d{8}T\d{6})\.json$")
CATEGORIES = ("inference", "fine_tune", "pretrain", "gpu_dev_optimize")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def parse_result_filename(path: Path) -> tuple[str, datetime | None]:
    """Return (notebook_stem, timestamp_from_filename_or_None)."""
    m = TS_FILENAME_RE.match(path.name)
    if not m:
        return path.stem, None
    ts = datetime.strptime(m.group("ts"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return m.group("stem"), ts


def load_results(results_dir: Path) -> dict[str, dict]:
    """Return {notebook_stem: latest_result_record}."""
    latest: dict[str, dict] = {}
    if not results_dir.exists():
        return latest

    for path in sorted(results_dir.glob("*.json")):
        stem, fname_ts = parse_result_filename(path)
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warn(f"could not parse {path.name}: {e}")
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        sort_ts = fname_ts or mtime
        record = {
            "stem": stem,
            "path": path,
            "payload": payload,
            "sort_ts": sort_ts,
            "display_ts": fname_ts or mtime,
            "mtime": mtime,
        }
        prev = latest.get(stem)
        if prev is None:
            latest[stem] = record
            continue
        # Tie-break on mtime if sort_ts equal
        if (record["sort_ts"], record["mtime"]) > (prev["sort_ts"], prev["mtime"]):
            latest[stem] = record
    return latest


def category_of(key: str) -> str:
    head = key.split("/", 1)[0]
    return head if head in CATEGORIES else "other"


def build_notebook_rows(routes: dict[str, tuple[str | None, str]],
                        results: dict[str, dict]) -> tuple[list[dict], dict[str, str]]:
    """Return (rows for notebooks a target can run, {key: reason} for the rest).

    `routes` comes from routing.routes(). Also warns about results whose
    notebook no longer exists.
    """
    rows: list[dict] = []
    no_target: dict[str, str] = {}
    for key, (target, reason) in routes.items():
        if reason:
            no_target[key] = reason
        else:
            rows.append(_build_row(key, target, results.get(Path(key).stem)))

    stems = {Path(key).stem for key in routes}
    for stem, rec in results.items():
        if stem not in stems:
            warn(f"orphan result {rec['path'].name}: no notebook '{stem}' in notebooks/")
    return rows, no_target


def failure_line(issues: list | None, message: str | None = None, width: int = 80) -> str:
    """One-line reason for a non-passing result: the first issue, else the
    runner's message. Shared by STATUS.md, the dashboard and the Teams card."""
    if issues:
        first = issues[0] or {}
        etype = first.get("error_type") or "unknown"
        desc = (first.get("description") or "").strip().replace("\n", " ")
        if len(desc) > width:
            desc = desc[:width - 3] + "..."
        return f"{etype}: {desc}" if desc else etype
    return (message or "").strip()


def _build_row(key: str, target: str | None, result: dict | None) -> dict:
    payload = result["payload"] if result else {}
    status = payload.get("status", "unknown") if result else "never-tested"
    return {
        "notebook": key,
        "category": category_of(key),
        "target": target,
        "status": status,
        "counts_as_pass": status == "pass",
        "last_run": result["display_ts"] if result else None,
        "image_tested": payload.get("docker_image_resolved"),
        "top_failure": failure_line(payload.get("issues"))
        if status not in ("pass", "never-tested") else "",
        "cost_usd": payload.get("cost_usd"),
    }


def compute_summary(rows: list[dict]) -> dict:
    total = len(rows)
    pass_count = sum(1 for r in rows if r["counts_as_pass"])
    fail_count = sum(1 for r in rows if r["status"] not in ("pass", "never-tested"))
    never_count = sum(1 for r in rows if r["status"] == "never-tested")

    now = datetime.now(tz=timezone.utc)
    week_ago = now - timedelta(days=7)
    recent = [r for r in rows if r["last_run"] and r["last_run"] >= week_ago]
    recent_count = len(recent)
    recent_pass = sum(1 for r in recent if r["counts_as_pass"])

    total_cost = 0.0
    have_cost = False
    for r in rows:
        c = r.get("cost_usd")
        if isinstance(c, (int, float)):
            total_cost += float(c)
            have_cost = True

    last_runs = [r["last_run"] for r in rows if r["last_run"]]
    most_recent = max(last_runs) if last_runs else None

    def pct(n: int, d: int) -> float:
        return (100.0 * n / d) if d else 0.0

    return {
        "total_testable": total,
        "pass": pass_count,
        "pass_pct": pct(pass_count, total),
        "fail": fail_count,
        "fail_pct": pct(fail_count, total),
        "never_tested": never_count,
        "never_tested_pct": pct(never_count, total),
        "recent_7d": recent_count,
        "recent_7d_pass": recent_pass,
        "recent_7d_pass_pct": pct(recent_pass, recent_count),
        "total_cost_usd": total_cost if have_cost else None,
        "most_recent_run": most_recent,
    }


def fmt_ts(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M")


def fmt_cost(cost: float | None) -> str:
    if cost is None:
        return ""
    return f"${cost:.2f}"


def render_markdown(rows: list[dict], summary: dict, no_target: dict[str, str]) -> str:
    lines: list[str] = []

    lines.append("# Notebook Regression Status")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    total = summary["total_testable"]
    lines.append(f"- Total testable notebooks: **{total}**")
    lines.append(f"- Pass: **{summary['pass']}** ({summary['pass_pct']:.1f}%)")
    lines.append(f"- Fail: **{summary['fail']}** ({summary['fail_pct']:.1f}%)")
    lines.append(f"- Never-tested: **{summary['never_tested']}** ({summary['never_tested_pct']:.1f}%)")
    lines.append(
        f"- Results in last 7 days: **{summary['recent_7d']}** "
        f"(pass rate within those: {summary['recent_7d_pass_pct']:.1f}%)"
    )
    if summary["total_cost_usd"] is None:
        lines.append("- Total cost across latest results: —")
    else:
        lines.append(f"- Total cost across latest results: **${summary['total_cost_usd']:.2f}**")
    lines.append(f"- Most recent run: **{fmt_ts(summary['most_recent_run'])}**")
    lines.append("")

    lines.append("## Notebooks")
    lines.append("")
    lines.append("| Notebook | Category | Status | Target | Last run (UTC) | Image tested | Top failure | Cost |")
    lines.append("|---|---|---|---|---|---|---|---|")

    sorted_rows = sorted(rows, key=lambda r: (r["category"], r["notebook"]))
    prev_cat: str | None = None
    for r in sorted_rows:
        if prev_cat is not None and r["category"] != prev_cat:
            lines.append("| | | | | | | | |")
        prev_cat = r["category"]
        lines.append(
            "| {nb} | {cat} | {status} | {target} | {ts} | {img} | {fail} | {cost} |".format(
                nb=r["notebook"],
                cat=r["category"],
                status=r["status"],
                target=r["target"] or "—",
                ts=fmt_ts(r["last_run"]) if r["last_run"] else "—",
                img=r["image_tested"] or "—",
                fail=r["top_failure"].replace("|", "\\|"),
                cost=fmt_cost(r["cost_usd"]),
            )
        )

    if no_target:
        lines += ["", "## No target", "",
                  "Notebooks whose declared requirements no configured target meets.", "",
                  "| Notebook | Why |", "|---|---|"]
        lines += [f"| {k} | {why.replace('|', chr(92) + '|')} |"
                  for k, why in sorted(no_target.items())]

    return "\n".join(lines) + "\n"


def render_gap(rows: list[dict]) -> str:
    now = datetime.now(tz=timezone.utc)
    week_ago = now - timedelta(days=7)

    never = [r for r in rows if r["status"] == "never-tested"]
    stale = [r for r in rows if r["last_run"] and r["last_run"] < week_ago]

    never.sort(key=lambda r: r["notebook"])
    stale.sort(key=lambda r: r["last_run"])  # oldest first

    return "\n".join([r["notebook"] for r in never + stale]) + ("\n" if (never or stale) else "")


def render_failing(rows: list[dict]) -> str:
    """Print the keys of notebooks whose latest result is not a pass."""
    failing = [r for r in rows if r["status"] not in ("pass", "never-tested")]
    failing.sort(key=lambda r: r["notebook"])
    return "\n".join(r["notebook"] for r in failing) + ("\n" if failing else "")


def render_json(rows: list[dict], summary: dict) -> str:
    def ts_iso(ts: datetime | None) -> str | None:
        return ts.isoformat() if ts else None

    summary_out = dict(summary)
    summary_out["most_recent_run"] = ts_iso(summary["most_recent_run"])

    notebooks_out = []
    for r in sorted(rows, key=lambda r: (r["category"], r["notebook"])):
        notebooks_out.append({
            "notebook": r["notebook"],
            "category": r["category"],
            "status": r["status"],
            "target": r["target"],
            "counts_as_pass": r["counts_as_pass"],
            "last_run": ts_iso(r["last_run"]),
            "image_tested": r["image_tested"],
            "top_failure": r["top_failure"],
            "cost_usd": r["cost_usd"],
        })

    return json.dumps({"summary": summary_out, "notebooks": notebooks_out}, indent=2) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate notebook regression results.")
    p.add_argument("--gap", action="store_true", help="Print only never-tested/stale notebook paths.")
    p.add_argument("--failing", action="store_true",
                   help="Print only currently-failing notebook paths (latest result "
                        "is not a pass).")
    p.add_argument("--json", dest="as_json", action="store_true", help="Print machine-readable JSON.")
    p.add_argument("--write", action="store_true", help="Also write STATUS.md to repo root.")
    p.add_argument("--results-dir", default=None, help="Override results directory.")
    args = p.parse_args()

    if sum([args.gap, args.failing, args.as_json]) > 1:
        print("error: --gap, --failing, and --json are mutually exclusive", file=sys.stderr)
        return 2

    results_dir = Path(args.results_dir).resolve() if args.results_dir else DEFAULT_RESULTS_DIR
    results = load_results(results_dir)
    rows, no_target = build_notebook_rows(routing.routes(), results)
    summary = compute_summary(rows)

    if args.gap:
        sys.stdout.write(render_gap(rows))
        return 0

    if args.failing:
        sys.stdout.write(render_failing(rows))
        return 0

    if args.as_json:
        sys.stdout.write(render_json(rows, summary))
        return 0

    md = render_markdown(rows, summary, no_target)
    sys.stdout.write(md)

    if args.write:
        header = "<!-- Auto-generated by tools/status.py — do not edit by hand -->\n\n"
        STATUS_MD_PATH.write_text(header + md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
