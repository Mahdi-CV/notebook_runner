"""
Aggregate per-notebook regression results into a status dashboard.

Reads results/*.json, cross-references against manifest.yaml, and prints a
markdown summary + table. Supports --gap (work-order list), --json
(machine-readable), --write (also write STATUS.md), --results-dir PATH.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = REPO_ROOT / "manifest.yaml"
STATUS_MD_PATH = REPO_ROOT / "STATUS.md"

TS_FILENAME_RE = re.compile(r"^(?P<stem>.+)_(?P<ts>\d{8}T\d{6})\.json$")
CATEGORIES = ("inference", "fine_tune", "pretrain", "gpu_dev_optimize")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        warn(f"manifest not found at {MANIFEST_PATH}")
        return {}
    with open(MANIFEST_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("notebooks", {}) or {}


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


def category_of(manifest_key: str) -> str:
    head = manifest_key.split("/", 1)[0]
    return head if head in CATEGORIES else "other"


def build_notebook_rows(manifest: dict, results: dict[str, dict]) -> list[dict]:
    """Build per-notebook rows, excluding skipped notebooks from totals.

    Also logs warnings for orphan results (manifest skipped, or notebook not in manifest).
    """
    # Map manifest keys → their stem (filename without .ipynb)
    manifest_stem_to_key: dict[str, str] = {}
    for key in manifest:
        stem = Path(key).stem
        manifest_stem_to_key[stem] = key

    rows: list[dict] = []
    used_stems: set[str] = set()

    for key, entry in manifest.items():
        entry = entry or {}
        stem = Path(key).stem
        if entry.get("skip"):
            if stem in results:
                warn(f"manifest entry {key} is skip:true but has result file {results[stem]['path'].name}")
            used_stems.add(stem)
            continue

        result = results.get(stem)
        used_stems.add(stem)
        rows.append(_build_row(key, entry, result))

    # Orphans: results without a (non-skipped) manifest entry
    for stem, rec in results.items():
        if stem in used_stems:
            continue
        warn(f"orphan result {rec['path'].name}: notebook stem '{stem}' not in manifest (or is skipped)")

    return rows


def _build_row(manifest_key: str, entry: dict, result: dict | None) -> dict:
    category = category_of(manifest_key)
    if result is None:
        return {
            "notebook": manifest_key,
            "category": category,
            "status": "never-tested",
            "raw_status": None,
            "counts_as_pass": False,
            "last_run": None,
            "image_tested": None,
            "top_failure": "",
            "cost_usd": None,
            "expected_partial": entry.get("expected_result") == "partial",
        }

    payload = result["payload"]
    raw_status = payload.get("status", "unknown")
    expected_partial = entry.get("expected_result") == "partial"

    if raw_status == "pass":
        status, counts_as_pass = "pass", True
    elif raw_status == "partial":
        # partial counts as pass only when manifest says so; status column shows partial regardless
        status, counts_as_pass = "partial", expected_partial
    elif raw_status == "fail":
        status, counts_as_pass = "fail", False
    else:
        status, counts_as_pass = raw_status, False

    issues = payload.get("issues") or []
    top_failure = ""
    if status in ("fail", "partial") and issues:
        first = issues[0] or {}
        etype = first.get("error_type", "unknown")
        desc = (first.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        top_failure = f"{etype}: {desc}" if desc else etype

    return {
        "notebook": manifest_key,
        "category": category,
        "status": status,
        "raw_status": raw_status,
        "counts_as_pass": counts_as_pass,
        "last_run": result["display_ts"],
        "image_tested": payload.get("docker_image_resolved"),
        "top_failure": top_failure,
        "cost_usd": payload.get("cost_usd"),
        "expected_partial": expected_partial,
    }


def compute_summary(rows: list[dict]) -> dict:
    total = len(rows)
    pass_count = sum(1 for r in rows if r["counts_as_pass"])
    fail_count = sum(1 for r in rows if r["status"] == "fail")
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


def fmt_cost(cost: float | int | None) -> str:
    if cost is None:
        return ""
    return f"${cost:.2f}"


def render_markdown(rows: list[dict], summary: dict) -> str:
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
    lines.append("| Notebook | Category | Status | Last run (UTC) | Image tested | Top failure | Cost |")
    lines.append("|---|---|---|---|---|---|---|")

    sorted_rows = sorted(rows, key=lambda r: (r["category"], r["notebook"]))
    prev_cat: str | None = None
    for r in sorted_rows:
        if prev_cat is not None and r["category"] != prev_cat:
            lines.append("| | | | | | | |")
        prev_cat = r["category"]
        lines.append(
            "| {nb} | {cat} | {status} | {ts} | {img} | {fail} | {cost} |".format(
                nb=r["notebook"],
                cat=r["category"],
                status=r["status"],
                ts=fmt_ts(r["last_run"]) if r["last_run"] else "—",
                img=r["image_tested"] or "—",
                fail=r["top_failure"].replace("|", "\\|"),
                cost=fmt_cost(r["cost_usd"]),
            )
        )

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
    """Print manifest-relative paths of notebooks whose latest result is a hard fail.

    A 'partial' that counts as pass (expected_result: partial) is NOT failing.
    A 'partial' that does NOT count as pass is treated as failing — it needs work.
    """
    failing = [
        r for r in rows
        if r["status"] == "fail"
        or (r["status"] == "partial" and not r["counts_as_pass"])
    ]
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
            "raw_status": r["raw_status"],
            "counts_as_pass": r["counts_as_pass"],
            "last_run": ts_iso(r["last_run"]),
            "image_tested": r["image_tested"],
            "top_failure": r["top_failure"],
            "cost_usd": r["cost_usd"],
            "expected_partial": r["expected_partial"],
        })

    return json.dumps({"summary": summary_out, "notebooks": notebooks_out}, indent=2) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate notebook regression results.")
    p.add_argument("--gap", action="store_true", help="Print only never-tested/stale notebook paths.")
    p.add_argument("--failing", action="store_true",
                   help="Print only currently-failing notebook paths (status fail, "
                        "or partial that does not count as pass).")
    p.add_argument("--json", dest="as_json", action="store_true", help="Print machine-readable JSON.")
    p.add_argument("--write", action="store_true", help="Also write STATUS.md to repo root.")
    p.add_argument("--results-dir", default=None, help="Override results directory.")
    args = p.parse_args()

    if sum([args.gap, args.failing, args.as_json]) > 1:
        print("error: --gap, --failing, and --json are mutually exclusive", file=sys.stderr)
        return 2

    results_dir = Path(args.results_dir).resolve() if args.results_dir else DEFAULT_RESULTS_DIR
    manifest = load_manifest()
    results = load_results(results_dir)
    rows = build_notebook_rows(manifest, results)
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

    md = render_markdown(rows, summary)
    sys.stdout.write(md)

    if args.write:
        header = "<!-- Auto-generated by tools/status.py — do not edit by hand -->\n\n"
        STATUS_MD_PATH.write_text(header + md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
