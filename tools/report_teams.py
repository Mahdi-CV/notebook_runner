"""
Post a regression run's summary to a Microsoft Teams channel.

Builds one Adaptive Card per run from results/runs.db: ci_run.py posts one per
pass (Phase E), `run.py --notify` one per invocation.

Usage:
  python3 tools/report_teams.py --run-id ID                # post (needs TEAMS_WEBHOOK_URL)
  python3 tools/report_teams.py --run-id ID --dry-run      # write the card to logs/, post nothing
  python3 tools/report_teams.py --run-id ID --always-post  # post even with nothing to report

Design notes:

  * Never fatal. The run already did its work; only publication failed.

  * One post per run, not per notebook. Forty notebooks would otherwise mean
    forty messages and a muted channel.

  * Quiet by default: with nothing failing, nothing changed and nothing left
    untested there is no post (--always-post overrides), so the channel only
    speaks when something needs attention.

  * Infra-blocked failures are listed apart from content failures: a wedged GPU
    node is not tutorial rot, and authors should not chase it.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_db
import routing
from status import failure_line
from teams_alert import post, text_block, wrap_card

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = REPO_ROOT / "logs" / "teams_preview"

# Adaptive Card colors are a fixed vocabulary, not hex.
GOOD, WARN, BAD, PLAIN = "Good", "Warning", "Attention", "Default"


def log(msg: str) -> None:
    print(f"[teams] {msg}")


def _names(names: list[str], limit: int) -> str:
    shown = ", ".join(f"`{n}`" for n in names[:limit])
    return shown + (f" (+{len(names) - limit} more)" if len(names) > limit else "")


def _detail_blocks(title: str, entries: list[dict], color: str, limit: int = 15) -> list[dict]:
    """Section heading, then one line per failure: notebook, then why. Capped so
    a catastrophic run stays readable; the overflow count is always stated."""
    blocks = [text_block(title, weight="Bolder", color=color, spacing="Medium")]
    for e in entries[:limit]:
        blocks.append(text_block(f"**`{e['name']}`**\n{e['why'] or 'no error detail captured'}",
                                 color=color, spacing="Small"))
    if len(entries) > limit:
        blocks.append(text_block(f"_+{len(entries) - limit} more — see the dashboard_",
                                 isSubtle=True, size="Small"))
    return blocks


def build_card(conn, run: dict, rows: list[dict], log_name: str | None) -> tuple[dict, bool]:
    """Return (card, quiet). Reports the notebooks this run touched, their
    change since the previous result, and the dashboard-wide totals."""
    passing, failing, infra, skipped = [], [], [], []
    for r in rows:
        name = r["notebook"]
        if r["phase"] == "skip":
            skipped.append(name)
        elif r["status"] == "pass":
            passing.append(name)
        else:
            entry = {"name": name, "why": failure_line(r["issues"], r["message"])}
            # Infra-blocked only if every issue is; one content bug is the author's to fix.
            if r["issues"] and all(i["fixability"] == "infra_blocked" for i in r["issues"]):
                infra.append(entry)
            else:
                failing.append(entry)

    prev = results_db.previous_status(conn, run["run_id"])
    newly_failing = sorted(n for n in [e["name"] for e in failing + infra]
                           if prev.get(n) == "pass")
    newly_passing = sorted(n for n in passing if n in prev and prev[n] != "pass")

    latest = results_db.latest_by_notebook(conn)
    tracked = [k for k, (_, reason) in routing.routes().items() if not reason]
    dash_pass = sum(k in latest and latest[k]["status"] == "pass" for k in tracked)
    never = sum(k not in latest for k in tracked)

    tested = len(passing) + len(failing) + len(infra)
    if failing:
        color, headline = BAD, f"{len(failing)} failing"
    elif infra:
        color, headline = WARN, f"{len(infra)} infra-blocked"
    elif not tested:
        color, headline = WARN, "nothing tested"
    else:
        color, headline = GOOD, "all passing"

    dashboard = f"{dash_pass}/{len(tracked)} passing" + (f", {never} never tested" if never else "")
    facts = [("Mode", run.get("mode") or "?"), ("Dashboard", dashboard)]
    if run.get("total_cost_usd"):
        facts.append(("Cost", f"${run['total_cost_usd']:.2f}"))
    if newly_failing:
        facts.append(("Newly failing", _names(newly_failing, 15)))
    if newly_passing:
        facts.append(("Newly passing", _names(newly_passing, 5)))
    if skipped:
        facts.append(("No target", _names(skipped, 5)))
    if passing and (failing or infra):
        facts.append(("Passing", _names(passing, 10)))

    body = [
        text_block(f"Notebook regression run `{run['run_id']}`", weight="Bolder", size="Medium"),
        text_block(f"**{len(passing)}/{tested} passing** — {headline}",
                   color=color, size="Medium", weight="Bolder"),
        {"type": "FactSet", "facts": [{"title": t, "value": v} for t, v in facts]},
    ]
    if failing:
        body += _detail_blocks("Failing — content bugs for authors", failing, BAD)
    if infra:
        body += _detail_blocks("Infra-blocked — not the notebook's fault", infra, WARN)
    footer = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if log_name:
        footer += f" · log: {log_name}"
    body.append(text_block(footer, isSubtle=True, size="Small", color=PLAIN))

    quiet = not (failing or infra or newly_failing or newly_passing or never)
    return wrap_card(body), quiet


def main() -> int:
    p = argparse.ArgumentParser(description="Post a run's summary card to Teams.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--log-name", default=None, help="Log filename shown in the card footer.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write the card to logs/teams_preview/ and post nothing.")
    p.add_argument("--always-post", action="store_true",
                   help="Post even when nothing is failing, changed or untested.")
    args = p.parse_args()

    conn = results_db.connect()
    run, rows = results_db.run_results(conn, args.run_id)
    if run is None:
        log(f"run {args.run_id} not in {results_db.DEFAULT_DB} (nothing ran) — nothing to post")
        return 0
    card, quiet = build_card(conn, run, rows, args.log_name)
    conn.close()

    if args.dry_run:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out = PREVIEW_DIR / f"card_{args.run_id}.json"
        out.write_text(json.dumps(card, indent=2))
        log(f"dry-run — card written to {out}")
    elif quiet and not args.always_post:
        log("nothing failing, changed or untested — skipping post (--always-post to override)")
    else:
        post(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
