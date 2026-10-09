"""
Post a CI pass summary to a Microsoft Teams channel.

Reads the same results/*.json that status.py aggregates and posts one Adaptive
Card per pass. Intended as Phase E of tools/ci_run.py, after the dashboard is
regenerated.

Usage:
  python3 tools/report_teams.py                 # post (needs TEAMS_WEBHOOK_URL)
  python3 tools/report_teams.py --dry-run       # write the card to logs/, post nothing
  python3 tools/report_teams.py --always-post   # post even when nothing changed
  python3 tools/report_teams.py --run-id ID     # card for one run, from results/runs.db
                                                # (what `run.py --notify` calls)

Design notes:

  * The webhook URL is a bearer credential — anyone holding it can post to the
    channel. It is read from TEAMS_WEBHOOK_URL (via .env) and never committed.

  * Failure here is never fatal. The pass already did its work; only
    publication failed. Same reasoning as git_push() in ci_run.py.

  * One post per pass, not per notebook. Forty notebooks would otherwise mean
    forty messages and a muted channel.

  * By default a post is suppressed when nothing changed and nothing is
    failing. A channel that only speaks when something needs attention keeps
    being read. --always-post overrides for a heartbeat.

  * The payload is a Power Automate ("Workflows") Adaptive Card, NOT the legacy
    Office 365 connector MessageCard — Microsoft retired connectors, and the
    two schemas are not interchangeable. Verified against the live webhook.

  * stdlib only (urllib). Keeps uv.lock at two runtime dependencies.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Reuse status.py's aggregation rather than reimplementing it — a second copy
# of "what counts as failing" would drift from the dashboard it summarizes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_db  # noqa: E402
import status as status_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = REPO_ROOT / "logs" / "teams_preview"
HTTP_TIMEOUT = 30

# Adaptive Card colors are a fixed vocabulary, not hex.
GOOD, WARN, BAD, PLAIN = "Good", "Warning", "Attention", "Default"


def log(msg: str) -> None:
    print(f"[teams] {msg}")


def previous_status_md() -> str | None:
    """STATUS.md as of the last commit, for a pass-over-pass delta.

    Phase D commits STATUS.md every run, so git history *is* the record of
    previous passes — no extra state file to keep in sync.
    """
    proc = subprocess.run(
        ["git", "show", "HEAD:STATUS.md"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def parse_status_table(md: str) -> dict[str, str]:
    """Extract {notebook: status} from a rendered STATUS.md table.

    Deliberately forgiving: this drives a "what changed" line, so a parse miss
    should degrade to "no delta shown", never crash the pass.
    """
    out: dict[str, str] = {}
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("Notebook", "---"):
            continue
        name = cells[0].strip("`").strip()
        for cell in cells[1:]:
            token = cell.lower().strip("*_` ")
            if token in ("pass", "fail", "partial", "never-tested"):
                out[name] = token
                break
    return out


def classify(rows: list[dict], results: dict[str, dict]) -> dict:
    """Split rows into the buckets the card reports on.

    Infra failures are separated from content failures on purpose: a wedged GPU
    node is not tutorial rot, and conflating them would send authors chasing
    bugs that aren't in their notebooks.

    status.py's rows don't carry `issues` (it folds them into `top_failure`),
    so fixability is read from the raw result payloads keyed by stem.

    Each bucket entry keeps the notebook's `top_failure` line — the card is
    quiet on success but has to be diagnostic on failure, and "which notebook"
    without "what broke" just sends the reader to the logs anyway.
    """
    failing, infra = [], []
    for r in rows:
        if r["status"] == "fail" or (r["status"] == "partial" and not r["counts_as_pass"]):
            name = r.get("notebook") or "?"
            stem = Path(name).stem
            entry = {"name": name, "why": (r.get("top_failure") or "").strip()}
            if _is_infra_blocked(results.get(stem)):
                infra.append(entry)
            else:
                failing.append(entry)
    key = lambda e: e["name"]  # noqa: E731
    return {"failing": sorted(failing, key=key), "infra": sorted(infra, key=key)}


def _is_infra_blocked(result: dict | None) -> bool:
    payload = (result or {}).get("payload") or {}
    issues = payload.get("issues") or []
    if not issues:
        return False
    # Infra-blocked only if EVERY issue is infra — one real content bug in the
    # set means the author still has something to fix.
    return all((i or {}).get("fixability") == "infra_blocked" for i in issues)


def text_block(text: str, **kw) -> dict:
    return {"type": "TextBlock", "text": text, "wrap": True, **kw}


def build_card(summary: dict, buckets: dict, delta: dict, log_name: str | None) -> dict:
    # Key names come from status.py's compute_summary(); read them without
    # .get() defaults so a rename upstream fails loudly here instead of
    # silently rendering "0/0 — all passing", which reads as success when it
    # actually means "nothing was measured".
    total = summary["total_testable"]
    passing = summary["pass"]
    never = summary["never_tested"]
    failing = len(buckets["failing"])
    infra = len(buckets["infra"])

    if failing:
        color, headline = BAD, f"{failing} failing"
    elif infra:
        color, headline = WARN, f"{infra} infra-blocked"
    elif never == total and total:
        # Everything untested is not a green pass — say so plainly.
        color, headline = WARN, "none tested yet"
    else:
        color, headline = GOOD, "all passing"

    body = [
        text_block("Notebook regression pass", weight="Bolder", size="Medium"),
        text_block(
            f"**{passing}/{total} passing** — {headline}",
            color=color, size="Medium", weight="Bolder",
        ),
    ]

    # Deltas are the one thing worth saying even on a green pass — a newly
    # passing notebook closes the loop on a fix.
    facts = []
    if never:
        facts.append({"title": "Never tested", "value": str(never)})
    if delta.get("newly_failing"):
        # Regressions are the headline — never abbreviate these.
        facts.append({"title": "Newly failing",
                      "value": _name_list(delta["newly_failing"], limit=15)})
    if delta.get("newly_passing"):
        # Recoveries are good news, not a to-do list. Capped tightly so a
        # first green run doesn't paste 30+ names into the success card.
        facts.append({"title": "Newly passing",
                      "value": _name_list(delta["newly_passing"], limit=5)})
    if facts:
        body.append({"type": "FactSet", "facts": facts})

    # Detail is asymmetric by design: on success the headline is the whole
    # message, and on failure the card carries enough to act on without
    # opening the log — notebook name plus the top error line per failure.
    if buckets["failing"]:
        body.append(text_block("Failing — content bugs for authors",
                               weight="Bolder", color=BAD, spacing="Medium"))
        body.extend(_detail_blocks(buckets["failing"], BAD))

    if buckets["infra"]:
        body.append(text_block("Infra-blocked — not the notebook's fault",
                               weight="Bolder", color=WARN, spacing="Medium"))
        body.extend(_detail_blocks(buckets["infra"], WARN))

    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = f"{stamp}"
    if log_name:
        footer += f" · log: {log_name}"
    if buckets["failing"] or buckets["infra"]:
        footer += " · full detail in STATUS.md"
    body.append(text_block(footer, isSubtle=True, size="Small", color=PLAIN))

    return wrap_card(body)


def _name_list(names: list[str], limit: int) -> str:
    shown = ", ".join(f"`{n}`" for n in names[:limit])
    if len(names) > limit:
        shown += f" (+{len(names) - limit} more)"
    return shown


def _detail_blocks(entries: list[dict], color: str, limit: int = 15) -> list[dict]:
    """One line per failure: notebook name, then why it failed.

    Capped so a catastrophic pass (every notebook failing) still produces a
    readable card rather than an unscrollable wall. The overflow count is
    always stated — silently truncating would read as "that's all of them".
    """
    blocks = []
    for e in entries[:limit]:
        why = e["why"] or "no error detail captured"
        blocks.append(text_block(f"**`{e['name']}`**\n{why}",
                                 color=color, spacing="Small"))
    if len(entries) > limit:
        blocks.append(text_block(
            f"_+{len(entries) - limit} more — see STATUS.md_",
            isSubtle=True, size="Small"))
    return blocks


def wrap_card(body: list[dict]) -> dict:
    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.4",
                "body": body,
            },
        }],
    }


def build_run_card(run: dict, rows: list[dict], manifest: dict) -> dict:
    """Card for one run.py invocation, read from the results DB.

    Unlike build_card() (the whole-dashboard view ci_run.py posts), this
    reports only the notebooks this run touched — an ad-hoc `run.py --notify`
    on two notebooks should say what happened to those two, not restate the
    other forty.
    """
    expected = {Path(k).name: (v or {}).get("expected_result")
                for k, v in (manifest.get("notebooks") or {}).items()}

    passing, failing, infra, skipped = [], [], [], []
    for r in rows:
        name = r["notebook"]
        if r["phase"] == "skip":
            skipped.append(name)
            continue
        status = r["status"]
        if status == "pass" or (status == "partial"
                                and expected.get(Path(name).name) == "partial"):
            passing.append(name)
            continue
        entry = {"name": name, "why": _run_row_why(r)}
        issues = r["issues"]
        if issues and all(i["fixability"] == "infra_blocked" for i in issues):
            infra.append(entry)
        else:
            failing.append(entry)

    tested = len(passing) + len(failing) + len(infra)
    if failing:
        color, headline = BAD, f"{len(failing)} failing"
    elif infra:
        color, headline = WARN, f"{len(infra)} infra-blocked"
    elif not tested:
        color, headline = WARN, "nothing tested"
    else:
        color, headline = GOOD, "all passing"

    body = [
        text_block(f"Notebook regression run `{run['run_id']}`",
                   weight="Bolder", size="Medium"),
        text_block(f"**{len(passing)}/{tested} passing** — {headline}",
                   color=color, size="Medium", weight="Bolder"),
    ]
    facts = [{"title": "Mode", "value": run.get("mode") or "?"}]
    if run.get("total_cost_usd"):
        facts.append({"title": "Cost", "value": f"${run['total_cost_usd']:.2f}"})
    if skipped:
        facts.append({"title": "Skipped", "value": _name_list(skipped, limit=5)})
    if passing and (failing or infra):
        facts.append({"title": "Passing", "value": _name_list(passing, limit=10)})
    body.append({"type": "FactSet", "facts": facts})

    if failing:
        body.append(text_block("Failing — content bugs for authors",
                               weight="Bolder", color=BAD, spacing="Medium"))
        body.extend(_detail_blocks(failing, BAD))
    if infra:
        body.append(text_block("Infra-blocked — not the notebook's fault",
                               weight="Bolder", color=WARN, spacing="Medium"))
        body.extend(_detail_blocks(infra, WARN))

    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body.append(text_block(f"{stamp} · results DB run_id {run['run_id']}",
                           isSubtle=True, size="Small", color=PLAIN))
    return wrap_card(body)


def _run_row_why(row: dict) -> str:
    """Top failure line for a DB row — same shape as status.py's top_failure."""
    if row["issues"]:
        first = row["issues"][0]
        etype = first["error_type"] or "unknown"
        desc = (first["description"] or "").strip().replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        return f"{etype}: {desc}" if desc else etype
    # No issues means the runner itself failed (crash, no result file written).
    return (row["message"] or "").strip()


def post(url: str, card: dict) -> bool:
    data = json.dumps(card).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            log(f"posted (HTTP {resp.status})")
            return True
    except urllib.error.HTTPError as e:
        # Never echo the URL — it carries the signature that authorizes posting.
        log(f"FAILED (HTTP {e.code}) — pass itself was unaffected")
        return False
    except Exception as e:  # noqa: BLE001 — publication must never fail a pass
        log(f"FAILED ({type(e).__name__}) — pass itself was unaffected")
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Post a CI summary to Teams.")
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--dry-run", action="store_true",
                   help="Write the card to logs/teams_preview/ and post nothing.")
    p.add_argument("--always-post", action="store_true",
                   help="Post even when nothing changed and nothing is failing.")
    p.add_argument("--log-name", default=None,
                   help="Pass log filename, shown in the card footer.")
    p.add_argument("--run-id", default=None,
                   help="Post a card for this one run (from the results DB) instead "
                        "of the whole-dashboard summary. Always posts: asking for a "
                        "specific run's card is itself the request to hear about it.")
    p.add_argument("--db", default=str(results_db.DEFAULT_DB),
                   help="Results database for --run-id.")
    args = p.parse_args()

    manifest = status_mod.load_manifest()

    if args.run_id:
        conn = results_db.connect(args.db)
        run, rows = results_db.run_results(conn, args.run_id)
        conn.close()
        if run is None:
            log(f"run {args.run_id} not found in {args.db} — nothing to post")
            return 0
        return publish(build_run_card(run, rows, manifest), args.dry_run)

    results = status_mod.load_results(Path(args.results_dir))
    rows = status_mod.build_notebook_rows(manifest, results)
    summary = status_mod.compute_summary(rows)
    buckets = classify(rows, results)

    current = {(r.get("notebook") or "?"): r["status"] for r in rows}
    prev_md = previous_status_md()
    delta: dict[str, list[str]] = {}
    if prev_md:
        prev = parse_status_table(prev_md)
        delta["newly_failing"] = sorted(
            n for n, s in current.items()
            if s == "fail" and prev.get(n) not in (None, "fail")
        )
        delta["newly_passing"] = sorted(
            n for n, s in current.items()
            if s == "pass" and prev.get(n) not in (None, "pass")
        )

    # "Quiet" means genuinely nothing to report. A pass where everything is
    # still never-tested is NOT quiet — staying silent there would let a
    # pipeline that never actually ran look like a healthy one.
    quiet = (
        not buckets["failing"] and not buckets["infra"]
        and not delta.get("newly_failing") and not delta.get("newly_passing")
        and summary["never_tested"] == 0
    )
    if quiet and not args.always_post and not args.dry_run:
        log("nothing failing and nothing changed — skipping post (--always-post to override)")
        return 0

    return publish(build_card(summary, buckets, delta, args.log_name), args.dry_run)


def publish(card: dict, dry_run: bool) -> int:
    if dry_run:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = PREVIEW_DIR / f"card_{stamp}.json"
        out.write_text(json.dumps(card, indent=2))
        log(f"dry-run — card written to {out}")
        return 0

    url = os.getenv("TEAMS_WEBHOOK_URL")
    if not url:
        # Not an error: a repo without a webhook configured should still pass.
        log("TEAMS_WEBHOOK_URL unset — skipping notification")
        return 0

    post(url, card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
