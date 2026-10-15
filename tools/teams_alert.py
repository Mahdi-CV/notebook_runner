"""
Post a card to the Teams channel (TEAMS_WEBHOOK_URL, a Workflows webhook).

As a script: a one-off alert, used by docker/run-ci.sh for failures that stop a
pass before ci_run.py can post its own summary (e.g. the host disk is full).
As a module: wrap_card() and post(), shared with report_teams.py.

Stdlib only, and no imports from the rest of tools/: it runs on the bare host
before the container starts, where the project's dependencies may not exist.

Usage:
  python3 tools/teams_alert.py "Title" "Detail text"
  python3 tools/teams_alert.py "Title" "Detail text" --dry-run
"""

import argparse
import json
import os
import socket
import sys
import urllib.request

HTTP_TIMEOUT = 30


def text_block(text: str, **kw) -> dict:
    return {"type": "TextBlock", "text": text, "wrap": True, **kw}


def wrap_card(body: list[dict]) -> dict:
    """Workflows webhooks take an Adaptive Card; the Office 365 connector
    MessageCard schema is not accepted."""
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


def post(card: dict) -> None:
    """Post to TEAMS_WEBHOOK_URL. Never raises and never echoes the URL: it is
    a bearer credential, and publication must not fail the caller."""
    url = os.environ.get("TEAMS_WEBHOOK_URL")
    if not url:
        print("[teams] TEAMS_WEBHOOK_URL unset — not posting", file=sys.stderr)
        return
    req = urllib.request.Request(url, data=json.dumps(card).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            print(f"[teams] posted (HTTP {resp.status})", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — see docstring
        print(f"[teams] FAILED ({type(e).__name__})", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="Post a one-off alert card to Teams.")
    p.add_argument("title")
    p.add_argument("detail")
    p.add_argument("--dry-run", action="store_true", help="Print the card, post nothing.")
    args = p.parse_args()

    card = wrap_card([
        text_block(f"⚠️ {args.title}", weight="Bolder", size="Medium", color="Attention"),
        text_block(args.detail),
        text_block(f"Host: {socket.gethostname()}", isSubtle=True, size="Small"),
    ])
    if args.dry_run:
        print(json.dumps(card, indent=2, ensure_ascii=False))
    else:
        post(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
