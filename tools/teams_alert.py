"""
Post a one-off alert card to the Teams channel.

Used by docker/run-ci.sh for failures that stop a pass before ci_run.py can
post its own summary card (e.g. the host disk is full) — otherwise those
failures are silent, since the summary card is the last thing a pass does.

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
import urllib.error
import urllib.request

HTTP_TIMEOUT = 15


def build_card(title: str, detail: str) -> dict:
    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "text": f"⚠️ {title}", "wrap": True,
                     "weight": "Bolder", "size": "Medium", "color": "Attention"},
                    {"type": "TextBlock", "text": detail, "wrap": True},
                    {"type": "TextBlock", "text": f"Host: {socket.gethostname()}",
                     "wrap": True, "isSubtle": True, "size": "Small"},
                ],
            },
        }],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Post a one-off alert card to Teams.")
    p.add_argument("title")
    p.add_argument("detail")
    p.add_argument("--dry-run", action="store_true", help="Print the card, post nothing.")
    args = p.parse_args()

    card = build_card(args.title, args.detail)
    if args.dry_run:
        print(json.dumps(card, indent=2, ensure_ascii=False))
        return 0

    url = os.environ.get("TEAMS_WEBHOOK_URL")
    if not url:
        print("[teams-alert] TEAMS_WEBHOOK_URL unset — not posting", file=sys.stderr)
        return 0
    req = urllib.request.Request(url, data=json.dumps(card).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            print(f"[teams-alert] posted (HTTP {resp.status})", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — never echo the URL; it is a credential
        print(f"[teams-alert] FAILED ({type(e).__name__})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
