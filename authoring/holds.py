"""Holds set on the dashboard: notebooks kept off runners for a known
infrastructure problem, without editing runner.yaml.

jobs/holds.json is {notebook key: {reason, targets, since, by, at}}; routing
(tools/routing.py) adds it to runner.yaml's known_issues, which win for a
notebook in both and can only be changed in runner.yaml. Every hold and lift
is appended to jobs/holds-log.jsonl, which the notebook's page shows.

There is no login, so a hold may only be set on a notebook whose latest result
is blocked (it could not be evaluated), never on one that fails on its own
content, and always with a name and a reason that the dashboard shows.
"""

import json

from . import store

HOLDS = "holds.json"
LOG = "holds-log.jsonl"


def read() -> dict:
    try:
        return json.loads((store.JOBS_DIR / HOLDS).read_text())
    except (OSError, ValueError):
        return {}


def history(key: str | None = None) -> list[dict]:
    """Hold and lift events, newest first, for one notebook or all."""
    try:
        lines = (store.JOBS_DIR / LOG).read_text().splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if key is None or event.get("notebook") == key:
            events.append(event)
    return events[::-1]


def _write(holds: dict) -> None:
    path = store.JOBS_DIR / HOLDS
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(holds, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _log(event: dict) -> None:
    with open(store.JOBS_DIR / LOG, "a") as f:
        f.write(json.dumps({"at": store.now(), **event}, ensure_ascii=False) + "\n")


def hold(key: str, targets: list[str], reason: str, by: str) -> None:
    holds = read()
    holds[key] = {"targets": targets, "reason": reason, "since": store.now()[:10],
                  "by": by, "at": store.now()}
    _write(holds)
    _log({"action": "hold", "notebook": key, "targets": targets, "reason": reason, "by": by})


def lift(key: str, by: str) -> dict | None:
    holds = read()
    entry = holds.pop(key, None)
    if entry is not None:
        _write(holds)
        _log({"action": "lift", "notebook": key, "targets": entry.get("targets"),
              "reason": entry.get("reason"), "by": by})
    return entry
