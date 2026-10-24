"""The words both apps show for the system's internal names, as Jinja filters.

The regression dashboard and the authoring service register these, so a
category, an issue type, a status or a count reads the same everywhere and
no snake_case key reaches the page.
"""

import re
from datetime import datetime, timezone

from markupsafe import Markup, escape

from .render import _markdown_html

CATEGORIES = {"inference": "Inference", "fine_tune": "Fine-tuning", "pretrain": "Training",
              "gpu_dev_optimize": "GPU development and optimization"}
ISSUE_TYPES = {"version_incompatibility": "Version mismatch", "deprecated_api": "Deprecated API",
               "missing_dependency": "Missing dependency", "content_error": "Notebook bug",
               "other": "Other"}
KINDS = {"verify": "Check", "fix": "Fix attempt", "full": "Check and fix", "skip": "Skipped",
         "interactive": "Interactive"}
# Authoring job kinds (authoring/store.py), as a job's label.
JOB_KINDS = {"validate": "Validate", "annotate": "Annotate", "fix": "Fix",
             "regression": "Nightly pass"}
STATUSES = {"pass": "Passed", "passed": "Passed", "fail": "Failed", "failed": "Failed",
            "blocked": "Blocked", "never-tested": "Never tested", "skip": "Skipped",
            "queued": "Queued", "running": "Running", "cancelled": "Cancelled",
            "error": "Crashed", "done": "Done"}


def plural(n, word: str, words: str | None = None) -> str:
    return f"{n} {word if n == 1 else (words or word + 's')}"


def codespans(text) -> Markup:
    """Escape text, then show `backticked` spans as code."""
    return Markup(re.sub(r"`([^`\n]+)`", r"<code>\1</code>", str(escape(text or ""))))


_UNSAFE_HREF = re.compile(r'href="(?!https?://|#)[^"]*"')
_IMG = re.compile(r"<img\b[^>]*>")
_DOUBLE = re.compile(r"&amp;(lt;|gt;|amp;|#39;|#34;)")


def markdown(text) -> Markup:
    """Markdown from an agent's notes, which are written from author content:
    raw HTML is escaped first, links may only be http(s) or in-page, and
    images are dropped, so nothing in it can run or load on this page."""
    html = _markdown_html(str(escape(text or "")))
    html = _DOUBLE.sub(r"&\1", html)  # code spans escape our escapes again
    return Markup(_IMG.sub("", _UNSAFE_HREF.sub('href="#"', html)))


def parse_ts(ts) -> datetime | None:
    """Stored timestamps (naive UTC, `Z`, or ISO with an offset) as naive UTC."""
    if not ts:
        return None
    try:
        dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def when(ts) -> Markup:
    """One format for a moment everywhere: "today 14:05 UTC", or "Oct 23, 14:05 UTC"
    on other days (with the year when it is not this one); the full time is the
    tooltip and the <time> element's datetime."""
    dt = parse_ts(ts)
    if not dt:
        return Markup("—")
    now = _now()
    day = ("today" if dt.date() == now.date() else f"{dt:%b} {dt.day}," if dt.year == now.year
           else f"{dt:%b} {dt.day}, {dt.year},")
    iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return Markup('<time datetime="{0}" title="{0}">{1} {2} UTC</time>').format(iso, day, f"{dt:%H:%M}")


def age(ts) -> str:
    """How long ago, as "5m ago", "3h ago" or "2d ago"."""
    dt = parse_ts(ts)
    if not dt:
        return ""
    secs = (_now() - dt).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def elapsed(start, end=None) -> float | None:
    """Seconds from one timestamp to another, or to now: feed it to `dur`."""
    a, b = parse_ts(start), parse_ts(end) if end else _now()
    return max((b - a).total_seconds(), 0) if a and b else None


def dur(value, unit: str = "s") -> str:
    """A duration in seconds (or minutes, with unit="min") as "45 s", "12 min"
    or "1 h 05 min"."""
    try:
        secs = max(float(value) * (60 if unit == "min" else 1), 0)
    except (TypeError, ValueError):
        return "—"
    if secs < 59.5:
        return f"{int(secs + 0.5)} s"
    mins = int(secs / 60 + 0.5)  # half up: 12.5 min reads as 13
    return f"{mins} min" if mins < 60 else f"{mins // 60} h {mins % 60:02d} min"


def register(env) -> None:
    env.filters["category"] = lambda k: CATEGORIES.get(k, k)
    env.filters["issue_type"] = lambda k: ISSUE_TYPES.get(k, (k or "").replace("_", " "))
    env.filters["kind"] = lambda k: KINDS.get(k, k)
    env.filters["jobkind"] = lambda k: JOB_KINDS.get(k, k)
    env.filters["statuslabel"] = lambda k: STATUSES.get(k, k)
    env.filters["plural"] = plural
    env.filters["codespans"] = codespans
    env.filters["md"] = markdown
    env.filters["ts"] = when
    env.filters["age"] = age
    env.filters["elapsed"] = elapsed
    env.filters["dur"] = dur
