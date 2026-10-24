"""The words both apps show for the system's internal names, as Jinja filters.

The regression dashboard and the authoring service register these, so a
category, an issue type, a status or a count reads the same everywhere and
no snake_case key reaches the page.
"""

import re

from markupsafe import Markup, escape

from .render import _markdown_html

CATEGORIES = {"inference": "Inference", "fine_tune": "Fine-tuning", "pretrain": "Training",
              "gpu_dev_optimize": "GPU development and optimization"}
ISSUE_TYPES = {"version_incompatibility": "Version mismatch", "deprecated_api": "Deprecated API",
               "missing_dependency": "Missing dependency", "content_error": "Notebook bug",
               "other": "Other"}
KINDS = {"verify": "Check", "fix": "Fix attempt", "full": "Check and fix", "skip": "Skipped",
         "interactive": "Interactive"}
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


def register(env) -> None:
    env.filters["category"] = lambda k: CATEGORIES.get(k, k)
    env.filters["issue_type"] = lambda k: ISSUE_TYPES.get(k, (k or "").replace("_", " "))
    env.filters["kind"] = lambda k: KINDS.get(k, k)
    env.filters["statuslabel"] = lambda k: STATUSES.get(k, k)
    env.filters["plural"] = plural
    env.filters["codespans"] = codespans
    env.filters["md"] = markdown
