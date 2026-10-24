"""Render a notebook as a standalone HTML page for the job page.

The page is shown in a sandboxed iframe and served with a Content-Security-
Policy that allows no scripts: notebooks come from authors, so raw HTML in
their markdown or outputs must not run on this site. Images are inline data
URIs; everything else is text.
"""

import html
import re

try:
    import markdown as _markdown
except ImportError:  # the page still works, with markdown shown as text
    _markdown = None

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Cells papermill adds to an executed notebook. They are not shown, and the
# rest are numbered as in the original notebook, as tools/verdict.py and the
# agents' issues number them.
INSERTED_TAGS = {"injected-parameters", "papermill-error-cell-tag"}
CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'"

STYLE = """
body { margin: 0; padding: 16px 20px; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1c1d21; background: #fff; }
.cell { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: 0 10px; margin: 0 0 14px; scroll-margin-top: 12px; }
.n { color: #6b6f7a; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; padding-top: 6px; }
.cell.flag { box-shadow: -3px 0 0 #a3222b; border-radius: 4px; }
.cell.flag .n { color: #a3222b; font-weight: 700; }
.md { padding: 0 2px; overflow-x: auto; }
.md img { max-width: 100%; }
.md pre, .src, .out pre { font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; word-break: break-word; margin: 0; }
.src { background: #f4f5f7; border: 1px solid #e2e3e7; border-radius: 6px; padding: 8px 10px; }
.tags { font-size: 11px; color: #6b6f7a; margin-bottom: 2px; }
.out { margin-top: 4px; padding: 4px 10px; border-left: 3px solid #e2e3e7; overflow-x: auto; }
.out.err { border-color: #a3222b; background: #fbe4e5; }
.out.err pre { color: #a3222b; }
.out img { max-width: 100%; }
.out table { border-collapse: collapse; font-size: 12.5px; }
.out td, .out th { border: 1px solid #e2e3e7; padding: 2px 6px; }
.more { color: #6b6f7a; font-size: 12px; }
.warn-mark { display: block; margin-top: 2px; }
.md { max-width: 80ch; overflow-wrap: anywhere; }
.md h1 { font-size: 20px; margin: 4px 0 8px; } .md h2 { font-size: 17px; margin: 14px 0 6px; }
.md h3, .md h4 { font-size: 15px; margin: 12px 0 4px; } .md p, .md li { line-height: 1.55; }
.md code { font: .88em ui-monospace, SFMono-Regular, Menlo, monospace; background: #eceef1; padding: 1px 4px; border-radius: 4px; }
.md pre { background: #f4f5f7; border: 1px solid #e2e3e7; border-radius: 6px; padding: 8px 10px; }
.md pre code { background: none; padding: 0; font-size: inherit; }
.out pre { max-height: 26em; overflow: auto; }
.issue { margin-top: 6px; padding: 8px 12px; border: 1px solid #e2e3e7; border-left: 4px solid #a3222b; border-radius: 8px; background: #fff; max-width: 80ch; overflow-wrap: anywhere; }
.issue .head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 8px; font-weight: 600; color: #a3222b; }
.issue p { margin: 4px 0 0; }
.lbl { color: #6b6f7a; font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }
.issue code { font: .88em ui-monospace, SFMono-Regular, Menlo, monospace; background: #eceef1; padding: 1px 4px; border-radius: 4px; }
.loose { margin: 0 0 14px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.badge.info { color: #1d4f91; background: #e2ecf9; } .badge.warn { color: #8a5a00; background: #fdf1d8; }
.badge.muted { color: #585c66; background: #eceef1; }
@media (max-width: 500px) { .cell { grid-template-columns: 32px minmax(0, 1fr); } body { padding: 12px; } }
"""

MAX_OUTPUT_CHARS = 20000
# Who fixes an issue, in the words of each app: authors on the authoring
# service, maintainers on the dashboard.
FIXABILITY = {
    "authors": {"auto_fixable": "Can be fixed automatically", "needs_author": "Needs your change",
                "infra_blocked": "Platform issue, not your notebook"},
    "maintainers": {"auto_fixable": "Auto-fixable", "needs_author": "Needs the author",
                    "infra_blocked": "Platform issue"},
}
FIXABILITY_CLASS = {"auto_fixable": "info", "needs_author": "warn", "infra_blocked": "muted"}


def _text(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def _markdown_html(source: str) -> str:
    if _markdown is None:
        return f"<pre>{html.escape(source)}</pre>"
    try:  # superfences also handles code blocks inside list items
        return _markdown.markdown(source, extensions=["pymdownx.superfences", "tables"])
    except ImportError:
        return _markdown.markdown(source, extensions=["fenced_code", "tables"])


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return html.escape(text)
    cut = len(text) - MAX_OUTPUT_CHARS
    return (html.escape(text[:MAX_OUTPUT_CHARS // 2]) + f"\n<span class=more>… {cut:,} characters "
            "not shown …</span>\n" + html.escape(text[-MAX_OUTPUT_CHARS // 2:]))


def _output(out: dict, cell: int) -> str:
    kind = out.get("output_type")
    if kind == "stream":
        return f"<div class=out><pre>{_clip(ANSI.sub('', _text(out.get('text'))))}</pre></div>"
    if kind == "error":
        tb = ANSI.sub("", "\n".join(out.get("traceback") or [])) or \
            f"{out.get('ename')}: {out.get('evalue')}"
        return f"<div class='out err'><pre>{_clip(tb)}</pre></div>"
    data = out.get("data") or {}
    for mime in ("image/png", "image/jpeg", "image/gif"):
        if mime in data:
            return (f"<div class=out><img alt='Cell {cell} output image' "
                    f"src='data:{mime};base64,{_text(data[mime]).strip()}'></div>")
    if "text/html" in data:  # inert in the sandbox: no scripts, no external loads
        return f"<div class=out>{_text(data['text/html'])[:MAX_OUTPUT_CHARS]}</div>"
    if "text/markdown" in data:
        return f"<div class='out md'>{_markdown_html(_text(data['text/markdown']))}</div>"
    if "text/plain" in data:
        return f"<div class=out><pre>{_clip(ANSI.sub('', _text(data['text/plain'])))}</pre></div>"
    return ""


def _issue(issue: dict, audience: str) -> str:
    """An issue as a callout under its cell: its type, what broke and the
    proposed fix. Agents write these from author content: all of it is escaped."""
    from .labels import ISSUE_TYPES, codespans  # labels imports this module

    kind = str(issue.get("error_type") or "other")
    fix = issue.get("fixability")
    who = (f"<span class='badge {FIXABILITY_CLASS.get(fix, 'muted')}'>"
           f"{html.escape(FIXABILITY[audience].get(fix, fix.replace('_', ' ')))}</span>"
           if isinstance(fix, str) and fix else "")
    proposed = issue.get("proposed_fix")
    return ("<div class=issue role=note><div class=head>"
            f"<span>{html.escape(ISSUE_TYPES.get(kind, kind.replace('_', ' ')))}</span>{who}</div>"
            f"<p>{codespans(issue.get('description'))}</p>"
            + (f"<p><span class=lbl>Proposed fix</span><br>{codespans(proposed)}</p>" if proposed else "")
            + "</div>")


def _coalesce(outputs: list[dict]) -> list[dict]:
    """Merge consecutive stream outputs and apply carriage returns, as Jupyter
    does, so a progress bar shows its final state instead of every update."""
    merged: list[dict] = []
    for out in outputs:
        prev = merged[-1] if merged else None
        if (out.get("output_type") == "stream" and prev and prev.get("output_type") == "stream"
                and prev.get("name") == out.get("name")):
            prev["text"] = _text(prev["text"]) + _text(out.get("text"))
        else:
            merged.append(dict(out))
    for out in merged:
        if out.get("output_type") == "stream":
            out["text"] = "\n".join(line.rsplit("\r", 1)[-1] if "\r" in line.rstrip("\r") else line.rstrip("\r")
                                    for line in _text(out["text"]).split("\n"))
    return merged


def notebook_html(nb: dict, title: str, issues: list[dict] = (), audience: str = "authors") -> str:
    """The notebook as a page. Cells with an issue or an error output are
    marked, each issue is shown under its cell (at the top when it names no
    cell, or one the notebook does not have), and every cell has an #cell-N
    anchor. `audience` ("authors" or "maintainers") words who fixes an issue."""
    callouts: dict[int | None, list[str]] = {}
    for issue in issues:
        n = issue.get("cell_index")
        n = n if isinstance(n, int) and not isinstance(n, bool) else None
        callouts.setdefault(n, []).append(_issue(issue, audience))
    parts = []
    i = -1
    for cell in nb.get("cells", []):
        if set((cell.get("metadata") or {}).get("tags") or []) & INSERTED_TAGS:
            continue
        i += 1
        src = _text(cell.get("source"))
        outputs = _coalesce(cell.get("outputs") or [])
        notes = "".join(callouts.pop(i, []))
        flag = bool(notes) or any(o.get("output_type") == "error" for o in outputs)
        tags = (cell.get("metadata") or {}).get("tags") or []
        tag_line = f"<div class=tags>tags: {html.escape(', '.join(tags))}</div>" if tags else ""
        if cell.get("cell_type") == "markdown":
            body = f"<div><div class=md>{tag_line}{_markdown_html(src)}</div>{notes}</div>"
            label = ""
        else:
            count = cell.get("execution_count")
            label = f"[{count if count is not None else ' '}]"
            body = (f"<div>{tag_line}<pre class=src>{html.escape(src)}</pre>"
                    + "".join(_output(o, i) for o in outputs) + notes + "</div>")
        mark = "<span class=warn-mark title='error or issue'>⚠</span>" if flag else ""
        parts.append(f"<div class='cell{' flag' if flag else ''}' id=cell-{i}>"
                     f"<div class=n>{i}<br>{label}{mark}</div>{body}</div>")
    loose = "".join(c for n in sorted(callouts, key=lambda n: (n is not None, n or 0)) for c in callouts[n])
    if loose:  # issues with no cell of their own in this notebook
        parts.insert(0, f"<div class=loose><div class=lbl>Not tied to a cell</div>{loose}</div>")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{STYLE}</style></head><body>{''.join(parts)}</body></html>")
