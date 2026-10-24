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
.cell { display: grid; grid-template-columns: 64px 1fr; gap: 0 10px; margin: 0 0 14px; scroll-margin-top: 12px; }
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
.md { max-width: 80ch; }
.md h1 { font-size: 20px; margin: 4px 0 8px; } .md h2 { font-size: 17px; margin: 14px 0 6px; }
.md h3, .md h4 { font-size: 15px; margin: 12px 0 4px; } .md p, .md li { line-height: 1.55; }
.md code { font: .88em ui-monospace, SFMono-Regular, Menlo, monospace; background: #eceef1; padding: 1px 4px; border-radius: 4px; }
.md pre { background: #f4f5f7; border: 1px solid #e2e3e7; border-radius: 6px; padding: 8px 10px; }
.md pre code { background: none; padding: 0; font-size: inherit; }
.out pre { max-height: 26em; overflow: auto; }
@media (max-width: 500px) { .cell { grid-template-columns: 32px 1fr; } body { padding: 12px; } }
"""

MAX_OUTPUT_CHARS = 20000


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


def notebook_html(nb: dict, title: str, flagged: set[int] = frozenset()) -> str:
    """The notebook as a page; cells in `flagged` (the issues' cells, and any
    cell with an error output) are marked, and every cell has an #cell-N anchor."""
    parts = []
    i = -1
    for cell in nb.get("cells", []):
        if set((cell.get("metadata") or {}).get("tags") or []) & INSERTED_TAGS:
            continue
        i += 1
        src = _text(cell.get("source"))
        outputs = _coalesce(cell.get("outputs") or [])
        flag = i in flagged or any(o.get("output_type") == "error" for o in outputs)
        tags = (cell.get("metadata") or {}).get("tags") or []
        tag_line = f"<div class=tags>tags: {html.escape(', '.join(tags))}</div>" if tags else ""
        if cell.get("cell_type") == "markdown":
            body = f"<div class=md>{tag_line}{_markdown_html(src)}</div>"
            label = ""
        else:
            count = cell.get("execution_count")
            label = f"[{count if count is not None else ' '}]"
            body = (f"<div>{tag_line}<pre class=src>{html.escape(src)}</pre>"
                    + "".join(_output(o, i) for o in outputs) + "</div>")
        mark = "<span class=warn-mark title='error or issue'>⚠</span>" if flag else ""
        parts.append(f"<div class='cell{' flag' if flag else ''}' id=cell-{i}>"
                     f"<div class=n>{i}<br>{label}{mark}</div>{body}</div>")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{STYLE}</style></head><body>{''.join(parts)}</body></html>")
