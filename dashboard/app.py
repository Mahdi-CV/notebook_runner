"""
Read-only web dashboard for the notebook regression pipeline.

Serves results/runs.db (written by run.py and tools/results_db.py), each
notebook's declared requirements, and where the runner's targets (runner.yaml)
would run it. Every route is GET and the database is opened with a read-only
URI, so the dashboard can never change what it reports on.

Run:
  uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
Environment:
  NBR_ROOT  repo root to read results/, notebooks/ and runner.yaml from (default: this repo)
  NBR_AUTHORING_UPSTREAM  the authoring service (authoring/app.py), e.g. http://127.0.0.1:8081,
                     served under /authoring/ here; NBR_AUTHORING_URL links elsewhere instead
"""

import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.exception_handlers import http_exception_handler
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates

ROOT = Path(os.environ.get("NBR_ROOT") or Path(__file__).resolve().parent.parent)
DB_PATH = ROOT / "results" / "runs.db"
NOTEBOOKS_DIR = ROOT / "notebooks"
CONFIG_PATH = ROOT / "runner.yaml"
STALE_AFTER = timedelta(days=7)  # matches tools/status.py --gap
# The authoring service (authoring/app.py). With NBR_AUTHORING_UPSTREAM set, this
# app serves it under /authoring/ on its own port, so both share one address.
AUTHORING_UPSTREAM = (os.environ.get("NBR_AUTHORING_UPSTREAM") or "").rstrip("/")
AUTHORING_URL = os.environ.get("NBR_AUTHORING_URL") or ("/authoring/" if AUTHORING_UPSTREAM else None)
MAX_PROXY_BODY = 12 * 1024 * 1024

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authoring import labels
from authoring import render as nbrender  # the authoring service's notebook view
import results_db
import routing
from status import category_of, result_reason, verdict_status

app = FastAPI(title="Notebook CI", docs_url=None, redoc_url=None,
              openapi_url=None)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


# ── data access ──────────────────────────────────────────────────────────────

def db() -> sqlite3.Connection | None:
    """Open the results DB read-only. None if it doesn't exist yet."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def catalog() -> dict:
    """Targets, and for every notebook its requirements and route."""
    config = routing.load_config(CONFIG_PATH)
    notebooks = {}
    for key, (target, reason) in routing.routes(NOTEBOOKS_DIR, config).items():
        notebooks[key] = {"req": routing.resolve(NOTEBOOKS_DIR / key, {})[0],
                          "target": target, "reason": reason}
    return {"targets": config["targets"], "notebooks": notebooks}


# ── formatting ───────────────────────────────────────────────────────────────

def parse_ts(ts: str | None) -> datetime | None:
    """Parse stored timestamps (naive UTC, or ISO with offset) to naive UTC."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def fmt_ts(ts: str | None) -> str:
    dt = parse_ts(ts)
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "—"


def fmt_age(ts: str | None) -> str:
    dt = parse_ts(ts)
    if not dt:
        return ""
    secs = (datetime.now(timezone.utc).replace(tzinfo=None) - dt).total_seconds()
    if secs < 3600:
        return f"{max(int(secs // 60), 0)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def fmt_cost(v) -> str:
    return f"${v:.2f}" if v else "—"


def why(result: dict | None) -> str:
    """One-line reason for a non-passing result (same as STATUS.md and Teams)."""
    if not result:
        return ""
    try:
        payload = json.loads(result.get("payload_json") or "{}")
    except json.JSONDecodeError:
        payload = {}
    merged = {**result, **payload}
    reason = result_reason(merged, width=110)
    head, sep, rest = reason.partition(": ")
    if sep and head in labels.ISSUE_TYPES:
        reason = f"{labels.ISSUE_TYPES[head]}: {rest}"
    summary = (merged.get("summary") or "").strip()
    if verdict_status(merged.get("status")) != "pass" and not merged.get("issues") and summary:
        first = summary.split(". ")[0][:160]
        if first and first not in reason:
            reason = f"{reason} — {first}"
    return reason


for name, fn in {"ts": fmt_ts, "age": fmt_age, "cost": fmt_cost, "why": why,
                 "verdict": verdict_status}.items():
    templates.env.filters[name] = fn
labels.register(templates.env)


def notebook_row(key: str, latest: dict | None) -> dict:
    status = verdict_status(latest["status"] if latest else None)
    ts = parse_ts(latest["recorded_at"]) if latest else None
    stale = bool(latest) and (
        ts is None or ts < datetime.now(timezone.utc).replace(tzinfo=None) - STALE_AFTER)
    return {
        "key": key,
        "name": Path(key).name,
        "category": category_of(key),
        "status": status,
        "latest": latest,
        "stale": stale,
    }


def render(request: Request, template: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {"db_missing": not DB_PATH.exists(),
                                                          "authoring_url": AUTHORING_URL, **ctx})


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def overview(request: Request, category: str = "", status: str = ""):
    m = catalog()
    conn = db()
    latest = results_db.latest_by_notebook(conn) if conn else {}

    rows, no_target = [], []
    for key, nb in sorted(m["notebooks"].items()):
        if nb["reason"]:
            no_target.append({"key": key, "category": category_of(key),
                              "reason": nb["reason"], "latest": latest.get(key)})
            continue
        rows.append({**notebook_row(key, latest.get(key)), "target": nb["target"]})
    # What needs attention first: failures, then blocked, untested and stale results.
    rank = {"fail": 0, "blocked": 1, "never-tested": 2}
    rows.sort(key=lambda r: (rank.get(r["status"], 4 if not r["stale"] else 3),
                             r["category"], r["name"]))

    summary = {
        "total": len(rows),
        "passing": sum(r["status"] == "pass" for r in rows),
        "pass": sum(r["status"] == "pass" for r in rows),
        "fail": sum(r["status"] == "fail" for r in rows),
        "blocked": sum(r["status"] == "blocked" for r in rows),
        "never": sum(r["status"] == "never-tested" for r in rows),
        "stale": sum(r["stale"] for r in rows),
        "no_target": len(no_target),
        "last": max((r["latest"]["recorded_at"] for r in rows if r["latest"]), default=None),
    }
    categories = sorted({r["category"] for r in rows})
    shown = [r for r in rows
             if (not category or r["category"] == category)
             and (not status or r["status"] == status or (status == "stale" and r["stale"]))]
    no_target = [] if status else [n for n in no_target if not category or n["category"] == category]
    return render(request, "overview.html", rows=shown, no_target=no_target, summary=summary,
                  categories=categories, category=category, status=status)


@app.get("/notebook/{key:path}", response_class=HTMLResponse)
def notebook(request: Request, key: str):
    m = catalog()
    nb = m["notebooks"].get(key)
    conn = db()
    history = results_db.notebook_history(conn, key) if conn else []
    if nb is None and not history:
        raise HTTPException(404, f"Unknown notebook {key}")
    row = notebook_row(key, history[0] if history else None)
    return render(request, "notebook.html", key=key, nb=nb, row=row, history=history)


@app.get("/result/{result_id}", response_class=HTMLResponse)
def result(request: Request, result_id: int):
    conn = db()
    res = results_db.get_result(conn, result_id) if conn else None
    if res is None:
        raise HTTPException(404, f"No result {result_id}")
    try:
        payload = json.loads(res.get("payload_json") or "{}")
    except json.JSONDecodeError:
        payload = {}
    targets = routing.load_config(CONFIG_PATH)["targets"]
    return render(request, "result.html", res=res, payload=payload,
                  payload_pretty=json.dumps(payload, indent=2, ensure_ascii=False),
                  fixes=payload.get("fixes") or [], status=verdict_status(res["status"]),
                  verdict=payload.get("verdict"), executed=_executed_path(payload) is not None,
                  target=targets.get(res.get("server") or ""))


def _executed_path(payload: dict) -> Path | None:
    """The executed notebook a verdict came from, if it is under results/executed."""
    name = Path(payload.get("executed_notebook") or "").name
    path = (ROOT / "results" / "executed" / name) if name else None
    return path if path and path.is_file() else None


@app.get("/result/{result_id}/notebook")
def executed_notebook(result_id: int):
    conn = db()
    res = results_db.get_result(conn, result_id) if conn else None
    path = _executed_path(json.loads(res.get("payload_json") or "{}")) if res else None
    if path is None:
        raise HTTPException(404, f"No executed notebook for result {result_id}")
    return FileResponse(path, media_type="application/x-ipynb+json", filename=path.name)


@app.get("/result/{result_id}/view", response_class=HTMLResponse)
def executed_notebook_view(result_id: int):
    """The executed notebook as a page, for the result page's sandboxed frame.
    Notebooks are authored content, so the page may run no scripts."""
    conn = db()
    res = results_db.get_result(conn, result_id) if conn else None
    payload = json.loads(res.get("payload_json") or "{}") if res else {}
    path = _executed_path(payload)
    if path is None:
        raise HTTPException(404, f"No executed notebook for result {result_id}")
    try:
        nb = json.loads(path.read_text())
    except ValueError:
        raise HTTPException(422, "The executed notebook is not valid JSON")
    page = nbrender.notebook_html(nb, f"{res['notebook']} (executed)", payload.get("issues") or [],
                                  audience="maintainers")
    return HTMLResponse(page, headers={"Content-Security-Policy": nbrender.CSP,
                                       "X-Content-Type-Options": "nosniff"})


def run_counts(results: list[dict]) -> dict:
    """pass / fail / blocked by each notebook's last result in a run, and no-target skips."""
    last: dict[str, dict] = {}
    skipped = set()
    for r in sorted(results, key=lambda r: r["id"]):
        if r["phase"] == "skip":
            skipped.add(r["notebook"])
        else:
            last[r["notebook"]] = r
    counts = {"pass": 0, "fail": 0, "blocked": 0}
    for r in last.values():
        counts[verdict_status(r["status"])] = counts.get(verdict_status(r["status"]), 0) + 1
    counts["skipped"] = len(skipped - set(last))
    return counts


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    conn = db()
    rows = results_db.list_runs(conn, 200) if conn else []
    by_run: dict[str, list] = {}
    if rows:
        marks = ", ".join("?" * len(rows))
        for r in conn.execute(f"SELECT id, run_id, notebook, phase, status FROM results "
                              f"WHERE run_id IN ({marks})", [r["run_id"] for r in rows]):
            by_run.setdefault(r["run_id"], []).append(dict(r))
    for r in rows:
        r["counts"] = run_counts(by_run.get(r["run_id"], []))
    return render(request, "runs.html", runs=rows)


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run(request: Request, run_id: str):
    conn = db()
    run_row, results = results_db.run_results(conn, run_id) if conn else (None, [])
    if run_row is None:
        raise HTTPException(404, f"No run {run_id}")
    try:
        argv = " ".join(json.loads(run_row.get("argv") or "[]"))
    except json.JSONDecodeError:
        argv = run_row.get("argv") or ""
    order = {"fail": 0, "blocked": 1}
    results = sorted(results, key=lambda r: (order.get(verdict_status(r["status"]), 2), r["notebook"]))
    return render(request, "run.html", run=run_row, results=results, argv=argv,
                  counts=run_counts(results))


@app.get("/targets", response_class=HTMLResponse)
def targets_page(request: Request):
    m = catalog()
    groups: dict[str, list] = {}
    for key, nb in sorted(m["notebooks"].items()):
        groups.setdefault(category_of(key), []).append({"key": key, **nb})
    return render(request, "targets.html", targets=m["targets"], groups=groups)


# ── the authoring service, on this port ──────────────────────────────────────
# Its own process keeps the write access and job queue; this one only relays
# requests to it, so the dashboard stays read-only.

class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # hand redirects to the browser


_opener = urllib.request.build_opener(_NoRedirects)
_PASS_BACK = ("content-type", "location", "content-security-policy", "x-content-type-options",
              "content-disposition", "cache-control")


def _relay(method: str, url: str, body: bytes | None, headers: dict) -> Response:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        resp = _opener.open(req, timeout=120)
    except urllib.error.HTTPError as e:
        resp = e
    except OSError as e:
        return PlainTextResponse(f"The authoring service is not reachable ({e}).", status_code=502)
    out = Response(resp.read(), status_code=resp.status if hasattr(resp, "status") else resp.code)
    for name in _PASS_BACK:
        if resp.headers.get(name):
            out.headers[name] = resp.headers[name]
    for cookie in resp.headers.get_all("set-cookie") or []:
        out.headers.append("set-cookie", cookie)
    return out


@app.get("/authoring")
def authoring_root():
    return RedirectResponse("/authoring/")


@app.api_route("/authoring/{path:path}", methods=["GET", "POST"])
async def authoring(path: str, request: Request):
    if not AUTHORING_UPSTREAM:
        raise HTTPException(404, "The authoring service is not configured here")
    body = await request.body() if request.method == "POST" else None
    if body is not None and len(body) > MAX_PROXY_BODY:
        raise HTTPException(413, "Upload too large")
    query = f"?{request.url.query}" if request.url.query else ""
    headers = {k: v for k, v in request.headers.items()
               if k.lower() in ("content-type", "cookie", "accept", "user-agent")}
    return await run_in_threadpool(_relay, request.method, f"{AUTHORING_UPSTREAM}/{path}{query}",
                                   body, headers)


@app.exception_handler(StarletteHTTPException)
async def html_errors(request: Request, exc: StarletteHTTPException):
    """A page for people, JSON for scripts."""
    if "text/html" in request.headers.get("accept", ""):
        title = {404: "Not found", 403: "Not allowed"}.get(exc.status_code, "Something went wrong")
        return templates.TemplateResponse(request, "error.html", {
            "db_missing": False, "authoring_url": AUTHORING_URL, "title": title,
            "detail": exc.detail, "back_url": "/", "back_label": "Back to the notebooks"},
            status_code=exc.status_code)
    return await http_exception_handler(request, exc)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
