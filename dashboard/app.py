"""
Read-only web dashboard for the notebook regression pipeline.

Serves results/runs.db (written by run.py and tools/results_db.py) and
manifest.yaml. Every route is GET and the database is opened with a read-only
URI, so the dashboard can never change what it reports on.

Run:
  uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
Environment:
  NBR_ROOT  repo root to read results/ and manifest.yaml from (default: this repo)
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(os.environ.get("NBR_ROOT") or Path(__file__).resolve().parent.parent)
DB_PATH = ROOT / "results" / "runs.db"
MANIFEST_PATH = ROOT / "manifest.yaml"
STALE_AFTER = timedelta(days=7)  # matches tools/status.py --gap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import results_db  # noqa: E402
from status import category_of, effective_status  # noqa: E402

app = FastAPI(title="Notebook regression dashboard", docs_url=None, redoc_url=None,
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


def manifest() -> dict:
    data = yaml.safe_load(MANIFEST_PATH.read_text()) or {}
    return {"servers": data.get("servers") or {}, "notebooks": data.get("notebooks") or {}}


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
    """One-line reason for a non-passing result (same shape as the Teams card)."""
    if not result:
        return ""
    issues = result.get("issues") or []
    if issues:
        first = issues[0]
        etype = first.get("error_type") or "issue"
        desc = (first.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 110:
            desc = desc[:107] + "…"
        return f"{etype}: {desc}" if desc else etype
    return (result.get("message") or "").strip()


for name, fn in {"ts": fmt_ts, "age": fmt_age, "cost": fmt_cost, "why": why}.items():
    templates.env.filters[name] = fn


def notebook_row(key: str, entry: dict, latest: dict | None) -> dict:
    expected_partial = (entry or {}).get("expected_result") == "partial"
    status, counts = effective_status(latest["status"] if latest else None, expected_partial)
    stale = bool(latest) and (parse_ts(latest["recorded_at"]) or datetime.min) < (
        datetime.now(timezone.utc).replace(tzinfo=None) - STALE_AFTER)
    return {
        "key": key,
        "name": Path(key).name,
        "category": category_of(key),
        "status": status,
        "counts_as_pass": counts,
        "expected_partial": expected_partial,
        "latest": latest,
        "stale": stale,
    }


def render(request: Request, template: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {"db_missing": not DB_PATH.exists(), **ctx})


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def overview(request: Request, category: str = "", status: str = ""):
    m = manifest()
    conn = db()
    latest = results_db.latest_by_notebook(conn) if conn else {}

    rows, skipped = [], []
    for key, entry in sorted(m["notebooks"].items()):
        entry = entry or {}
        if entry.get("skip"):
            skipped.append({"key": key, "category": category_of(key),
                            "reason": entry.get("skip_reason") or "", "latest": latest.get(key)})
            continue
        rows.append(notebook_row(key, entry, latest.get(key)))

    summary = {
        "total": len(rows),
        "passing": sum(r["counts_as_pass"] for r in rows),
        "pass": sum(r["status"] == "pass" for r in rows),
        "partial": sum(r["status"] == "partial" for r in rows),
        "fail": sum(r["status"] not in ("pass", "partial", "never-tested") for r in rows),
        "never": sum(r["status"] == "never-tested" for r in rows),
        "stale": sum(r["stale"] for r in rows),
        "skipped": len(skipped),
        "last": max((r["latest"]["recorded_at"] for r in rows if r["latest"]), default=None),
    }
    categories = sorted({r["category"] for r in rows})
    shown = [r for r in rows
             if (not category or r["category"] == category)
             and (not status or r["status"] == status
                  or (status == "fail" and r["status"] not in ("pass", "partial", "never-tested")))]
    return render(request, "overview.html", rows=shown, skipped=skipped, summary=summary,
                  categories=categories, category=category, status=status)


@app.get("/notebook/{key:path}", response_class=HTMLResponse)
def notebook(request: Request, key: str):
    m = manifest()
    entry = m["notebooks"].get(key)
    conn = db()
    history = results_db.notebook_history(conn, key) if conn else []
    if entry is None and not history:
        raise HTTPException(404, f"Unknown notebook {key}")
    row = notebook_row(key, entry or {}, history[0] if history else None)
    return render(request, "notebook.html", key=key, entry=entry or {}, row=row,
                  history=history, servers=m["servers"])


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
    m = manifest()
    entry = m["notebooks"].get(res["notebook"]) or {}
    status, counts = effective_status(res["status"], entry.get("expected_result") == "partial")
    return render(request, "result.html", res=res, payload=payload,
                  payload_pretty=json.dumps(payload, indent=2, ensure_ascii=False),
                  fixes=payload.get("fixes") or [], status=status, counts_as_pass=counts,
                  server=m["servers"].get(res.get("server") or ""), entry=entry)


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    conn = db()
    return render(request, "runs.html", runs=results_db.list_runs(conn, 200) if conn else [])


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
    notebooks = manifest()["notebooks"]
    for r in results:
        entry = notebooks.get(r["notebook"]) or {}
        r["display_status"], r["counts_as_pass"] = effective_status(
            r["status"], entry.get("expected_result") == "partial")
    return render(request, "run.html", run=run_row, results=results, argv=argv)


@app.get("/manifest", response_class=HTMLResponse)
def manifest_page(request: Request):
    m = manifest()
    groups: dict[str, list] = {}
    for key, entry in sorted(m["notebooks"].items()):
        groups.setdefault(category_of(key), []).append({"key": key, **(entry or {})})
    return render(request, "manifest.html", servers=m["servers"], groups=groups)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
