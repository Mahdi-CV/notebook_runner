"""Self-serve notebook authoring: submit validate and annotate jobs, follow them.

Authors upload a notebook or paste a GitHub URL, then watch the job's stage
and log until it ends with a summary: pass, or each issue with its fix. The
Runners page shows how busy each GPU target is and how long the queue is.

There is no login: the author's name identifies their jobs and is remembered
in a cookie. This app holds no credentials; the worker and the job containers
do the work (authoring/worker.py).

Run:
  uvicorn authoring.app:app --host 0.0.0.0 --port 8081
Environment:
  NBR_JOBS_DIR       where jobs/ lives (default: this repo's jobs/)
  NBR_DASHBOARD_URL  the regression dashboard, linked from the nav
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.exception_handlers import http_exception_handler
from jinja2 import ChoiceLoader, FileSystemLoader
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import capacity, labels, pipeline, store
from .labels import CATEGORIES
from . import render as nbrender

HERE = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = int(os.environ.get("NBR_MAX_UPLOAD_MB") or 10) * 1024 * 1024
DASHBOARD_URL = os.environ.get("NBR_DASHBOARD_URL")
# NBR_AUTHORING_PREFIX: the path the regression dashboard serves this app under
# when it proxies it on its own port (e.g. "/authoring"); links and redirects
# carry it, the routes do not (the dashboard strips it).
PREFIX = (os.environ.get("NBR_AUTHORING_PREFIX") or "").rstrip("/")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@'-]{0,59}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.ipynb$")

app = FastAPI(title="Notebook CI · Authoring", docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.loader = ChoiceLoader([FileSystemLoader(str(HERE / "templates")),
                                     FileSystemLoader(str(HERE.parent / "dashboard" / "templates"))])


def db():
    return store.connect()


def render(request: Request, template: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {
        "db_missing": False, "dashboard_url": DASHBOARD_URL, "categories": CATEGORIES,
        "prefix": PREFIX,
        "author": request.cookies.get("author", ""), **ctx})


def queue_position(conn, job: dict) -> int | None:
    """1-based place among GPU jobs waiting ahead of this one."""
    if job["status"] != "queued" or job["phase"] != "gpu":
        return None
    ahead = [j for j in store.active(conn) if j["status"] == "queued" and j["phase"] == "gpu"]
    ids = [j["id"] for j in ahead]
    return ids.index(job["id"]) + 1 if job["id"] in ids else None


# ── pages ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request, error: str = ""):
    author = request.cookies.get("author", "")
    conn = db()
    mine = store.list_jobs(conn, author=author, limit=20) if author else []
    return render(request, "home.html", mine=mine, error=error,
                  availability=capacity.read_snapshot())


@app.get("/jobs", response_class=HTMLResponse)
def jobs(request: Request, author: str = ""):
    return render(request, "jobs.html", jobs=store.list_jobs(db(), author=author or None, limit=200),
                  filter_author=author)


@app.post("/jobs")
async def submit(request: Request, kind: str = Form(...), author: str = Form(...),
                 category: str = Form(...), url: str = Form(""),
                 notebook: UploadFile | None = File(None)):
    author = author.strip()
    problem = None
    if kind not in ("validate", "annotate"):
        problem = "Choose validate or annotate."
    elif not NAME_RE.match(author):
        problem = "Enter your name or email (letters, digits, spaces, . _ @ ' -)."
    elif category not in CATEGORIES:
        problem = "Choose the tutorial's category."
    conn = db()
    if not problem and store.active_for_author(conn, author, kind):
        problem = (f"You already have a {kind} job queued or running. Wait for it to finish, "
                   "or cancel it, before submitting another.")
    data, name, source = None, None, None
    if not problem:
        if notebook is not None and notebook.filename:
            name = Path(notebook.filename).name
            data = await notebook.read(MAX_UPLOAD_BYTES + 1)
            source = "upload"
            if not FILE_RE.match(name):
                problem = "Upload a .ipynb file whose name has only letters, digits, . _ -"
            elif len(data) > MAX_UPLOAD_BYTES:
                problem = (f"The notebook is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB. "
                           "Clear its outputs and upload it again.")
            else:
                try:
                    assert isinstance(json.loads(data).get("cells"), list)
                except (ValueError, AssertionError, AttributeError):
                    problem = "That file is not a Jupyter notebook."
        elif url.strip():
            url = url.strip()
            m = re.match(r"https://github\.com/[^/]+/[^/]+/(?:pull/\d+|blob/.+?/([^/]+\.ipynb))$", url)
            if not m:
                problem = ("Paste a GitHub pull request URL (…/pull/123) or a notebook file URL "
                           "(…/blob/<branch>/docs/notebooks/<category>/<name>.ipynb).")
            name, source = (m.group(1) if m and m.group(1) else "pending.ipynb"), "pr"
        else:
            problem = "Upload a notebook or paste a GitHub URL."
    if problem:
        return RedirectResponse(f"{PREFIX}/?error={quote(problem)}", status_code=303)

    if source == "pr" and name == "pending.ipynb":
        name = f"pr-{url.rstrip('/').rsplit('/', 1)[-1]}.ipynb"
    job_id = store.create(conn, kind, name, author=author, category=category,
                          source=source, source_url=url or None)
    if data is not None:
        (store.job_dir(job_id) / "input.ipynb").write_bytes(data)
    resp = RedirectResponse(f"{PREFIX}/jobs/{job_id}", status_code=303)
    resp.set_cookie("author", author, max_age=365 * 24 * 3600, samesite="lax")
    return resp


def load_job(job_id: str) -> tuple:
    if not re.match(r"^[a-z][0-9a-f]+$", job_id):
        raise HTTPException(404, "No such job")
    conn = db()
    job = store.get(conn, job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return conn, job


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    conn, job = load_job(job_id)
    d = store.job_dir(job_id)
    log_text, log_next = read_log(job_id)
    final = job["status"] in store.FINAL
    # Recompute an annotate job's diff, so jobs from before a cell_diff change show it too.
    if job["kind"] == "annotate" and (job.get("summary") or {}).get("diff") \
            and (d / "input.ipynb").exists() and (d / "annotated.ipynb").exists():
        try:
            job["summary"]["diff"] = pipeline.cell_diff(json.loads((d / "input.ipynb").read_text()),
                                                        json.loads((d / "annotated.ipynb").read_text()))
        except ValueError:
            pass
    # The static checks are ready before the GPU run; show them as soon as they are.
    static = (job.get("summary") or {}).get("static")
    if static is None and (d / "static.json").exists():
        try:
            static = json.loads((d / "static.json").read_text())
        except ValueError:
            static = None
    return render(request, "job.html", job=job, position=queue_position(conn, job),
                  log=log_text, log_next=log_next, final=final, steps=job_steps(job, final, log_text),
                  static=static,
                  has={name: (d / name).exists() for name in
                       ("annotated.ipynb", "static.json", "CHANGES.md")},
                  executed=executed_path(job), views=list(notebook_views(job)), view_names=VIEWS,
                  estimates=(capacity.read_snapshot() or {}).get("estimates", {}))


@app.get("/api/jobs/{job_id}")
def job_api(job_id: str, since: int = 0):
    conn, job = load_job(job_id)
    try:
        lines = (store.job_dir(job_id) / "log.txt").read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    return JSONResponse({"status": job["status"], "stage": job["stage"], "target": job["target"],
                         "phase": job["phase"],
                         "final": job["status"] in store.FINAL,
                         "position": queue_position(conn, job),
                         "log": lines[since:since + 2000], "next": min(len(lines), since + 2000)})


@app.post("/jobs/{job_id}/cancel")
def cancel(request: Request, job_id: str):
    conn, job = load_job(job_id)
    if job["status"] in store.ACTIVE and job["kind"] != "regression":
        if request.cookies.get("author") != job["author"]:
            raise HTTPException(403, "Only the author who submitted this job can cancel it.")
        store.update(conn, job_id, cancel=1)
        store.log(job_id, f"Cancel requested by {job['author']}")
    return RedirectResponse(f"{PREFIX}/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}/annotated.ipynb")
def annotated(job_id: str):
    _, job = load_job(job_id)
    path = store.job_dir(job_id) / "annotated.ipynb"
    if not path.exists():
        raise HTTPException(404, "No annotated notebook")
    return FileResponse(path, media_type="application/x-ipynb+json", filename=job["notebook"])


@app.get("/jobs/{job_id}/executed.ipynb")
def executed(job_id: str):
    _, job = load_job(job_id)
    path = executed_path(job)
    if path is None:
        raise HTTPException(404, "No executed notebook")
    return FileResponse(path, media_type="application/x-ipynb+json",
                        filename=f"{Path(job['notebook']).stem}.executed.ipynb")


VIEWS = {"executed": "Executed", "annotated": "Annotated", "submitted": "Submitted"}


def notebook_views(job: dict) -> dict[str, Path]:
    """The notebooks a job page can show, best first."""
    d = store.job_dir(job["id"])
    views = {"executed": executed_path(job), "annotated": d / "annotated.ipynb",
             "submitted": d / "input.ipynb"}
    return {k: p for k, p in views.items() if p is not None and p.is_file()}


@app.get("/jobs/{job_id}/view/{which}", response_class=HTMLResponse)
def view_notebook(job_id: str, which: str):
    _, job = load_job(job_id)
    path = notebook_views(job).get(which)
    if path is None:
        raise HTTPException(404, "No such notebook for this job")
    try:
        nb = json.loads(path.read_text())
    except ValueError:
        raise HTTPException(422, "The notebook is not valid JSON")
    issues = ((job.get("summary") or {}).get("run") or {}).get("issues") or []
    flagged = {i["cell_index"] for i in issues if isinstance(i.get("cell_index"), int)} \
        if which != "annotated" else set()
    page = nbrender.notebook_html(nb, f"{job['notebook']} ({VIEWS[which].lower()})", flagged)
    # Notebooks come from authors: no scripts, no external loads, in a sandboxed frame.
    return HTMLResponse(page, headers={"Content-Security-Policy": nbrender.CSP,
                                       "X-Content-Type-Options": "nosniff"})


@app.get("/runners", response_class=HTMLResponse)
def runners(request: Request):
    return render(request, "runners.html", availability=capacity.read_snapshot())


@app.get("/api/runners")
def runners_api():
    return JSONResponse(capacity.read_snapshot() or {})


@app.get("/healthz")
def healthz():
    return "ok"


# ── helpers ──────────────────────────────────────────────────────────────────

def read_log(job_id: str, max_lines: int = 2000) -> tuple[str, int]:
    """The last max_lines of the job's log, and how many lines it has."""
    try:
        lines = (store.job_dir(job_id) / "log.txt").read_text(errors="replace").splitlines()
    except OSError:
        return "", 0
    return "\n".join(lines[-max_lines:]), len(lines)


def executed_path(job: dict) -> Path | None:
    run = ((job.get("summary") or {}).get("run") or {})
    name = Path(run.get("executed") or "").name
    if not name:
        return None
    path = store.job_dir(job["id"]) / "results" / "executed" / name
    return path if path.is_file() else None


OUTCOME = {"passed": "Passed", "done": "Done", "failed": "Failed", "blocked": "Blocked",
           "cancelled": "Cancelled", "error": "Crashed"}


def job_steps(job: dict, final: bool, log: str = "") -> list[dict]:
    """The job page's progress steps: [{"name", "state"}], where state is done,
    now, or (where a job stopped short) its outcome's badge class."""
    kind, phase, status, stage = job["kind"], job["phase"], job["status"], job.get("stage") or ""
    names = {"validate": ["Queued", "Static checks", "Waiting for a GPU", "Running"],
             "annotate": ["Queued", "Static checks", "Annotating", "Checking the result"],
             "regression": ["Queued", "Waiting for a GPU", "Running"]}[kind]
    names = names + [OUTCOME.get(status, "Done") if final else "Done"]
    gpu_wait = names.index("Waiting for a GPU") if "Waiting for a GPU" in names else None
    if kind == "annotate":
        seen = stage if not final else "\n".join(
            line[3:] for line in log.splitlines() if line.startswith("── "))
        reached = 3 if "Checking" in seen else 2 if "Annotating" in seen \
            else 1 if status == "running" or job.get("started_at") else 0
    elif phase == "gpu":
        ran = status == "running" or (job.get("summary") or {}).get("run")
        reached = gpu_wait + 1 if ran else gpu_wait
    else:
        reached = 1 if status == "running" or job.get("started_at") else 0
    if final and status in ("passed", "done"):
        return [{"name": n, "state": "done"} for n in names]
    if final:
        reached = min(reached, len(names) - 2)
        cls = job_class(status)
        # Only the step it stopped at is coloured; the outcome box below states the result.
        return [{"name": n, "state": "done" if i < reached else cls if i == reached else ""}
                for i, n in enumerate(names)]
    return [{"name": n, "state": "done" if i < reached else "now" if i == reached else ""}
            for i, n in enumerate(names)]


def severity_badge(sev: str) -> str:
    return {"error": "bad", "warning": "warn", "info": "muted"}.get(sev, "muted")


def job_class(status: str) -> str:
    return {"passed": "ok", "done": "ok", "failed": "bad", "error": "bad", "blocked": "warn",
            "cancelled": "muted", "queued": "muted", "running": "info"}.get(status, "muted")


templates.env.filters["sevclass"] = severity_badge
templates.env.filters["jobclass"] = job_class
templates.env.filters["diffrows"] = pipeline.diff_rows
# The same verdict words as the regression dashboard (pass/fail).
templates.env.filters["joblabel"] = lambda s: labels.STATUSES.get(s, s)
labels.register(templates.env)


@app.exception_handler(StarletteHTTPException)
async def html_errors(request: Request, exc: StarletteHTTPException):
    """A page for people, JSON for the live API and scripts."""
    if "text/html" in request.headers.get("accept", "") and "/api/" not in request.url.path:
        title = {404: "Not found", 403: "Not allowed"}.get(exc.status_code, "Something went wrong")
        return templates.TemplateResponse(request, "error.html", {
            "db_missing": False, "dashboard_url": DASHBOARD_URL, "prefix": PREFIX,
            "author": request.cookies.get("author", ""), "title": title, "detail": exc.detail,
            "back_url": f"{PREFIX}/jobs", "back_label": "Back to the jobs", "layout": "layout.html"},
            status_code=exc.status_code)
    return await http_exception_handler(request, exc)
