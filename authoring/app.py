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


def render(request: Request, template: str, status_code: int = 200, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {
        "db_missing": False, "dashboard_url": DASHBOARD_URL, "categories": CATEGORIES,
        "prefix": PREFIX,
        "author": request.cookies.get("author", ""), **ctx}, status_code=status_code)


def queue_position(conn, job: dict) -> int | None:
    """1-based place among GPU jobs waiting ahead of this one."""
    if job["status"] != "queued" or job["phase"] != "gpu":
        return None
    ahead = [j for j in store.active(conn) if j["status"] == "queued" and j["phase"] == "gpu"]
    ids = [j["id"] for j in ahead]
    return ids.index(job["id"]) + 1 if job["id"] in ids else None


# ── pages ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request, url: str = "", category: str = ""):
    # Other pages (the dashboard's notebook page) link here with both forms pre-filled.
    prefill = {"url": url.strip()[:500], "category": category if category in CATEGORIES else ""}
    return home_page(request, {"validate": prefill, "annotate": prefill})


def home_page(request: Request, values: dict, error: dict | None = None) -> HTMLResponse:
    """The submit page. values: what each form shows ({kind: {author, category,
    url}}); error: a rejected submission ({kind, text, blocking, file}), shown
    next to its form, with status 400."""
    author = request.cookies.get("author", "")
    mine = store.list_jobs(db(), author=author, limit=20) if author else []
    return render(request, "home.html", status_code=400 if error else 200, mine=mine,
                  values=values, error=error, availability=capacity.read_snapshot())


@app.get("/jobs", response_class=HTMLResponse)
def jobs(request: Request, author: str = ""):
    return render(request, "jobs.html", jobs=store.list_jobs(db(), author=author or None, limit=200),
                  filter_author=author)


@app.post("/jobs")
async def submit(request: Request, kind: str = Form(...), author: str = Form(...),
                 category: str = Form(...), url: str = Form(""),
                 notebook: UploadFile | None = File(None)):
    author = author.strip()
    problem, blocking = None, None
    if kind not in ("validate", "annotate"):
        problem = "Choose validate or annotate."
    elif not NAME_RE.match(author):
        problem = "Enter your name or email (letters, digits, spaces, . _ @ ' -)."
    elif category not in CATEGORIES:
        problem = "Choose the tutorial's category."
    conn = db()
    if not problem:
        blocking = active_job(conn, author, kind)
        if blocking:
            problem = one_at_a_time(kind)
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
        # Back to the same form with what the author typed; a browser cannot
        # refill a file input, so say which file to choose again.
        form = kind if kind in ("validate", "annotate") else "validate"
        return home_page(request, {form: {"author": author, "category": category, "url": url}},
                         {"kind": form, "text": problem, "blocking": blocking,
                          "file": notebook.filename if notebook is not None else None})

    if source == "pr" and name == "pending.ipynb":
        name = f"pr-{url.rstrip('/').rsplit('/', 1)[-1]}.ipynb"
    job_id = create_job(conn, kind, name, author, category, source, url or None, data)
    resp = RedirectResponse(f"{PREFIX}/jobs/{job_id}", status_code=303)
    resp.set_cookie("author", author, max_age=365 * 24 * 3600, samesite="lax")
    return resp


def create_job(conn, kind: str, name: str, author: str, category: str, source: str,
               url: str | None, data: bytes | None) -> str:
    job_id = store.create(conn, kind, name, author=author, category=category,
                          source=source, source_url=url)
    if data is not None:
        (store.job_dir(job_id) / "input.ipynb").write_bytes(data)
    return job_id


def one_at_a_time(kind: str) -> str:
    return (f"You already have {'an' if kind == 'annotate' else 'a'} {kind} job queued or running. Wait for it to finish, "
            "or cancel it, before submitting another.")


def active_job(conn, author: str, kind: str) -> dict | None:
    """The author's queued or running job of this kind: one at a time per kind."""
    return next((j for j in store.active(conn) if j["author"] == author and j["kind"] == kind),
                None)


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
                  estimates=(capacity.read_snapshot() or {}).get("estimates", {}),
                  source=github_source(job.get("source_url")), blocked_by=blocked_reason(job),
                  again=again_state(conn, job, request.cookies.get("author", "")) if final else None)


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


@app.post("/jobs/{job_id}/again")
def run_again(request: Request, job_id: str):
    """A new job of the same kind on the same input: the same uploaded file, or
    the same GitHub URL fetched again (so it picks up new commits)."""
    conn, job = load_job(job_id)
    state = again_state(conn, job, request.cookies.get("author", ""))
    if state is None:
        raise HTTPException(409, "This job cannot be run again.")
    if state["not_author"]:
        raise HTTPException(403, "Only the author who submitted this job can run it again.")
    if state["blocking"]:
        kind = job["kind"]
        return home_page(request, {kind: {"author": job["author"], "category": job["category"],
                                          "url": job["source_url"] or ""}},
                         {"kind": kind, "text": one_at_a_time(kind),
                          "blocking": state["blocking"], "file": None})
    data = (store.job_dir(job_id) / "input.ipynb").read_bytes() if job["source"] == "upload" else None
    new_id = create_job(conn, job["kind"], job["notebook"], job["author"], job["category"],
                        job["source"], job["source_url"], data)
    store.log(new_id, f"Runs job {job_id} again")
    return RedirectResponse(f"{PREFIX}/jobs/{new_id}", status_code=303)


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
    # The annotated notebook numbers its cells differently: its issues would land on the wrong cells.
    issues = ((job.get("summary") or {}).get("run") or {}).get("issues") or [] \
        if which != "annotated" else []
    page = nbrender.notebook_html(nb, f"{job['notebook']} ({VIEWS[which].lower()})", issues)
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


def again_state(conn, job: dict, author: str) -> dict | None:
    """Whether a finished author job can be run again by this visitor: None if
    it cannot be at all (still active, a nightly job, its upload is gone),
    else {"not_author": bool, "blocking": the author's active job of that kind}."""
    if job["status"] not in store.FINAL or job["kind"] == "regression" or not job["author"]:
        return None
    if job["source"] == "upload" and not (store.job_dir(job["id"]) / "input.ipynb").is_file():
        return None
    if job["source"] != "upload" and not job["source_url"]:
        return None
    return {"not_author": author != job["author"],
            "blocking": active_job(conn, job["author"], job["kind"])}


def blocked_reason(job: dict) -> str | None:
    """Why a blocked job stopped, for the job page's next step: no_runner,
    metadata (metadata.ci unreadable), fetch (GitHub or the input), or run."""
    if job["status"] != "blocked":
        return None
    stage = job.get("stage") or ""
    if "no runner fits" in stage:
        return "no_runner"
    if "metadata.ci" in stage:
        return "metadata"
    return "run" if (job.get("summary") or {}).get("run") else "fetch"


def github_source(url: str | None) -> dict | None:
    """A GitHub source URL as words: {"label", "title"}: "pull request #57" with
    the repository as the title, or "gpuaidev@main" with the file's path."""
    m = re.match(r"https://github\.com/([^/]+/([^/]+))/(?:pull/(\d+)|blob/([^/]+)/(.+))$", url or "")
    if not m:
        return None
    if m[3]:
        return {"label": f"pull request #{m[3]}", "title": m[1]}
    return {"label": f"{m[2]}@{m[4]}", "title": f"{m[1]}: {m[5]}"}


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
            "cancelled": "muted", "queued": "pending", "running": "info"}.get(status, "muted")


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
