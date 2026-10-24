"""The job queue: one SQLite database and one directory per job.

jobs/jobs.db lives on local disk (not the NFS results share): the web app, the
worker and the job containers all write it, and SQLite is only safe for
concurrent writers on a local filesystem.

A job's kind is one of:
  validate    static checks, then a verify run on a GPU target (author)
  annotate    the annotate-notebook-for-ci skill on the notebook, no GPU (author)
  fix         the fixer agent on a failed validate job's notebook (its parent),
              then a re-run on a GPU target (author)
  regression  one notebook of the nightly pass on a GPU target (tools/ci_run.py)

Its status moves queued -> running -> one of the final states. A validate or
regression job waits for a GPU slot in `stage` "Waiting for a GPU".
"""

import json
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(os.environ.get("NBR_JOBS_DIR") or REPO_ROOT / "jobs")
# The regression notebooks (the gpuaidev checkout's docs/notebooks).
NOTEBOOKS_DIR = Path(os.environ.get("NBR_NOTEBOOKS") or REPO_ROOT / "notebooks")

KINDS = ("validate", "annotate", "fix", "regression")
AUTHOR_KINDS = ("validate", "annotate", "fix")
ACTIVE = ("queued", "running")
FINAL = ("passed", "failed", "blocked", "done", "error", "cancelled")

# Lower runs first: authors before the nightly pass.
PRIORITY = {"validate": 0, "annotate": 0, "fix": 0, "regression": 10}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,
    stage         TEXT,
    author        TEXT,
    notebook      TEXT NOT NULL,   -- file name, or the key under notebooks/ for regression
    category      TEXT,
    source        TEXT,            -- upload | pr | checkout
    source_url    TEXT,
    priority      INTEGER NOT NULL,
    phase         TEXT NOT NULL,   -- prepare (static checks, annotate) | gpu
    cancel        INTEGER NOT NULL DEFAULT 0,
    mode          TEXT,            -- regression: verify | full
    run_id        TEXT,            -- regression: the nightly pass it belongs to
    only_target   TEXT,            -- regression: ci_run.py --target
    gpus          INTEGER,
    target        TEXT,            -- runner.yaml target it runs on
    parent        TEXT,            -- fix: the validate job whose issues it fixes
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    summary_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_author ON jobs(author, created_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or JOBS_DIR / "jobs.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    # Switching to WAL takes an exclusive lock and does not wait for one, so two
    # processes opening a new database at once can collide: retry briefly.
    for attempt in range(50):
        try:
            if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            # Columns added since the table was first created.
            have = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
            if "parent" not in have:
                conn.execute("ALTER TABLE jobs ADD COLUMN parent TEXT")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or attempt == 49:
                raise
            time.sleep(0.1)
    if os.geteuid() == 0:
        # The worker runs as root on the host; the web app, the nightly pass and
        # the job containers write as uid 1001, the owner of jobs/. Keep the
        # database theirs, or they could no longer write it.
        owner = path.parent.stat()
        for f in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            if f.exists() and f.stat().st_uid != owner.st_uid:
                os.chown(f, owner.st_uid, owner.st_gid)
    return conn


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def new_id(kind: str) -> str:
    return f"{kind[0]}{datetime.now(timezone.utc):%Y%m%d%H%M%S}{secrets.token_hex(3)}"


def create(conn: sqlite3.Connection, kind: str, notebook: str, **fields) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind}")
    job_id = new_id(kind)
    job_dir(job_id).mkdir(parents=True)
    row = {"id": job_id, "kind": kind, "status": "queued", "stage": "Queued",
           "notebook": notebook, "priority": PRIORITY[kind], "created_at": now(),
           "phase": "gpu" if kind == "regression" else "prepare", **fields}
    # What the job's container reads: it never opens jobs.db.
    request = {k: row.get(k) for k in ("kind", "notebook", "category", "source",
                                         "source_url", "mode", "run_id", "author", "parent")}
    (job_dir(job_id) / "request.json").write_text(json.dumps(request, indent=1))
    cols = ", ".join(row)
    conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({', '.join('?' * len(row))})",
                 list(row.values()))
    log(job_id, f"Job created: {kind} {notebook}")
    return job_id


def get(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _with_summary(row) if row else None


def update(conn: sqlite3.Connection, job_id: str, **fields) -> None:
    if "summary" in fields:
        fields["summary_json"] = json.dumps(fields.pop("summary"), ensure_ascii=False)
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*fields.values(), job_id])


def set_stage(conn: sqlite3.Connection, job_id: str, stage: str) -> None:
    update(conn, job_id, stage=stage)
    log(job_id, f"── {stage}")


def finish(conn: sqlite3.Connection, job_id: str, status: str, stage: str,
           summary: dict | None = None) -> None:
    if status not in FINAL:
        raise ValueError(f"{status} is not a final status")
    fields = {"status": status, "stage": stage, "finished_at": now()}
    if summary is not None:
        fields["summary"] = summary
    update(conn, job_id, **fields)
    log(job_id, f"── {stage}")


def log(job_id: str, text: str) -> None:
    """Append a line to the job's progress log, which the web page tails."""
    path = job_dir(job_id) / "log.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text.rstrip("\n") + "\n")


def list_jobs(conn: sqlite3.Connection, author: str | None = None,
              kinds: tuple[str, ...] = AUTHOR_KINDS, limit: int = 100) -> list[dict]:
    marks = ", ".join("?" * len(kinds))
    sql = f"SELECT * FROM jobs WHERE kind IN ({marks})"
    args: list = list(kinds)
    if author:
        sql += " AND author = ?"
        args.append(author)
    sql += " ORDER BY created_at DESC LIMIT ?"
    return [_with_summary(r) for r in conn.execute(sql, [*args, limit])]


def active(conn: sqlite3.Connection) -> list[dict]:
    """Queued and running jobs, in the order the scheduler considers them."""
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
        "ORDER BY priority, created_at")
    return [_with_summary(r) for r in rows]


def active_for_author(conn: sqlite3.Connection, author: str, kind: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE author = ? AND kind = ? "
        "AND status IN ('queued', 'running')", (author, kind)).fetchone()[0]


def children(conn: sqlite3.Connection, parent: str) -> list[dict]:
    """The fix jobs started from a validate job, newest first."""
    rows = conn.execute("SELECT * FROM jobs WHERE parent = ? ORDER BY created_at DESC", (parent,))
    return [_with_summary(r) for r in rows]


def recent_durations(conn: sqlite3.Connection, kind: str, limit: int = 20) -> list[float]:
    """Seconds the last finished jobs of this kind took, for wait estimates."""
    rows = conn.execute(
        "SELECT started_at, finished_at FROM jobs WHERE kind = ? AND started_at IS NOT NULL "
        "AND finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT ?", (kind, limit))
    out = []
    for r in rows:
        start, end = _parse(r["started_at"]), _parse(r["finished_at"])
        if start and end:
            out.append((end - start).total_seconds())
    return out


def _parse(ts: str | None) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _with_summary(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["summary"] = json.loads(d.get("summary_json") or "null")
    except json.JSONDecodeError:
        d["summary"] = None
    return d
