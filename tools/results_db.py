"""
SQLite history of every notebook regression run.

results/*.json is a latest-wins cache that feeds STATUS.md; this database is the
append-only record behind it. run.py writes to it unconditionally — one `runs`
row per invocation and one `results` row per notebook per phase (verify / fix /
skip) — so questions like "when did
this notebook start failing?" or "what did last week's pass cost?" are a query,
not an archaeology dig through JSON files.

Usage:
  python3 tools/results_db.py import              # backfill from results/*.json
  sqlite3 results/runs.db 'SELECT * FROM latest_results'

Design notes:

  * stdlib only (sqlite3). Same reasoning as report_teams.py.

  * Writes from run.py are never fatal. A locked or corrupt database must not
    throw away an hour of GPU time; the JSON result file is still written by
    the agent regardless, and `import` can backfill later.

  * The full result payload is kept verbatim in `results.payload_json`. The
    columns are for querying; the payload is the source of truth, so a field
    added to write_result.py later is never silently dropped.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "results" / "runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    mode            TEXT,
    host            TEXT,
    argv            TEXT,
    passed          INTEGER,
    failed          INTEGER,
    skipped         INTEGER,
    total_cost_usd  REAL
);

CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    notebook        TEXT NOT NULL,      -- path under the notebook root, e.g. inference/foo.ipynb
    notebook_path   TEXT,
    phase           TEXT NOT NULL,      -- verify | fix | skip
    server          TEXT,               -- runner.yaml target it ran on (ssh host or k8s pool)
    agent           TEXT,
    status          TEXT NOT NULL,      -- pass | fail | partial | skip | unknown
    summary         TEXT,
    message         TEXT,               -- runner-level error (crash, no result file, skip reason)
    docker_image    TEXT,
    cost_usd        REAL,
    num_turns       INTEGER,
    result_file     TEXT,
    payload_json    TEXT,
    recorded_at     TEXT NOT NULL,
    UNIQUE (result_file)
);

-- Issues live in payload_json; query them with json_each(payload_json, '$.issues').
DROP TABLE IF EXISTS issues;

CREATE INDEX IF NOT EXISTS idx_results_notebook ON results(notebook, recorded_at);
CREATE INDEX IF NOT EXISTS idx_results_run      ON results(run_id);

-- Most recent non-skip result per notebook: the DB-side analogue of STATUS.md.
-- Ordered by recorded_at, not id: a backfilled import gets higher ids than live
-- rows recorded earlier. Recreated on every connect so the definition stays current.
DROP VIEW IF EXISTS latest_results;
CREATE VIEW latest_results AS
SELECT r.* FROM results r
WHERE r.phase != 'skip' AND r.id = (
    SELECT r2.id FROM results r2
    WHERE r2.notebook = r.notebook AND r2.phase != 'skip'
    ORDER BY r2.recorded_at DESC, r2.id DESC LIMIT 1
);
"""

def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def warn(msg: str) -> None:
    print(f"[results_db] warning: {msg}", file=sys.stderr)


def normalize_ts(ts: str | None) -> str:
    """Return a naive-UTC ISO timestamp. Agent payloads write naive UTC while
    now_iso() carries '+00:00'; recorded_at must use one form to sort as text."""
    if not ts:
        return datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat()
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def start_run(conn: sqlite3.Connection, run_id: str, mode: str,
              host: str | None, argv: list[str]) -> None:
    # INSERT OR IGNORE: ci_run.py runs several run.py batches under one run_id.
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, started_at, mode, host, argv) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, now_iso(), mode, host, json.dumps(argv)),
        )


def finish_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Recompute run totals from the results table (correct across batches).

    A notebook counts once, by its final phase: in --mode full a failed verify
    followed by a passing fix is a pass.
    """
    with conn:
        conn.execute(
            """
            UPDATE runs SET
                finished_at    = ?,
                passed         = (SELECT COUNT(*) FROM results r WHERE r.id IN
                                    (SELECT MAX(id) FROM results WHERE run_id = runs.run_id
                                     AND phase != 'skip' GROUP BY notebook)
                                  AND r.status = 'pass'),
                failed         = (SELECT COUNT(*) FROM results r WHERE r.id IN
                                    (SELECT MAX(id) FROM results WHERE run_id = runs.run_id
                                     AND phase != 'skip' GROUP BY notebook)
                                  AND r.status != 'pass'),
                skipped        = (SELECT COUNT(DISTINCT notebook) FROM results
                                  WHERE run_id = runs.run_id AND phase = 'skip'),
                total_cost_usd = (SELECT SUM(cost_usd) FROM results
                                  WHERE run_id = runs.run_id)
            WHERE run_id = ?
            """,
            (now_iso(), run_id),
        )


def record_result(conn: sqlite3.Connection, run_id: str | None, notebook: str,
                  notebook_path: str | None, phase: str, result: dict,
                  server: str | None = None) -> int | None:
    """Insert one phase outcome. `result` is run.py's result dict: the agent's
    JSON payload (if one was written) plus cost_usd/num_turns/message/result_path.

    Returns the new row id, or None if this result file was already recorded
    (e.g. env_check blocked, so verify and the old result are the same file).
    """
    result_file = result.get("result_path")
    payload = {k: v for k, v in result.items()
               if k not in ("cost_usd", "num_turns", "result_path")}
    with conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO results (
                run_id, notebook, notebook_path, phase, server, agent, status,
                summary, message, docker_image, cost_usd, num_turns, result_file,
                payload_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, notebook, notebook_path, phase, server,
                result.get("agent"),
                result.get("status") or "unknown",
                result.get("summary"),
                result.get("message"),
                result.get("docker_image_resolved"),
                result.get("cost_usd"),
                result.get("num_turns"),
                result_file,
                json.dumps(payload),
                normalize_ts(result.get("timestamp")),
            ),
        )
    return cur.lastrowid if cur.rowcount else None


def run_results(conn: sqlite3.Connection, run_id: str) -> tuple[dict | None, list[dict]]:
    """Return (run row, final result per notebook) for one run. Used by the
    Teams card; the final phase is what the run concluded about the notebook."""
    run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    rows = conn.execute(
        """
        SELECT r.* FROM results r WHERE r.id IN (
            SELECT MAX(id) FROM results WHERE run_id = ? GROUP BY notebook
        ) ORDER BY r.notebook
        """,
        (run_id,),
    ).fetchall()
    return (dict(run) if run else None), _with_issues(rows)


ISSUE_FIELDS = ("cell_index", "error_type", "fixability", "description", "proposed_fix")


def _with_issues(rows) -> list[dict]:
    """Rows as dicts, each with `issues` parsed from its payload."""
    out = []
    for row in rows:
        d = dict(row)
        try:
            issues = json.loads(d.get("payload_json") or "{}").get("issues") or []
        except (json.JSONDecodeError, AttributeError):
            issues = []
        d["issues"] = [{k: i.get(k) for k in ISSUE_FIELDS}
                       for i in issues if isinstance(i, dict)]
        out.append(d)
    return out


def latest_by_notebook(conn: sqlite3.Connection) -> dict[str, dict]:
    """Latest non-skip result per notebook key, with its issues."""
    rows = conn.execute("SELECT * FROM latest_results").fetchall()
    return {r["notebook"]: r for r in _with_issues(rows)}


def notebook_history(conn: sqlite3.Connection, notebook: str) -> list[dict]:
    """Every non-skip result for one notebook, newest first, with issues."""
    rows = conn.execute(
        "SELECT * FROM results WHERE notebook = ? AND phase != 'skip' "
        "ORDER BY recorded_at DESC, id DESC", (notebook,)).fetchall()
    return _with_issues(rows)


def previous_status(conn: sqlite3.Connection, run_id: str) -> dict[str, str]:
    """{notebook: status} of each notebook's last non-skip result recorded
    before `run_id` started, for the notebooks that run touched."""
    run = conn.execute("SELECT started_at FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        return {}
    rows = conn.execute(
        """
        SELECT r.notebook, (
            SELECT p.status FROM results p
            WHERE p.notebook = r.notebook AND p.phase != 'skip'
              AND p.recorded_at < ? AND p.run_id IS NOT ?
            ORDER BY p.recorded_at DESC, p.id DESC LIMIT 1
        ) AS status
        FROM (SELECT DISTINCT notebook FROM results WHERE run_id = ?) r
        """,
        (normalize_ts(run["started_at"]), run_id, run_id),
    ).fetchall()
    return {r["notebook"]: r["status"] for r in rows if r["status"]}


def get_result(conn: sqlite3.Connection, result_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
    return _with_issues([row])[0] if row else None


def list_runs(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))]


def notebook_keys(notebooks_dir: Path) -> dict[str, str]:
    """Map notebook filename -> key (e.g. 'foo.ipynb' -> 'inference/foo.ipynb').
    Filenames that appear under more than one key are left out as ambiguous."""
    by_name: dict[str, list[str]] = {}
    for p in sorted(notebooks_dir.rglob("*.ipynb")) if notebooks_dir.is_dir() else []:
        by_name.setdefault(p.name, []).append(str(p.relative_to(notebooks_dir)))
    return {name: keys[0] for name, keys in by_name.items() if len(keys) == 1}


def import_json(conn: sqlite3.Connection, results_dir: Path,
                notebooks_dir: Path = REPO_ROOT / "notebooks") -> int:
    """Backfill results/*.json not yet in the database (e.g. from another machine).

    Imported rows have no run_id and no cost (cost lived only in run.py's
    memory). The JSON records only an absolute local notebook path, so the key
    is resolved against the notebook tree to match what run.py writes. Files are
    deduplicated by filename, not absolute path, so the same result imported on
    two hosts is recorded once.
    """
    keys = notebook_keys(notebooks_dir)
    seen = {Path(r["result_file"]).name for r in
            conn.execute("SELECT result_file FROM results WHERE result_file IS NOT NULL")}
    n = 0
    for path in sorted(results_dir.glob("*.json")):
        if path.name in seen:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            warn(f"could not parse {path.name}: {e}")
            continue
        nb_path = payload.get("notebook") or ""
        name = Path(nb_path).name if nb_path else f"{path.stem.rsplit('_', 1)[0]}.ipynb"
        notebook = keys.get(name, name)
        phase = "fix" if payload.get("agent") == "claude_code_fix" else "verify"
        if record_result(conn, None, notebook, nb_path, phase,
                         {**payload, "result_path": str(path.resolve())}):
            n += 1
            seen.add(path.name)
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Notebook regression results database.")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="Backfill from results/*.json (idempotent).")
    imp.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    imp.add_argument("--notebooks-dir", default=str(REPO_ROOT / "notebooks"),
                     help="Notebook tree used to map result filenames to notebook keys.")
    args = p.parse_args()

    conn = connect(args.db)
    if args.cmd == "import":
        n = import_json(conn, Path(args.results_dir), Path(args.notebooks_dir))
        print(f"imported {n} new result(s) into {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
