"""
SQLite history of every notebook regression run.

results/*.json is a latest-wins cache that feeds STATUS.md; this database is the
append-only record behind it. run.py writes to it unconditionally — one `runs`
row per invocation, one `results` row per notebook per phase (verify / fix /
skip), and one `issues` row per diagnosed issue — so questions like "when did
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
    notebook        TEXT NOT NULL,      -- manifest key, e.g. inference/foo.ipynb
    notebook_path   TEXT,
    phase           TEXT NOT NULL,      -- verify | fix | skip
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

CREATE TABLE IF NOT EXISTS issues (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id       INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    cell_index      INTEGER,
    error_type      TEXT,
    fixability      TEXT,
    description     TEXT,
    proposed_fix    TEXT
);

CREATE INDEX IF NOT EXISTS idx_results_notebook ON results(notebook, recorded_at);
CREATE INDEX IF NOT EXISTS idx_results_run      ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_issues_result    ON issues(result_id);

-- Most recent non-skip result per notebook: the DB-side analogue of STATUS.md.
CREATE VIEW IF NOT EXISTS latest_results AS
SELECT r.* FROM results r
JOIN (
    SELECT notebook, MAX(id) AS id FROM results
    WHERE phase != 'skip' GROUP BY notebook
) latest ON latest.id = r.id;
"""


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def warn(msg: str) -> None:
    print(f"[results_db] warning: {msg}", file=sys.stderr)


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
                  notebook_path: str | None, phase: str, result: dict) -> int | None:
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
                run_id, notebook, notebook_path, phase, agent, status, summary,
                message, docker_image, cost_usd, num_turns, result_file,
                payload_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, notebook, notebook_path, phase,
                result.get("agent"),
                result.get("status") or "unknown",
                result.get("summary"),
                result.get("message"),
                result.get("docker_image_resolved"),
                result.get("cost_usd"),
                result.get("num_turns"),
                result_file,
                json.dumps(payload),
                result.get("timestamp") or now_iso(),
            ),
        )
        if cur.rowcount == 0:
            return None
        result_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO issues (result_id, cell_index, error_type, fixability, "
            "description, proposed_fix) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (result_id, i.get("cell_index"), i.get("error_type"),
                 i.get("fixability"), i.get("description"), i.get("proposed_fix"))
                for i in (result.get("issues") or []) if isinstance(i, dict)
            ],
        )
    return result_id


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
    out = []
    for row in rows:
        d = dict(row)
        d["issues"] = [dict(i) for i in conn.execute(
            "SELECT * FROM issues WHERE result_id = ? ORDER BY id", (row["id"],))]
        out.append(d)
    return (dict(run) if run else None), out


def import_json(conn: sqlite3.Connection, results_dir: Path) -> int:
    """Backfill results/*.json written before the database existed.

    Imported rows have no run_id and no cost (cost lived only in run.py's
    memory). The notebook key is the bare filename stem's .ipynb — the JSON
    doesn't record the manifest-relative path, only the absolute local one.
    """
    n = 0
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            warn(f"could not parse {path.name}: {e}")
            continue
        nb_path = payload.get("notebook") or ""
        notebook = Path(nb_path).name if nb_path else f"{path.stem}.ipynb"
        phase = "fix" if payload.get("agent") == "claude_code_fix" else "verify"
        if record_result(conn, None, notebook, nb_path, phase,
                         {**payload, "result_path": str(path.resolve())}):
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Notebook regression results database.")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="Backfill from results/*.json (idempotent).")
    imp.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    conn = connect(args.db)
    if args.cmd == "import":
        n = import_json(conn, Path(args.results_dir))
        print(f"imported {n} new result(s) into {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
