# Plan — timeout ceiling for CI passes

Status: **proposed, not implemented.** Written for review because it changes
when a pass is allowed to give up, which is a policy decision about the
pipeline, not a refactor.

## The problem

`tools/ci_run.py:99` streams a child process with no upper bound:

```python
proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, ...)
for line in proc.stdout:
    tee.write(line)
proc.wait()          # <-- blocks forever
```

If `run.py` (or the Claude CLI, or an `ssh` it spawned) wedges — a GPU node
that stops responding mid-`docker run`, a half-open TCP session, an agent stuck
in a retry loop — the read loop simply never returns.

Consequences, in order of how much they hurt:

1. **The pass never ends.** No verdicts are written, `STATUS.md` is never
   regenerated, nothing is committed.
2. **Every later pass is silently skipped.** `ci_cron.sh` / `run-ci.sh` hold a
   `flock` for the process lifetime. A wedged pass holds it forever, so each
   weekly fire logs "previous pass still running" and exits 0. The pipeline
   looks healthy while it has actually been dead for weeks.
3. **It is invisible.** A hang produces no error, no non-zero exit, no alert.

This is the failure mode most likely to take the pipeline down quietly, and it
is currently unguarded at every level.

## Design constraints

- **Per-notebook, not just per-pass.** A single wedged notebook must not consume
  the entire budget; the remaining notebooks deserve their turn.
- **Generous by default.** Fine-tune and pretrain notebooks legitimately run for
  hours (see the Timeouts table in `CLAUDE.md`). A ceiling that trips on healthy
  work is worse than none — it would manufacture false failures in a tool whose
  entire purpose is honest reporting.
- **A timeout is a result, not a crash.** It must land in `results/` as a
  real verdict so `STATUS.md` shows it and the next pass retries it. Dying
  without writing anything reproduces problem 1 at a smaller scale.
- **Kill the whole tree.** `run.py` spawns `claude`, which spawns `ssh`. Killing
  only the direct child orphans the rest, which keep holding the GPU node.

## Proposed change

### 1. `stream()` grows a deadline (`tools/ci_run.py`)

Add a `timeout: int | None` parameter. Implementation notes that matter:

- Start the child in **its own process group** (`start_new_session=True`) and
  signal the group with `os.killpg`, so `claude` and `ssh` die with it.
- Escalate: `SIGTERM`, grace period (~30s), then `SIGKILL`.
- The blocking `for line in proc.stdout` cannot itself observe a deadline —
  move the drain to a reader thread, or poll with `select`. A naive
  `proc.wait(timeout=...)` after the read loop does **not** fix the hang,
  because the read loop is where it blocks. This is the subtle part and the
  reason this is a plan rather than a two-line patch.
- Return a sentinel (e.g. `-1`) so callers can distinguish timeout from a
  normal non-zero exit.

### 2. Per-notebook budget, by category

`run_batch()` currently passes every notebook to one `run.py` invocation, so a
per-notebook ceiling means either batching per notebook or having `run.py`
enforce it. Recommended: **batch per notebook in CI** — the pass is already
sequential, and it makes the budget and the blast radius the same thing.

Budgets keyed off the existing category split in `manifest.yaml`:

| Category | Budget | Rationale |
|---|---|---|
| `inference`, `gpu_dev_optimize` | 60 min | Measured: median 8 min, p90 16 min, max 45 min (see below). |
| `fine_tune` | 3 h | Matches the "background + poll" guidance in `CLAUDE.md`. Not yet measured. |
| `pretrain` | 4 h | Longest observed legitimate work. Not yet measured. |

**Measured inference timings (2026-09-16, 1×MI300X, wall clock per notebook
including image pull and model download):**

| min | notebook | note |
|---:|---|---|
| 0.1–4.7 | 5 notebooks | env-blocked on HF gating — fail fast, never reach Docker |
| 7.6 | `build_airbnb_agent_mcp` | |
| 7.7 | `t2v_comfyui_api_mode_instinct` | |
| 7.9 | `deepseek_janus_cpu_gpu` | |
| 8.0 | `voice_pipeline_rag_ollama` | |
| 9.6 | `graph_multiagent_openclaw` | |
| 15.9 | `rag_ollama_llamaindex` | |
| 45.4 | `deepseekr1_sglang` | server start + a 20-min internal poll timeout |

Eight notebooks that actually executed: 107 min total, mean 13.3 min. The
45-minute outlier is the one that matters for budget sizing — it is legitimate
work (large MoE model load), not a hang, so a ceiling below ~50 min would
manufacture a false failure. 60 min gives that case ~30% headroom while still
bounding a wedged notebook to roughly one hour instead of forever.

Note these numbers are *warm-cache* for several images. A cold node re-pulls
40–90GB per image, so first-run times can be substantially higher; size the
ceiling for the cold case or pre-pull images before a pass.

Override per notebook with a `timeout_minutes:` key in `manifest.yaml`, so a
known-slow tutorial does not force the whole category up.

### 3. Whole-pass ceiling

A `--max-duration` argument (default **8 h**) checked between notebooks and
between phases. On expiry: stop scheduling new work, still run **Phase C**
(regenerate `STATUS.md`) and **Phase D** (commit) so a partial pass publishes
what it learned. Log exactly which notebooks were skipped — silent truncation
would read as "covered everything".

### 4. Timeouts become verdicts

On a timeout, write a result via `tools/write_result.py` with:

- `status: fail`
- one issue, `error_type: other`, `fixability: infra_blocked`
- a summary naming the budget and the phase it died in

`infra_blocked` is deliberate: the fixer agent must not try to "fix" a
notebook because the node wedged. This keeps the hang visible in `STATUS.md`
instead of leaving the notebook looking never-tested.

### 5. Container and lock guards (defense in depth)

- `docker run --stop-timeout` plus an outer `timeout(1)` in `run-ci.sh`, so a
  hang *below* Python (a wedged `docker` client) still terminates.
- Log the lock holder's PID and start time when `flock` is skipped, so a stuck
  pass is diagnosable from `logs/cron_skipped.log` alone.

## Testing

1. **Synthetic hang** — a fake `run.py` that `sleep`s past its budget; assert
   the pass terminates, writes a timeout verdict, and releases the lock.
2. **Process-tree kill** — confirm no orphaned `claude`/`ssh` survive
   (`pgrep -g <pgid>` empty after the kill).
3. **No false positives** — run a known-slow `pretrain` notebook and confirm it
   finishes inside its budget. This is the test that protects the tool's
   credibility; do not skip it.

## Open question for you

What should a timeout do to the **exit code** of the pass? Options: exit
non-zero (a monitor notices, but a single flaky node reds the whole run), or
exit 0 with the failure recorded in `STATUS.md` (matches today's "run.py exit=1
is expected" handling). My recommendation is **exit 0, record the verdict, and
alert via the Teams webhook** once that is wired up — the dashboard, not the
exit code, is the product.
