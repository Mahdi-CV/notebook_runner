# notebook_runner

An autonomous notebook regression testing pipeline for AMD ROCm GPU tutorials. Built on [Claude Code](https://claude.ai/claude-code).

> **Health auditor, not a CI runner.** It surfaces bugs so tutorial authors can fix them — it does not silently patch content errors to make numbers look green.

---

## Architecture at a glance

The pipeline is built from **two specialized agents** with a strict division of
labour, orchestrated by a thin launcher and fed into a GitHub reporter. Each
agent is driven entirely by its own playbook (`CLAUDE.md` / `CLAUDE_FIX.md`) —
the launcher only handles infrastructure.

- **Verify agent** (`CLAUDE.md`, identity `claude_code_verify`) — the
  *diagnostician*. Runs each notebook once as published, collects cell-level
  errors, and classifies every issue by `fixability`. It never patches content.
- **Fix agent** (`CLAUDE_FIX.md`, identity `claude_code_fix`) — the *surgeon*.
  Takes the verification diagnosis and applies **only** `auto_fixable` fixes,
  validates each by re-running, and retrieves a clean fix notebook for a PR.

Separating diagnosis from repair keeps each agent simple, auditable, and hard
to corrupt: the verifier has no incentive to green-wash because it cannot fix,
and the fixer cannot invent new diagnoses because it only acts on the verifier's
classified issues.

```
                          ┌──────────────────────────┐
                          │        run.py            │
                          │  (launcher / plumbing)   │
                          │  SCP notebook + tools →   │
                          │  spawn agents → parse →   │
                          │  read result JSON         │
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────┴───────────────────┐
                    │  per notebook, --mode full            │
                    ▼                                       │
      ┌──────────────────────────────┐                     │
      │  VERIFY AGENT                 │                     │
      │  playbook: CLAUDE.md          │                     │
      │  identity: claude_code_verify │                     │
      │                               │                     │
      │  1. Read every cell           │                     │
      │  2. Detect pattern A/B/C/D    │                     │
      │  3. Resolve Docker image      │                     │
      │  4. Pre-flight patch (4 only) │                     │
      │  5. Run ONCE via papermill    │                     │
      │  6. Collect cell errors       │                     │
      │  7. Diagnose + classify       │                     │
      │     fixability per issue      │                     │
      └───────────────┬───────────────┘                    │
                      │ writes result JSON                  │
                      ▼                                     │
             ┌──────────────────┐                          │
             │  results/*.json   │  ◄───────────────────────┘
             │  (diagnosis)      │
             └────────┬──────────┘
                      │
         status != pass AND ≥1 auto_fixable issue?
                      │ yes                        │ no
                      ▼                            ▼
      ┌──────────────────────────────┐      (stop — carry
      │  FIX AGENT                    │       diagnosis through)
      │  playbook: CLAUDE_FIX.md      │
      │  identity: claude_code_fix    │
      │                               │
      │  1. Load verification result  │
      │  2. Filter to auto_fixable    │
      │  3. Validate version pins     │
      │  4. Apply fixes → _fixed.ipynb│
      │  5. Re-run ONCE, validate     │
      │  6. Retrieve genuine-fix nb   │
      └───────────────┬───────────────┘
                      │ writes updated result JSON
                      ▼                    + fixes/<stem>.ipynb
             ┌──────────────────┐
             │  results/*.json   │
             │  (diagnosis+fixes)│
             └────────┬──────────┘
                      │
                      ▼
      ┌──────────────────────────────┐        ┌─────────────────────┐
      │  tools/status.py              │        │ tools/report_github │
      │  → STATUS.md dashboard        │        │  needs_author  → ISSUE
      │  (latest result per notebook) │        │  validated fix → PR │
      └──────────────────────────────┘        │  now passes    → close
                                               └─────────────────────┘
```

`tools/ci_run.py` chains the whole thing on a schedule: run the gap set, re-run
persistent failures, regenerate `STATUS.md`, and commit — all in `--mode full`.

---

## What it does

For each notebook, the **verify agent**:

1. Reads every cell and identifies the execution pattern (Docker / server+client / host Python)
2. Resolves the correct versioned Docker image tag for the target hardware via the Docker Hub API
3. Applies only the 4 allowed pre-flight patches
4. Runs the notebook end-to-end **once** via papermill inside the correct Docker container
5. Collects cell-level errors from the output notebook
6. Analyses each failure on two independent axes: structural blockers and code quality issues
7. Classifies each issue by `fixability` and writes a structured JSON result to `results/`

Then, if the verify result is not a pass and contains at least one `auto_fixable`
issue, the **fix agent**:

8. Loads the verification diagnosis and applies **only** the `auto_fixable` fixes
9. Validates version pins against the actual Docker image before committing to them
10. Re-runs the patched notebook **once** and marks each fix `validated: true/false`
11. Retrieves a clean genuine-fix notebook (no scaffolding) into `fixes/` for a PR
12. Writes an updated result linking back to the verification result

Both agents always clean up remote files and containers after any outcome.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | |
| `claude` CLI + AMD LLM gateway config | AMD employees: already provisioned via company offering. Verify with `claude -p "say hello"` |
| `pyyaml` | `pip install pyyaml` |
| `python-dotenv` | `pip install python-dotenv` |
| SSH access to an AMD GPU server | Key-based auth recommended |
| HuggingFace token | For gated model downloads |
| `gh` CLI | Only for `tools/report_github.py --publish` |

---

## Setup

```bash
git clone https://github.com/Mahdi-CV/notebook-runner-agent.git
cd notebook-runner-agent

cp .env.example .env
# Edit .env and fill in GPU_HOST, GPU_USER, HF_TOKEN
```

**.env keys:**

| Variable   | Required | Description |
|------------|----------|-------------|
| `GPU_HOST` | yes      | Hostname or IP of the AMD GPU server |
| `GPU_USER` | yes      | SSH username on the GPU server |
| `HF_TOKEN` | yes      | HuggingFace access token for gated model downloads |

---

## Usage

```bash
# Single notebook — verify then auto-fix (default mode)
python3 run.py path/to/notebook.ipynb

# Verify only — diagnose, never fix
python3 run.py path/to/notebook.ipynb --mode verify

# Fix only — apply fixes from a prior verification result
python3 run.py path/to/notebook.ipynb --mode fix --verification-result results/foo.json

# All notebooks in a directory
python3 run.py --dir /path/to/notebooks/

# Filter to one category
python3 run.py --dir /path/to/notebooks/ --category inference

# Only the gap set (never-tested + stale >7d)
python3 run.py --dir /path/to/notebooks/ --gap

# Only notebooks whose latest result is a hard fail
python3 run.py --dir /path/to/notebooks/ --failing

# Skip notebooks that need more GPUs than the target node has
python3 run.py --dir /path/to/notebooks/ --max-gpus 1

# Override server credentials for this run
python3 run.py path/to/notebook.ipynb --host my-server.example.com --user amd

# Interactive mode — opens a live Claude session with full runtime context pre-loaded
python3 run.py path/to/notebook.ipynb --interactive
```

**Modes:** `verify` (diagnosis only) | `fix` (apply fixes from prior result) | `full` (verify then fix — default)

**Categories:** `inference` | `fine_tune` | `pretrain` | `gpu_dev_optimize`

### CI orchestrator

`tools/ci_run.py` runs an automated regression pass and updates the dashboard in four phases:

```bash
python3 tools/ci_run.py                       # full pass over notebooks/
python3 tools/ci_run.py --category inference  # one category only
python3 tools/ci_run.py --no-commit           # update files, don't commit
python3 tools/ci_run.py --skip-gap            # only attack current failures
python3 tools/ci_run.py --dry-run             # print the plan, run nothing
```

| Phase | Name | What happens |
|-------|------|--------------|
| A | Freshness | Run the gap set (never-tested + stale >7d) in `--mode full` |
| B | Failures | Re-aggregate, then re-run notebooks still failing (deduped vs Phase A) |
| C | Dashboard | Regenerate `STATUS.md` from fresh results |
| D | Commit | Auto-commit `STATUS.md` + `manifest.yaml` (skip with `--no-commit`) |

A weekly cron (`tools/ci_cron.sh`, `flock`-guarded) drives this unattended.

### Manifest (optional)

Create `manifest.yaml` in the same directory as `run.py` to control per-notebook behaviour:

```yaml
servers:
  mi300x-box:
    host: gpu1.example.com
    user: amd
    hardware: mi300x

notebooks:
  inference/my_notebook.ipynb:
    server: mi300x-box
    expected_result: partial   # "partial" counts as passing
    skip: false
    notes: "Known issue: cell 3 uses deprecated vllm flag --max-num-seqs"

  fine_tune/slow_notebook.ipynb:
    skip: true
    skip_reason: "Requires H100, not available on ROCm server"
```

---

## Output

Results are written to `results/` as `{notebook_stem}_{timestamp_UTC}.json`. A
fix-agent result carries the full diagnosis **and** the applied fixes, and links
back to the verification result it acted on:

```json
{
  "notebook": "/path/to/rag_ollama_llamaindex.ipynb",
  "status": "pass",
  "summary": "Phase 1 failed at cell 6 (systemctl hang) and cell 19 (flatbuffers PEP 440 error). Both auto_fixable issues were fixed and validated.",
  "issues": [
    {
      "cell_index": 6,
      "error_type": "content_error",
      "description": "Cell uses sudo systemctl start ollama — hangs when Ollama is already running on port 11434.",
      "proposed_fix": "Replace with !ollama list to verify Ollama is running without relying on systemd.",
      "fixability": "auto_fixable"
    },
    {
      "cell_index": 19,
      "error_type": "version_incompatibility",
      "description": "pip install chromadb fails: system flatbuffers has a non-PEP 440 version string, which pip 24.1+ rejects.",
      "proposed_fix": "Add --ignore-installed flatbuffers to the pip install command.",
      "fixability": "auto_fixable"
    }
  ],
  "fixes": [
    {"cell_index": 6,  "description": "Replaced systemctl cell with !ollama list", "patch": "!ollama list", "validated": true},
    {"cell_index": 19, "description": "Added --ignore-installed flatbuffers to pip install chromadb", "patch": "...", "validated": true}
  ],
  "docker_image_resolved": "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0",
  "verification_result": "results/rag_ollama_llamaindex_20260322T200105Z.json",
  "timestamp": "2026-03-22T20:07:20.511678",
  "agent": "claude_code_fix"
}
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `pass` | All cells ran cleanly, or all fixable errors were validated |
| `fail` | Errors remain that could not be fixed |
| `partial` | Some errors fixed, some remain (used with `expected_result: partial` in manifest) |

---

## How the playbooks work

Each agent follows a strict playbook. It never deviates. The verify agent's is
`CLAUDE.md`; the fix agent's is `CLAUDE_FIX.md` (which reuses CLAUDE.md's SSH,
Docker, timeout, and cleanup patterns).

### Verify agent — `CLAUDE.md`

**Step 1 — Read and Plan.** Read every cell before touching the server, then
identify the execution pattern:

| Pattern | Condition | Execution |
|---------|-----------|-----------|
| A | GPU libraries imported (torch, vllm, etc.) | papermill inside Docker |
| B | Explicit server+client split (`docker run -d` in markdown) | server in Docker background, client on host Python |
| C | `%%bash` cell contains `docker run -d` | run that bash directly, poll health |
| D | No Docker in any cell | papermill on host Python |

Then call `resolve_docker_image.py` for the exact versioned tag — never `:latest`.

**Phase 1 — Baseline Run.** Apply only the 4 allowed pre-flight patches (via
`preflight_patch.py`); everything else is a reported bug:

1. `notebook_login()` → `login(token=os.environ["HF_TOKEN"])`
2. `input()` calls → static stub
3. Gradio `launch()` cells → skip
4. Audio playback cells → skip

Run the notebook exactly **once**. Infrastructure failures (SSH drop, disk full,
Docker daemon error) trigger a retry. Content failures do not.

**Phase 2 — Analysis.** For every failing cell, perform **two independent checks**:

1. **Structural** (`content_error`): does the cell block sequential execution? (foreground server, `input()`, GUI)
2. **Code quality** (`deprecated_api`, `version_incompatibility`, `missing_dependency`): is there a code-level bug independent of the structural issue?

Both checks are mandatory. Never collapse them into one entry. Each issue is
classified by `fixability` (see below). The verify agent then writes the result
and stops — **it does not fix**.

### Fix agent — `CLAUDE_FIX.md`

**Step 0 — Load context.** Read the verification result and filter issues to
those marked `auto_fixable`. If there are none, write a `fail` result carrying
all issues through unchanged and exit — no re-run.

**Step 1–2 — Plan and validate.** Plan every fix in `cell_index` order. For any
version pin, confirm the candidate actually imports inside the resolved Docker
image before committing to it.

**Step 3 — Apply and run.** Write the fixes to `<nb>_fixed.ipynb`, apply the 4
pre-flight patches, and re-run **once** with the same pattern and Docker image
the verifier used.

**Step 4 — Validate and report.** Mark each fix `validated: true` if its cell no
longer errors. Carry `needs_author` / `infra_blocked` issues through unchanged.
Retrieve the genuine `<nb>_fixed.ipynb` (never the `_patched` scaffolded one)
into `fixes/` so the reporter can build a clean author-facing PR.

### Fixability classification

The verify agent tags every issue so the fixer knows what to attempt:

| Fixability | Meaning |
|------------|---------|
| `auto_fixable` | Applied mechanically: version pin, import rename, flag update, clear API migration — **the fixer only touches these** |
| `needs_author` | Human judgment: placeholder content, architectural redesign, foreground-server restructuring |
| `infra_blocked` | Cannot be fixed in the notebook: missing system package, wrong GPU count, Docker image bug |

### Reporting to GitHub — `tools/report_github.py`

The reporter turns the latest result per notebook into GitHub artifacts (nothing
is auto-merged; `--dry-run` is the default):

- Content bug needing a human → a GitHub **issue** (diagnosis)
- Validated mechanical fix → a GitHub **pull request** (proposal, built from the clean `_fixed.ipynb`)
- Notebook that now passes → its open issue/PR is **closed**

Each artifact carries a hidden fingerprint marker so re-runs update in place
instead of opening duplicates.

---

## Project structure

```
notebook-runner-agent/
├── CLAUDE.md              # VERIFY agent playbook — diagnose & classify
├── CLAUDE_FIX.md         # FIX agent playbook — apply auto_fixable fixes & validate
├── run.py                 # Launcher: SCP → spawn verify/fix agents → parse → read result
├── manifest.yaml          # Optional: per-notebook server/skip/expectation config
├── .claude/
│   └── settings.json      # Permitted tools: Bash, Read, Write, Glob, Grep
├── .env.example           # Secret variable documentation
├── fixes/                 # Genuine-fix notebooks retrieved by the fixer (for PRs)
├── results/               # JSON result files (gitignored)
├── logs/                  # CI run logs (gitignored)
└── tools/
    ├── resolve_docker_image.py  # Queries Docker Hub for the versioned tag matching AMD hardware
    ├── preflight_patch.py       # Applies exactly the 4 allowed pre-flight patches, deterministically
    ├── write_result.py          # Writes structured JSON result to results/
    ├── status.py                # Aggregates latest result per notebook → STATUS.md dashboard
    ├── report_github.py         # Files issues / PRs / closes them from results
    ├── ci_run.py                # CI orchestrator: gap → failures → dashboard → commit
    └── ci_cron.sh               # flock-guarded weekly cron wrapper around ci_run.py
```

---

## Benchmark

Head-to-head against the previous 2,200-line custom-loop agent across 24 AMD ROCm notebooks:

| Metric | Custom agent | notebook_runner |
|--------|-------------|-----------------|
| Wins | 3 | 20 |
| Pass rate | 25% (6/24) | 48% (11/23) |
| Issues found | 23 | 68 |
| Validated fixes | 8 | 15 |
| `deprecated_api` found | 1 | 14 |
| `version_incompatibility` found | 3 | 14 |

The playbooks (`CLAUDE.md` + `CLAUDE_FIX.md`) are the valuable artifact. The
launcher is ~200 lines of plumbing.

---

## Modifying agent behaviour

Each agent's reasoning, phases, and rules live entirely in its playbook. To
change how the **verify** agent behaves — new execution pattern, different
timeout, additional pre-flight patch — edit `CLAUDE.md`. To change how the
**fix** agent behaves — pin-validation strategy, what it retrieves for PRs —
edit `CLAUDE_FIX.md`.

The launcher (`run.py`) handles infrastructure only: SCP, spawning each agent,
parsing output, and reading the result file. Keep reasoning out of `run.py`.

---

## Security

- The agents run with `--dangerously-skip-permissions` so they can operate headlessly. The permitted tools are restricted in `.claude/settings.json` to `Bash`, `Read`, `Write`, `Glob`, and `Grep`.
- Never commit `.env`. The file is gitignored.
- SSH keys for GPU server access should be managed outside this repository.
- `tools/report_github.py` defaults to `--dry-run`; it calls `gh` only with `--publish`. Nothing is ever auto-merged.
