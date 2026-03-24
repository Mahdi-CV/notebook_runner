# notebook_runner

An autonomous notebook regression testing agent for AMD ROCm GPU tutorials. Built on [Claude Code](https://claude.ai/claude-code).

> **Health auditor, not a CI runner.** It surfaces bugs so tutorial authors can fix them — it does not silently patch content errors to make numbers look green.

---

## What it does

For each notebook, the agent:

1. Copies the `.ipynb` to the remote AMD GPU server via SCP
2. Reads every cell and identifies the execution pattern (Docker / server+client / host Python)
3. Resolves the latest Docker image tag for the target hardware via Docker Hub API
4. Runs the notebook end-to-end via papermill inside the correct Docker container
5. Collects cell-level errors from the output notebook
6. Analyses each failure on two independent axes: structural blockers and code quality issues
7. Applies validated fixes to a `_patched.ipynb` and re-runs
8. Writes a structured JSON result to `results/`
9. Cleans up all remote files and containers

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | |
| `claude` CLI | Install via `npm install -g @anthropic-ai/claude-code` |
| `pyyaml` | `pip install pyyaml` |
| `python-dotenv` | `pip install python-dotenv` |
| SSH access to an AMD GPU server | Key-based auth recommended |
| HuggingFace token | For gated model downloads |

---

## Setup

```bash
git clone <this-repo>
cd notebook_runner

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
# Single notebook — fully autonomous
python3 run.py path/to/notebook.ipynb

# All notebooks in a directory
python3 run.py --dir /path/to/notebooks/

# Filter to one category
python3 run.py --dir /path/to/notebooks/ --category inference

# Override server credentials for this run
python3 run.py path/to/notebook.ipynb --host my-server.example.com --user amd

# Interactive mode — opens a live Claude session with full runtime context pre-loaded
python3 run.py path/to/notebook.ipynb --interactive
```

**Categories:** `inference` | `fine_tune` | `pretrain` | `gpu_dev_optimize`

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

Results are written to `results/` as `{notebook_stem}_{timestamp_UTC}.json`:

```json
{
  "notebook": "/path/to/rag_ollama_llamaindex.ipynb",
  "status": "pass",
  "summary": "The notebook builds a RAG pipeline using LlamaIndex and Ollama. Phase 1 failed at cell 6 (systemctl hang) and cell 19 (flatbuffers PEP 440 error). Both were fixed and validated in Phase 3. All 47 cells ran to completion.",
  "issues": [
    {
      "cell_index": 6,
      "error_type": "content_error",
      "description": "Cell uses sudo systemctl start ollama — hangs when Ollama is already running on port 11434.",
      "proposed_fix": "Replace with !ollama list to verify Ollama is running without relying on systemd."
    },
    {
      "cell_index": 19,
      "error_type": "version_incompatibility",
      "description": "pip install chromadb fails: system flatbuffers has a non-PEP 440 version string, which pip 24.1+ rejects.",
      "proposed_fix": "Add --ignore-installed flatbuffers to the pip install command."
    }
  ],
  "fixes": [
    {"description": "Replaced systemctl cell with !ollama list", "validated": true},
    {"description": "Added --ignore-installed flatbuffers to pip install chromadb", "validated": true}
  ],
  "timestamp": "2026-03-22T20:07:20.511678",
  "agent": "claude_code"
}
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `pass` | All cells ran cleanly, or all fixable errors were validated |
| `fail` | Errors remain that could not be fixed |
| `partial` | Some errors fixed, some remain (used with `expected_result: partial` in manifest) |

---

## How the playbook works

The agent follows a strict 3-phase playbook defined in `CLAUDE.md`. It never deviates.

### Phase 0 — Read and Plan

- Read every cell (code and markdown) with the `Read` tool before touching the server
- Identify the execution pattern:

| Pattern | Condition | Execution |
|---------|-----------|-----------|
| A | GPU libraries imported (torch, vllm, etc.) | papermill inside Docker |
| B | Explicit server+client split (`docker run -d` in markdown) | server in Docker background, client on host Python |
| C | `%%bash` cell contains `docker run -d` | run that bash directly, poll health |
| D | No Docker in any cell | papermill on host Python |

- Call `resolve_docker_image.py` to get the latest tag for the target hardware

### Phase 1 — Baseline Run

Apply only the 4 allowed pre-flight patches (everything else is a reported bug):

1. `notebook_login()` → `login(token=os.environ["HF_TOKEN"])`
2. `input()` calls → static stub
3. Gradio `launch()` cells → skip
4. Audio playback cells → skip

Run the notebook exactly **once**. Infrastructure failures (SSH drop, disk full, Docker daemon error) trigger a retry. Content failures do not.

### Phase 2 — Analysis

For every failing cell, perform **two independent checks**:

1. **Structural** (`content_error`): does the cell block sequential execution? (foreground server, `input()`, GUI)
2. **Code quality** (`deprecated_api`, `version_incompatibility`, `missing_dependency`): is there a code-level bug independent of the structural issue?

Both checks are mandatory. Never collapse them into one entry.

### Phase 3 — Fix and Validate

- Patch to `_patched.ipynb` on the remote server
- Re-run Docker with the patched notebook
- Mark each fix `validated: true` if the error is gone, `validated: false` if still failing
- One fix cycle per run — no recursion

---

## Project structure

```
notebook_runner/
├── CLAUDE.md              # Agent playbook — the source of truth for agent behaviour
├── run.py                 # Launcher: SCP → spawn claude → parse stream-json → read result
├── .claude/
│   └── settings.json      # Permitted tools: Bash, Read, Write, Glob, Grep
├── .env.example           # Secret variable documentation
├── results/               # JSON result files (gitignored)
└── manifest.yaml          # Optional: per-notebook config (not committed if absent)
```

Shared tools live one level up in `../../tools/` and are referenced by absolute path at runtime:

| Tool | Purpose |
|------|---------|
| `tools/resolve_docker_image.py` | Queries Docker Hub for the latest tag matching the AMD hardware |
| `tools/write_result.py` | Writes the structured JSON result to `results/` |

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

The playbook (CLAUDE.md) is the valuable artifact. The launcher is ~200 lines of plumbing.

---

## Modifying agent behaviour

The agent's reasoning, phases, and rules all live in `CLAUDE.md`. To change how the agent behaves — new execution pattern, different timeout, additional pre-flight patch — edit `CLAUDE.md`.

The launcher (`run.py`) handles infrastructure only: SCP, spawning Claude, parsing output, and reading the result file. Keep reasoning out of `run.py`.

---

## Security

- The agent runs with `--dangerously-skip-permissions` so it can operate headlessly. The permitted tools are restricted in `.claude/settings.json` to `Bash`, `Read`, `Write`, `Glob`, and `Grep`.
- Never commit `.env`. The file is gitignored.
- SSH keys for GPU server access should be managed outside this repository.
