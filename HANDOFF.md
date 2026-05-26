# Handoff: notebook-runner-agent

**Owner passing off:** Mahdi Ghodsi (AI Solution Architect, AMD)
**Taking over:** [intern name]
**Repo:** https://github.com/Mahdi-CV/notebook-runner-agent

---

## What this is

An autonomous agent that regression-tests AMD ROCm Jupyter notebooks on remote GPU servers. You point it at a notebook, it copies it to a GPU box, runs every cell via papermill inside the correct Docker container, diagnoses failures, attempts fixes, and writes a structured JSON report — all without human input.

The philosophy: **health auditor, not a CI runner**. It surfaces bugs so tutorial authors can fix them. It does not silently patch content errors to make numbers look green.

---

## Repo layout

```
notebook-runner-agent/
├── CLAUDE.md              # The agent playbook — read this first
├── run.py                 # Launcher: SCP + spawn Claude + parse output + collect result
├── tools/
│   ├── resolve_docker_image.py  # Queries Docker Hub for latest AMD-hardware-specific tag
│   └── write_result.py          # Writes structured JSON to results/
├── .claude/settings.json  # Restricts Claude to: Bash, Read, Write, Glob, Grep
├── .env.example           # Documents required secrets
├── .gitignore
└── results/               # JSON result files land here (gitignored)
```

**The two files that matter most are `CLAUDE.md` and `run.py`.** Everything else is supporting infrastructure.

---

## How to set it up

```bash
git clone https://github.com/Mahdi-CV/notebook-runner-agent.git
cd notebook-runner-agent

pip install python-dotenv pyyaml

cp .env.example .env
# Open .env and fill in the three values:
#   GPU_HOST — hostname or IP of the AMD GPU server
#   GPU_USER — SSH username (must have Docker access)
#   HF_TOKEN — HuggingFace token for gated model downloads
```

You need SSH key-based access to the GPU server already configured. If `ssh GPU_USER@GPU_HOST` works without a password prompt, you are set.

You also need the `claude` CLI:
```bash
npm install -g @anthropic-ai/claude-code
claude login   # authenticates with your Anthropic account
```

---

## How to run it

```bash
# Single notebook — fully autonomous, no interaction needed
python3 run.py path/to/notebook.ipynb

# All notebooks in a directory
python3 run.py --dir /path/to/notebooks/

# Filter to one category
python3 run.py --dir /path/to/notebooks/ --category inference

# Override server credentials for a single run
python3 run.py path/to/notebook.ipynb --host gpu1.example.com --user amd

# Interactive mode — opens a live Claude session with runtime context pre-loaded
# Useful for debugging a specific notebook
python3 run.py path/to/notebook.ipynb --interactive
```

**Categories:** `inference` | `fine_tune` | `pretrain` | `gpu_dev_optimize`

---

## What happens when you run it

1. `run.py` copies the notebook to the GPU server via SCP into `/home/$USER/tutorial_agent_runs/<notebook_stem>/`
2. It builds a task prompt and a runtime context block (SSH commands, paths, hardware info)
3. It spawns `claude -p <prompt>` in headless mode with `--dangerously-skip-permissions`
4. Claude reads `CLAUDE.md` (the playbook) and executes the 3-phase protocol:
   - **Phase 0:** Read every cell, identify execution pattern, resolve Docker image tag
   - **Phase 1:** Apply the 4 allowed pre-flight patches, run notebook once via papermill
   - **Phase 2:** For every failing cell, perform two independent checks (structural + code quality)
   - **Phase 3:** Patch, re-run, validate fixes
5. Claude calls `tools/write_result.py` to write a JSON report to `results/`
6. `run.py` reads that JSON and prints a summary

---

## Reading the output

Result files land in `results/` as `{notebook_stem}_{timestamp_UTC}.json`:

```json
{
  "notebook": "/path/to/notebook.ipynb",
  "status": "pass",
  "summary": "The notebook builds a RAG pipeline ...",
  "issues": [
    {
      "cell_index": 6,
      "error_type": "content_error",
      "description": "Cell uses sudo systemctl start ollama — hangs when Ollama is already running.",
      "proposed_fix": "Replace with !ollama list to verify Ollama is running."
    }
  ],
  "fixes": [
    {"description": "Replaced systemctl cell with !ollama list", "validated": true}
  ],
  "timestamp": "2026-03-22T20:07:20.511678",
  "agent": "claude_code"
}
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `pass` | All cells ran, or all fixable errors were validated |
| `fail` | Errors remain that could not be fixed |
| `partial` | Some fixed, some remain (used with `expected_result: partial` in manifest) |

**Error types:**

| Type | Meaning |
|------|---------|
| `content_error` | Cell blocks sequential execution (foreground server, `input()`, GUI) |
| `deprecated_api` | Uses an API the library has deprecated or removed |
| `version_incompatibility` | Package version mismatch or invalid version string |
| `missing_dependency` | Package not installed in the container |

---

## Optional: manifest.yaml

Create `manifest.yaml` next to `run.py` to control per-notebook behavior. This file is gitignored (it's a local override, not part of the repo).

```yaml
servers:
  mi300x-box:
    host: gpu1.example.com
    user: amd
    hardware: mi300x

notebooks:
  inference/my_notebook.ipynb:
    server: mi300x-box
    expected_result: partial   # partial counts as passing
    skip: false
    notes: "Known issue: cell 3 uses deprecated vllm flag --max-num-seqs"

  fine_tune/slow_notebook.ipynb:
    skip: true
    skip_reason: "Requires H100, not available on ROCm server"
```

---

## How to change agent behavior

**All agent reasoning lives in `CLAUDE.md`.** The launcher (`run.py`) is plumbing only: SCP, spawning Claude, parsing the stream-json output, and reading the result file.

To change what the agent does — add a new execution pattern, adjust a timeout, allow an additional pre-flight patch, change the error classification rules — **edit `CLAUDE.md`**. Do not put reasoning logic in `run.py`.

To change the output schema or add a new field — edit `tools/write_result.py`.

To add support for a new Docker registry or hardware target — edit `tools/resolve_docker_image.py`.

---

## Debugging a run

**Agent didn't write a result file:**
Run with `--interactive` to open a live session. The runtime context is still pre-loaded, so you can walk through the playbook manually or ask Claude what went wrong.

**SSH / SCP failures:**
Check that `ssh GPU_USER@GPU_HOST` works from your machine without a password prompt. The agent assumes key-based auth.

**Docker image not found:**
Run the resolver manually:
```bash
python3 tools/resolve_docker_image.py rocm/pytorch mi300x
python3 tools/resolve_docker_image.py vllm/vllm-openai-rocm mi300x
```
If it returns `ERROR`, the Docker Hub API couldn't find a matching tag. Check the hardware name spelling (`mi300x`, `mi308x`, `mi355x`, `mi350x`).

**Agent hit max turns (200):**
The notebook is probably very long or has multiple infrastructure retries. You can bump `--max-turns` in `run.py` line 373, or break the notebook into sections.

**Cost is unexpectedly high:**
Check `num_turns` in the result. If it's near 200, the agent was working hard. A typical run is 30-80 turns. Very high turn counts usually mean the notebook has a hanging cell or the agent is retrying an infrastructure issue.

---

## What to work on next

The agent is functional and has been benchmarked against 30+ AMD ROCm notebooks. Areas to extend:

- **Multi-server routing** — the manifest supports server routing per-notebook but there is no parallelism yet. Notebooks currently run sequentially.
- **Result aggregation** — there is no dashboard or summary report across many runs. Results are individual JSON files.
- **Notification** — no Slack/email/JIRA integration when a notebook fails.
- **Scheduled runs** — no cron or CI trigger yet. Currently run manually.

---

## Who to ask

Mahdi Ghodsi — for architecture questions or if something is fundamentally broken.
For everything else, the answer is usually in `CLAUDE.md`.
