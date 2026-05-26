# Handoff: notebook-runner-agent

**Owner passing off:** Mahdi Ghodsi (AI Solution Architect, AMD)
**Taking over:** [intern name]
**Repo:** https://github.com/Mahdi-CV/notebook_runner

---

## What this is

An autonomous agent that regression-tests AMD ROCm Jupyter notebooks on remote GPU servers. You point it at a notebook, it copies it to a GPU box, runs every cell via papermill inside the correct Docker container, diagnoses failures, attempts fixes, and writes a structured JSON report — all without human input.

The philosophy: **health auditor, not a CI runner**. It surfaces bugs so tutorial authors can fix them. It does not silently patch content errors to make numbers look green.

---

## How we got here (read this before anything else)

This project went through two generations. Understanding why matters for understanding what you're working on.

### Generation 1: custom ReAct agent

The first version was a hand-rolled ReAct loop — the standard AI agent pattern: send prompt → model responds with a tool call → execute the tool → append result → repeat until done. ~2,200 lines across 7 modules:

```
run.py        (348 LOC)  — CLI and manifest loading
agent.py      (437 LOC)  — the ReAct loop itself
tools.py      (410 LOC)  — 5 custom tools with JSON schemas
context.py    (179 LOC)  — message history, context window trimming
llm.py         (83 LOC)  — HTTP client for AMD's LLM gateway
events.py      (71 LOC)  — event bus for the dashboard
dashboard.py  (104 LOC)  — Flask/SSE real-time browser UI
SKILL.md      (267 LOC)  — the actual playbook (what the agent should do)
```

It worked. Benchmarked against 24 AMD ROCm notebooks: **25% pass rate, 23 issues found, 8 validated fixes**.

The problem: roughly 1,300 of those 2,200 lines — the loop, the tool schemas, the XML parser, the context trimmer, the HTTP client — solve problems that are not specific to our notebooks at all. That's infrastructure overhead.

### Generation 2: this repo

The second version replaces the entire custom loop with Claude Code's headless mode. `run.py` builds a prompt, calls `claude -p` as a subprocess, and reads the result file. Claude Code handles the loop, tool execution, context management, and streaming internally.

Same benchmark, 24 notebooks: **48% pass rate, 68 issues found, 15 validated fixes**.

```
CLAUDE.md    (386 LOC)  — the full playbook (equivalent to the original SKILL.md)
run.py       (533 LOC)  — subprocess launcher + stream-json parser + result collector
tools/resolve_docker_image.py  — Docker Hub API (called by the agent via Bash)
tools/write_result.py          — JSON result writer (called by the agent via Bash)
```

No custom loop. No tool schemas. No XML parser. No context trimmer. No HTTP client.

The jump in results comes from two things: a stronger underlying model (Claude vs the previous gateway model), and a playbook that enforces two independent checks per failing cell instead of stopping at the first issue found.

### What the benchmark proved

The custom infrastructure was not the bottleneck. The playbook and the model were. That is the central lesson, and it's why the playbook (`CLAUDE.md`) is the most important file in this repo — not the launcher.

The full benchmark writeup, including case studies and a head-to-head table, lives in the original `team-agents` repo under `docs/lessons-learned-agents-comparison.md`.

---

## Architecture

### The mental model

Every agentic project of this type has the same shape:

```
[List of items to process]
    │
    ▼  for each item:
    │
    ├─ deterministic steps  →  scripts (tools/)
    └─ variable-depth reasoning  →  Claude Code agent (CLAUDE.md)
            │
            ▼
    [Structured output per item]  →  results/*.json
```

The separator between deterministic and reasoning is the most important design decision. Get it wrong and you either over-build the scripts (too rigid) or under-build them (the agent inventing schemas on every run).

**In this project:**

| Step | Type | Where it lives |
|------|------|---------------|
| Copy notebook to GPU server | Deterministic | `run.py` (SCP) |
| Resolve latest Docker image tag | Deterministic, external API | `tools/resolve_docker_image.py` |
| Read notebook cells, identify execution pattern | Reasoning | `CLAUDE.md` Phase 0 |
| Run notebook via papermill | Deterministic shell | `CLAUDE.md` Phase 1 (agent uses Bash tool) |
| Diagnose cell failures, classify error types | Reasoning | `CLAUDE.md` Phase 2 |
| Apply patches, validate fixes | Reasoning + shell | `CLAUDE.md` Phase 3 |
| Write JSON result | Deterministic, fixed schema | `tools/write_result.py` |

### The files and what they do

**`CLAUDE.md` — the playbook**

This is the agent's system prompt. Claude reads it automatically when the session starts. It defines:
- What the agent is and what it must never do
- What runtime variables are available (SSH commands, paths, hardware)
- The exact 3-phase workflow, step by step, with explicit commands
- The 4 allowed pre-flight patches (everything else is a reported bug, not a fix)
- Hard rules that override everything else

If you want to change how the agent behaves, you edit this file. Not `run.py`. Not a Python config. This file.

**`run.py` — the launcher**

Pure plumbing. It does four things:
1. Reads `.env` and `manifest.yaml` (if present) to get server config for this notebook
2. Copies the notebook to the GPU server via SCP
3. Builds the task prompt + runtime context block and spawns `claude -p` as a subprocess
4. Parses the `stream-json` output to print live progress, then reads the result JSON the agent wrote

There is no reasoning logic in `run.py`. If you find yourself adding if/else branches about what the agent should do, that logic belongs in `CLAUDE.md`.

**`tools/resolve_docker_image.py` — Docker Hub resolver**

Queries the Docker Hub API for the latest image tag matching the target AMD hardware. Called by the agent via the Bash tool:

```bash
python3 tools/resolve_docker_image.py rocm/pytorch mi300x
# → rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch
```

Hardware mappings: `mi300x`/`mi308x` → `mi30x` tag suffix; `mi355x`/`mi350x` → `mi35x`. The agent always resolves the tag at runtime — never trusts what the notebook itself says, because notebooks go stale.

**`tools/write_result.py` — result writer**

Writes the structured JSON result file to `results/`. The agent always calls this at the end, even on failure. This is what makes results machine-readable and aggregatable downstream.

The schema is fixed and validated inside the script. The agent cannot write a malformed result even if its reasoning goes wrong.

**`.claude/settings.json` — permissions**

Restricts the agent to exactly five tools: `Bash`, `Read`, `Write`, `Glob`, `Grep`. The agent cannot browse the web, call external APIs directly, or do anything outside those five primitives. This is the safety boundary for headless autonomous operation.

The `--dangerously-skip-permissions` flag used in `run.py` skips interactive confirmation prompts — it does NOT bypass these deny rules. Deny rules always apply.

### How the agent executes

When `run.py` spawns `claude -p`, the following happens entirely inside Claude Code:

**Phase 0 — Read and Plan**
- Reads every cell of the notebook locally with the `Read` tool
- Identifies the execution pattern based on cell content (not just markdown headers):
  - Pattern A: GPU libraries imported (torch, vllm, etc.) → papermill inside Docker
  - Pattern B: explicit server+client split (`docker run -d` in markdown) → server in Docker background, client cells on host Python
  - Pattern C: `%%bash` cell contains `docker run -d` → run that bash directly, poll health
  - Pattern D: no Docker → papermill on host Python
- Resolves the correct Docker image tag via `resolve_docker_image.py`

**Phase 1 — Baseline Run**
- Applies exactly 4 allowed pre-flight patches (no others):
  1. `notebook_login()` → `login(token=os.environ["HF_TOKEN"])`
  2. `input()` calls → static stub value
  3. Gradio `launch()` cells → skip
  4. Audio playback cells → skip
- Runs the notebook exactly once via papermill inside Docker
- If it fails for a content reason, that failure is the result — no retry, no workaround

**Phase 2 — Analysis**
For every failing cell, performs two independent checks — both are mandatory:
- Check 1 (structural): does the cell block sequential execution? (foreground server, `input()`, GUI) → `content_error`
- Check 2 (code quality): is there a deprecated API, version mismatch, or missing dependency, independent of the structural issue? → `deprecated_api`, `version_incompatibility`, `missing_dependency`

The two-check discipline is why this agent finds 3× more issues than the previous one. A cell that hangs because of a foreground server may also be using a deprecated CLI flag. Both get reported.

**Phase 3 — Fix and Validate**
- Writes a patched notebook to `_patched.ipynb` on the remote server
- Re-runs Docker with the patched notebook
- Marks each fix `validated: true` or `validated: false`
- One fix cycle per run — no recursion

---

## The manifest and why it exists

`manifest.yaml` is the per-item configuration file. It answers: what varies across notebooks or environments, that shouldn't be hardcoded?

The agent's reasoning rules go in `CLAUDE.md`. But things like "this specific notebook needs a different server," "this notebook is known-broken upstream so partial is acceptable," "skip this one because it requires hardware we don't have" — those are configuration, not reasoning. They go in the manifest.

```yaml
servers:
  mi300x-primary:
    host: gpu1.example.com
    user: amd
    hardware: mi300x
  mi308x-secondary:
    host: gpu2.example.com
    user: amd
    hardware: mi308x

notebooks:
  inference/vllm_v1_DSR1.ipynb:
    server: mi300x-primary
    expected_result: partial     # partial counts as passing for this notebook
    notes: "Known issue: huggingface_hub 1.x breaks CLI entrypoint"

  fine_tune/huge_model.ipynb:
    skip: true
    skip_reason: "Requires 8x GPU, not available on current test server"

  inference/my_notebook.ipynb:
    server: mi308x-secondary
    docker_overrides:
      "rocm/pytorch:rocm6\\.3.*": "rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch"
```

`run.py` reads the manifest before doing anything. Notebooks marked `skip: true` never touch the LLM. Per-notebook server config overrides `.env`. `expected_result: partial` tells the launcher to treat partial as pass in the final summary.

The manifest is gitignored because it's environment-specific — different people have different servers. When you set up your environment, create your own.

---

## Repo layout

```
notebook-runner-agent/
├── CLAUDE.md              # The agent playbook — the source of truth for agent behavior
├── run.py                 # Launcher: SCP → spawn claude → parse stream-json → read result
├── tools/
│   ├── resolve_docker_image.py  # Docker Hub tag resolution for AMD hardware
│   └── write_result.py          # Structured JSON result writer
├── .claude/settings.json  # Permitted tools: Bash, Read, Write, Glob, Grep
├── .env.example           # Documents required secrets
├── .gitignore
└── results/               # JSON result files land here (gitignored)
```

---

## How to set it up

```bash
git clone https://github.com/Mahdi-CV/notebook_runner.git
cd notebook_runner

pip install python-dotenv pyyaml

cp .env.example .env
# Fill in:
#   GPU_HOST — hostname or IP of the AMD GPU server
#   GPU_USER — SSH username (must have Docker access)
#   HF_TOKEN — HuggingFace token for gated model downloads
```

You need SSH key-based access to the GPU server already configured. Test with:
```bash
ssh $GPU_USER@$GPU_HOST 'echo ok'
```
If that requires a password, set up key-based auth first.

You also need the `claude` CLI. As an AMD employee you already have this configured through the AMD company offering — the CLI routes through AMD's internal LLM gateway using environment variables set in your shell profile. Verify:

```bash
claude --version
claude -p "say hello"   # should respond without any auth error
```

If either fails, the AMD Claude env vars are not active in your current shell. Check your `.bashrc` / `.zshrc`, or ask Mahdi.

---

## How to run it

```bash
# Single notebook — fully autonomous
python3 run.py path/to/notebook.ipynb

# All notebooks in a directory
python3 run.py --dir /path/to/notebooks/

# Filter to one category
python3 run.py --dir /path/to/notebooks/ --category inference

# Override server for a single run
python3 run.py path/to/notebook.ipynb --host gpu1.example.com --user amd

# Interactive mode — opens a live Claude session with runtime context pre-loaded
# Good for debugging a specific notebook
python3 run.py path/to/notebook.ipynb --interactive
```

**Categories:** `inference` | `fine_tune` | `pretrain` | `gpu_dev_optimize`

---

## Reading the output

Result files land in `results/` as `{notebook_stem}_{timestamp_UTC}.json`:

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
      "description": "pip install chromadb blocked by system flatbuffers with a non-PEP 440 version string, rejected by pip 24.1+.",
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

| Status | Meaning |
|--------|---------|
| `pass` | All cells ran, or all fixable errors were validated |
| `fail` | Errors remain that could not be fixed |
| `partial` | Some fixed, some remain (used with `expected_result: partial` in manifest) |

| Error type | Meaning |
|------------|---------|
| `content_error` | Cell blocks sequential execution (foreground server, `input()`, GUI, audio) |
| `deprecated_api` | Uses an API the library has deprecated or removed |
| `version_incompatibility` | Package version mismatch or non-standard version string |
| `missing_dependency` | Package not installed in the container |

---

## How to change agent behavior

Edit `CLAUDE.md`. That is the complete answer.

The launcher (`run.py`) is plumbing. The tools are deterministic scripts with fixed schemas. All reasoning logic — what execution patterns to look for, what patches are allowed, what counts as a content error vs a code quality issue, how many fix cycles to attempt — lives in `CLAUDE.md`.

Specific cases:
- Add a new execution pattern → add a Pattern E section to `CLAUDE.md`
- Allow an additional pre-flight patch → add it to the Hard Rules section in `CLAUDE.md`
- Change timeout values → update the timeout table in `CLAUDE.md`
- Add a new error type → update the Phase 2 section and the `--status` choices in `tools/write_result.py`
- Support a new Docker registry → add it to `tools/resolve_docker_image.py`
- Support a new hardware target → add it to the `_HW_TAG` dict in `tools/resolve_docker_image.py`

---

## Debugging a run

**Agent didn't write a result file:**
Run with `--interactive`. The runtime context is pre-loaded, so you can step through the playbook manually or ask Claude what it would do next.

**SSH / SCP failures:**
Confirm `ssh $GPU_USER@$GPU_HOST 'echo ok'` works without a password prompt. The launcher retries SCP 3 times — if all 3 fail, check connectivity and key-based auth.

**Docker image not found:**
```bash
python3 tools/resolve_docker_image.py rocm/pytorch mi300x
python3 tools/resolve_docker_image.py vllm/vllm-openai-rocm mi300x
```
If this returns `ERROR`, the hardware name spelling is wrong or Docker Hub is unreachable. Valid hardware names: `mi300x`, `mi308x`, `mi355x`, `mi350x`.

**Agent hit max turns (200):**
Usually means a hanging cell (foreground server, very long training run) or the agent retrying an infrastructure issue. Check the partial output in the terminal. You can increase `--max-turns` in `run.py` line 373 or use `--interactive` to investigate.

**Cost is higher than expected:**
A typical run is 30–80 turns. Near-200 turns means the agent worked hard or got stuck. Check `num_turns` in the result JSON. Fine-tuning notebooks legitimately need more turns due to longer waits.

---

## What to work on next

The agent is validated against 30+ AMD ROCm notebooks. The areas most ready for extension:

**Parallelism** — the architecture is already designed for it. Each agent run is one subprocess processing one notebook and writing to its own result file. Running N notebooks simultaneously is a `ThreadPoolExecutor` wrapping the existing `run_notebook()` call. The only constraint is Docker container name conflicts on a single GPU server — use the manifest to route different notebooks to different servers.

**Result aggregation** — results are individual JSON files. There is no dashboard or batch summary report. A simple script that reads all JSONs, groups by status, and outputs a markdown table would be immediately useful.

**Scheduled runs** — nothing triggers this automatically. A cron job or CI pipeline calling `run.py --dir` on a schedule is the natural next step. The result files are already structured for this.

**Notification** — no Slack, email, or JIRA integration when a notebook fails. The result JSON has everything needed to file a ticket or send a message.

**memory.md** — the agent currently has no persistent memory across runs. Adding a `memory.md` would let it accumulate known infrastructure facts (Docker images that fail on this server, ROCm driver versions, known-broken upstream packages) and avoid re-diagnosing the same infrastructure issues on every run.

---

## Who to ask

Mahdi Ghodsi — for architecture questions, the history of decisions, or if something is fundamentally broken.

The full design context — the pattern, the component decisions, the benchmark data, the field guide for building new agents — is in the `team-agents` repo under `docs/`. Start with `agentic-projects-field-guide.md`.

For everything else, the answer is in `CLAUDE.md`.
