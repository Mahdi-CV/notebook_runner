# Handoff — Autonomous Regression Testing for AMD ROCm AI Tutorials

**Author:** Kalyan Archakam (AI Software Intern)
**Repo:** `notebook_runner`
**Active branch at handoff:** `feature/two-agent-verify-fix`
**Status:** Working end-to-end. 33 of 40 hub tutorials are testable and run on every pass; 18/33 pass end-to-end today.

This document is the "start here" orientation for whoever picks this up next. It explains what the
system is, how it was built and why, how to run it, and exactly where to continue. For deep reference
on individual pieces, this doc points you at the authoritative files (`README.md`, `CLAUDE.md`,
`CLAUDE_FIX.md`) rather than duplicating them.

---

## 1. TL;DR — the 60-second version

The AMD AI Tutorial Hub has ~40 Jupyter notebooks. They are not linear scripts; most need
human-style, multi-step setup (Docker, vLLM servers, model downloads, tool calls). Testing them by
hand costs ~1–4 hours per notebook, every release. Nothing was automated.

This repo is an **agent that regression-tests those notebooks autonomously** on AMD Instinct GPU
servers over SSH. It runs each notebook as published, decides pass/fail, and produces a structured,
per-cell report of what broke and whether it is auto-fixable.

It is a **two-agent pipeline**:

1. **Verify agent** (playbook: `CLAUDE.md`) — runs the notebook once, diagnoses failures, classifies
   each issue. Never fixes.
2. **Fix agent** (playbook: `CLAUDE_FIX.md`) — applies only the mechanically-fixable issues, re-runs
   once to validate, reports. Never re-diagnoses.

A launcher (`run.py`) spawns each agent as a headless Claude Code process. Results are JSON in
`results/`, rolled up into `STATUS.md`, and optionally reported to GitHub as issues (content bugs) and
PRs (validated fixes).

**Guiding principle: this is a health auditor, not a CI runner.** It surfaces content bugs to tutorial
authors instead of silently patching notebooks green and shipping hidden defects to users.

---

## 2. How we got here (design history — read this before changing the architecture)

The two-agent design is not arbitrary. It's the result of testing three approaches against the same 5
notebooks, then scaling:

| Approach | Result | Problem |
|---|---|---|
| **Manual testing** | 2/5 passed end-to-end | ~1–4 hrs/notebook, not scalable |
| **Naive single agent** (Claude Code, no structure) | 4/5 reported pass | 2 were **false positives** — it green-washed |
| **Prior single-agent** (baseline it was forked from) | 2/5 passed (+1 partial) | Flags problems but **misdiagnoses cause** and keeps applying wrong fixes |

Neither manual nor a naive agent was reliable *or* scalable.

The single-agent baseline was then run across all 33 testable notebooks → **48.5% baseline pass rate
(16/33)**. Improvements applied at that stage: two independent checks per failing cell (structural +
code-quality), better timeouts, a hardened docker-image resolver, and a single fix cycle (no
recursion).

**The failure mode that forced two agents:** one agent doing verify *and* fix would lose context at
the fixing stage, hallucinate, and fix the wrong issue. Refining the playbook helped but not enough.
The fix was **separation of concerns**:

- Verify agent diagnoses only → can't green-wash.
- Fix agent repairs only, and only issues the verifier already classified as `auto_fixable` → can't
  invent problems.

Each agent has a focused, auditable playbook. This is why behavior lives in `CLAUDE.md` /
`CLAUDE_FIX.md`, **not** in `run.py`. Keep it that way.

---

## 3. Architecture

```
                         run.py  (launcher / CLI)
                            │  reads manifest.yaml (server · hardware · skips · expected result)
                            │  runs tools/env_check.py first (fail fast)
                            ▼
        ┌──────────────────────────────┐        ┌──────────────────────────────┐
        │  VERIFY AGENT (CLAUDE.md)     │        │  FIX AGENT (CLAUDE_FIX.md)    │
        │  "The Diagnostician"          │  ───▶  │  "The Surgeon"                │
        │  • read all cells, ID pattern │  only  │  • load verification result   │
        │  • apply 4 pre-flight patches │  if    │  • filter to auto_fixable     │
        │  • run notebook ONCE          │  fail  │  • validate version pins in   │
        │  • 2 checks: structural +     │        │    the real Docker image      │
        │    code quality               │        │  • apply + run patched nb once│
        │  • classify fixability        │        │  • validate + report          │
        └───────────────┬──────────────┘        └───────────────┬──────────────┘
                        │ result JSON                            │ result JSON
                        ▼                                        ▼
                     results/*.json  ──▶  tools/status.py ──▶  STATUS.md (dashboard)
                                     └──▶  tools/report_github.py ──▶ GitHub issues + fix PRs

  Shared tools (tools/):  resolve_docker_image.py · env_check.py · preflight_patch.py · write_result.py
  CI:  tools/ci_run.py (4-phase pass)  ←  tools/ci_cron.sh (weekly, flock-guarded)
```

### Execution phases inside the verify agent
- **Phase 0 (env_check):** cheap pre-flight — HF token access to gated models, GPU count vs required,
  disk space. Blocks the run early on unwinnable cases.
- **Phase 1 (baseline run):** classify the notebook's execution pattern (A: papermill in Docker; B:
  server+client split; C: notebook launches Docker itself; D: host Python), apply the **only 4
  allowed pre-flight patches**, then run **once**. No retries on content failures.
- **Phase 2 (analysis):** for every failed/timed-out cell, do **two independent checks** —
  structural (does it block sequential execution?) and code-quality (deprecated API, version
  mismatch, missing dep?). Report all findings, not just the first.

### Fixability classification (routes the fixer)
- `auto_fixable` → fixer attempts it (version pin, import rename, flag update, clear API migration).
- `needs_author` → raised as an issue for the human author (placeholder content, redesign, foreground
  server, unknown replacement API).
- `infra_blocked` → reported, not fixable in the notebook (missing system package, wrong GPU count,
  base-image bug).

The four allowed pre-flight patches (and nothing else) are: `notebook_login()` → `login(token=...)`,
`input()` → static value/skip, Gradio launch → skip, audio playback → skip. Everything else is a
content bug and gets reported. See `CLAUDE.md` "HARD RULES" for the exact list.

---

## 4. Repo layout

```
notebook_runner/
├── run.py                 # Launcher/CLI. Spawns claude headless. Modes: verify | fix | full.
├── CLAUDE.md              # VERIFY agent playbook (source of truth for verify behavior)
├── CLAUDE_FIX.md          # FIX agent playbook (source of truth for fix behavior)
├── README.md              # Current, authoritative usage reference
├── HANDOFF.md             # This file
├── STATUS.md              # Auto-generated dashboard (do not hand-edit)
├── manifest.yaml          # Per-notebook config: server, hardware, skip, expected_result, notes
├── .env                   # GPU_HOST, GPU_USER, HF_TOKEN  (gitignored; copy from .env.example)
├── tools/
│   ├── env_check.py           # Phase-0 pre-flight checks (token/GPU/disk)
│   ├── resolve_docker_image.py# Resolve latest repo:tag from Docker Hub for a hardware family
│   ├── preflight_patch.py     # Applies exactly the 4 allowed patches, prints JSON report
│   ├── write_result.py        # Writes results/<stem>_<ts>.json
│   ├── status.py              # Aggregates results → STATUS.md; also --gap / --failing / --json
│   ├── ci_run.py              # 4-phase CI pass (freshness → failures → dashboard → commit)
│   ├── ci_cron.sh             # Weekly cron wrapper (flock, venv, logging, --push)
│   └── report_github.py       # Reconciles results ↔ GitHub issues/PRs (idempotent, marker-based)
├── notebooks/             # Local copies of the hub notebooks under test (by category)
│   ├── inference/  fine_tune/  pretrain/  gpu_dev_optimize/
├── results/               # One JSON per (notebook, run). status.py reads the latest per stem.
├── fixes/                 # Genuine fixed notebooks (_fixed.ipynb) retrieved for PR diffs
├── logs/                  # Run logs, CI logs, github_preview/ (dry-run PR/issue bodies)
└── upstream_prs/          # Scratch/notes for upstream PR work
```

CI now runs two ways: (a) the original local cron job (`tools/ci_cron.sh`), and (b) GitHub Actions
on a **self-hosted runner that is itself the MI300X GPU box** (`.github/workflows/nightly.yml` here,
plus a PR-trigger workflow in `gpuaidev-internal`). See §9 for the deployed setup and secrets.

---

## 5. Setup

**Requirements**
- Python 3.10+ with `pyyaml` and `python-dotenv` (a `venv/` already exists in the repo).
- **Claude Code CLI** on PATH (`claude`). `run.py` spawns it headless with
  `--dangerously-skip-permissions --output-format stream-json --max-turns 200`. In production this
  must be an **AMD Gateway-certified Claude** (see §9).
- SSH access to an AMD Instinct GPU server (key-based, no password prompts).
- `gh` CLI authenticated (`gh auth login`) — only needed for GitHub reporting.

**Configure**
```bash
cp .env.example .env
# then fill in:
#   GPU_HOST=<ip-or-host>
#   GPU_USER=<ssh-user>
#   HF_TOKEN=<huggingface-token>   # for gated model downloads
source venv/bin/activate
```

The workspace on the server is always `/home/amd/tutorial_agent_runs/<stem>/`, regardless of SSH
user, because the playbooks reference that literal path. `run.py` scp's the notebook and
`preflight_patch.py` there before each run.

---

## 6. How to run it

```bash
# One notebook, full pipeline (verify, then fix if it failed):
python run.py notebooks/inference/foo.ipynb

# Verify only (diagnosis, no fixing):
python run.py notebooks/inference/foo.ipynb --mode verify

# Fix only, from a prior verification result:
python run.py notebooks/inference/foo.ipynb --mode fix --verification-result results/foo_<ts>.json

# A whole directory / one category:
python run.py --dir notebooks/ --category inference

# Only the work that matters:
python run.py --dir notebooks/ --gap       # never-tested or stale (>7d)
python run.py --dir notebooks/ --failing   # latest result is a hard fail

# Node capacity filter (skip notebooks needing more GPUs than the node has):
python run.py --dir notebooks/ --max-gpus 1

# Point at a specific server (overrides .env / manifest):
python run.py notebooks/inference/foo.ipynb --host <ip> --user root

# Interactive (opens a full Claude session with the runtime context loaded):
python run.py notebooks/inference/foo.ipynb -i
```

**Modes:** `verify` = diagnosis only · `fix` = apply fixes from a prior result · `full` = verify then
fix (default; fix only runs if verify didn't pass and there are `auto_fixable` issues).

### CI (automated passes)
```bash
python tools/ci_run.py            # full 4-phase pass, commit locally
python tools/ci_run.py --push     # + push to origin (used by cron)
python tools/ci_run.py --dry-run  # print the plan, run nothing
```
Phases: **A** freshness (gap set) → **B** current failures → **C** regenerate `STATUS.md` → **D**
commit (`--push` to push). The weekly cron wrapper `tools/ci_cron.sh` adds a `flock` guard (fine-tune
notebooks can run for hours, so overlapping passes must be prevented), venv resolution, and logging to
`logs/cron_<ts>.log`. Install it via `crontab -e`; on WSL that only fires while WSL is running, so use
Windows Task Scheduler (`wsl.exe bash -lc '.../tools/ci_cron.sh'`) if the box sleeps.

### Status & reporting
```bash
python tools/status.py                 # print dashboard;  --write updates STATUS.md
python tools/status.py --gap|--failing # work-order lists (paths, one per line)

python tools/report_github.py --mode audit                       # dry-run; previews in logs/github_preview/
python tools/report_github.py --mode audit --repo AMD-ROCm-Internal/gpuaidev-internal \
    --upstream-checkout <checkout-of-gpuaidev-internal> --publish  # create/update/close issues + PRs
```
The GitHub reporter is idempotent: every artifact carries a hidden marker
(`<!-- agent-managed: issue:STEM -->`), so re-runs update the existing issue/PR instead of duplicating.
Issues = content bugs for authors; PRs (branch `agent/fix/<stem>`) = validated fixes; both auto-close
when the notebook starts passing.

---

## 7. The manifest (`manifest.yaml`)

Per-notebook config the launcher reads. Top-level keys: `servers` (named host/hardware defs) and
`notebooks` (path → metadata). Supported notebook fields:

- `server` — named entry from the `servers` block (sets host/user/hardware).
- `hardware_required` — list, e.g. `[mi300x]` or `[mi355x, mi350x]`.
- `gpus_required` — integer; used by `--max-gpus` filtering (default 1 if absent).
- `expected_result: partial` — a "partial" outcome counts as passing (for notebooks with genuinely
  untestable cells, e.g. a final Gradio UI).
- `skip: true` + `skip_reason: "..."` — exclude from runs (report, don't fail).
- `docker_overrides` — regex→replacement map applied to image tags (rarely needed; resolver handles
  the common case).
- `notes` — free-form, surfaced into the agent prompt (external services, known untestable cells,
  large-model warnings).

Docker images are **never** taken from the notebook. The agent always calls
`resolve_docker_image.py <repo> <hardware>` and uses that exact tag. Repos: `rocm/pytorch` (torch/
transformers/diffusers), `vllm/vllm-openai-rocm` (vLLM), `lmsysorg/sglang` (SGLang).

---

## 8. Result JSON & where the numbers come from

Each run writes `results/<stem>_<ISO-timestamp>.json`. `status.py` keeps the **latest per stem**.
Shape (see `tools/write_result.py`):

```json
{
  "notebook": "notebooks/inference/foo.ipynb",
  "status": "pass | fail | partial",
  "summary": "one paragraph: what ran, what broke",
  "issues": [{"cell_index": 8, "error_type": "deprecated_api",
              "description": "...", "proposed_fix": "...", "fixability": "auto_fixable"}],
  "fixes":  [{"cell_index": 8, "fix_description": "...", "patch": "...", "validated": true}],
  "docker_image_resolved": "vllm/vllm-openai-rocm:v0.23.0",
  "agent": "claude_code_verify | claude_code_fix",
  "timestamp": "..."
}
```

**Impact at handoff (from `STATUS.md`):** 0% of tutorials were automatically tested before this; now
**33 of 40** are tested on every run. **54.5% pass end-to-end (18/33)** — those 18 are ready for
end-users. Of the 15 that fail, 14 need author intervention (which the agent raises) and 1 the agent
auto-fixes. **Determinism was verified: running any notebook 20× gives the same verdict.**

---

## 9. What's next / path to production (the ask)

**The production goal is to run this agent continuously against the internal tutorials repo,
`AMD-ROCm-Internal/gpuaidev-internal`** (the internal staging of the public `ROCm/gpuaidev` AI
Developer Hub), so every tutorial is regression-tested on a schedule and every content bug is filed
against that repo before it reaches users.

### 9a. Deployed self-hosted CI (built — see also §4)

The runner + Gateway-Claude items below are now DONE. The CI is a **GitHub Actions self-hosted
runner that is itself the MI300X box** (`134.199.202.143`), running as the `gh-runner` user. Because
the runner IS the GPU box, `run.py` talks to `localhost` (no SSH hop): `GPU_HOST=localhost`,
`GPU_USER=gh-runner`, `WORKSPACE_BASE=/home/gh-runner`.

Deployed pieces:
- **Harness** at `/home/gh-runner/notebook_runner` (public HTTPS clone of `Mahdi-CV/notebook_runner`),
  with a `.venv` (pyyaml, python-dotenv). Workflows do `git pull --ff-only` to refresh it.
- **Claude** via the native installer at `~/.local/bin/claude`. `run.py` spawns bare `claude` as a
  LOCAL subprocess, and the runner's base PATH does NOT include `~/.local/bin`, so every workflow
  prepends it: `echo "$HOME/.local/bin" >> "$GITHUB_PATH"`. (This is why the earlier scaffolded
  `regression.yml` could never find claude.)
- **`gh` CLI** installed at `/usr/bin/gh` (on the base PATH) for `report_github.py --publish`.
- **Passwordless self-SSH** (`gh-runner@localhost`) + `gh-runner` in the `docker` group, so the agent
  can drive docker/papermill on the 8 MI300X GPUs (1.8 TB free on `/home/gh-runner`).

Two workflows:
- **Nightly** (`notebook_runner/.github/workflows/nightly.yml`): cron `17 3 * * *` + manual dispatch.
  Runs `tools/ci_run.py --push` (gap + failing sets in `--mode full`, regenerates `STATUS.md`,
  commits/pushes it back to the harness repo) then `report_github.py --mode audit --publish`.
- **PR** (`gpuaidev-internal/.github/workflows/notebook-ci.yml`): `pull_request` on
  `docs/notebooks/**/*.ipynb`. Diffs the PR to find changed notebooks, runs only those in
  `--mode full`, then `report_github.py --mode pr-comment --publish` posts ONE idempotent status
  comment, and gates the check on pass/partial.

Both use `concurrency.group: notebook-regression` so only one job ever touches the single GPU node.

Runner registration: the box has a prepared but unregistered `actions-runner-internal/` slot (the two
live runners — `amd/skills` and a personal tutorials fork — must NOT be touched). Register it against
`AMD-ROCm-Internal/gpuaidev-internal` with labels `self-hosted,mi300x,gpuaidev-internal` using a
runner token from that repo's Settings > Actions > Runners.

Secrets to set on both repos (Actions secrets):
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_MODEL` (AMD Gateway),
`HF_TOKEN`. Nightly (harness repo) also needs `GPUAIDEV_GH_TOKEN` (read gpuaidev-internal + `gh` auth
for reporting) and `HARNESS_PUSH_TOKEN` (write access to `Mahdi-CV/notebook_runner` for the STATUS.md
push). The PR workflow uses the built-in `github.token` for its PR comment.

### 9b. Still open

1. **More hardware** for the 7 non-testable tutorials: notebooks that don't run on MI300X (Radeon
   Cloud, MI355X, multi-node) and GUI/Gradio/browser-only notebooks that need a headed harness.
2. **Org approval** to register the internal runner and let the reporter open issues/PRs against
   `AMD-ROCm-Internal/gpuaidev-internal` (runner token + repo write scope).

**Where it goes after that:**
- Gate new tutorial proposals: require a CI pass before human review (already used on 2 notebooks that
  were verified, fixed, and published).
- Extend to AI Academy notebooks (minimal agent changes) to catch outages before Discord does.
- Scale across the full catalog and additional repos.

Concrete near-term code work if you want to keep improving the agent itself:
- Parallelize runs across multiple GPU nodes (currently sequential).
- Tighten fixer version-pin validation and broaden `auto_fixable` coverage where it's safe.

---

## 10. A worked example: the tutorial authored *with* the agent

A new inference tutorial was written and validated using this pipeline and is up for review to
publish to the AI Developer Hub: **"Graph Engineering with OpenClaw: a Multi-Agent Incident Triage
System."** It is maintained outside this repo and submitted through the normal hub review process.

It's a useful reference for two reasons: (a) it's an example of the agent catching real content bugs
during authoring (version drift in OpenClaw 2026.7.1-2 — installer TUI, gateway process-name change,
lean-mode hiding spawn tools, a workspace sandbox), and (b) it shows the hub's expected structure
(Prerequisites with Hardware/Software subsections, AMD Developer Cloud credits, numbered Parts). If you
extend the agent to *gate* new tutorials, this notebook is a good end-to-end test case.

---

## 11. Debugging a run (quick reference)

- **SSH hangs / drops:** the agent redirects stdin for background processes; if you see hangs, check
  the `SSH_CMD` flags and that key-based auth works non-interactively.
- **`env_check` blocked the run:** the result JSON says why (token/GPU/disk). Fix the environment or
  use `--skip-env-check` for development only.
- **Agent finished but no result file:** it hit `--max-turns` (200) or crashed; check the run log in
  `logs/` and the stream-json output.
- **Wrong Docker image:** never hand-set it; confirm `resolve_docker_image.py <repo> <hardware>`
  returns a sane tag. `:latest` is a rule violation.
- **Behavior is wrong:** edit the **playbooks** (`CLAUDE.md` for verify, `CLAUDE_FIX.md` for fix), not
  `run.py`. `run.py` only wires context and streams output; all reasoning lives in the playbooks.

---

## 12. Where to continue — a checklist for the next person

1. Read `README.md` (authoritative usage), then `CLAUDE.md` and `CLAUDE_FIX.md` (the two playbooks —
   this is where all agent behavior is defined).
2. `source venv/bin/activate`, fill in `.env`, confirm `claude` is on PATH and SSH to the GPU box works.
3. Run one notebook end-to-end: `python run.py notebooks/inference/build_airbnb_agent_mcp.ipynb`.
4. Regenerate the dashboard: `python tools/status.py --write` and read `STATUS.md`.
5. Do a dry-run report: `python tools/report_github.py --mode audit` and inspect `logs/github_preview/`.
6. For CI: read §9a (self-hosted runner is deployed). Remaining production work is in §9b — register
   the `actions-runner-internal` slot against `gpuaidev-internal` and set the Actions secrets.

If in doubt: the philosophy is **surface bugs to authors, never green-wash**. Every issue you hide is
a defect that ships to users.
