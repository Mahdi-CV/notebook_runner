# PR: fix(rag_ollama_llamaindex): Fix systemctl pager hang; use a venv instead of --break-system-packages

## Regression Test Finding

**Notebook**: `inference/rag_ollama_llamaindex.ipynb`
**Error type**: content_error (auto_fixable) — two independent issues, one PR
**Tested against**: Host Python (Pattern D, no Docker)

This PR bundles two fixes needed to get past the notebook's setup section
and reach its actual RAG pipeline logic for the first time. Both were
required — fixing only the first still left the notebook unable to run to
completion.

---

## Issue 1: `systemctl status` hangs indefinitely

### What broke

`!sudo systemctl start ollama` / `!sudo systemctl status ollama` hung for
the full 600s papermill timeout and aborted the entire run before any of
the notebook's actual content executed.

### Root cause — verified by direct reproduction, not just inferred

`systemctl status` decides whether to page its output based on whether
stdout looks like an interactive terminal. On this host,
**`/etc/sudoers` has `Defaults use_pty`**, which makes `sudo` allocate a
real pseudo-terminal for every command it runs — including this one, even
though papermill's own stdin/stdout have no real terminal behind them.
`systemctl status` sees that pty and decides to page; the resulting
`pager` (`less`) process sits waiting for `q`/EOF that never comes,
because nothing on the other end of the kernel's stdin can send it.

Reproduced directly: running the two-line cell through real `papermill`
against this host times out and leaves a live process tree with a `pager`
process attached to a `pts/N` device, sitting in state `S+` (sleeping,
foreground) — confirmed via `pstree`/`ps` while the hang was in progress.

**A plain non-TTY `subprocess.run(..., stdin=DEVNULL)` reproduces
nothing** — the hang is specific to the pty `sudo`'s `use_pty` setting
allocates, not to "no TTY" in general.

**`SYSTEMD_PAGER=cat` does NOT fix this** — also confirmed by direct test.
`/etc/sudoers` here also has `Defaults env_reset`, which strips
environment variables (no `env_keep` entry for `SYSTEMD_PAGER`) before the
command inside `sudo` runs, so the env var never reaches `systemctl`.
`--no-pager` works because it's a `systemctl` CLI flag, immune to `sudo`'s
environment stripping.

Both `use_pty` and `env_reset` are common, unremarkable `sudoers`
hardening defaults, not a quirk of this one test host.

### Fix

```diff
 !sudo systemctl start ollama
-!sudo systemctl status ollama
+!sudo systemctl status ollama --no-pager
```

---

## Issue 2: pip installs fail with PEP 668 `externally-managed-environment`

### What broke

Once the pager hang was fixed, the run reached two `!pip install` cells
(ROCm-build torch, then llama-index) that both fail with PEP 668
`externally-managed-environment` errors on this Debian-based host — the
system Python refuses `pip install` outside a virtual environment. The
llama-index failure is fatal: every downstream cell (indexing, vector
store, the four query examples) depends on it.

### Fix: a dedicated venv, not `--break-system-packages`

`--break-system-packages` would "work" but defeats the protection PEP 668
exists to provide, on the reader's own machine, for no reason beyond
convenience. This notebook has no venv step anywhere (its own
"Prerequisites" section assumes bare host Python), so the real fix is to
introduce one — done as a self-contained addition, not a rewrite of the
tutorial's structure:

1. **New markdown + code cell, added before the PyTorch section**, asks
   the reader to install `python3-venv` if needed (many minimal
   distros/containers don't ship it by default) and creates a dedicated
   venv:
   ```bash
   !python3 -m venv $HOME/.venv-rag-ollama
   ```
2. **Both `pip install` cells** (torch, llama-index) and both `pip list`
   verification cells now source the venv's `activate` script first, so
   `pip` (and anything else on `PATH`) resolves from the venv:
   ```bash
   !. $HOME/.venv-rag-ollama/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```
   `.` (not `source`) is used deliberately — see below.
3. **The GPU-check cell** (`import torch`, right after the torch install)
   gets a few lines prepended that ask the activated venv directly for its
   site-packages path via `subprocess.check_output`, then inject that path
   into the kernel's `sys.path` — no temp file needed:
   ```python
   import subprocess, sys
   venv_site_packages = subprocess.check_output(
       '. $HOME/.venv-rag-ollama/bin/activate && python3 -c "import site; print(site.getsitepackages()[0])"',
       shell=True, text=True,
   ).strip()
   if venv_site_packages not in sys.path:
       sys.path.insert(0, venv_site_packages)
   ```

### `.` not `source` — the shell running `!` cells matters

An earlier draft of this fix used `source venv/bin/activate`. `source` is
a bash-only builtin. Papermill/IPython's `!` shell-out does not run bash —
it runs `subprocess.Popen(cmd, shell=True)`, which on this (and most)
Linux systems resolves to `/bin/sh` → `dash`, not `bash`. Confirmed
directly: `subprocess.run("source /etc/os-release && echo done",
shell=True)` fails with `/bin/sh: 1: source: not found`, while the POSIX
equivalent `subprocess.run(". /etc/os-release && echo done",
shell=True)` succeeds. `.` is the POSIX-standard dot-sourcing builtin,
available in `dash`, `bash`, and every other `/bin/sh` implementation —
using it instead of `source` makes the cell portable to whatever shell
`!` actually invokes, rather than assuming bash.

### Why the sys.path step is necessary — this is not a simple flag swap

Activating the venv only affects the `!` cell's own subprocess — it does
**not** make the packages importable from the notebook's own kernel. The
Jupyter kernel is a single long-lived Python process (`/usr/bin/python3`)
that was never launched inside the venv; a `!` cell that activates the
venv and installs `X` puts `X` in the venv's *separate* `site-packages`,
which the kernel process never sees on its own `sys.path`. A later cell's
plain `import X` still fails, even though the install "succeeded" —
confirmed directly: installing a test package into a fresh venv and then
trying to import it from a plain `python3 -c` process reproduces
`ModuleNotFoundError` until the venv's site-packages directory is
explicitly added to `sys.path`.

The `sys.path.insert` cell only needs to run **once** — `sys.path`
mutations persist for the kernel's lifetime, and papermill executes the
whole notebook in one kernel (no restart between cells), so the later
`import chromadb` / `from llama_index...` cell (which installs into the
same venv) picks up the same path without needing its own injection.
Confirmed by the full run below: execution proceeded cleanly through both
install blocks and both import blocks.

An alternative — registering the venv as its own Jupyter kernel and
switching to it — was considered and rejected: it would require changing
how papermill invokes this notebook (`--kernel`), which is an
execution-pipeline change, not a notebook-content fix.

---

## Verification

Four rounds of testing:

1. **Minimal repro** of the pager hang alone (see mechanism above),
   confirming `--no-pager` fixes it and `SYSTEMD_PAGER=cat` does not.
2. **Manual end-to-end mechanism check** of the venv + `sys.path`
   approach: created a venv, installed a package into it, confirmed it
   was *not* importable from a plain Python process, then confirmed it
   *was* importable after inserting the venv's site-packages into
   `sys.path` — the exact pattern used in the fix.
3. **Shell-compatibility check for `.` vs `source`**: confirmed directly
   that `subprocess.run(..., shell=True)` — what papermill/IPython's `!`
   actually uses — invokes `/bin/sh` (`dash` on this host), which does
   not understand `source`, but does understand the POSIX `.` builtin.
   Verified both the failure (`source` → `not found`) and the fix (`.` →
   succeeds) directly, then verified a `. activate && pip install X`
   compound command correctly resolves `pip` from the venv's `PATH`.
4. **Full notebook, all fixes in place, run end-to-end** via the real
   verification pipeline on 2026-09-17 (`results/rag_ollama_llamaindex_20260917T220835.json`):
   execution proceeded cleanly through Ollama setup, venv creation, and
   both the torch and llama-index install/import blocks — the
   `--no-pager` and venv fixes (including the simplified `sys.path` cell)
   are confirmed working together. The run fails only at a **third,
   independent, pre-existing bug**: the LlamaIndex install cell omits
   `llama-index-readers-file` (and its `pypdf` dependency), so
   `SimpleDirectoryReader` can't properly parse the downloaded PDF and
   falls back to a single-blob document, and the notebook's hardcoded
   `documents[10]` then raises `IndexError`. That is out of scope for
   this PR.

   **Caveat caught during this round of testing**: an *earlier* attempt
   at this same verification run (`results/rag_ollama_llamaindex_20260917T220259.json`)
   reported a false PASS — all 49 cells succeeded, including the
   `documents[10]` line. Investigating why (rather than accepting a
   green run at face value) turned up the cause: `$HOME/.venv-rag-ollama`
   lives in the server's home directory, outside the per-run workspace
   (`/home/amd/tutorial_agent_runs/<stem>/`) that gets wiped after each
   test, so it survives between independent runs. An earlier manual
   debugging session on this same server had `pip install`ed
   `llama-index-readers-file` directly into that same venv while
   diagnosing the original bug, and that stale, already-patched venv was
   silently reused. Confirmed by inspecting the venv directly
   (`pip list` showed `llama-index-readers-file` and `pypdf` already
   present) and reproduced the real failure only after deleting
   `$HOME/.venv-rag-ollama` and re-running from scratch. The result
   quoted above is from that clean re-run.
