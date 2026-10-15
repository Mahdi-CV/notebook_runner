# PR: fix(rag_ollama_llamaindex): Add --no-pager to systemctl status command

## Regression Test Finding

**Notebook**: `inference/rag_ollama_llamaindex.ipynb`
**Error type**: content_error
**Cell**: 6 (systemctl start + status)
**Tested against**: Host Python (Pattern D, no Docker)

## What broke

Cell 6 runs `!sudo systemctl status ollama` without `--no-pager`. In a non-TTY environment (papermill, script execution, headless Jupyter), `systemctl status` launches a pager (less) that blocks indefinitely waiting for interactive input. This causes a 600s timeout, and the entire RAG pipeline (cells 8-45) is never reached.

## Root cause

`systemctl status` defaults to piping through a pager when output exceeds terminal height. In a non-interactive environment, the pager hangs waiting for stdin. The `--no-pager` flag disables this behavior.

## Fix applied

**Before:**
```bash
!sudo systemctl start ollama
!sudo systemctl status ollama
```

**After (option A — minimal):**
```bash
!sudo systemctl start ollama
!sudo systemctl status ollama --no-pager
```

**After (option B — more robust):**
```bash
!sudo systemctl start ollama
!sudo systemctl is-active ollama
```

**After (option C — health check, recommended since Prerequisites says Ollama should already be running):**
```python
import requests, time

# Start Ollama if not already running
import subprocess
subprocess.run(["sudo", "systemctl", "start", "ollama"], check=False)
time.sleep(2)

# Verify it's running
r = requests.get("http://localhost:11434")
assert r.status_code == 200, "Ollama is not running on port 11434"
print("Ollama is running")
```

## Verification

Agent confirmed on 2026-06-08 that Ollama was pre-installed and both required models (llama3.1:8b, nomic-embed-text) were already pulled. The notebook fails purely due to the pager issue in cell 6 — the actual RAG pipeline was never reached.
