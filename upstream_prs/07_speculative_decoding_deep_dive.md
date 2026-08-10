# PR: fix(speculative_decoding_deep_dive): Fix Docker-in-Docker pattern and stale image tags

## Regression Test Finding

**Notebook**: `inference/speculative_decoding_deep_dive.ipynb`
**Error type**: content_error + deprecated_api (multiple cells)
**Cells**: 8, 14, 17
**Tested against**: vllm/vllm-openai-rocm:v0.22.0

## What broke

Cell 8 (`%%bash` cell with `docker run`) failed with exit status 127 — `docker: command not found`. The notebook is designed to be run on a host machine with Docker installed, but the regression test ran it inside a Docker container (as is standard for notebooks that import GPU libraries). Additionally, cells contain placeholder paths, `-it` flags that block automation, missing `%%bash` prefixes, and outdated image tags.

## Root cause

This notebook orchestrates multiple Docker containers (vLLM, SGLang) from within Jupyter cells. It assumes the notebook runs on bare metal with Docker available. This is a Pattern D notebook (host Python) that was incorrectly classified during initial testing. The structural issues remain regardless of where it runs.

## Fix applied

### Issue 1: Placeholder paths not substituted (cell 8)

**Before:**
```bash
MODEL_PATH=/path/to/model/weights/cache/directory/
WORK_PATH=/path/to/workspace/
```

**After:**
```bash
MODEL_PATH=${MODEL_PATH:-$HOME/.cache/huggingface/hub}
WORK_PATH=${WORK_PATH:-/tmp/vllm_workspace}
mkdir -p $WORK_PATH
```

Or add an instruction cell before cell 8:
```python
# Set these paths before running the Docker cells below
import os
os.environ["MODEL_PATH"] = os.path.expanduser("~/.cache/huggingface/hub")
os.environ["WORK_PATH"] = "/tmp/vllm_workspace"
```

### Issue 2: `docker run -it` blocks papermill (cell 14)

**Before:**
```bash
docker run -it --rm \
    ...
```

**After:**
```bash
docker run -d --rm \
    ...
```

Then add a wait/poll step after the container starts:
```bash
# Wait for benchmark to complete
docker wait <container_name>
```

### Issue 3: Missing `%%bash` prefix (cell 17)

**Before (code cell, no magic):**
```
docker run -d --rm \
    --name vllm_speculative \
    ...
```

**After:**
```bash
%%bash
docker run -d --rm \
    --name vllm_speculative \
    ...
```

### Issue 4: Stale Docker image tags (cells 8, 14, 17)

**Before:**
```
vllm/vllm-openai-rocm:v0.15.0
lmsysorg/sglang:v0.5.8-rocm700-mi30x
```

**After:**
```
vllm/vllm-openai-rocm:v0.22.0
lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x
```

Consider using variables at the top of the notebook:
```python
VLLM_IMAGE = "vllm/vllm-openai-rocm:v0.22.0"
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x"
```

## Verification

Agent confirmed on 2026-06-04 that the Docker-not-found error is the primary blocker. The structural issues (placeholder paths, -it flag, missing %%bash) were identified via static analysis and would each independently cause failures even if Docker were available.

## Note for authors

This notebook should be run on a host with Docker installed, not inside a container. Consider adding a prerequisite note at the top:
```markdown
> **Prerequisites**: This notebook must be run on a machine with Docker installed and the AMD GPU exposed via `--device=/dev/kfd --device=/dev/dri`. It cannot be run inside a Docker container.
```
