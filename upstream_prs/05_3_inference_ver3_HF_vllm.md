# PR: fix(3_inference_ver3_HF_vllm): Fix foreground server blocking sequential execution

## Regression Test Finding

**Notebook**: `inference/3_inference_ver3_HF_vllm.ipynb`
**Error type**: content_error
**Cell**: 8 (vLLM server launch)
**Tested against**: vllm/vllm-openai-rocm:v0.22.0

## What broke

Cell 8 launches the vLLM server as a foreground inline shell command:
```
!HIP_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct ...
```

This blocks all subsequent cells — the server never exits, so the client code in cell 11 never executes. In practice during testing, the server crashed immediately due to `HIP_VISIBLE_DEVICES=2` being unavailable in the container, causing a `ConnectionRefusedError` on the client side.

## Root cause

Two issues:

1. **Structural**: The notebook embeds both a foreground server launch and client code in sequential cells. This design requires running them in separate terminals/sessions. Papermill (or any sequential executor) cannot handle a foreground server followed by a client.

2. **Hardcoded GPU index**: `HIP_VISIBLE_DEVICES=2` selects GPU 2, which may not be available inside a standard Docker container. This causes `AssertionError: DP adjusted local rank 0 is out of bounds`.

## Fix applied

### Option A: Convert server cell to markdown (minimal change)

Change cell 8 from a code cell to a markdown cell, explaining that the server must be started in a separate terminal:

```markdown
**Start the vLLM server in a separate terminal:**

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-model-len 4096 \
    --dtype float16
```

> Note: Remove `HIP_VISIBLE_DEVICES=2` unless you specifically need to pin to GPU 2.
> Wait until you see "Uvicorn running on http://0.0.0.0:8000" before proceeding.
```

### Option B: Background server with health poll (larger refactor)

Replace the server cell with a background launch + health check:
```python
import subprocess, time, requests

proc = subprocess.Popen(
    ["python3", "-m", "vllm.entrypoints.openai.api_server",
     "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "--max-model-len", "4096", "--dtype", "float16"],
    stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT
)

# Wait for server to be ready
for _ in range(60):
    try:
        r = requests.get("http://localhost:8000/health")
        if r.status_code == 200:
            print("vLLM server is ready!")
            break
    except:
        pass
    time.sleep(5)
```

### Issue 2: Remove hardcoded HIP_VISIBLE_DEVICES=2

Regardless of which option above is chosen, remove `HIP_VISIBLE_DEVICES=2` or replace with `HIP_VISIBLE_DEVICES=0` (first available GPU). Document that users should set this based on their system configuration.

## Verification

Agent confirmed on 2026-06-04 that the structural issue (foreground server) is the primary blocker. The server cell must be either separated or backgrounded for the notebook to execute end-to-end.
