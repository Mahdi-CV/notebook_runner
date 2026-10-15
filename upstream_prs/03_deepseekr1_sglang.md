# PR: fix(deepseekr1_sglang): Fix port mismatch and update stale Docker image tag

## Regression Test Finding

**Notebook**: `inference/deepseekr1_sglang.ipynb`
**Error type**: content_error + deprecated_api
**Cells**: 2 (docker launch) and 8 (placeholder URL)
**Tested against**: lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x

## What broke

The SGLang server never becomes reachable because the port mapping is inconsistent:
- `INFERENCE_PORT=30000` (line 105)
- Docker maps `-p 3000:3000` (line 125)
- Server starts with `--port 3000` (line 130)
- Health check polls `localhost:30000/health_generate` (cell 4)

The server listens on 3000 inside the container, is mapped to 3000 on host, but the client expects it on 30000.

## Root cause

Port number mismatch between the `INFERENCE_PORT` environment variable and the Docker/server configuration. Likely a typo — `3000` should be `30000` (or vice versa, but `INFERENCE_PORT` is the downstream reference).

## Fix applied

### Issue 1: Port mismatch (cell 2)

**Before:**
```bash
    -p 3000:3000 \
    ...
    --port 3000 \
```

**After:**
```bash
    -p 30000:30000 \
    ...
    --port 30000 \
```

This makes the Docker mapping and server port match `INFERENCE_PORT=30000`.

### Issue 2: Stale Docker image tag (cell 2)

**Before:**
```bash
    --name sglang_server "lmsysorg/sglang:v0.4.5.post3-rocm630" \
```

**After:**
```bash
    --name sglang_server "$SGLANG_DIMG" \
```

The `SGLANG_DIMG` variable is already defined earlier in the cell (line 111). Use the variable instead of hardcoding it a second time. Also update `SGLANG_DIMG` to the current tag:

```bash
export SGLANG_DIMG="lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x"
```

### Issue 3: Placeholder URL (cell 8)

Cell 8 contains `curl http://YOUR_SERVER_PUBLIC_IP:PORT_NUMBER/v1/models` which is a literal placeholder. Convert this cell to markdown with instructions for the user to substitute their server IP and port.

## Verification

Agent confirmed on 2026-05-29 that SGLang starts and becomes healthy when launched with the correct port mapping and current image tag. The port mismatch is the primary blocking issue.
