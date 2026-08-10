# PR: fix(opea_deployment_and_evaluation): Fix non-executable cells in troubleshooting section

## Regression Test Finding

**Notebook**: `inference/opea_deployment_and_evaluation.ipynb`
**Error type**: content_error (multiple cells)
**Cells**: 58, 70/72/74/76, 87, 103
**Tested against**: Host Python (Pattern D)

## What broke

Cell 58 contains raw vLLM CLI flags as plain Python code (no `!` prefix, no `%%bash` magic), causing a `SyntaxError` that halts all subsequent cell execution. This stopped the notebook at the troubleshooting section — the core tutorial (deployment, verification, benchmarking) had already completed successfully.

## Root cause

The "Common Issues and Solutions" section (cells 58–105) contains several cells that are documentation/examples mistakenly left as executable code cells. Some would cause syntax errors, some would block indefinitely, and one would reboot the server.

## Fix applied

### Issue 1: Raw CLI flags as Python code (cell 58)

**Before (code cell):**
```
--max-model-len 2048 --tensor-parallel-size 1
```

**After (markdown cell):**
```markdown
In `compose_vllm.yaml`, modify the vLLM service command to include:

```yaml
command: [..., "--max-model-len", "2048", "--tensor-parallel-size", "1"]
```
```

### Issue 2: `!export` doesn't persist (cells 70, 72, 74, 76)

**Before:**
```python
!export host_ip=$(hostname -I | awk '{print $1}')
```

**After:**
```python
import subprocess
host_ip = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True).decode().strip()
import os
os.environ["host_ip"] = host_ip
```

Or use the `%env` magic:
```python
%env host_ip={host_ip}
```

### Issue 3: `docker compose logs -f` blocks forever (cell 87)

**Before:**
```python
!docker compose -f compose_vllm.yaml logs -f chatqna-vllm-service
```

**After:**
```python
!docker compose -f compose_vllm.yaml logs --tail 50 chatqna-vllm-service
```

### Issue 4: `sudo reboot` in executable cell (cell 103)

**Before (code cell):**
```python
!sudo reboot
```

**After (markdown cell):**
```markdown
If all else fails, reboot the server:

```bash
sudo reboot
```

> **Warning**: This will terminate all running containers and the notebook kernel.
```

## Verification

Agent confirmed on 2026-05-29 that the core tutorial (cells 1–57) ran successfully — OPEA ChatQnA deployed, API verified, Apache Bench completed 100/100 requests. The failures are all in the troubleshooting section (cells 58+) which contains documentation that should not be executable code.
