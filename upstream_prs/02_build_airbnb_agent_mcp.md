# PR: fix(build_airbnb_agent_mcp): Use `python3` instead of `python` in MCPServerStdio

## Regression Test Finding

**Notebook**: `inference/build_airbnb_agent_mcp.ipynb`
**Error type**: content_error
**Cell**: 31 (line 513 in raw JSON — the time_server definition cell)
**Tested against**: vllm/vllm-openai-rocm:v0.22.0 (server) + host Python (client)

## What broke

`MCPServerStdio("python", args=["-m", "mcp_server_time", ...])` raises `FileNotFoundError: [Errno 2] No such file or directory: 'python'` when `run_async()` is called at cell 35. The MCP server subprocess cannot start.

## Root cause

On Ubuntu 22.04/24.04, the `python` binary does not exist by default — only `python3` is available. The notebook hardcodes `"python"` as the executable for the MCP server subprocess.

## Fix applied

**Before:**
```python
time_server = MCPServerStdio(
    "python",
    args=["-m", "mcp_server_time", "--local-timezone=US/Pacific"],
)
```

**After:**
```python
time_server = MCPServerStdio(
    "python3",
    args=["-m", "mcp_server_time", "--local-timezone=US/Pacific"],
)
```

Alternative (more portable): use `sys.executable` to reference whatever Python is running the notebook.

## Verification

Fix validated by notebook_runner agent on 2026-05-29. With `"python3"`, the full notebook ran end-to-end: MCP time server queries, Airbnb MCP server queries, and the final listing search all completed successfully.
