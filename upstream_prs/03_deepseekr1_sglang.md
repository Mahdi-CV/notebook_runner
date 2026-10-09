# PR: fix(deepseekr1_sglang): Fix port mismatch in Docker launch cell

## Regression Test Finding

**Notebook**: `inference/deepseekr1_sglang.ipynb`
**Error type**: content_error (auto_fixable)
**Cell**: 2 (Docker launch — `docker run` port mapping and `--port` flag)
**Tested against**: lmsysorg/sglang:v0.4.5.post3-rocm630

## What broke

The health-check cell (cell 4) hung for its full 20-minute timeout and then
failed with `SystemExit(1)`, even though the SGLang server itself started
and served inference successfully.

## Root cause

**SGLang's actual default port is 30000**, not 3000 — confirmed against
[SGLang's own docs](https://docs.sglang.io) and server source
(`python -m sglang.launch_server` binds `30000` unless `--port` overrides
it). Cell 2 (a `%%bash` cell) gets this right in its own header:

```bash
export INFERENCE_PORT=30000
```

...but then hardcodes the wrong number in the `docker run` itself:

```bash
docker run -d --rm \
    ...
    -p 3000:3000 \
    ...
    python3 -m sglang.launch_server \
    --port 3000 \
    ...
```

Cell 4's health check reads `os.getenv("INFERENCE_PORT", 30000)`. The
`INFERENCE_PORT` shell export in cell 2 never reaches the Python kernel
(it's set inside a `%%bash` subprocess that exits before the next cell
runs), so cell 4 always falls through to its **default**, `30000` — which
is correct, and matches SGLang's real default. The bug is entirely in cell
2: the container is mapped and the server launched on `3000` instead of
`30000`, so the health check (correctly polling `30000`) never finds
anything listening.

**An earlier draft of this PR (and a since-reverted local fix) went the
opposite direction** — changing cell 4's fallback default to `3000` to
match cell 2's `3000`. That "worked" in the sense that the health check
then found the misconfigured server, but it's fixing the wrong cell: it
papers over cell 2's bug instead of fixing it, and it hardcodes a port
number that contradicts both `INFERENCE_PORT` and SGLang's documented
default. Caught by re-reading SGLang's actual default rather than
assuming the notebook's `3000` was intentional.

## Fix applied

### Cell 2: port mapping and `--port` flag, 3000 → 30000

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

Two lines changed, both in cell 2. Cell 4 is untouched — its existing
`os.getenv("INFERENCE_PORT", 30000)` default was already correct.

## Verification

Verified end-to-end on 2026-09-17 with a fresh verification agent (no
memory of the prior incorrect fix, no bias toward either direction): ran
the corrected notebook via papermill against a live SGLang server. All 10
cells completed with zero errors in ~1m02s, health check succeeded in 60s.
Result: `partial` (matches manifest's `expected_result: partial`), **zero
issues** — cell 2 and cell 4 are now consistent with each other and with
SGLang's actual default port. Full result at
`results/deepseekr1_sglang_20260917T192902.json`.

Cell 8 (placeholder public IP/port for the reader's own device) and cell 9
(Open WebUI / VS Code AI Toolkit, markdown-only) remain out of scope — both
are expected, reader-side manual follow-up steps, not content bugs.
