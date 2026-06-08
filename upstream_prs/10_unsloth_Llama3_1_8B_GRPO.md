# PR: fix(unsloth_Llama3_1_8B_GRPO): Update unsloth_zoo pin or constrain transformers

## Regression Test Finding

**Notebook**: `fine_tune/unsloth_Llama3_1_8B_GRPO.ipynb`
**Error type**: version_incompatibility
**Cell**: 4 (install cell with version pins)
**Tested against**: rocm/pytorch:latest

## What broke

`from unsloth import FastLanguageModel` raises `ImportError: cannot import name 'HybridCache' from 'transformers.models.gemma3.modeling_gemma3'`. No model loading, training, or inference executes.

## Root cause

The install cell pins `unsloth_zoo==2025.3.17`. This version's `temporary_patches.py:162` imports `HybridCache` from `transformers.models.gemma3.modeling_gemma3`, but that symbol was removed in `transformers>=4.50`.

The unsloth ROCm fork (billhe/rocm branch) pulls in `transformers>=4.50` as a transitive dependency during its own install, creating a version conflict with the pinned `unsloth_zoo`.

The notebook comments say it was verified under `unsloth==2025.3.19` + `unsloth_zoo==2025.3.17` — the environment has since drifted.

## Fix applied

### Option A: Upgrade unsloth_zoo (preferred if a compatible version exists)

**Before:**
```python
!pip install unsloth_zoo==2025.3.17
```

**After:**
```python
!pip install "unsloth_zoo>=2025.4.0"
```

(Verify that a version >=2025.4.0 exists on PyPI and is compatible with the ROCm unsloth fork.)

### Option B: Pin transformers to compatible range

**Before:**
```python
!git clone https://github.com/billishyahao/unsloth.git && cd unsloth && git checkout billhe/rocm && pip install .
```

**After:**
```python
!git clone https://github.com/billishyahao/unsloth.git && cd unsloth && git checkout billhe/rocm && pip install .
!pip install "transformers<4.50"
```

This downgrades transformers after unsloth installs it, ensuring compatibility with `unsloth_zoo==2025.3.17`.

## Known secondary issue

If the unsloth_zoo fix is resolved, a second issue may surface: the bitsandbytes ROCm fork (built from source via `git clone https://github.com/ROCm/bitsandbytes`) has version string "N/A" in its metadata, which causes `packaging.version.parse()` to raise `InvalidVersion`. This is a packaging bug in the ROCm bitsandbytes fork — not fixable in the notebook. It would need to be fixed upstream in ROCm/bitsandbytes.

## Verification

Agent confirmed on 2026-06-05 and 2026-06-08 that unsloth_zoo==2025.3.17 is incompatible with transformers>=4.50. The June 5 run also validated that removing the pin and using a newer unsloth_zoo resolves the HybridCache import (but then hit the bitsandbytes version issue downstream).
