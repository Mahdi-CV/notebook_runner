# PR: fix(qwen_image): Split pip install to fix editable build failure

## Regression Test Finding

**Notebook**: `fine_tune/qwen_image.ipynb`
**Error type**: missing_dependency
**Cells**: 5 (install), 8 (import fails as downstream consequence)
**Tested against**: rocm/pytorch:latest

## What broke

Cell 8 fails with `ModuleNotFoundError: No module named 'imageio'` when importing `diffsynth.pipelines.qwen_image`. DiffSynth-Studio's dependencies were never installed because the editable install in cell 5 silently failed.

## Root cause

Cell 5 generates a `requirements-amd.txt` that sets `--index-url https://download.pytorch.org/whl/rocm6.4` as the **primary** pip index. When pip attempts the editable install (`-e .`) of DiffSynth-Studio, the build backend needs to resolve its own build-time dependencies (setuptools, wheel, etc.) but can only find them on PyPI — not on the ROCm wheel index. This causes:

```
ERROR: Failed to build 'file:///DiffSynth-Studio' when getting requirements to build editable
```

Since the cell uses `!pip` (shell escape), the error is swallowed — Python doesn't raise an exception — and execution continues with DiffSynth's dependencies (imageio, einops, etc.) never installed.

## Fix applied

**Before (cell 5 generates a single requirements-amd.txt):**
```
--index-url https://download.pytorch.org/whl/rocm6.4
--extra-index-url https://pypi.org/simple
torch
torchvision
-e .
```

**After (split into two install steps):**

```python
# Step 1: Install torch with ROCm wheels
!pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# Step 2: Install DiffSynth-Studio in editable mode (uses standard PyPI for build deps)
!pip install -e .
```

This ensures the editable build backend can resolve its dependencies from PyPI while torch still comes from the ROCm index.

## Secondary note

The ROCm wheel URL uses `rocm6.4` but the Docker image is `rocm/pytorch:latest` (ROCm 7.2.4). Since torch is already pre-installed in the Docker image, the torch install line could be removed entirely:

```python
# torch/torchvision already provided by Docker image
!pip install -e .
```

## Verification

Agent confirmed on 2026-06-08 that the editable install fails with the combined requirements file. The fix (splitting installs) has not been validated end-to-end yet, but the root cause is clearly the `--index-url` override interfering with build isolation.
