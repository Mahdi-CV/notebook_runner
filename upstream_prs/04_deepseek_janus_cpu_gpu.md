# PR: fix(deepseek_janus_cpu_gpu): Pin transformers<4.49 and update ROCm wheel URL

## Regression Test Finding

**Notebook**: `inference/deepseek_janus_cpu_gpu.ipynb`
**Error type**: version_incompatibility
**Cells**: 6 (pip install) and 22 (missing demo image)
**Tested against**: rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

## What broke

Importing janus models fails with `ValueError: mutable default <class 'dict'> for field params is not allowed: use default_factory`. The notebook's import cell crashes before any inference can run.

## Root cause

`transformers>=4.49` applies Python `@dataclass` to all `PretrainedConfig` subclasses. `janus==1.0.0` defines `VisionConfig` with `params: dict = {}` as a class-level field, which violates dataclass rules (mutable defaults require `default_factory`). Since the notebook installs `transformers` unpinned, it pulls the latest (5.9.0), triggering the incompatibility.

Additionally, the pip install cell uses `--index-url https://download.pytorch.org/whl/rocm6.3` which is stale — the Docker image ships ROCm 7.2.4.

## Fix applied

### Issue 1: Pin transformers (cell 6)

**Before:**
```python
!pip install transformers ipywidgets
```

**After:**
```python
!pip install "transformers<4.49" ipywidgets
```

### Issue 2: Update ROCm wheel URL (cell 6)

**Before:**
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

**After (option A — remove, rely on Docker-provided torch):**
```python
# torch and torchvision are pre-installed in the rocm/pytorch Docker image
# !pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

**After (option B — update URL):**
```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2.4
```

Option A is preferred since the Docker image already has the correct torch.

### Issue 3: Missing demo image (cell 22)

The notebook references `../assets/deepseek_janus_demo_small.jpg` which does not exist in the working directory and has no download step.

**Suggested fix:** Add a download cell before cell 22:
```python
import urllib.request, os
os.makedirs("../assets", exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/deepseek-ai/Janus/main/images/demo.jpg",
    "../assets/deepseek_janus_demo_small.jpg"
)
```

Or include the image in the repository's assets directory.

## Verification

Fix for issues 1+2 validated by notebook_runner agent on 2026-05-29. With `transformers==4.48.3` installed, the janus import succeeded and model loading proceeded. Issue 3 (missing image) was identified but not fixed in that run.
