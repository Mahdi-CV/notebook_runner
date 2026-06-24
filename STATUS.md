<!-- Auto-generated — do not edit by hand -->

# Notebook Regression Status

## Summary

- Total testable notebooks: **32**
- Pass: **15** (46.9%)
- Partial: **4** (12.5%)
- Fail: **13** (40.6%)
- Most recent run: **2026-06-24 22:26**
- Architecture: **Two-agent verify/fix** (branch: `feature/two-agent-verify-fix`)

### Changes from previous run (single-agent architecture)

| Notebook | Old status | New status | Reason |
|---|---|---|---|
| `llama_factory_llama3` | fail | **pass** | Wrong Docker image (Python 3.10) → correct image (Python 3.12) |
| `vllm_v1_DSR1` | fail | **pass** | Fixer replaced deprecated `huggingface-cli` → `hf` |
| `fp8_quantization_quark_vllm` | fail | **partial** | Fixer fixed quark API imports; export `needs_author` |
| `llama4_profiling_vllm` | fail | **partial** | Fixer fixed CLI flags; GPU memory `infra_blocked` |

No regressions — all previously passing notebooks remain passing.

## Notebooks

| Notebook | Category | Status | Agent | Last run (UTC) | Image tested | Top failure |
|---|---|---|---|---|---|---|
| fine_tune/fine_tuning_lora_qwen2vl.ipynb | fine_tune | pass | verify | 2026-06-05 16:29 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | |
| fine_tune/llama_factory_llama3.ipynb | fine_tune | pass | verify | 2026-06-23 17:49 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | |
| fine_tune/qwen_image.ipynb | fine_tune | fail | verify | 2026-06-22 22:48 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | version_incompatibility: pip install -e . fails — DiffSynth-Studio setup.py uses pkg_resources (Python 3.12) |
| fine_tune/slime_qwen3_4B_GRPO.ipynb | fine_tune | fail | fix | 2026-06-24 20:20 | rlsys/slime:latest | content_error: %%bash cells hardcode cd /workspace/notebooks/slime — path doesn't exist |
| fine_tune/torchtune_llama3.ipynb | fine_tune | fail | verify | 2026-06-23 22:09 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | content_error: Training runs 3235 steps (~6.7h), no max_steps_per_epoch limit |
| fine_tune/unsloth_Llama3_1_8B_GRPO.ipynb | fine_tune | fail | verify | 2026-06-22 22:40 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | version_incompatibility: unsloth_zoo 2025.3.17 incompatible with transformers 5.x |
| | | | | | | |
| gpu_dev_optimize/aiter_mla_decode_kernel.ipynb | gpu_dev_optimize | pass | — | 2026-05-27 18:53 | — | |
| gpu_dev_optimize/fp8_quantization_quark_vllm.ipynb | gpu_dev_optimize | partial | fix | 2026-06-23 20:52 | vllm/vllm-openai-rocm:v0.23.0 | version_incompatibility: quark 0.8.1 Config/QuantizationConfig imports moved (fixed); ModelExporter removed (needs_author) |
| gpu_dev_optimize/helion_gpu_kernel_dev.ipynb | gpu_dev_optimize | fail | — | 2026-05-27 20:44 | — | version_incompatibility: triton==3.5.1 conflicts with Docker image |
| gpu_dev_optimize/llama4_profiling_vllm.ipynb | gpu_dev_optimize | partial | fix | 2026-06-24 22:26 | vllm/vllm-openai-rocm:v0.23.0 | other: --disable-log-requests and --max_num_batched_tokens flags wrong for v0.23.0 (fixed); VLLM_TORCH_PROFILER_DIR needs_author |
| gpu_dev_optimize/triton_kernel_dev.ipynb | gpu_dev_optimize | pass | — | 2026-05-28 19:02 | — | |
| | | | | | | |
| inference/1_inference_ver3_HF_transformers.ipynb | inference | pass | verify | 2026-06-22 22:54 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | |
| inference/3_inference_ver3_HF_vllm.ipynb | inference | fail | verify | 2026-06-23 18:52 | vllm/vllm-openai-rocm:v0.23.0 | content_error: Foreground vLLM server blocks papermill — needs_author restructure |
| inference/SGlang_PD_Disagg_On_AMD_GPU.ipynb | inference | fail | verify | 2026-06-24 18:15 | lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi30x-20260623 | content_error: Placeholder paths, servers in markdown cells, RDMA driver missing |
| inference/amd_comfyui_rocm_tutorial.ipynb | inference | pass | — | 2026-06-09 17:40 | rocm/comfyui:comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04 | |
| inference/build_airbnb_agent_mcp.ipynb | inference | fail | — | 2026-05-29 22:33 | vllm/vllm-openai-rocm:v0.22.0 | content_error: MCPServerStdio uses "python" executable instead of "python3" |
| inference/deepseek_janus_cpu_gpu.ipynb | inference | fail | verify | 2026-06-22 22:14 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | version_incompatibility: janus 1.0.0 incompatible with transformers 5.x (mutable dict default) |
| inference/deepseekr1_sglang.ipynb | inference | fail | — | 2026-05-29 22:57 | lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x | content_error: Port mismatch in docker run command |
| inference/opea_deployment_and_evaluation.ipynb | inference | fail | — | 2026-05-29 18:55 | — | content_error: Raw vLLM CLI flags in non-shell cell |
| inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.ipynb | inference | pass | — | 2026-05-27 23:25 | — | |
| inference/rag_ollama_llamaindex.ipynb | inference | fail | fix | 2026-06-24 19:12 | — | content_error: systemctl status ollama hangs papermill (pager in non-TTY) |
| inference/rapbot_vllm.ipynb | inference | partial | verify | 2026-06-22 23:20 | vllm/vllm-openai-rocm:v0.22.0 | content_error: Interactive input() chat loop |
| inference/speculative_decoding_deep_dive.ipynb | inference | fail | fix | 2026-06-23 19:02 | vllm/vllm-openai-rocm:v0.23.0 | content_error: Docker-in-Docker unavailable, placeholder paths, stale image tags |
| inference/triton_inference_server_benchmark.ipynb | inference | pass | — | 2026-06-12 18:13 | rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12 | |
| inference/vllm_v1_DSR1.ipynb | inference | pass | fix | 2026-06-24 21:08 | rocm/vllm-dev:nightly | |
| inference/voice_pipeline_rag_ollama.ipynb | inference | partial | verify | 2026-06-22 23:03 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | missing_dependency: ffmpeg not installed in Docker image |
| | | | | | | |
| pretrain/ddim_pretrain.ipynb | pretrain | fail | verify | 2026-06-22 22:33 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | deprecated_api: huggingface_hub.Repository removed in huggingface_hub >= 0.19.0 |
| pretrain/se3transform_intro.ipynb | pretrain | pass | — | 2026-06-08 17:45 | rocm/dgl:dgl-2.4.0.amd0_rocm7.0.0_ubuntu24.04_py3.12_pytorch_2.8.0 | |
| pretrain/setup_tutorial.ipynb | pretrain | pass | — | 2026-06-16 19:50 | rocm/pytorch-training:latest | |
| pretrain/torchtitan_llama3.ipynb | pretrain | fail | verify | 2026-06-24 20:29 | rocm/pytorch-training:latest | content_error: Relative paths fail in Docker, unpinned torch nightly breaks torchtitan |
| pretrain/train_llama_mock_data.ipynb | pretrain | pass | — | 2026-06-16 20:22 | rocm/megatron-lm:24.12-dev | |
| pretrain/training_with_primus.ipynb | pretrain | pass | — | 2026-06-16 20:59 | rocm/primus:v25.9_gfx942 | |

## Failure Classification

| Fixability | Count | Description |
|---|---|---|
| `auto_fixable` | 8 | Fixer agent can apply mechanically (version pin, import rename, flag update) |
| `needs_author` | 24 | Requires human judgment (structural redesign, placeholder content, missing assets) |
| `infra_blocked` | 2 | Cannot fix in notebook (GPU memory, system packages) |

Most failures are `needs_author` — structural issues like foreground servers blocking papermill,
placeholder paths, unpinned dependencies, and interactive widgets that can't run headlessly.
