# Notebook Regression Status

## Summary

- Total testable notebooks: **20**
- Pass: **8** (40.0%)
- Fail: **12** (60.0%)
- Never-tested: **0** (0.0%)
- Results in last 7 days: **10** (pass rate within those: 50.0%)
- Total cost across latest results: —
- Most recent run: **2026-06-08 22:41**

## Notebooks

| Notebook | Category | Status | Last run (UTC) | Image tested | Top failure | Cost |
|---|---|---|---|---|---|---|
| fine_tune/fine_tuning_lora_qwen2vl.ipynb | fine_tune | pass | 2026-06-05 16:29 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 |  |  |
| fine_tune/qwen_image.ipynb | fine_tune | fail | 2026-05-27 17:33 | — | version_incompatibility: The requirements-amd.txt constructed in the install cell uses pip editable in... |  |
| fine_tune/unsloth_Llama3_1_8B_GRPO.ipynb | fine_tune | fail | 2026-06-05 16:53 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | version_incompatibility: unsloth_zoo==2025.3.17 (pinned in cell 5) is incompatible with transformers>=... |  |
| | | | | | | |
| gpu_dev_optimize/aiter_mla_decode_kernel.ipynb | gpu_dev_optimize | pass | 2026-05-27 18:53 | — |  |  |
| gpu_dev_optimize/fp8_quantization_quark_vllm.ipynb | gpu_dev_optimize | fail | 2026-06-04 21:34 | vllm/vllm-openai-rocm:v0.22.0 | version_incompatibility: pip install amd-quark==0.8.1 fails in Python 3.12 (vllm/vllm-openai-rocm:v0.2... |  |
| gpu_dev_optimize/helion_gpu_kernel_dev.ipynb | gpu_dev_optimize | fail | 2026-05-27 20:44 | — | version_incompatibility: pip install triton==3.5.1 conflicts with the Docker image rocm/pytorch:rocm7.... |  |
| gpu_dev_optimize/triton_kernel_dev.ipynb | gpu_dev_optimize | pass | 2026-05-28 19:02 | — |  |  |
| | | | | | | |
| inference/1_inference_ver3_HF_transformers.ipynb | inference | pass | 2026-06-05 22:46 | rocm/pytorch:latest |  |  |
| inference/3_inference_ver3_HF_vllm.ipynb | inference | fail | 2026-06-04 18:08 | vllm/vllm-openai-rocm:v0.22.0 | content_error: Cell 8 launches the vLLM server as a foreground inline shell command (!HIP_VI... |  |
| inference/build_airbnb_agent_mcp.ipynb | inference | fail | 2026-05-29 22:33 | vllm/vllm-openai-rocm:v0.22.0 | content_error: MCPServerStdio is initialized with "python" as the executable (time_server = ... |  |
| inference/deepseek_janus_cpu_gpu.ipynb | inference | fail | 2026-05-29 22:43 | rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 | version_incompatibility: Cell installs torch from rocm6.3 wheel index (https://download.pytorch.org/wh... |  |
| inference/deepseekr1_sglang.ipynb | inference | fail | 2026-05-29 22:57 | lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x | content_error: Port mismatch: the docker run command maps -p 3000:3000 and starts the server... |  |
| inference/opea_deployment_and_evaluation.ipynb | inference | fail | 2026-05-29 18:55 | — | content_error: Cell contains raw vLLM CLI flags (--max-model-len 2048 --tensor-parallel-size... |  |
| inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.ipynb | inference | pass | 2026-05-27 23:25 | — |  |  |
| inference/rag_ollama_llamaindex.ipynb | inference | fail | 2026-06-08 22:41 | — | content_error: Cell runs `!sudo systemctl start ollama` then `!sudo systemctl status ollama`... |  |
| inference/rapbot_vllm.ipynb | inference | pass | 2026-06-05 22:27 | vllm/vllm-openai-rocm:v0.22.0 |  |  |
| inference/speculative_decoding_deep_dive.ipynb | inference | fail | 2026-06-04 17:50 | vllm/vllm-openai-rocm:v0.22.0 | content_error: %%bash cell runs docker commands but Docker is not available inside the vllm ... |  |
| inference/voice_pipeline_rag_ollama.ipynb | inference | pass | 2026-06-08 17:39 | rocm/pytorch:latest |  |  |
| | | | | | | |
| pretrain/ddim_pretrain.ipynb | pretrain | fail | 2026-05-27 19:36 | — | deprecated_api: Cell imports huggingface_hub.Repository which was removed in huggingface_hub>... |  |
| pretrain/se3transform_intro.ipynb | pretrain | pass | 2026-06-08 17:45 | rocm/dgl:dgl-2.4.0.amd0_rocm7.0.0_ubuntu24.04_py3.12_pytorch_2.8.0 |  |  |
