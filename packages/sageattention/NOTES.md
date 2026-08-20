# sageattention — build notes
- **Source:** thu-ml/SageAttention (v2.2.0)
- **Quirks:** `max_jobs: 1`. Patch fixes arch parser (space-separated TORCH_CUDA_ARCH_LIST), MSVC CXX flags, Windows ABI guard, nvcc --threads=8->4, sm_90-only gencode for _qattn_sm90.
- **Arch list:** cu124-cu126 use `"8.0 8.6 8.9 9.0"`. cu128+ use `"8.0 8.6 8.9 9.0 10.0 12.0"` (Blackwell).
