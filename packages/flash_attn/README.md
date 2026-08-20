# flash_attn — build notes
- **Source:** Dao-AILab/flash-attention (v2.8.3)
- **Quirks:** SM >= 8.0 (Ampere+). Patch inits only csrc/cutlass submodule (skips composable_kernel — ROCm only, breaks Windows paths). Bridges TORCH_CUDA_ARCH_LIST -> FLASH_ATTN_CUDA_ARCHS.
- **Arch list:** cu124-cu126 use default `"8.0 9.0"`. cu128+ use `"8.0 9.0 10.0 12.0"` (Blackwell).

**CUDA-version-specific behavior:** setup.py explicitly handles CUDA 12.9+ (SM 101/Thor, family-specific 100f/120f gencode flags).

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. 
