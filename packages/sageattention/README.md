# sageattention — build notes
- **Source:** thu-ml/SageAttention (v2.2.0)
- **Quirks:** `max_jobs: 1`. Patch fixes arch parser (space-separated TORCH_CUDA_ARCH_LIST), MSVC CXX flags, Windows ABI guard, nvcc --threads=8->4, sm_90-only gencode for _qattn_sm90.
- **Arch list:** cu124-cu126 use `"8.0 8.6 8.9 9.0"`. cu128+ use `"8.0 8.6 8.9 9.0 10.0 12.0"` (Blackwell).

**CUDA-version-specific behavior:** Needs CUDA >= 12.0; SM 12.0 (Blackwell) needs >= 12.8. No upper CUDA bound.

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. Must include 8.9 (Ada / RTX 40-series) — sage 2.x's _qattn_sm89 extension contains Ada-only FP8 mma instructions gated on __CUDA_ARCH__ >= 890. Without an sm_89 cubin the loader falls back to sm_80 binary (forward-compatible Ampere → Ada), where those gated paths compile to stubs and `cudaErrorLaunchFailure` surfaces at runtime on real Ada hardware. 8.6 added too so Ampere consumer cards (3090/3080/3070/3060) get a native cubin instead of relying on JIT.
