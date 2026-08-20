# sageattn3 — build notes
- **Source:** thu-ml/SageAttention (v2.2.0), `build_subdir: sageattention3_blackwell`
- **Quirks:** Blackwell-only (SM 10.0, 12.0, 12.1). Complex MSVC patches: kernel_traits.h dependent-name workaround, kernel_ws.h parameter passing (pointer vs CUTE_GRID_CONSTANT), launch.h device-side parameter packing.
- **Arch list:** Global `"10.0 12.0"`. Only builds for cu128+.

**CUDA-version-specific behavior:** Blackwell-only (SM 10.0, 12.0). Needs CUDA >= 12.8. Patch replaces runtime GPU detection with TORCH_CUDA_ARCH_LIST parsing.

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `build_matrix.combinations`, `build_matrix.platforms`.
- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list`. 
