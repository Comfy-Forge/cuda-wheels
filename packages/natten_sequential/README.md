# natten_sequential — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `build_matrix.platforms`.
- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. NATTEN's documented build floor is sm_80 (Ampere). Pre-Ampere arches aren't listed as valid NATTEN_CUDA_ARCH values on natten.org/install. Hopper (9.0) auto-enables -DNATTEN_WITH_HOPPER_FNA=1 in setup.py. Blackwell DC (10.0/10.3) auto-enables -DNATTEN_WITH_BLACKWELL_FNA=1 and requires CUDA 12.8+. Blackwell consumer (12.0, sm_120 / RTX 5090) doesn't trigger any arch-specific NATTEN flags.
