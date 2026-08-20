# torchao — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `min_pytorch`.
- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. setup.py inspects _get_cuda_arch_flags() for sm90a/sm100a cutlass paths.
