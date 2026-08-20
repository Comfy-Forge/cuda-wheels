# nunchaku — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `min_pytorch`.
- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. INT4/FP4 SVDQuant kernels need Ampere or newer; FP4 paths need Blackwell.
