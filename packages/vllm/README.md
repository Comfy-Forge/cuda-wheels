# vllm — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `build_matrix.combinations`, `build_matrix.platforms`. vLLM has no Windows support upstream -- no win32 branch anywhere in setup.py.
