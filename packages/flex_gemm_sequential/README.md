# flex_gemm_sequential — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `build_matrix.platforms`. Sequential-checkpoint path is Linux-only for the POC; Windows version of the timeout/trigger plumbing is deferred until the Linux mechanism is proven.
