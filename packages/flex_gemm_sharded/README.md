# flex_gemm_sharded — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `build_matrix.platforms`. Sharded build path is Linux-only in this iteration. Windows compile-shard adapter not implemented yet; restricting platforms prevents the matrix from generating windows entries for this test bed.
