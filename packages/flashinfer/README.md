# flashinfer — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — restricts `build_matrix.platforms` to linux+windows only: the AOT kernel-cache build takes hours and runs through the sequential-checkpoint chain, which has no aarch64 lane. The torch-axis collapse itself needs no override — `links_torch: false` triggers it automatically in generate_matrix.
