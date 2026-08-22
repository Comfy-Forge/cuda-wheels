# flashinfer — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — restricts `build_matrix.platforms` to linux+windows only: the AOT kernel-cache build takes hours and runs through the sequential-checkpoint chain, which has no aarch64 lane. The torch-axis collapse itself needs no override — `links_torch: false` triggers it automatically in generate_matrix.

## Package name is flashinfer_jit_cache

Upstream ships TWO dists: `flashinfer-python` (pure python, PyPI) and
`flashinfer-jit-cache` (the AOT-compiled kernel cache -- what this farm
builds, from build_subdir flashinfer-jit-cache). The package name matches
the actual dist: the generate_matrix perpetual-rebuild warning caught the
mismatch (wheels named flashinfer_jit_cache-* could never satisfy a
"flashinfer-" prefix check, which is why this package never released).
Users install BOTH: flashinfer-python from PyPI + this wheel from the farm.

## Overrides

`arch_override.yml`: flashinfer 0.6.17 refuses SM 12.x below CUDA 12.9,
so cu12.8 drops Blackwell-consumer (upstream's cu128 wheels do the same).
