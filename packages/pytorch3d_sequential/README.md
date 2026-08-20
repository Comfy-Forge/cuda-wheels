# pytorch3d_sequential — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `build_matrix.platforms`. nvcc_flags: --threads=1 was DROPPED for Windows. Run 26037716007 (Windows cu13.0 torch2.11 py3.10) failed at LINK with LNK1120: 7 unresolved externals for pulsar template specializations (e.g. norm_sphere_gradients<1>). Canonical pytorch3d ships ~81 Windows wheels for the exact same matrix cell, so it's not an upstream incompatibility -- it's something we introduced. The only emitted-code-affecting diff was --threads=1 (max_jobs is a ninja knob, can't change per-file nvcc output). Documented as a parallelism knob, but it's the only changing variable, so drop it and observe. If Linux now OOMs without --threads=1 we'll need a per-platform `nvcc_flags` mechanism (currently flat in the YAML schema).
