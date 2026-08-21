# torchao — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `min_pytorch`.
- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. setup.py inspects _get_cuda_arch_flags() for sm90a/sm100a cutlass paths.

## OPEN: CUDA extensions did not build on the farm (2026-08-21)

The first farm build produced `torchao-0.18.0+git5f2baf9d5-py3-none-any.whl`
-- pure python, git-suffixed version, no CUDA extensions -- despite
`USE_CPP` defaulting to 1 and `use_cuda = torch.version.cuda and CUDA_HOME`
both holding in the build env. The verify gate blocked it. Needs a proper
read of the build log to see which of setup.py's skip branches fired
(v0.18.0 setup.py:505-525) before the next attempt.
