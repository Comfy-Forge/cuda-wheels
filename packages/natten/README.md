# natten — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`pcto_override.yml`** — overrides the cell axes (which cuda/torch/python/platform combos build): `min_pytorch`. NATTEN v0.21.6's setup.py line 53 hard-asserts `torch_ver >= [2, 5]`; any cu12.4/torch2.4.x dispatch fails immediately at metadata-generation with "AssertionError: NATTEN only supports PyTorch >= 2.5". Floor was bumped from 2.0 -> 2.5 around NATTEN v0.20.0. Skip those combos.
- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. NATTEN's documented build floor is sm_80 (Ampere). Pre-Ampere arches aren't listed as valid NATTEN_CUDA_ARCH values on natten.org/install, even though the runtime check at backends/configs/checks.py is permissive to sm_60. We track NATTEN's own supported matrix, not the runtime gate. Hopper (9.0) auto-enables -DNATTEN_WITH_HOPPER_FNA=1 in setup.py. Blackwell DC (10.0/10.3) auto-enables -DNATTEN_WITH_BLACKWELL_FNA=1 and requires CUDA 12.8+ — hence omitted from cu124/cu126 rows. Stripped from Windows builds via patches/natten.py (MSVC can't parse the sm_100 FMHA backward kernel templates).
