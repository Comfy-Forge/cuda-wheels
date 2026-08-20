# pyg_lib — build notes
- **Source:** pyg-team/pyg-lib (0.5.0)
- **Quirks:** Recursive clone. Patch (`patches/pyg_lib.py`): neutralizes legacy `libnvToolsExt` references in torch 2.4-2.6's CMake config -- CUDA >= 12.5 dropped that library, so `find_package(Torch)` otherwise blows up at configure time.

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list`. 
