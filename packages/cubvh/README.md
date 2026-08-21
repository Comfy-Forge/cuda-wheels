# cubvh — build notes
- **Source:** ashawkey/cubvh
- **Quirks:** Recursive clone. Patch (`patches/cubvh.py`): bumps `cpp_standard` 17 -> 20 -- torch 2.13's headers use C++20 features that MSVC/Windows-nvcc hard-error on under c++17.

## Curated Requires-Dist

AOT-compiled wheel: upstream's ninja/pybind11 are build-time leakage.
`requires_dist` in package.yml trims to torch, numpy, trimesh, kiui.
