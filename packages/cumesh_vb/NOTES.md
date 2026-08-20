# cumesh_vb — build notes
- **Source:** visualbruno/CuMesh
- **Quirks:** Same CUDA components as cumesh. Patch fetches Eigen submodule, renames package to cumesh_vb, MSVC CXX/NVCC flag fixes. Forces C++17 for nvcc (CUB bugs in CUDA 12.4 with C++20).
