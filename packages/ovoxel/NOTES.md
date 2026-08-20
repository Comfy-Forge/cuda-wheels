# ovoxel / ovoxel_vb — build notes
- **Source:** microsoft/TRELLIS.2 / PozzettiAndrea/Trellis.2.sparseflex
- **Quirks:** `build_subdir: o-voxel`. Recursive clone. Patch removes git URL deps, adds batched BVH queries (batch_size=500000) to avoid kernel timeout. MSVC fixes for double literals and size_t narrowing.
