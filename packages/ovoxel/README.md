# ovoxel / ovoxel_vb — build notes
- **Source:** microsoft/TRELLIS.2 / PozzettiAndrea/Trellis.2.sparseflex
- **Quirks:** `build_subdir: o-voxel`. Recursive clone. Patch removes git URL deps, adds batched BVH queries (batch_size=500000) to avoid kernel timeout. MSVC fixes for double literals and size_t narrowing.

## Curated Requires-Dist

Upstream declares `cumesh`/`flex_gemm` as git URLs; our patch made them
bare names so the build works, but bare names resolve against PyPI, not
the farm. `requires_dist` in package.yml replaces the list with runtime
deps plus exact local-version sibling pins (`cumesh==0.0.1+cu128torch2.8`
style) -- PyPI forbids local versions, so those pins resolve from our
index or fail loudly.
