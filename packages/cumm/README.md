# cumm — build notes
- **Source:** FindDefinition/cumm (v0.8.2)
- **Quirks:** Generates BF16 GEMM kernels at build time. Needs `pccm>=0.4.16 ccimport>=0.4.4`. Patch adds BF16 Ampere TensorOp + Simt fallback configs, fixes pybind11 `zero_whole_storage_` binding.
