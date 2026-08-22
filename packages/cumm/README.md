# cumm — build notes
- **Source:** FindDefinition/cumm (v0.8.2)
- **Quirks:** Generates BF16 GEMM kernels at build time. Needs `pccm>=0.4.16 ccimport>=0.4.4`. Patch adds BF16 Ampere TensorOp + Simt fallback configs, fixes pybind11 `zero_whole_storage_` binding.

## Kernels are JIT-compiled at runtime (no baked SASS)

The shipped wheels contain only an sm_52 placeholder in `core_cc`; the real
kernels are compiled at runtime through the vendored NVRTC. This was
verified against the legacy farm's wheels too -- it has always been so,
and users work because of the JIT path. OPEN QUESTION: the build exports
`CUMM_DISABLE_JIT=1` intending AOT baking, which has silently never
happened; if AOT is ever fixed upstream-side, drop `verify.skip_arch`.

## Curated Requires-Dist

Upstream leaks pccm/pybind11 (codegen-time) and fire (CLI helper) as
runtime deps. `requires_dist` in package.yml trims to numpy + sympy.

## Overrides

`arch_override.yml`: cumm 0.8.2's arch table does not know 11.0 (Thor),
which the aarch64 policy adds on CUDA 13.x -- dropped there so setup
parses. Inert for wheel content (runtime NVRTC JIT, no AOT SASS).
