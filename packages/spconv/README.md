# spconv — build notes
- **Source:** traveller59/spconv (v2.3.8)
- **Quirks:** Depends on cumm (`extra_deps: "pccm>=0.4.16 ccimport>=0.4.4 cumm"`). Patch adds BF16 sparse convolution (Simt fallback + Ampere TensorOp + ImplGemm params).

## Kernels are JIT-compiled at runtime (no baked SASS)

The shipped wheels contain only an sm_52 placeholder in `core_cc`; the real
kernels are compiled at runtime through the vendored NVRTC. This was
verified against the legacy farm's wheels too -- it has always been so,
and users work because of the JIT path. OPEN QUESTION: the build exports
`CUMM_DISABLE_JIT=1` intending AOT baking, which has silently never
happened; if AOT is ever fixed upstream-side, drop `verify.skip_arch`.

## Curated Requires-Dist

Upstream leaks its build stack (pccm, ccimport, pybind11, fire) as
runtime deps and pins `cumm<0.8.0` -- which our own farm cumm (0.8.2)
does not satisfy. `requires_dist` in package.yml replaces the list with
numpy plus an exact local-version pin on the farm's cumm.
