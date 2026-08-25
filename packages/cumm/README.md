# cumm — build notes
- **Source:** FindDefinition/cumm (v0.7.11 -- package.yml has pinned 0.7.11 since the 2026-08-22 revert; this line said v0.8.2 and was wrong)
- **Quirks:** Generates BF16 GEMM kernels at build time. Needs `pccm>=0.4.16 ccimport>=0.4.4`. Patch adds BF16 Ampere TensorOp + Simt fallback configs, fixes pybind11 `zero_whole_storage_` binding.

## Kernels are JIT-compiled at runtime (no baked SASS)

The shipped wheels contain **no device code at all** in `core_cc` -- not
even a placeholder. `cuobjdump --list-elf` and `--list-ptx` both answer
"does not contain device code" for the Linux .so *and* the Windows .pyd
(checked 2026-08-25 on cu130; the earlier "only an sm_52 placeholder"
claim in this file was wrong). The real kernels are generated and compiled
at run time through NVRTC. OPEN QUESTION: the build exports
`CUMM_DISABLE_JIT=1` intending AOT baking, which has silently never
happened; if AOT is ever fixed upstream-side, drop `verify.skip_arch`.

## The Windows wheel cannot load without a CUDA Toolkit (OPEN DEFECT)

The NVRTC that makes the JIT path work is **only vendored on Linux**.
auditwheel grafts `cumm.libs/libnvrtc.so.13` (~109MB) and
`libnvrtc-builtins.so.13` (~6MB) into the Linux wheels. The Windows wheel
bundles nothing -- one `.pyd`, zero DLLs -- while
`cumm/core_cc.cp*-win_amd64.pyd` carries a **load-time** PE import of
`nvrtc64_130_0.dll` (`objdump -p`: it sits in the import table, and the
Delay Import Directory is empty). On a Windows box without CUDA Toolkit
13.0 on PATH, `import cumm` raises
`ImportError: DLL load failed while importing core_cc`. `spconv`'s Windows
wheel inherits this -- its `__init__` loads cumm first.

CI does not catch it because the Windows runner has the toolkit installed,
so the gate's import test resolves the DLL from PATH. The fix is DLL
vendoring (`delvewheel`), which no wheel in this farm does yet; see the
`skip_arch.windows` note in package.yml. The arch waiver does **not** cover
this.

## Curated Requires-Dist

Upstream leaks pccm/pybind11 (codegen-time) and fire (CLI helper) as
runtime deps. `requires_dist` in package.yml trims to numpy + sympy.

## Overrides

`arch_override.yml`: cumm 0.7.11's arch table ends at 9.0 (plus 8.7)
-- Blackwell and Thor tokens are trimmed on every lane so setup parses.
Inert for wheel content (runtime NVRTC JIT, no AOT SASS).
