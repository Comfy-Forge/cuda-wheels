# spconv — build notes
- **Source:** traveller59/spconv (v2.3.8)
- **Quirks:** Depends on cumm (`extra_deps: "pccm>=0.4.16 ccimport>=0.4.4 cumm"`). Patch adds BF16 sparse convolution (Simt fallback + Ampere TensorOp + ImplGemm params).

## Kernels ARE baked ahead of time (this section was wrong until 2026-08-24)

This previously read "the shipped wheels contain only an sm_52 placeholder;
the real kernels are compiled at runtime through the vendored NVRTC". Every
clause was false, and the error propagated into `verify.skip_arch` and
`arch_override.yml`, which between them hid a real regression.

Measured with `cuobjdump --list-elf` on the published aarch64 wheel's
`core_cc` (116 MB): **478 `sm_80` + 478 `sm_90` real cubins**, plus 221
`sm_52` entries that are 872-byte artifacts of nvcc's default-arch pass over
device-code-free translation units (`readelf -S` shows no `.text` at all).
Counting only those 221 and stopping is what produced the "placeholder" myth.

There is **no libnvrtc in any spconv wheel** — `readelf -d` on `core_cc`
gives `libcudart, libstdc++, libm, libgcc_s, libc` and nothing else, with no
RPATH. The 99 MB NVRTC belongs to **cumm**, and only on Linux (Windows cumm
wheels are 1.3 MB and resolve `nvrtc64_*.dll` at import).

`CUMM_DISABLE_JIT=1` is unrelated to AOT baking either way: cumm reads it in
one place, to suppress the *editable-install* pccm rebuild. Gencode comes
unconditionally from `get_cuda_arch_flags()`.

`verify.skip_arch` has been removed, so C7 now checks these wheels for real.

## Blackwell coverage was lost, and is being restored

The legacy farm built spconv against cumm **0.8.2** and shipped
`sm_70/80/90/100/120`. The farm pins cumm **0.7.11** (0.8.2's GEMM headers
are uncompilable by spconv's harness), and 0.7.11's `supported_arches` table
simply stops at 9.0 — Blackwell tokens raise `Unknown CUDA arch`. The arch
list was therefore trimmed, and the trim was documented as "inert for wheel
content". It was not: today's wheels carry no Blackwell SASS where the
legacy ones did.

Fixed at the source rather than by narrowing policy:
`packages/cumm/patches/cumm.py` restores `'10.0'/'11.0'/'12.0'` to that table
(exactly what 0.8.2 carries), and `arch_override.yml` is back on the policy
rows. **Ordering matters** — spconv installs the farm's own cumm *wheel*, so
cumm must be rebuilt and republished before spconv can use the wider list.

## Curated Requires-Dist

Upstream leaks its build stack (pccm, ccimport, pybind11, fire) as
runtime deps and pins `cumm<0.8.0` -- which our own farm cumm (0.8.2)
does not satisfy. `requires_dist` in package.yml replaces the list with
numpy plus an exact local-version pin on the farm's cumm.

## Overrides

`arch_override.yml`: cumm 0.7.11 (upstream's pinned codegen stack)
rejects Blackwell arch tokens ("Unknown CUDA arch (10.0)"), so the arch
list is trimmed to cumm-known archs. Inert for the wheel content: spconv
bakes no AOT SASS (runtime NVRTC JIT, `verify.skip_arch`).

`pcto_override.yml`: no aarch64 -- the required cumm<0.8.0 has no
aarch64 wheels on PyPI (and upstream never shipped ARM spconv).
