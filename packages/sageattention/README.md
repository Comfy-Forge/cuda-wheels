# sageattention — build notes
- **Source:** thu-ml/SageAttention (v2.2.0)
- **Quirks:** `max_jobs: 2`. Patch fixes arch parser (space-separated TORCH_CUDA_ARCH_LIST), MSVC CXX flags, Windows ABI guard, nvcc `--threads` -> 1, sm_90a-only gencode for _qattn_sm90 (wgmma cannot target anything else).
- **Arch list:** cu124-cu126 `"8.0 8.6 8.9 9.0"`; cu128+ `"8.0 8.6 8.9 9.0 12.0+PTX"`; ARM `"8.0 9.0 12.0+PTX"`. See `arch_override.yml` for the authoritative rows.

**No sm_100 (B200/GB200), and it is an upstream gap, not a farm trim.**
Corrected 2026-08-24 — this line previously claimed cu128+ used
`"... 9.0 10.0 12.0"`. Upstream `setup.py`'s capability map whitelists only
8.0/8.6/8.9/9.0/12.0 and silently `continue`s on anything else, so no sm_100
cubin can be produced; and `sageattention/core.py`'s `sageattn()` dispatches
on `sm{major}{minor}` and raises `ValueError: Unsupported CUDA architecture`
for anything outside {sm80, sm86, sm89, sm90, sm120}. On a B200 the package
therefore fails in Python before any kernel launch. Restoring sm_100 needs
three coupled source edits (setup.py map, core.py dispatch, arch rows), not
an arch-list change — tracked, not done.

**PTX caveat:** the cu124/cu126 rows resolve to `9.0+PTX`, which upstream
maps to `90a`. `sm_90a.ptx` is arch-conditional and loads only on sm_90, so
those wheels have no forward-compat JIT tail despite the `+PTX` request. C7
does not currently notice: `arch_list_to_sm` discards the `+PTX` suffix.

**CUDA-version-specific behavior:** Needs CUDA >= 12.0; SM 12.0 (Blackwell) needs >= 12.8. No upper CUDA bound.

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. Must include 8.9 (Ada / RTX 40-series) — sage 2.x's _qattn_sm89 extension contains Ada-only FP8 mma instructions gated on __CUDA_ARCH__ >= 890. Without an sm_89 cubin the loader falls back to sm_80 binary (forward-compatible Ampere → Ada), where those gated paths compile to stubs and `cudaErrorLaunchFailure` surfaces at runtime on real Ada hardware. 8.6 added too so Ampere consumer cards (3090/3080/3070/3060) get a native cubin instead of relying on JIT.
