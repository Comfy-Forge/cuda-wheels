"""Patch cumesh for Windows MSVC compatibility.

Fix: Move #if CUDART_VERSION directives outside CUDA_CHECK macro calls.
MSVC doesn't handle preprocessor directives inside macro arguments correctly.
"""
from pathlib import Path

atlas_file = Path("src/atlas.cu")
content = atlas_file.read_text()

# The issue: #if directives inside CUDA_CHECK() macro don't work on MSVC
# Solution: Define a type alias before the macro call, reuse it in both calls

old_block = '''    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
#if CUDART_VERSION >= 12090
        ::cuda::std::plus(),
#else
        cub::Sum(),
#endif
        M
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
#if CUDART_VERSION >= 12090
        ::cuda::std::plus(),
#else
        cub::Sum(),
#endif
        M
    ));'''

new_block = '''#if CUDART_VERSION >= 12090
    using ReduceOp = ::cuda::std::plus<>;
#else
    using ReduceOp = cub::Sum;
#endif
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
        ReduceOp(),
        M
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
        ReduceOp(),
        M
    ));'''

if old_block in content:
    content = content.replace(old_block, new_block)
    atlas_file.write_text(content)
    print("Fixed MSVC preprocessor issue in atlas.cu")
else:
    print("WARNING: Could not find expected code block in atlas.cu - may already be patched or source changed")

# --- CUDA 13.2 / torch 2.13 compatibility (found by the first cu132/2.13 run) ---

# (1) torch 2.13's headers use C++20 features (designated initializers,
# bit-field default member init). GCC tolerates them under c++17; MSVC and
# Windows nvcc hard-error. cumesh pins c++17 in four places in setup.py.
#
# GATED on the cells that need it (torch >= 2.13 or CUDA >= 13.2): applied
# unconditionally, this rewrite killed every Windows cu12.4/12.6 old-torch
# cell with cudafe++ 0xC0000409 -- under the pinned MSVC 14.29 host, nvcc
# rejects -std=c++20 ("flag will be ignored", demoting cudafe++ to C++17)
# yet still forwards /std:c++20 to the host compiler, and a C++17 EDG
# parsing a 14.29 STL forced into _MSVC_LANG=202002L fastfails. Old torch
# headers aren't C++20-clean either even on 14.4x hosts (ivalue_inl.h).
# Fallback is "0.0" so a missing env FAILS CLOSED (no rewrite): the safe
# mode is upstream's own c++17, proven green on every legacy cell.
import os as _os

def _mm(env: str) -> tuple[int, int]:
    parts = _os.environ.get(env, "0.0").split(".")[:2]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return (0, 0)

if _mm("CUW_TORCH_VERSION") >= (2, 13) or _mm("CUW_CUDA_VERSION") >= (13, 2):
    _setup = Path("setup.py")
    _t = _setup.read_text()
    _t2 = _t.replace("c++17", "c++20")
    if _t2 != _t:
        _setup.write_text(_t2)
        print("cumesh patch: setup.py c++17 -> c++20")
else:
    print("cumesh patch: keeping upstream c++17 (torch < 2.13, CUDA < 13.2)")

# (1a) On the CUDA < 12.6 Windows lanes (pinned MSVC 14.29), cumesh's
# /permissive- makes 'cuda' ambiguous between libcu++'s ::cuda and
# c10::cuda from torch <= 2.6 headers -- 303x C2872 across CUDA 12.4's
# bundled CUB/libcu++, whose 'cuda::std' references are unqualified
# (upstream CCCL qualified them to ::cuda::std later, which is why
# 12.6+ toolkits are immune). Strict conformance is a style choice, not
# a build requirement: drop /permissive- (and the /Zc:__cplusplus that
# only matters alongside it) on these lanes; everything else keeps
# upstream's flags byte-identical.
if _mm("CUW_CUDA_VERSION") < (12, 6):
    import re as _re2
    _setup = Path("setup.py")
    _t = _setup.read_text()
    # Drop the whole list element (string, trailing comma/space) in both the
    # cxx_flags and -Xcompiler= spellings; keep list syntax valid whether or
    # not the element is last (the flag lists end with a non-stripped flag).
    _t2 = _re2.sub(r'\s*"(?:-Xcompiler=)?/(?:permissive-|Zc:__cplusplus)"\s*,?', '', _t)
    if _t2 != _t:
        _setup.write_text(_t2)
        print("cumesh patch: dropped /permissive- + /Zc:__cplusplus for CUDA < 12.6 (C2872 'cuda' ambiguity)")

# (1b) The CUDA 12.9 clusterlaunchcontrol.h LLP64 fix that used to live
# here is now farm-wide in scripts/patch_cuda_toolkit.py, run by the
# setup-cuda action: it is NVIDIA's bug, and cubvh hit it too (2026-08-23).

# (2) CCCL 3.x (shipped with CUDA 13.2) removed DeviceScan::ExclusiveSum's
# 4-arg IN-PLACE overload (d_temp, bytes, d_data, num). The arguments then
# bind to the new env-based overload and produce garbage template errors
# (InputIteratorT=nullptr_t, NumItemsT=int*). The 5-arg form with
# d_in == d_out is still in-place and works on every CUDA line, so duplicate
# the data argument. Idempotent: 5-arg calls are left alone.
import re as _re

def _fix_inplace_exclusive_sum(text: str) -> tuple[str, int]:
    out, pos, fixed = [], 0, 0
    for m in _re.finditer(r"cub::DeviceScan::ExclusiveSum\(", text):
        start, depth, j = m.end(), 1, m.end()
        while depth and j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        argstr = text[start:j-1]
        args, d, last = [], 0, 0
        for k, ch in enumerate(argstr):
            if ch in "([":
                d += 1
            elif ch in ")]":
                d -= 1
            elif ch == "," and d == 0:
                args.append(argstr[last:k])
                last = k + 1
        args.append(argstr[last:])
        if len(args) == 4:
            data = args[2]
            new_argstr = ",".join([args[0], args[1], data, " " + data.strip(), args[3]])
            out.append(text[pos:start]); out.append(new_argstr); out.append(")")
            pos = j
            fixed += 1
    out.append(text[pos:])
    return "".join(out), fixed

_total = 0
for _f in ["src/shared.h", "src/clean_up.cu", "src/connectivity.cu",
           "src/remesh/svox2vert.cu", "src/simplify.cu", "src/atlas.cu"]:
    _path = Path(_f)
    if not _path.exists():
        continue
    _text = _path.read_text()
    _new, _n = _fix_inplace_exclusive_sum(_text)
    if _n:
        _path.write_text(_new)
        _total += _n
        print(f"cumesh patch: {_f}: {_n} in-place ExclusiveSum call(s) -> 5-arg form")
if _total == 0:
    raise SystemExit("cumesh patch: no in-place ExclusiveSum calls found -- "
                     "the CCCL-3.2 breakage lives in src/shared.h; upstream changed?")
print(f"cumesh patch: {_total} CCCL-3.x call sites fixed")
