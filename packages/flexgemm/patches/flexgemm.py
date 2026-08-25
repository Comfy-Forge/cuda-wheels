"""Patch FlexGEMM for Windows compatibility.

Fix: Make triton dependency platform-specific (triton on Linux, triton-windows on Windows).
"""
import re
from pathlib import Path

pyproject = Path("pyproject.toml")
content = pyproject.read_text()

# Replace triton dependency with platform-specific versions
# "triton>=X.Y.Z" -> "triton>=X.Y.Z; platform_system != 'Windows'", "triton-windows>=X.Y.Z; platform_system == 'Windows'"
content = re.sub(
    r'"triton(>=[\d.]+)"',
    r'"triton\1; platform_system != \'Windows\'", "triton-windows\1; platform_system == \'Windows\'"',
    content
)

pyproject.write_text(content)
print("Patched pyproject.toml for triton-windows compatibility")


# ── MSVC: dependent template-arg member calls need `template` ────────────
# hash.cu / sparse_neighbor_map.cu call `x.data_ptr<T>()` (and `item<T>`)
# where T is deduced inside a generic lambda (`using T = decltype(tag)`),
# making it a dependent name. GCC/Clang accept the shorthand; MSVC's
# parser rejects it ("type name is not allowed", wave-2 Windows failure
# at sparse_neighbor_map.cu:327). `.template member<T>()` is the
# conforming spelling and is valid on every compiler, so it is applied
# unconditionally. Idempotent: already-patched calls are skipped.
#
# WIDENED 2026-08-25: the old pattern only matched `<T>`/`<K>`, so it fixed
# 114/128/327/341 and left 309/313/326/342 -- which spell CONCRETE types
# (`<int64_t>`, `<int32_t>`) -- untouched. Those failed identically on
# Windows torch2.12.1 cu12.6/cu13.0 (10 cells). What makes a call dependent
# is the OBJECT, not the template argument: `expanded_size`/`expanded_start`/
# `out_coords` are `auto` variables deduced from dependent expressions, so
# they need `.template` no matter how the argument is spelled. Match any
# single type-name argument. Harmless where the object turns out to be
# non-dependent: `data_ptr`/`item` are member templates of Tensor, so
# `.template` is well-formed on every compiler either way.
import re as _re
from pathlib import Path as _P

_fixed = 0
for _f in [_P("flex_gemm/kernels/cuda/hash/hash.cu"),
           _P("flex_gemm/kernels/cuda/spconv/sparse_neighbor_map.cu")]:
    _s = _f.read_text()
    _new, _n = _re.subn(
        r"(?<!template )\.(?!template )(data_ptr|item)<([A-Za-z_][\w:]*)>",
        r".template \1<\2>", _s)
    if _n:
        _f.write_text(_new)
        print(f"  {_f}: {_n} dependent member call(s) got `.template`")
        _fixed += _n
if _fixed == 0:
    raise SystemExit("flexgemm patch: no data_ptr<T>/item<T> calls found -- "
                     "upstream changed; update this patch")
print(f"flexgemm patch: {_fixed} MSVC dependent-template fixes")

# ── C++ standard + triton compatibility (review board 2026-08-24) ───────
# 1. setup.py hardcodes the C++ standard, which overrides torch's own choice
#    (cpp_extension only appends one when the caller supplied none). Pinned
#    at c++17 that breaks torch >= 2.13 (whose headers need C++20); pinned at
#    c++20 it breaks torch < 2.7 (nvcc's EDG misparses ivalue_inl.h). Drop it.
# 2. The Autotuner subclass forwards 13 positional args to triton's base
#    __init__, which triton 3.0/3.1 (torch 2.4/2.5) does not accept. That is
#    a real runtime defect, not a CI artifact, so it is FIXED here rather
#    than forgiven by the verify gate.
import os as _os
import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (fix_triton_autotuner_super_auto, require, strip_std_flags,
                       translate_cxx_flags_for_msvc)

_setup = Path("setup.py")
_t = _setup.read_text()
_t, _n_std = strip_std_flags(_t)
require(_n_std > 0,
        "no hardcoded C++-standard flag in setup.py -- upstream changed; "
        "refusing to build against an unverified flag set")
_setup.write_text(_t)
print(f"patch: dropped {_n_std} hardcoded std flag(s); torch now selects it")

# Locate the autotuner wherever the fork put it: vb renames the package
# directory before this runs, and the ap fork has no Autotuner subclass at
# all. The helper fails loud only if a file exists with an unrecognised
# super() forward.
_n_at = fix_triton_autotuner_super_auto(".")
print(f"patch: triton Autotuner handling done ({_n_at} file(s) rewritten)")
