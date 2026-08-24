"""Patch FlexGEMM-ap for the farm build matrix.

The ap fork already carries the MSVC dependent-template fixes vanilla
FlexGEMM needs, so this patch only covers the two farm-matrix issues:
the hardcoded C++ standard and the triton Autotuner signature.
"""
from pathlib import Path

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
from patch_lib import (fix_triton_autotuner_super, require, strip_std_flags,
                       translate_cxx_flags_for_msvc)

_setup = Path("setup.py")
_t = _setup.read_text()
_t, _n_std = strip_std_flags(_t)
require(_n_std > 0,
        "no hardcoded C++-standard flag in setup.py -- upstream changed; "
        "refusing to build against an unverified flag set")
_setup.write_text(_t)
print(f"patch: dropped {_n_std} hardcoded std flag(s); torch now selects it")

_n_at = fix_triton_autotuner_super(_pl.Path("flex_gemm/utils/autotuner.py"))
require(_n_at > 0,
        "triton Autotuner super().__init__ forward not found in "
        "flex_gemm/utils/autotuner.py -- the wheel would fail to import "
        "against triton 3.0/3.1")
print("patch: triton Autotuner call is now signature-filtered")
