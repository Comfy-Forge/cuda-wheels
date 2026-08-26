"""Patch flex_gemm_vb (visualbruno fork) for wheel building:
1. Rename package from flex_gemm to flex_gemm_vb to avoid conflicts
2. Update autotuner cache path and walk_package references

Note: visualbruno's fork already has triton-windows platform-specific deps
      and MSVC-compatible CXX_FLAGS in setup.py.
"""
from pathlib import Path

# --- 1. Rename package to flex_gemm_vb ---

# pyproject.toml
pyproject = Path("pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "flex_gemm"', 'name = "flex_gemm_vb"')
pyproject.write_text(content)
print("Renamed package to flex_gemm_vb in pyproject.toml")

# setup.py
setup_file = Path("setup.py")
content = setup_file.read_text()
content = content.replace('name="flex_gemm"', 'name="flex_gemm_vb"')
# Replace package list entries
content = content.replace('"flex_gemm"', '"flex_gemm_vb"')
content = content.replace('"flex_gemm.', '"flex_gemm_vb.')
# Replace source file paths (flex_gemm/kernels/cuda/...)
content = content.replace('"flex_gemm/', '"flex_gemm_vb/')
# Replace cache path
content = content.replace('~/.flex_gemm', '~/.flex_gemm_vb')
setup_file.write_text(content)
print("Renamed package to flex_gemm_vb in setup.py")

# Rename the actual package directory
src_dir = Path("flex_gemm")
dst_dir = Path("flex_gemm_vb")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed flex_gemm/ directory to flex_gemm_vb/")

# --- 2. Update autotuner walk_package references ---
autotuner = dst_dir / "utils" / "autotuner.py"
if autotuner.exists():
    content = autotuner.read_text()
    content = content.replace("walk_package('flex_gemm'", "walk_package('flex_gemm_vb'")
    autotuner.write_text(content)
    print("Updated walk_package references in autotuner.py")

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
from patch_lib import fix_triton_autotuner_super_auto, require, strip_std_flags

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
