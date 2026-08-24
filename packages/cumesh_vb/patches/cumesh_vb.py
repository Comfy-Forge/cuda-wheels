"""Patch cumesh_vb (visualbruno fork) for wheel building:
1. Fetch missing Eigen submodule (cubvh committed directly, not as git submodule)
2. Rename package from cumesh to cumesh_vb to avoid conflicts
3. Fix GCC-only CXX_FLAGS for Windows MSVC builds
"""
import os
import subprocess
from pathlib import Path

# --- 0. Fetch Eigen (nested submodule not auto-fetched) ---
# visualbruno committed third_party/cubvh directly into the tree rather than
# as a git submodule. The cubvh directory has its own .gitmodules pointing to
# eigen, but since cubvh isn't a submodule, --recursive doesn't fetch it.
eigen_dir = Path("third_party/cubvh/third_party/eigen")
if not eigen_dir.exists() or not any(eigen_dir.iterdir()):
    eigen_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "3.4.0",
         "https://gitlab.com/libeigen/eigen.git", str(eigen_dir)],
        check=True,
    )
    print(f"Cloned Eigen 3.4.0 into {eigen_dir}")
    # PINNED (review board 2026-08-24): the clone tracked master,
    # so every build got that day's Eigen -- unreproducible, and
    # current master hard-#errors below sm_70, which took out all
    # 96 cu12.4/12.6 cells. 3.4.0 has no such floor (plain cumesh
    # ships those cells today with the pinned tarball).

# --- 1. Rename package to cumesh_vb ---

# pyproject.toml
pyproject = Path("pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "cumesh"', 'name = "cumesh_vb"')
pyproject.write_text(content)
print("Renamed package to cumesh_vb in pyproject.toml")

# setup.py
setup_file = Path("setup.py")
content = setup_file.read_text()
content = content.replace('name="cumesh"', 'name="cumesh_vb"')
content = content.replace("'cumesh'", "'cumesh_vb'")
content = content.replace('name="cumesh._C"', 'name="cumesh_vb._C"')
content = content.replace("name='cumesh._cubvh'", "name='cumesh_vb._cubvh'")
content = content.replace("name='cumesh._xatlas'", "name='cumesh_vb._xatlas'")
setup_file.write_text(content)
print("Renamed package to cumesh_vb in setup.py")

# Rename the actual package directory
src_dir = Path("cumesh")
dst_dir = Path("cumesh_vb")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed cumesh/ directory to cumesh_vb/")

# --- 2. Fix CXX/NVCC flags for Windows ---
# MSVC: -O3 -> /O2, -std=c++20 -> /std:c++17
# nvcc on Windows: c++20 triggers cub header bugs in CUDA 12.4, downgrade to c++17
# (Review board 2026-08-24.) This used to force /std:c++17 on Windows to
# dodge CUDA 12.4 cub header bugs. That pin overrides torch's own choice and
# broke every torch >= 2.13 Windows cell, whose headers require C++20. Drop
# the opinion (cpp_extension picks the standard per torch) and keep only the
# genuinely MSVC-specific bits: /O2 for -O3, and --extended-lambda for nvcc.
import os as _os
import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (strip_std_flags, translate_cxx_flags_for_msvc,
                       require, fix_inplace_exclusive_sum_in_files)

content = setup_file.read_text()
content, _n_std = strip_std_flags(content)
require(_n_std > 0, "no hardcoded C++-standard flag in setup.py -- upstream changed")
_n_msvc = 0
if _os.name == "nt":
    content, _n_msvc = translate_cxx_flags_for_msvc(content)
    require(_n_msvc > 0, "no GCC cxx flags translated for MSVC -- block moved")
    content = content.replace('"nvcc": ["-O3",', '"nvcc": ["-O3", "--extended-lambda",')
setup_file.write_text(content)
print(f"Patched flags: dropped {_n_std} std flag(s), {_n_msvc} MSVC translation(s)")


# --- CCCL 3.x (CUDA 13.2) removed the 4-arg in-place ExclusiveSum ---------
# Ported from cumesh via the shared helper (review board 2026-08-24): this
# fork carries the same call sites and lost all 32 cu13.2 cells without it.
_n_cub = fix_inplace_exclusive_sum_in_files(
    ["src/shared.h", "src/atlas.cu", "src/simplify.cu", "src/connectivity.cu",
     "src/clean_up.cu", "src/remesh/svox2vert.cu"],
    required=True)
print(f"cumesh_vb patch: {_n_cub} CCCL-3.x call site(s) fixed")
