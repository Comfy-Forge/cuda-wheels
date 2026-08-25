"""Patch cumesh_vb (visualbruno fork) for wheel building:
1. Fetch missing Eigen submodule (cubvh committed directly, not as git submodule)
2. Rename package from cumesh to cumesh_vb to avoid conflicts
3. Fix GCC-only CXX_FLAGS for Windows MSVC builds
4. Port the CCCL-3.x in-place ExclusiveSum fix (CUDA 13.2)
5. Make every pybind11 py::class_ registration module_local() so this fork
   can be imported alongside plain `cumesh` in one interpreter
"""
import os
import subprocess
from pathlib import Path

# --- 0. Fetch Eigen (nested submodule not auto-fetched) ---
# visualbruno committed third_party/cubvh directly into the tree rather than
# as a git submodule. The cubvh directory has its own .gitmodules pointing to
# eigen, but since cubvh isn't a submodule, --recursive doesn't fetch it.
#
# EIGEN_PIN: the exact commit JeffreyXiang/cubvh @ ce92267 pins as its own
# `third_party/eigen` submodule. Plain `cumesh` gets this for free (its cubvh
# IS a submodule, so --recursive walks in and fetches it); the fork gets
# nothing, so we reproduce the pin by hand. Keeping the two identical is the
# whole point -- same cubvh sources compiled against the same Eigen.
#
# Do NOT "simplify" this to --branch 3.4.0. Eigen 3.4.0's arg_default_impl
# reaches `arg` through EIGEN_USING_STD, which expands to `using ::arg;` on the
# nvcc device pass -- and MSVC has no global ::arg. That is the C7555-adjacent
# break that killed all 30 Windows x torch>=2.12 cells. Eigen fixed it after
# 3.4.0 by hardcoding `using std::arg;` on the MSVC>=1920 branch
# (MathFunctions.h, "There is no official ::arg on device in CUDA/HIP").
#
# Do NOT track master either: current master hard-#errors below sm_70, which
# previously took out all 96 cu12.4/12.6 cells. e63d9f6 is 2024-03-29 and has
# no such floor.
EIGEN_PIN = "e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b"

eigen_dir = Path("third_party/cubvh/third_party/eigen")
if not eigen_dir.exists() or not any(eigen_dir.iterdir()):
    eigen_dir.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", "https://gitlab.com/libeigen/eigen.git"],
        ["git", "fetch", "-q", "--depth", "1", "origin", EIGEN_PIN],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(cmd, cwd=eigen_dir, check=True)
    print(f"Fetched Eigen {EIGEN_PIN[:10]} (cubvh's own pin) into {eigen_dir}")

# Belt and braces: prove we got the fixed tree, not a 3.4.0 fallback.
_mf = eigen_dir / "Eigen/src/Core/MathFunctions.h"
if _mf.exists() and "There is no official ::arg on device" not in _mf.read_text(
    encoding="utf-8", errors="replace"
):
    raise SystemExit(
        f"{_mf}: Eigen is missing the MSVC device-side `arg` fix. "
        f"Expected the tree at {EIGEN_PIN}; got something older (3.4.0?). "
        "Windows x torch>=2.12 will fail with 'the global scope has no \"arg\"'."
    )

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
                       require, fix_inplace_exclusive_sum_in_files,
                       add_pybind_module_local)

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


# --- 3. Make every pybind11 class registration module-local ---------------
# `cumesh` and `cumesh_vb` cannot be imported into the same interpreter:
#
#     ImportError: generic_type: type "CuMesh" is already registered!
#
# Both orders fail, and all three of this fork's extensions are affected.
# The Python-level rename above (cumesh -> cumesh_vb) does nothing about it:
# the C++ types keep their upstream identity in both builds, and the two .so
# sets carry the IDENTICAL pybind11 internals ID -- verified on the shipped
# cu130/torch2.11 wheels, where `strings` reports
# `__pybind11_internals_v11_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1__`
# in cumesh/_C and cumesh_vb/_C alike -- so they share ONE global type
# registry and both call py::class_<...>(m, "<same name>").
#
# The collision set at the pinned tag (d10e54c) is seven registrations:
#   src/ext.cpp                        CuMesh
#   third_party/cubvh/src/bindings.cpp cuBVH, cuHashTable, HashTable
#   third_party/xatlas/binding.cpp     ChartOptions, PackOptions, Atlas
# (the cubvh three also collide with the standalone `cubvh` package, which
# this fixes for free on the fork's side.)
#
# Fixed HERE, in the fork's patch, not in plain `cumesh`: this is exactly the
# "two node packs on different forks" case the fork exists to serve, and the
# upstream package should keep its types global so nothing else that depends
# on plain cumesh changes behaviour.
#
# module_local() over a distinct PYBIND11_INTERNALS_ID: see the long rationale
# on add_pybind_module_local() in scripts/patch_lib.py. Short form -- the
# internals route is a process-wide ABI split (severing the fork from torch's
# own pybind11 internals) to solve a seven-name clash, pybind11 v3 hard-errors
# below INTERNALS_VERSION 11 so the only free numbers are ones pybind11 will
# itself claim later, and it must be defined in every TU of all three
# extensions or it degrades silently. module_local() is confined to the
# registration sites and is what pybind11 documents for this case.
#
# It is free here because no bound signature crosses extension boundaries:
# bvh.py talks only to _cubvh, xatlas.py only to _xatlas, cumesh.py only to
# _C, and every class_ is constructed and consumed inside the module that
# registers it (checked at d10e54c).
_ml = add_pybind_module_local({
    "src/ext.cpp": 1,
    "third_party/cubvh/src/bindings.cpp": 3,
    "third_party/xatlas/binding.cpp": 3,
})
require(sum(_ml.values()) == 7,
        "cumesh_vb: expected 7 module-local pybind11 class registrations, "
        f"got {sum(_ml.values())} {_ml} -- an unlocalized py::class_ is a "
        "live ImportError for anyone with both cumesh and cumesh_vb installed")
print(f"cumesh_vb patch: {sum(_ml.values())} py::class_ registration(s) "
      f"are now module_local -- coexists with plain cumesh")
