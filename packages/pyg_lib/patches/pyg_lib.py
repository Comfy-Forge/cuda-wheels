"""Patch installed torch's CMake config to neutralize legacy nvToolsExt refs.

PyTorch 2.4/2.5/2.6 reference the legacy `libnvToolsExt` library in two
files. CUDA Toolkit >=12.5 dropped that library, so anything calling
`find_package(Torch)` against these torch versions blows up at configure
or generate time.

Files patched here (paths inside the installed torch package):
  1. share/cmake/Caffe2/public/cuda.cmake
       - removes the FATAL_ERROR check on CUDA::nvToolsExt
       - turns torch::nvtoolsext into an empty INTERFACE IMPORTED lib
  2. share/cmake/Torch/TorchConfig.cmake
       - drops `find_library(LIBNVTOOLSEXT ...)` + its use in
         TORCH_CUDA_LIBRARIES on Linux
       - drops the NVTOOLEXT_HOME setup, the .lib reference, and the
         /include reference on MSVC

PyTorch 2.7+ rewrote both files to use CUDA::nvtx3, so this patch is a
no-op on those versions (no substrings match).
"""
import re
from pathlib import Path
import torch

torch_share = Path(torch.__file__).parent / "share/cmake"

# ---------- 1) Caffe2/public/cuda.cmake ----------
CUDA_OLD_FATAL = """if(NOT TARGET CUDA::nvToolsExt)
  message(FATAL_ERROR "Failed to find nvToolsExt")
endif()"""

CUDA_NEW_FATAL = """# patched: nvToolsExt presence check removed (CUDA>=12.5 dropped it)
if(FALSE)
endif()"""

CUDA_OLD_LINK = """# nvToolsExt
add_library(torch::nvtoolsext INTERFACE IMPORTED)
set_property(
    TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES
    CUDA::nvToolsExt)"""

CUDA_NEW_LINK = """# nvToolsExt (patched: empty INTERFACE so downstream links are no-ops)
add_library(torch::nvtoolsext INTERFACE IMPORTED)"""

# torch 2.5 / 2.6 wraps the nvToolsExt fallback inside an NVTX3-detection
# if/else. When NVTX3 isn't found, the else branch creates the broken
# torch::nvtoolsext linked to CUDA::nvToolsExt. Replace just the else body
# with empty stubs so configure succeeds either way.
CUDA_OLD_NVTX3_FALLBACK = """  message(WARNING "Cannot find NVTX3, find old NVTX instead")
  add_library(torch::nvtoolsext INTERFACE IMPORTED)
  set_property(TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES CUDA::nvToolsExt)"""

CUDA_NEW_NVTX3_FALLBACK = """  # patched: NVTX3 not found, create empty stubs (CUDA>=12.5 dropped legacy nvToolsExt)
  add_library(torch::nvtx3 INTERFACE IMPORTED)
  add_library(torch::nvtoolsext INTERFACE IMPORTED)"""


def patch_cuda_cmake(path: Path) -> bool:
    text = path.read_text()
    original = text
    if CUDA_OLD_FATAL in text:
        text = text.replace(CUDA_OLD_FATAL, CUDA_NEW_FATAL, 1)
    if CUDA_OLD_LINK in text:
        text = text.replace(CUDA_OLD_LINK, CUDA_NEW_LINK, 1)
    if CUDA_OLD_NVTX3_FALLBACK in text:
        text = text.replace(CUDA_OLD_NVTX3_FALLBACK, CUDA_NEW_NVTX3_FALLBACK, 1)
    if text != original:
        path.write_text(text)
        return True
    return False


# ---------- 2) Torch/TorchConfig.cmake ----------
# Use line-level regex removal so we're tolerant of whitespace variation.
# Each pattern matches a full line (including trailing newline) that we
# want to drop entirely. Run sequentially; idempotent on torch >= 2.7.
TORCH_CONFIG_DROP_PATTERNS = [
    # Linux: find_library(LIBNVTOOLSEXT libnvToolsExt.so ...)
    r"^[ \t]*find_library\(\s*LIBNVTOOLSEXT[^\n]*\n",
    # Linux/MSVC: bare ${LIBNVTOOLSEXT} list entry
    r"^[ \t]*\$\{LIBNVTOOLSEXT\}[ \t]*\n",
    # MSVC: default NVTOOLEXT_HOME -> "C:/Program Files/NVIDIA Corporation/NvToolsExt"
    r"^[ \t]*if\(NOT NVTOOLEXT_HOME\)[ \t]*\n"
    r"[ \t]*set\(NVTOOLEXT_HOME[^\n]*NvToolsExt[^\n]*\)[ \t]*\n"
    r"[ \t]*endif\(\)[ \t]*\n",
    # MSVC: env var override of NVTOOLEXT_HOME
    r"^[ \t]*if\(DEFINED ENV\{NVTOOLSEXT_PATH\}\)[ \t]*\n"
    r"[ \t]*set\(NVTOOLEXT_HOME \$ENV\{NVTOOLSEXT_PATH\}\)[ \t]*\n"
    r"[ \t]*endif\(\)[ \t]*\n",
    # MSVC: nvToolsExt64_1.lib entry inside set(TORCH_CUDA_LIBRARIES ...)
    r"^[ \t]*\$\{NVTOOLEXT_HOME\}/lib/x64/nvToolsExt64_1\.lib[ \t]*\n",
    # MSVC: list(APPEND TORCH_INCLUDE_DIRS ${NVTOOLEXT_HOME}/include)
    r"^[ \t]*list\(APPEND TORCH_INCLUDE_DIRS \$\{NVTOOLEXT_HOME\}/include\)[ \t]*\n",
]


def patch_torch_config(path: Path) -> bool:
    text = path.read_text()
    original = text
    for pat in TORCH_CONFIG_DROP_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    if text != original:
        path.write_text(text)
        return True
    return False


cuda_cmake = torch_share / "Caffe2/public/cuda.cmake"
torch_config = torch_share / "Torch/TorchConfig.cmake"

cuda_changed = patch_cuda_cmake(cuda_cmake)
torch_config_changed = patch_torch_config(torch_config)

if cuda_changed:
    print(f"Patched {cuda_cmake}: neutralized nvToolsExt FATAL_ERROR + link interface")
else:
    print(f"No nvToolsExt block found in {cuda_cmake} (torch>=2.7?), skipping.")

if torch_config_changed:
    print(f"Patched {torch_config}: dropped legacy LIBNVTOOLSEXT / NVTOOLEXT_HOME refs")
else:
    print(f"No legacy nvToolsExt refs in {torch_config} (torch>=2.7?), skipping.")


# ── C++20 for torch >= 2.13 (review board 2026-08-24) ───────────────────
# pyg-lib is a CMake build: CMakeLists.txt hardcodes `set(CMAKE_CXX_STANDARD
# 17)` and sets no CUDA standard, so torch's cpp_extension cannot choose for
# it. torch 2.13's headers are C++20-only on MSVC (C7555 designated
# initializers in c10/util/StringUtil.h, C7582 bit-field NSDMIs in
# c10/core/AutogradState.h); GCC accepts both as extensions, which is why
# only Windows failed. Both standards must move: bumping CXX alone still
# leaves the .cu translation unit (cuda/hash_map.cu) failing in the nvcc
# frontend with "data member initializer is not allowed".
import os as _os
import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require as _require, torch_mm as _torch_mm

if _torch_mm() >= (2, 13):
    _cml = _pl.Path("CMakeLists.txt")
    _require(_cml.exists(), "pyg_lib: CMakeLists.txt not found at source root")
    _c = _cml.read_text()
    _c2 = _c.replace("set(CMAKE_CXX_STANDARD 17)", "set(CMAKE_CXX_STANDARD 20)")
    _require(_c2 != _c,
             "pyg_lib: 'set(CMAKE_CXX_STANDARD 17)' not found in CMakeLists.txt "
             "-- upstream changed; torch 2.13 cells would fail to compile")
    if "CMAKE_CUDA_STANDARD" not in _c2:
        _c2 = _c2.replace(
            "set(CMAKE_CXX_STANDARD 20)",
            "set(CMAKE_CXX_STANDARD 20)\n"
            "set(CMAKE_CUDA_STANDARD 20)\n"
            "set(CMAKE_CUDA_STANDARD_REQUIRED ON)", 1)
    _cml.write_text(_c2)
    print("pyg_lib patch: CMAKE_CXX_STANDARD/CMAKE_CUDA_STANDARD -> 20 "
          "(torch >= 2.13)")

    # Bumping the CUDA standard to 20 is required (hash_map.cu needs it), but
    # it makes nvcc emit a diagnostic it stays quiet about at C++17: the
    # vendored CUTLASS pinned by pyg-lib 0.5.0 marks
    # CudaHostAdapter::memsetDevice as CUTLASS_HOST_DEVICE while its only body
    # calls the host-only pure-virtual memsetDeviceImpl. torch's cmake injects
    # --Werror cross-execution-space-call, so it is fatal:
    #   cuda_host_adapter.hpp(398): error: calling a __host__ function
    #   ("memsetDeviceImpl") from a __host__ __device__ function
    #   ("memsetDevice") is not allowed
    # The function was never callable from device code anyway, so mark it
    # host-only. Preferred over stripping torch's --Werror, which would
    # disable a whole nvcc error class build-wide.
    _cha = _pl.Path("third_party/cutlass/include/cutlass/cuda_host_adapter.hpp")
    if _cha.exists():
        _h = _cha.read_text()
        _needle = "  CUTLASS_HOST_DEVICE\n  Status memsetDevice("
        if _needle in _h:
            _cha.write_text(_h.replace(
                _needle, "  CUTLASS_HOST\n  Status memsetDevice(", 1))
            print("pyg_lib patch: cutlass memsetDevice -> CUTLASS_HOST "
                  "(cross-execution-space-call under -std=c++20)")
        else:
            _require("CUTLASS_HOST\n  Status memsetDevice(" in _h,
                     "pyg_lib: cutlass cuda_host_adapter.hpp no longer has the "
                     "expected memsetDevice declaration -- the submodule pin "
                     "moved; torch 2.13 Windows cells will fail")
    else:
        _require(False,
                 "pyg_lib: third_party/cutlass not on disk -- clone_recursive "
                 "should have fetched it")
else:
    print(f"pyg_lib patch: keeping CMAKE_CXX_STANDARD 17 "
          f"(torch {_os.environ.get('CUW_TORCH_VERSION', '?')} < 2.13)")
