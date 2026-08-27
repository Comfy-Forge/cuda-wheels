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


# --- CUDA arch bridge ------------------------------------------------------
# pyg_lib picks CMAKE_CUDA_ARCHITECTURES from a hardcoded if-ladder keyed on
# the nvcc version (CMakeLists.txt:54-64) and never looks at
# TORCH_CUDA_ARCH_LIST. Two consequences, in opposite directions:
#
#   * On x86 it builds MORE than asked. cu13.0 gets "75;80;86;90;100;120", and
#     because those tags are UNSUFFIXED cmake emits BOTH -real and -virtual for
#     each -- 12 device passes where the config asks for 4, with PTX on every
#     arch instead of the one the config declares. The wheel's METADATA then
#     describes an arch set the wheel does not have.
#   * On aarch64 it builds the WRONG thing. The ladder has no sm_87 at any CUDA
#     version, while the farm's aarch64 policy asks for 8.7 (Orin/Thor) on
#     every row -- so ARM wheels ship without the one arch ARM most needs, and
#     C7 would reject them for "missing arch families sm_[87]".
#
# The fix is the farm's standard shape: the patch TRANSLATES the farm's list
# into upstream's form and never decides its contents. The upstream ladder is
# kept verbatim as the fallback for a bare (non-farm) build.
#
# arch_override.yml was widened at the same time so that honouring the config
# does not lose the x86 archs the ladder happened to build. See its comment.
_cml_arch = _pl.Path("CMakeLists.txt")
_require(_cml_arch.exists(), "pyg_lib: CMakeLists.txt not found at source root")
_ca = _cml_arch.read_text()

_arch_anchor = """  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90;100;120")"""

_require(
    _arch_anchor in _ca,
    "pyg_lib: the hardcoded CMAKE_CUDA_ARCHITECTURES ladder was not found in "
    "CMakeLists.txt. Without the bridge, aarch64 wheels ship no sm_87 and x86 "
    "wheels ship PTX for every arch -- re-check against the pinned source_tag.",
)

_arch_bridge = """  # --- cuda-wheels arch bridge (injected; see packages/pyg_lib/patches) ---
  # Translate the farm's TORCH_CUDA_ARCH_LIST into CMAKE_CUDA_ARCHITECTURES.
  # "8.0 8.7 9.0+PTX" -> "80-real;87-real;90-real;90-virtual".
  # Explicit -real/-virtual matters: a bare "80" makes cmake emit real AND
  # virtual, which is how this build ended up shipping PTX for every arch.
  set(_CUW_ARCHS "")
  if (NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
    string(REPLACE ";" " " _CUW_RAW "$ENV{TORCH_CUDA_ARCH_LIST}")
    string(REPLACE "," " " _CUW_RAW "${_CUW_RAW}")
    separate_arguments(_CUW_TOKENS UNIX_COMMAND "${_CUW_RAW}")
    foreach(_CUW_TOK IN LISTS _CUW_TOKENS)
      set(_CUW_PTX FALSE)
      if (_CUW_TOK MATCHES "\\\\+PTX$")
        set(_CUW_PTX TRUE)
        string(REPLACE "+PTX" "" _CUW_TOK "${_CUW_TOK}")
      endif()
      string(STRIP "${_CUW_TOK}" _CUW_TOK)
      string(REPLACE "." "" _CUW_NUM "${_CUW_TOK}")
      if (NOT _CUW_NUM STREQUAL "")
        list(APPEND _CUW_ARCHS "${_CUW_NUM}-real")
        if (_CUW_PTX)
          list(APPEND _CUW_ARCHS "${_CUW_NUM}-virtual")
        endif()
      endif()
    endforeach()
  endif()
  if (NOT _CUW_ARCHS STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${_CUW_ARCHS}")
    message(STATUS "cuda-wheels: CMAKE_CUDA_ARCHITECTURES from TORCH_CUDA_ARCH_LIST -> ${CMAKE_CUDA_ARCHITECTURES}")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;90;100;120")"""

_ca = _ca.replace(_arch_anchor, _arch_bridge, 1)
_cml_arch.write_text(_ca)
print("pyg_lib patch: CMAKE_CUDA_ARCHITECTURES bridged to TORCH_CUDA_ARCH_LIST")

# Prove it landed rather than trusting the replace.
_final_ca = _cml_arch.read_text()
_require("cuda-wheels arch bridge" in _final_ca,
         "pyg_lib: the arch bridge is NOT present in CMakeLists.txt on disk")
_require(_final_ca.count("set(CMAKE_CUDA_ARCHITECTURES") >= 6,
         "pyg_lib: the upstream ladder fallback was damaged by the bridge "
         "injection -- a bare build would now have no architectures at all")


# --- RPATH: $ORIGIN only, never the build machine's paths --------------------
# libpyg.so shipped with FOUR absolute RPATH entries and verify C4 rejected
# every wheel on both CUDA lines:
#
#   [elf_sanity] libpyg.so: non-$ORIGIN RPATH entry '/lib/intel64';
#     '/lib/intel64_win'; '/lib/win-x64';
#     '/opt/python/cp312-cp312/lib/python3.12/site-packages/torch/lib'
#
# CMakeLists.txt:118 links ${TORCH_LIBRARIES} wholesale. That variable carries
# torch's MKL search hints (the intel64/win-x64 triple, which are not even
# Linux paths) and resolves torch's own libraries by absolute path, and cmake's
# default is to bake the build-tree location of every linked library into the
# binary's RPATH. The wheel is packaged from the build tree, so those paths
# ship -- pointing at directories that exist only on the runner.
#
# Same defect and same fix as natten. BUILD_WITH_INSTALL_RPATH is the operative
# property: it applies INSTALL_RPATH to the build-tree binary, which is the one
# that gets packaged. Setting INSTALL_RPATH alone would change nothing, because
# no `make install` ever runs.
#
# $ORIGIN is sufficient at runtime: `import torch` has already loaded libtorch
# into the process before the extension is imported, so torch symbols resolve
# from the loaded image rather than from disk -- which is exactly why every
# setuptools-built extension in this farm works with no RPATH at all. auditwheel
# still sets its own rpath for anything it vendors into pyg_lib.libs/.
#
# NOTE: the previous attempt here was `target_link_options(--as-needed)`, meant
# to stop auditwheel vendoring 109MB of NVRTC that libpyg.so never calls (0 of
# its 398 undefined symbols). It did NOT work -- the build log still shows
# "libnvrtc.so shorthash is a49e67e8", so nvrtc was still vendored -- almost
# certainly because the flag is positional and torch's cmake re-enables
# --no-as-needed after it. Removed rather than left in place: it was unverified,
# achieved nothing measurable, and an inert flag that looks load-bearing is
# worse than no flag. The NVRTC bloat is real and still unfixed; the right lever
# is filtering TORCH_LIBRARIES, not a link flag whose position we do not control.
_cml_rp = _pl.Path("CMakeLists.txt")
_require(_cml_rp.exists(), "pyg_lib: CMakeLists.txt not found at source root")
_c_rp = _cml_rp.read_text()

_rp_anchor = "target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})"
_require(
    _rp_anchor in _c_rp,
    "pyg_lib: the TORCH_LIBRARIES link line was not found in CMakeLists.txt, so "
    "the RPATH block cannot be anchored. Without it C4 rejects every wheel for "
    "shipping the runner's own paths.",
)

if "cuda-wheels RPATH hygiene" not in _c_rp:
    _c_rp = _c_rp.replace(
        _rp_anchor,
        _rp_anchor + """

# --- cuda-wheels RPATH hygiene (injected; see packages/pyg_lib/patches) ---
# Do not bake the runner's MKL/torch paths into the shipped .so.
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
message(STATUS "cuda-wheels: ${PROJECT_NAME} RPATH pinned to $ORIGIN")
# --- end cuda-wheels RPATH hygiene ---""", 1)
    _cml_rp.write_text(_c_rp)
    print("pyg_lib patch: RPATH pinned to $ORIGIN (was MKL + build-tree torch)")
else:
    print("pyg_lib patch: RPATH hygiene block already present -- skipping")

_require("BUILD_WITH_INSTALL_RPATH TRUE" in _cml_rp.read_text(),
         "pyg_lib: the RPATH hygiene block is NOT PRESENT in CMakeLists.txt on "
         "disk -- C4 would block every wheel again")
