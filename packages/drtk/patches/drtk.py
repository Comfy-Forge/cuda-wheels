"""Patch DRTK: MSVC-compatible CRT/RTTI flags, half-operator macros, and
hand the choice of C++ standard back to torch.

1. Remove /GR- -- it disables RTTI, but PyTorch's headers need it
   (dynamic_cast / dynamic_pointer_cast -> C2280 without RTTI).
2. Replace /MT with /MD -- DRTK asks for the static CRT, nvcc compiles the
   .cu TUs against the dynamic one, and the mix is LNK2038.
3. Re-enable the half/bfloat16 operators torch's -D flags turn off.
4. Delete DRTK's hardcoded C++-standard flags (see the long note below).

Every mutation is asserted: an upstream edit that moves one of these strings
must fail the build loudly rather than silently ship a differently-compiled
wheel.
"""
import pathlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require, strip_std_flags

setup_file = Path("setup.py")
content = setup_file.read_text()

# Remove /GR- from the win32 flags -- PyTorch requires RTTI.
content, n_gr = re.subn(r'"/GR-",\s*|,\s*"/GR-"', "", content)
require(n_gr == 1,
        f'expected exactly one "/GR-" element in drtk setup.py, found {n_gr} '
        f'-- upstream changed the win32 cxx flag list; without the removal '
        f'every Windows TU that touches a torch header dies with C2280')
print("drtk patch: removed /GR- (RTTI required by PyTorch)")

# /MT (static CRT) -> /MD (dynamic CRT). nvcc compiles the .cu files with
# /MD; mixing the two CRTs is LNK2038 at link time.
n_mt = content.count('"/MT"')
require(n_mt == 1,
        f'expected exactly one "/MT" element in drtk setup.py, found {n_mt} '
        f'-- upstream changed the win32 cxx flag list; the CRT would no '
        f'longer match nvcc\'s /MD and the link would fail with LNK2038')
content = content.replace('"/MT"', '"/MD"')
print("drtk patch: /MT -> /MD (CRT must match nvcc's /MD)")

# --------------------------------------------------------------------------
# The C++ standard: delete DRTK's opinion, let torch's cpp_extension decide.
#
# DRTK hardcodes two DIFFERENT standards in one setup.py:
#     cxx_args["linux"] = ["-std=c++17", ...]      # host side
#     nvcc_args.append("-std=c++20")               # device side
# and passes nothing at all on win32 (upstream deleted its "/std:c++17"
# in 69bf36c, "C++20 is the default now" -- a Meta-internal style decision,
# not a language requirement).
#
# The nvcc c++20 carries an upstream comment pointing at
# pytorch/pytorch#122169. That issue is `namespace "thrust" has no member
# "swap"` while compiling ATen's OWN LinearAlgebra.cu with CUDA 12.4 against
# torch 2.1.2/2.2.1 -- a torch-source build, on torch versions below this
# farm's 2.4.1 floor. It says nothing about extensions, and DRTK's kernels
# never include thrust (they use cub::WarpReduce and nothing else).
#
# DRTK's own sources do not need C++20: no designated initializers, no
# concepts, no std::span, no three-way comparison, no bit-field NSDMIs
# anywhere under src/ at the pinned source_tag. Checked, not assumed.
#
# The farm previously RAISED both to c++20 for torch >= 2.7, to satisfy
# torch 2.13's MSVC headers (C7555 designated initializers in
# c10/util/StringUtil.h, C7582 bit-field NSDMIs in c10/core/AutogradState.h).
# That is a torch-side constraint and it is now handled where it belongs --
# the torch-tracking MSVC /std floor in .github/actions/build-wheel/action.yml
# (/std:c++17 below torch 2.12, /std:c++20 from 2.12). Keeping the bump here
# as well made every torch 2.7-2.11 cell compile c10/ATen headers at C++20
# while the libtorch it links against is built at C++17 (pytorch's
# CMAKE_CXX_STANDARD is 17 through at least 2.11, and cpp_extension still
# emits -std=c++17 there): one ODR-mismatched translation unit per extension,
# for no gain.
#
# strip_std_flags removes the list literal AND the whole
# `nvcc_args.append("-std=c++20")` statement (deleting only the literal would
# leave `append()` -> TypeError at setup.py import).
# --------------------------------------------------------------------------
content, n_std = strip_std_flags(content)
require(n_std == 2,
        f"expected 2 hardcoded C++-standard flags in drtk setup.py "
        f"(linux cxx -std=c++17 and nvcc_args.append('-std=c++20')), found "
        f"{n_std} -- upstream changed; refusing to build against an "
        f"unverified flag set")
setup_file.write_text(content)
print(f"drtk patch: dropped {n_std} hardcoded C++-standard flag(s); torch's "
      f"cpp_extension now selects the standard for host and device")

# --------------------------------------------------------------------------
# Re-enable half/bfloat16 operators in the .cu files.
# torch's cpp_extension puts -D__CUDA_NO_HALF_OPERATORS__ (and friends) on
# the nvcc command line, which breaks CUB headers (dispatch_histogram.cuh,
# agent_sub_warp_merge_sort.cuh) and disables native half-precision ops.
# An #undef at the top of the TU overrides the -D.
# --------------------------------------------------------------------------
UNDEF_BLOCK = (
    "// -- cuda-wheels patch: re-enable half/bfloat16 operators --\n"
    "#undef __CUDA_NO_HALF_OPERATORS__\n"
    "#undef __CUDA_NO_HALF2_OPERATORS__\n"
    "#undef __CUDA_NO_HALF_CONVERSIONS__\n"
    "#undef __CUDA_NO_BFLOAT16_CONVERSIONS__\n"
    "// -- end patch --\n\n"
)
cu_files = sorted(Path("src").rglob("*.cu"))
require(len(cu_files) > 0,
        "no src/**/*.cu found in the drtk tree -- the source layout changed; "
        "the half-operator #undef block would silently apply to nothing")
patched_cu = 0
for cu_file in cu_files:
    cu_content = cu_file.read_text()
    if "__CUDA_NO_HALF_OPERATORS__" not in cu_content:
        cu_file.write_text(UNDEF_BLOCK + cu_content)
        patched_cu += 1
require(patched_cu == len(cu_files),
        f"only {patched_cu}/{len(cu_files)} .cu files took the half/bfloat16 "
        f"#undef block -- some already mention __CUDA_NO_HALF_OPERATORS__; "
        f"check whether upstream now handles this itself")
print(f"drtk patch: {patched_cu} .cu file(s) got the half/bfloat16 "
      f"#undef block")
