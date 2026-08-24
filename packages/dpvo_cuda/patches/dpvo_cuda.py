"""Patch script for dpvo-cuda - downloads Eigen headers and fixes PyTorch API compatibility."""
import subprocess
import shutil
from pathlib import Path

# Download and extract Eigen (required for cuda_ba and lietorch_backends)
subprocess.run([
    "curl", "-sL",
    "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    "-o", "eigen.tar.gz"
], check=True)
subprocess.run(["tar", "-xzf", "eigen.tar.gz"], check=True)

# Move to thirdparty/eigen as DPVO expects
thirdparty = Path("thirdparty")
thirdparty.mkdir(exist_ok=True)
eigen_target = thirdparty / "eigen-3.4.0"
if eigen_target.exists():
    shutil.rmtree(eigen_target)
shutil.move("eigen-3.4.0", str(eigen_target))

print("Eigen headers installed successfully")

# Rename package to dpvo-cuda
setup_py = Path("setup.py")
content = setup_py.read_text()
content = content.replace("name='dpvo'", "name='dpvo_cuda'")
# Keep packages=find_packages() so the dpvo/ package ships and the
# compiled ext_modules (cuda_corr, cuda_ba, lietorch_backends) get bundled
# into the wheel. Without this the wheel is essentially empty.
setup_py.write_text(content)

print("setup.py patched: renamed to dpvo-cuda")

# Fix PyTorch API compatibility: .type() -> .scalar_type()
# This is needed for PyTorch 2.0+ which deprecated tensor.type()
files_to_patch = [
    # DPVO altcorr files
    Path("dpvo/altcorr/correlation_kernel.cu"),
    # DPVO fastba files - all of them
    Path("dpvo/fastba/ba.cpp"),
    Path("dpvo/fastba/ba_cuda.cu"),
    Path("dpvo/fastba/block_e.cu"),
    # Lietorch files (also use deprecated .type() API)
    Path("dpvo/lietorch/src/lietorch_cpu.cpp"),
    Path("dpvo/lietorch/src/lietorch_gpu.cu"),
]

for src_file in files_to_patch:
    if src_file.exists():
        content = src_file.read_text()
        original = content

        # Replace .type() with .scalar_type() in AT_DISPATCH macros
        content = content.replace(".type()", ".scalar_type()")

        # Fix Windows linker error: mutable_data_ptr<T> template not exported
        # Use data_ptr<T> instead which works on both platforms
        content = content.replace("mutable_data_ptr<", "data_ptr<")

        # Fix Windows linker error: long type not exported from PyTorch DLL
        # packed_accessor32<long,...> uses mutable_data_ptr<long> internally
        # Replace long with int64_t which is properly exported
        content = content.replace("<long,", "<int64_t,")

        # Also fix .item<long>() which has same Windows export issue
        content = content.replace(".item<long>()", ".item<int64_t>()")

        if content != original:
            src_file.write_text(content)
            print(f"Patched {src_file}")

print("PyTorch API compatibility patches applied")

# Fix MSVC compound literal syntax error in ba_cuda.cu
# MSVC doesn't support C99 compound literals: (float[6]){...}
# Replace with individual element assignments
ba_cuda = Path("dpvo/fastba/ba_cuda.cu")
if ba_cuda.exists():
    content = ba_cuda.read_text()
    original = content

    # Replace compound literal assignments with individual element assignments
    # Line ~323: Jj = (float[6]){fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};
    content = content.replace(
        "Jj = (float[6]){fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};",
        "Jj[0]=fx*W*d; Jj[1]=0; Jj[2]=fx*-X*W*d2; Jj[3]=fx*-X*Y*d2; Jj[4]=fx*(1+X*X*d2); Jj[5]=fx*-Y*d;"
    )

    # Line ~331: Jj = (float[6]){0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
    content = content.replace(
        "Jj = (float[6]){0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};",
        "Jj[0]=0; Jj[1]=fy*W*d; Jj[2]=fy*-Y*W*d2; Jj[3]=fy*(-1-Y*Y*d2); Jj[4]=fy*(X*Y*d2); Jj[5]=fy*X*d;"
    )

    if content != original:
        ba_cuda.write_text(content)
        print("Patched ba_cuda.cu: fixed MSVC compound literal syntax")

print("All patches applied successfully")

# ── Farm build-matrix fixes (review board 2026-08-24) ───────────────────
import os as _os
import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require

# 1. atomicAdd(double*, ...) needs sm_60, and the cu12.4/12.6 arch policy
#    starts at sm_50, so every one of those cells failed to compile. Rather
#    than drop Maxwell from the wheels, delete the write: r_total is only
#    ever read by a commented-out std::cout (ba_cuda.cu:515), so this is a
#    provably dead accumulation -- removing it also takes a single-address
#    global atomic out of a hot bundle-adjustment kernel.
_ba = _pl.Path("dpvo/fastba/ba_cuda.cu")
_t = _ba.read_text()
_dead = "      atomicAdd(&r_total[0],  w * r * r);"
if _dead in _t:
    _ba.write_text(_t.replace(
        _dead,
        "      // Farm patch: dead accumulation removed. r_total's only reader\n"
        "      // is the commented-out std::cout below, and atomicAdd(double*)\n"
        "      // needs sm_60 -- keeping this line would cost Maxwell support."))
    print("dpvo patch: removed dead atomicAdd(double*) (keeps sm_50/sm_52)")
else:
    require("Farm patch: dead accumulation removed" in _t,
            "the atomicAdd(&r_total[0], ...) line was not found in "
            "dpvo/fastba/ba_cuda.cu -- upstream changed; the sm<60 lanes "
            "would fail to compile")

# 2. Eigen 3.4.0's arg_default_impl does `using ::arg;` on the nvcc device
#    pass (EIGEN_USING_STD expands that way when EIGEN_CUDA_ARCH is set), and
#    MSVC has no global ::arg. It only surfaced once torch >= 2.12 raised the
#    Windows standard to C++20. Take the same branch HIP already takes -- one
#    function, not all 33 EIGEN_USING_STD call sites.
_mf = _pl.Path("thirdparty/eigen-3.4.0/Eigen/src/Core/MathFunctions.h")
if not _mf.exists():
    _mf = _pl.Path("thirdparty/eigen/Eigen/src/Core/MathFunctions.h")
if _mf.exists():
    _m = _mf.read_text()
    _old = ("    #if defined(EIGEN_HIP_DEVICE_COMPILE)\n"
            "    // HIP does not seem to have a native device side implementation for the math routine \"arg\"\n"
            "    using std::arg;\n"
            "    #else\n"
            "    EIGEN_USING_STD(arg);\n"
            "    #endif")
    _new = ("    // Farm patch: MSVC has no global ::arg, which is what\n"
            "    // EIGEN_USING_STD expands to on the nvcc device pass.\n"
            "    using std::arg;")
    _n_arg = 0
    if _old in _m:
        _m = _m.replace(_old, _new)
        _n_arg += 1
    _second = "    EIGEN_USING_STD(arg);\n    return arg(x);"
    if _second in _m:
        _m = _m.replace(_second, "    using std::arg;\n    return arg(x);")
        _n_arg += 1
    if _n_arg:
        _mf.write_text(_m)
        print(f"dpvo patch: Eigen arg() -> std::arg at {_n_arg} site(s) (MSVC)")
    else:
        print("dpvo patch: Eigen arg() sites already patched or restructured")
else:
    print(f"dpvo patch: WARNING Eigen MathFunctions.h not found; skipped arg() fix")
