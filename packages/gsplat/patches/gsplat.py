"""Patch gsplat for the farm's build matrix.

1. MSVC-only: translate the GCC-only -O3 cxx flag to /O2 (cl.exe merely warns
   D9002 and ships an UNOPTIMISED wheel otherwise).
2. Stop the hardcoded C++ standard from overriding torch's. Upstream appends
   "-std=c++17" to nvcc_flags AFTER torch's own -std, and nvcc honours the
   LAST one, so every torch >= 2.13 cell compiled C++20-only headers as C++17
   ("data member initializer is not allowed", "expected an expression").
3. Provide an sm<70 fallback for cg::labeled_partition instead of dropping
   old GPUs from the arch list. Upstream even left the guard commented out
   directly above the first use (Projection2DGSFused.cu). Farm rule: never
   shrink architecture coverage to make a build pass.

Review board 2026-08-24.
"""
import glob
import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import (guard_labeled_partition_in_files, require,
                       strip_std_flags, translate_cxx_flags_for_msvc)

setup_file = Path("setup.py")
content = setup_file.read_text()

if os.name == "nt":
    content, n_msvc = translate_cxx_flags_for_msvc(content)
    require(n_msvc > 0,
            "no GCC cxx flags translated for MSVC in gsplat setup.py -- the "
            "extra_compile_args block moved; refusing to ship an unoptimised "
            "wheel built with flags cl.exe silently ignores")
    print(f"gsplat patch: translated {n_msvc} cxx flag(s) for MSVC")

content, n_std = strip_std_flags(content)
require(n_std > 0,
        "no hardcoded -std flag found in gsplat setup.py -- upstream changed; "
        "refusing to build against an unverified flag set")
setup_file.write_text(content)
print(f"gsplat patch: dropped {n_std} hardcoded std flag(s); torch's "
      f"cpp_extension now selects the standard")

n_lp = guard_labeled_partition_in_files(
    sorted(glob.glob("gsplat/cuda/csrc/*.cu")), required=True)
print(f"gsplat patch: guarded {n_lp} cg::labeled_partition site(s) for sm<70")
