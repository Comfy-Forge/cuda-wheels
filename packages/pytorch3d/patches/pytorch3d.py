"""Patch pytorch3d for the farm's build matrix.

1. Stop hardcoding the C++ standard (review board 2026-08-24). The old patch
   rewrote `-std=c++17` to `/std:c++17` on Windows, which overrides torch's
   own choice -- cpp_extension only appends a standard when the caller gave
   none -- and so broke every torch >= 2.13 Windows cell (C7555 designated
   initializers in c10/util/StringUtil.h, C7582 bit-field NSDMIs in
   c10/core/AutogradState.h). Deleting the flag lets torch pick correctly on
   every platform; MSVC needs no -O3 translation here because pytorch3d's cxx
   list carries no optimisation flag.
2. CUDA 13 changed the default of --static-global-template-stub to `true`,
   giving the host-side stub of a __global__ function template internal
   linkage. pulsar explicitly instantiates 7 such templates in one TU and
   calls them from others, so the link fails with exactly 7 unresolved
   symbols (LNK2001/LNK1120 on Windows, `undefined reference` on Linux/ARM).
   NVIDIA's guidance is to restore the old behaviour with the flag; upstream
   setup.py honours the NVCC_FLAGS env var, so no source edit is needed.
"""
import os
import sys
import pathlib
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import strip_std_flags, require, cuda_mm

setup_file = Path("setup.py")
content = setup_file.read_text()
content, n_std = strip_std_flags(content)
require(n_std > 0,
        "no hardcoded C++-standard flag found in pytorch3d setup.py -- "
        "upstream changed; refusing to build against an unverified flag set")
setup_file.write_text(content)
print(f"pytorch3d patch: dropped {n_std} hardcoded std flag(s); "
      f"torch's cpp_extension now selects the standard")

if cuda_mm() >= (13, 0):
    env_file = os.environ.get("GITHUB_ENV")
    flag = "-static-global-template-stub=false"
    existing = os.environ.get("NVCC_FLAGS", "")
    value = (existing + " " + flag).strip()
    if env_file:
        with open(env_file, "a", encoding="utf-8") as fh:
            fh.write(f"NVCC_FLAGS={value}\n")
    os.environ["NVCC_FLAGS"] = value
    print(f"pytorch3d patch: CUDA >= 13.0 -> NVCC_FLAGS={value} "
          f"(restores pre-13 __global__ template stub linkage)")
else:
    print("pytorch3d patch: CUDA < 13.0, no template-stub flag needed")
