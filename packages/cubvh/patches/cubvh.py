"""Patch cubvh for torch 2.13 (same class as the cumesh fix).

torch 2.13's headers use C++20 features (designated initializers, bit-field
default member init); MSVC and Windows nvcc hard-error under c++17. cubvh
routes every std flag through one `cpp_standard` knob, so bump it.

SETTLED 2026-08-26 by build, in cubvh's favour. A cu12.4 / torch 2.4.1 Linux
cell compiled clean at c++20 (run 32968713626), so the EDG misparse of
`std::move(ivalue).to<List<Elem>>()` that patch_lib.py:55-57 describes is
MSVC-specific, not a property of c++20 on torch < 2.7 generally. The gate below
is correct as written; patch_lib's wording is over-broad and should say
"on MSVC" rather than describing it unqualified.

The record of the dispute, kept because it was a real contradiction: the gate is
`os.name == "nt" and torch_mm() < (2, 13)`, so off Windows EVERY torch gets
c++20, including the 2.4.1 floor. This docstring used to assert "GCC and nvcc
accept c++20 on every CUDA line in the grid, so this is unconditional".
scripts/patch_lib.py:55-57 says the opposite -- that pinning c++20 breaks
torch < 2.7, because nvcc's EDG misparses `std::move(ivalue).to<List<Elem>>()`
in ATen/core/ivalue_inl.h -- without limiting that to Windows. Both cannot be
right; the torch-2.4 Linux build above is the measurement that decided it.
"""
import os
import sys
import pathlib
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import strip_permissive_for_old_cuda, torch_mm

setup_file = Path("setup.py")
content = setup_file.read_text()

# GATED (review board 2026-08-24): bumping unconditionally broke every
# Windows torch <= 2.6 cell. Under MSVC 14.44 nvcc's EDG parses torch 2.6's
# ivalue_inl.h at C++20 and misparses `std::move(ivalue).to<List<Elem>>()`;
# under the pinned MSVC 14.29 (CUDA 12.4) nvcc rejects -std=c++20 outright
# and cudafe++ dies with 0xC0000409. Only torch >= 2.13 actually needs C++20.
# CUDA < 12.6 on Windows: cubvh's setup.py passes /permissive- (and
# -Xcompiler=/permissive-), which makes plain `cuda` ambiguous between
# libcu++'s ::cuda and torch's c10::cuda in the toolkit's own CCCL/thrust
# headers (error C2872). Same class cumesh hit; now shared logic.
strip_permissive_for_old_cuda(setup_file)

if os.name == "nt" and torch_mm() < (2, 13):
    print(f"cubvh patch: keeping upstream cpp_standard=17 "
          f"(Windows, torch {os.environ.get('CUW_TORCH_VERSION', '?')} < 2.13)")
    raise SystemExit(0)

new = content.replace("cpp_standard = 17", "cpp_standard = 20")
if new != content:
    setup_file.write_text(new)
    print("cubvh patch: cpp_standard 17 -> 20")
else:
    raise SystemExit(
        "cubvh patch: 'cpp_standard = 17' not found in setup.py -- upstream "
        "changed (HEAD already reads 'cpp_standard = 20 if IS_HIP_EXTENSION "
        "else 17'); update this patch before bumping the pin")
