"""Patch cubvh for torch 2.13 (same class as the cumesh fix).

torch 2.13's headers use C++20 features (designated initializers, bit-field
default member init); MSVC and Windows nvcc hard-error under c++17. cubvh
routes every std flag through one `cpp_standard` knob, so bump it. GCC and
nvcc accept c++20 on every CUDA line in the grid, so this is unconditional.
"""
import os
import sys
import pathlib
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import torch_mm

setup_file = Path("setup.py")
content = setup_file.read_text()

# GATED (review board 2026-08-24): bumping unconditionally broke every
# Windows torch <= 2.6 cell. Under MSVC 14.44 nvcc's EDG parses torch 2.6's
# ivalue_inl.h at C++20 and misparses `std::move(ivalue).to<List<Elem>>()`;
# under the pinned MSVC 14.29 (CUDA 12.4) nvcc rejects -std=c++20 outright
# and cudafe++ dies with 0xC0000409. Only torch >= 2.13 actually needs C++20.
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
