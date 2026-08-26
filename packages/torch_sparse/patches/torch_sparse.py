"""Patch pytorch_sparse to add BFloat16 support to CUDA kernels.

spmm_cuda.cu only dispatches for Float, Double, and Half.
BFloat16 is missing, causing NotImplementedError when models run in bf16.
"""
from pathlib import Path

cuda_dir = Path("csrc/cuda")
patched = 0

for cu_file in sorted(cuda_dir.glob("*.cu")):
    content = cu_file.read_text()
    fname = cu_file.name

    old = "_AND(at::ScalarType::Half,"
    new = "_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,"
    count = content.count(old)
    if count > 0:
        content = content.replace(old, new)
        cu_file.write_text(content)
        patched += count
        print(f"Patched {fname}: {count} dispatch site(s) -> +BFloat16")

if patched == 0:
    print("No dispatch sites needed patching - BFloat16 may already be supported.")
else:
    print(f"\nDone. Patched {patched} total dispatch sites.")


# --- Drop the never-loadable CPU-only extension twin -----------------------
# setup.py builds every extension twice (`_<name>_cpu` and `_<name>_cuda`) via
# product(main_files, suffices). The facade loads `cuda_spec or cpu_spec`, so
# the CPU twin is never loaded when a CUDA library is present -- and it is not
# a fallback either: the cuda build compiles csrc/cpu/*.cpp as well, making the
# CUDA library a strict superset. See patch_lib.force_only_cuda.
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import force_only_cuda  # noqa: E402

force_only_cuda("setup.py")
