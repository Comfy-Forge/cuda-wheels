"""Drop torch_scatter's never-loadable CPU-only extension twin.

setup.py builds every extension twice -- `_<name>_cpu` and `_<name>_cuda` --
via `product(main_files, suffices)`. In a CUDA wheel the `_cpu` half is dead:

  * torch_scatter/__init__.py:15 loads `cuda_spec or cpu_spec`, so whenever the
    CUDA library is present the CPU twin is never loaded by anything.
  * It is not a CPU fallback either. The `cuda` build compiles
    csrc/cpu/<name>_cpu.cpp AND csrc/cuda/<name>_cuda.cu (setup.py:90-96), so
    the CUDA library is a strict superset and serves CPU tensors itself.

Upstream's own FORCE_ONLY_CUDA=1 switch collapses it. See patch_lib.

This package previously had no patch script at all.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import force_only_cuda  # noqa: E402

force_only_cuda("setup.py")
