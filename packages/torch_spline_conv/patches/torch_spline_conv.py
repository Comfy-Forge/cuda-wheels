"""Drop torch_spline_conv's never-loadable CPU-only extension twin.

Same shape as torch_scatter: setup.py builds `_<name>_cpu` and `_<name>_cuda`
from `product(main_files, suffices)`, the facade prefers the CUDA spec, and the
CUDA build already compiles the CPU sources -- so the `_cpu` twin is compiled
and shipped but can never be loaded. Upstream's FORCE_ONLY_CUDA=1 collapses it.

See patch_lib.force_only_cuda for the full argument and the measurements.

This package previously had no patch script at all.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import force_only_cuda  # noqa: E402

force_only_cuda("setup.py")
