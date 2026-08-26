"""Let diso build for Maxwell.

diso's backward pass accumulates into double precision with atomicAdd, and
atomicAdd(double*, double) is a hardware intrinsic only from sm_60 (Pascal).
Below that nvcc fails outright:

    error: no instance of overloaded function "atomicAdd" matches the
    argument list -- argument types are: (double *, double)

The response used to be arch_override.yml pinning cu12.4/cu12.6 to a 6.0 floor,
which drops Maxwell -- an arch torch's own cu124/cu126 wheels ship. That is the
banned shape: verify_wheel's C7 compares the wheel against the RESOLVED list,
so narrowing moves both sides of the comparison and the gap leaves no trace.

NVIDIA publishes the alternative in the CUDA C Programming Guide (B.14): a
compare-and-swap loop over the same 64 bits. patch_lib.add_atomicadd_double_shim
injects it, guarded so it exists ONLY in device passes below sm_60 -- on sm_60+
the real intrinsic is used, and the host pass never sees it.

The helper hard-fails if it finds no atomicAdd to guard, because a silent no-op
would mean the arch list was widened on the strength of a fix that did not
apply, and the failure would surface as a compile error on the Maxwell cell
rather than here.
"""
import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import add_atomicadd_double_shim  # noqa: E402

_sources = sorted(_pl.Path(".").rglob("*.cu")) + sorted(_pl.Path(".").rglob("*.cuh"))
print(f"diso patch: scanning {len(_sources)} CUDA source file(s) for atomicAdd")
_n = add_atomicadd_double_shim(_sources)
print(f"diso patch: atomicAdd(double*) shim added to {_n} file(s)")
