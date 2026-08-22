"""Patch FlexGEMM for Windows compatibility.

Fix: Make triton dependency platform-specific (triton on Linux, triton-windows on Windows).
"""
import re
from pathlib import Path

pyproject = Path("pyproject.toml")
content = pyproject.read_text()

# Replace triton dependency with platform-specific versions
# "triton>=X.Y.Z" -> "triton>=X.Y.Z; platform_system != 'Windows'", "triton-windows>=X.Y.Z; platform_system == 'Windows'"
content = re.sub(
    r'"triton(>=[\d.]+)"',
    r'"triton\1; platform_system != \'Windows\'", "triton-windows\1; platform_system == \'Windows\'"',
    content
)

pyproject.write_text(content)
print("Patched pyproject.toml for triton-windows compatibility")


# ── MSVC: dependent template-arg member calls need `template` ────────────
# hash.cu / sparse_neighbor_map.cu call `x.data_ptr<T>()` (and `item<T>`)
# where T is deduced inside a generic lambda (`using T = decltype(tag)`),
# making it a dependent name. GCC/Clang accept the shorthand; MSVC's
# parser rejects it ("type name is not allowed", wave-2 Windows failure
# at sparse_neighbor_map.cu:327). `.template member<T>()` is the
# conforming spelling and is valid on every compiler, so it is applied
# unconditionally. Idempotent: already-patched calls are skipped.
import re as _re
from pathlib import Path as _P

_fixed = 0
for _f in [_P("flex_gemm/kernels/cuda/hash/hash.cu"),
           _P("flex_gemm/kernels/cuda/spconv/sparse_neighbor_map.cu")]:
    _s = _f.read_text()
    _new, _n = _re.subn(r"(?<!template )\.(?!template )(data_ptr|item)<([TK])>",
                        r".template \1<\2>", _s)
    if _n:
        _f.write_text(_new)
        print(f"  {_f}: {_n} dependent member call(s) got `.template`")
        _fixed += _n
if _fixed == 0:
    raise SystemExit("flexgemm patch: no data_ptr<T>/item<T> calls found -- "
                     "upstream changed; update this patch")
print(f"flexgemm patch: {_fixed} MSVC dependent-template fixes")
