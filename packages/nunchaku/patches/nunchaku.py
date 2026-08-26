"""Strip nunchaku's own +cuXX.YtorchM.m local version tag.

Upstream setup.py unconditionally appends `+cu{cuda}torch{major.minor}`
(dotted CUDA, e.g. +cu12.8torch2.8) to the version. The farm's rename step
then appends ITS tag, producing a double local version
(1.2.1+cu12.8torch2.8+cu128torch2.8) that fails PEP 440 and the verify
gate's filename check. Keep the base version; the farm owns the tag.
"""
from pathlib import Path

setup_py = Path("setup.py")
content = setup_py.read_text()
old = 'version = f"{version}+cu{cuda_version}torch{torch_major_minor_version}"'
if old not in content:
    raise SystemExit("nunchaku patch: version-suffix line not found -- "
                     "upstream setup.py changed; update this patch")
content = content.replace(old, "pass  # farm rename owns the +cuNNNtorchM.m tag")
setup_py.write_text(content)
print("nunchaku patch: upstream local-version suffix stripped")


# --- Release flags: this is a wheel, not a debug build ----------------------
# nunchaku hardcodes a debug-shaped flag set with no gate on it (setup.py:111,
# 116, 118). `-G` and `--generate-line-info` ARE gated (by DEBUG and by
# NUNCHAKU_BUILD_WHEELS respectively) and are left alone; these three are not:
#
#   -g        Full debug info on both the host and nvcc lines. auditwheel runs
#             --strip, so every byte of it is generated at full cost and then
#             discarded before the wheel ships. Pure waste.
#   -Og       Host optimisation "for debugging". It is appended AFTER the
#             default -O2, so it wins -- nunchaku's CPU-side code (Linear,
#             FluxModel, Serialization, the torch interop layer) ships built at
#             debug optimisation. Removing it costs a little compile time and
#             buys back runtime; it is the one flag here that trades in that
#             direction, and it is the right trade for a release artifact.
#   -UNDEBUG  Undefines NDEBUG, so assert() stays live in the shipped wheel.
#             Release builds define NDEBUG; this is the only thing keeping the
#             asserts in a user's inference path.
#
# NOT touched: --ptxas-options=--allow-expensive-optimizations=true. It merely
# restates ptxas's own default at -O2 and above, so removing it saves nothing,
# and setting it to false would trade shipped-kernel performance for compile
# time -- a real tradeoff that belongs to the owner, not to a cleanup patch.
import re as _re

_sp = Path("setup.py")
_t = _sp.read_text()

_flagsets = {
    "GCC_FLAGS": ('"-g", ', '"-UNDEBUG", ', '"-Og"'),
    "MSVC_FLAGS": ('"/UNDEBUG", ',),
}
_removed = []
for _name, _drop in _flagsets.items():
    _m = _re.search(rf"^(\s*{_name} = \[)(.*)(\]\s*)$", _t, _re.M)
    if not _m:
        raise SystemExit(
            f"nunchaku patch: {_name} assignment not found in setup.py -- "
            "upstream changed; refusing to ship a wheel built with an "
            "unverified flag set")
    _body = _m.group(2)
    for _f in _drop:
        if _f in _body:
            _body = _body.replace(_f, "", 1)
            _removed.append(f"{_name}:{_f.strip().strip(chr(34)).strip(',')}")
    _body = _re.sub(r",\s*\]", "]", _body.rstrip().rstrip(","))
    _t = _t[:_m.start()] + _m.group(1) + _body + _m.group(3) + _t[_m.end():]

# NVCC_FLAGS is a multi-line list; drop the two bare entries only.
for _f in ('        "-g",\n', '        "-UNDEBUG",\n'):
    if _f in _t:
        _t = _t.replace(_f, "", 1)
        _removed.append(f"NVCC_FLAGS:{_f.strip().strip(chr(34)).strip(',')}")

if not _removed:
    raise SystemExit(
        "nunchaku patch: none of the -g/-Og/-UNDEBUG debug flags were found. "
        "Either upstream already dropped them or the flag lists moved; verify "
        "before assuming the wheel is a release build.")

_sp.write_text(_t)
print(f"nunchaku patch: dropped {len(_removed)} debug flag(s): {', '.join(_removed)}")

# Prove it, and prove we did NOT disturb the two already-gated debug knobs.
_final = _sp.read_text()
_gcc = _re.search(r"^\s*GCC_FLAGS = \[(.*)\]\s*$", _final, _re.M).group(1)
for _bad in ('"-g"', '"-Og"', '"-UNDEBUG"'):
    if _bad in _gcc:
        raise SystemExit(f"nunchaku patch: {_bad} still present in GCC_FLAGS")
if '*cond("-G")' not in _final:
    raise SystemExit("nunchaku patch: the DEBUG-gated -G entry was damaged")
if 'allow-expensive-optimizations=true' not in _final:
    raise SystemExit("nunchaku patch: ptxas options were damaged")
