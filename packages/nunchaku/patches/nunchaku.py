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


# --- PTX tail ---------------------------------------------------------------
# nunchaku emits SASS only. Its gencode loop is
#     for target in sm_targets:
#         NVCC_FLAGS += ["-gencode", f"arch=compute_{target},code=sm_{target}"]
# -- `code=sm_X` on every line, `code=compute_X` on none -- so no wheel it has
# ever produced carried PTX, and verify C7 rejected all three platforms with
#   declared +PTX for ['sm_120'] but shipped NO PTX ... no JIT path onto newer GPUs
# The farm never asked for that tail explicitly either: the arch_override lists
# bare archs and generate_matrix's _ensure_ptx_on_highest_base appends +PTX to
# the highest base arch automatically.
#
# Emit a real tail where a real tail is possible. sm_targets is a mix of plain
# targets ("75","80","86","89") and ARCH-CONDITIONAL ones ("120a","121a", which
# upstream suffixes deliberately for the FP4 MMA path). Only the plain ones can
# produce portable PTX: `compute_120a` loads on sm_120 and nothing else, which
# is dead bytes plus a false forward-compat promise, and the base `compute_120`
# those kernels would need cannot be built from `a`-only instructions.
#
# So this adds `code=compute_X` for the highest PLAIN target, which is exactly
# what satisfies C7 on the cu12.4/12.6 rows (their arch list tops out at 8.9,
# built as plain sm_89). The cu12.8+ rows top out at 12.0, built as 120a, and
# are opted out via `no_ptx` in arch_override.yml -- see the note there.
_sp2 = Path("setup.py")
_t2 = _sp2.read_text()

_gc_anchor = '''    for target in sm_targets:
        NVCC_FLAGS += ["-gencode", f"arch=compute_{target},code=sm_{target}"]'''

if _gc_anchor not in _t2:
    raise SystemExit(
        "nunchaku patch: the -gencode loop was not found in setup.py. Without "
        "it no PTX tail can be added and C7 rejects every wheel for shipping "
        "none -- re-check the patch against the pinned source_tag.")

_gc_new = _gc_anchor + '''
    # cuda-wheels: portable PTX tail for the highest NON-arch-conditional
    # target. Arch-conditional targets (120a/121a) are skipped on purpose --
    # their PTX would be compute_120a, which loads only on sm_120.
    _cuw_plain = [t for t in sm_targets if t.isdigit()]
    if _cuw_plain:
        _cuw_top = max(_cuw_plain, key=int)
        NVCC_FLAGS += ["-gencode", f"arch=compute_{_cuw_top},code=compute_{_cuw_top}"]
        print(f"[cuda-wheels] PTX tail: compute_{_cuw_top} "
              f"(skipped arch-conditional {[t for t in sm_targets if not t.isdigit()]})")'''

_t2 = _t2.replace(_gc_anchor, _gc_new, 1)
_sp2.write_text(_t2)
print("nunchaku patch: PTX tail added for the highest plain sm target")

if "code=compute_{_cuw_top}" not in _sp2.read_text():
    raise SystemExit("nunchaku patch: the PTX tail is NOT PRESENT in setup.py on disk")
