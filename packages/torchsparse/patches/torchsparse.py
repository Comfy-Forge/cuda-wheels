"""Patch torchsparse v2.0.0 for modern torch (>= 2.8).

Upstream passes `tensor.type()` (a DeprecatedTypeProperties) as the first
argument of AT_DISPATCH_* macros; torch >= 2.8 removed the implicit
conversion to c10::ScalarType, failing with "no suitable conversion
function from 'const at::DeprecatedTypeProperties' to 'c10::ScalarType'".
The modern spelling is `tensor.scalar_type()`.

Only dispatch-argument positions are rewritten (`.type(), "` -- the macro
always follows the arg with the op-name string literal), so other
DeprecatedTypeProperties uses (`.type().is_cuda()` etc.) are untouched.
"""
import re
from pathlib import Path

pattern = re.compile(r"\.type\(\)(\s*,\s*\")")

total = 0
for f in sorted(Path("torchsparse/backend").rglob("*")):
    if f.suffix not in (".cu", ".cpp", ".cc", ".cuh", ".h"):
        continue
    text = f.read_text()
    new_text, n = pattern.subn(r".scalar_type()\1", text)
    if n:
        f.write_text(new_text)
        print(f"  {f}: {n} dispatch arg(s) .type() -> .scalar_type()")
        total += n

if total == 0:
    raise SystemExit(
        "torchsparse patch: no '.type(), \"' dispatch args found -- "
        "upstream changed; update this patch")
print(f"torchsparse patch: rewrote {total} dispatch argument(s)")


# ── Windows: vendor Google sparsehash (header-only) ──────────────────────
# hashmap_cpu.hpp includes <google/dense_hash_map>. Linux gets EPEL's
# sparsehash-devel via pre_build_script; Windows has no package manager
# in the lane, so vendor the 2.0.4 tarball and use upstream's shipped
# MSVC sparseconfig.h. setup.py has no include_dirs at all -- inject one.
import os as _os
if _os.name == "nt":
    import shutil as _sh
    import subprocess as _sp
    import tarfile as _tar
    from pathlib import Path as _P

    _dst = _P("third_party/sparsehash")
    if not (_dst / "google").exists():
        _sp.run(["curl", "-sfL", "--retry", "5",
                 "https://github.com/sparsehash/sparsehash/archive/refs/tags/sparsehash-2.0.4.tar.gz",
                 "-o", "_sparsehash.tar.gz"], check=True)
        with _tar.open("_sparsehash.tar.gz") as _tf:
            _tf.extractall("_sparsehash_src")
        _root = next(_P("_sparsehash_src").iterdir())
        _dst.mkdir(parents=True, exist_ok=True)
        for _d in ("google", "sparsehash"):
            _sh.copytree(_root / "src" / _d, _dst / _d, dirs_exist_ok=True)
        # The tarball's own windows sparseconfig targets pre-2015 MSVC
        # (stdext::hash_compare -- error C2039 on VS2022). Write a modern
        # config: std::hash from <functional>, stdint types.
        (_dst / "sparsehash" / "internal" / "sparseconfig.h").write_text(
            "#define GOOGLE_NAMESPACE ::google\n"
            "#define HASH_NAMESPACE std\n"
            "#define HASH_FUN_H <functional>\n"
            "#define SPARSEHASH_HASH HASH_NAMESPACE::hash\n"
            "#define HAVE_STDINT_H 1\n"
            "#define HAVE_UINT16_T 1\n"
            "#define HAVE_LONG_LONG 1\n"
            "#define HAVE_MEMCPY 1\n"
            "#define STL_NAMESPACE std\n"
            "#define _START_GOOGLE_NAMESPACE_ namespace google {\n"
            "#define _END_GOOGLE_NAMESPACE_ }\n")
        _sh.rmtree("_sparsehash_src")
        _P("_sparsehash.tar.gz").unlink()
        print("vendored sparsehash 2.0.4 into third_party/sparsehash")

    _setup = _P("setup.py")
    _s = _setup.read_text()
    _needle = ("extension_type('torchsparse.backend',\n"
               "                       sources,\n"
               "                       extra_compile_args=extra_compile_args)")
    _repl = ("extension_type('torchsparse.backend',\n"
             "                       sources,\n"
             "                       include_dirs=[os.path.abspath('third_party/sparsehash')],\n"
             "                       extra_compile_args=extra_compile_args)")
    if _repl in _s:
        print("include_dirs already injected")
    elif _needle not in _s:
        raise SystemExit("torchsparse patch: extension_type call not found -- "
                         "upstream changed; update this patch")
    else:
        _setup.write_text(_s.replace(_needle, _repl))
        print("injected sparsehash include_dirs into setup.py")


# ── Windows: atomic.cuh redefines CUDA's builtin atomicExch ──────────────
# On Linux uint64_t is `unsigned long` (a distinct overload); with MSVC
# uint64_t IS `unsigned long long`, so the helper collides with CUDA's
# builtin ("function has already been defined"). The builtin covers the
# Windows case entirely -- compile the helper out there.
_atomic = Path("torchsparse/backend/utils/atomic.cuh")
_as = _atomic.read_text()
if "#ifndef _WIN32" in _as:
    print("atomic.cuh guard already applied")
else:
    _as = _as.replace(
        "#pragma once\n",
        "#pragma once\n#ifndef _WIN32  // cuda-wheels: uint64_t==ULL on MSVC; "
        "CUDA's builtin already provides this overload\n", 1) + "#endif  // _WIN32\n"
    _atomic.write_text(_as)
    print("atomic.cuh: helper guarded out on Windows")
