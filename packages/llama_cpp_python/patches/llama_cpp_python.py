"""Patch llama-cpp-python's vendored llama.cpp for manylinux_2_28.

llama-sampler.cpp probes `std::random_device().entropy()` at static-init
time. `entropy()` is an inline that calls `_M_getentropy()`, which on
AlmaLinux 8 resolves against the SYSTEM libstdc++'s GLIBCXX_3.4.25 --
one notch above auditwheel's manylinux_2_28 cap (3.4.24), and the only
member of that symbol version. The wheel then fails repair with
"too-recent versioned symbols" no matter which gcc-toolset compiled it
(the symbol is not in libstdc++_nonshared.a, which only covers symbols
NEWER than the OS libstdc++).

The probe detects PRNG-backed std::random_device (a MinGW quirk). On
every platform this wheel targets (glibc Linux, MSVC Windows) the device
is entropy-backed, so the truthful constant is `false`.
"""
from pathlib import Path

target_file = Path("vendor/llama.cpp/src/llama-sampler.cpp")
old = "static bool is_rd_prng = std::random_device().entropy() == 0;"
new = ("static bool is_rd_prng = false;  "
       "// cuda-wheels: entropy() drags in _M_getentropy@GLIBCXX_3.4.25, "
       "one notch above the manylinux_2_28 cap; never PRNG-backed on our targets")

content = target_file.read_text()
if new in content:
    print("llama_cpp_python patch: already applied")
elif old not in content:
    raise SystemExit(
        "llama_cpp_python patch: entropy probe not found -- vendored "
        "llama.cpp changed; update this patch")
else:
    target_file.write_text(content.replace(old, new))
    print("llama_cpp_python patch: random_device entropy probe -> false "
          "(kills the GLIBCXX_3.4.25 verneed)")
