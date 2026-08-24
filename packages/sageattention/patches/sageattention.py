"""Patch SageAttention for build compatibility.

1. Fix arch parser to handle space-separated TORCH_CUDA_ARCH_LIST.
2. Replace GCC-specific CXX_FLAGS with MSVC equivalents on Windows.
3. Skip _GLIBCXX_USE_CXX11_ABI on Windows.
4. Reduce nvcc --threads from 8 to 4 to avoid OOM on CI runners.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

# Fix TORCH_CUDA_ARCH_LIST parser: upstream only handles comma/semicolon
# separators but PyTorch convention uses spaces (e.g. "7.0 8.0 9.0").
# Without this, HAS_SM80/89/90 are never set and qattn CUDA kernels are skipped.
old_parser = '    for item in arch_list_env.replace(",", ";").split(";"):'
new_parser = '    for item in arch_list_env.replace(",", " ").replace(";", " ").split():'
if old_parser in content:
    content = content.replace(old_parser, new_parser)
    print("Patched arch parser to handle space-separated TORCH_CUDA_ARCH_LIST")
else:
    print("WARNING: Could not find arch parser line - source may have changed")

# Replace hardcoded GCC CXX_FLAGS with platform-aware version
old_flags = '    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]'

new_flags = """    import platform
    if platform.system() == "Windows":
        CXX_FLAGS = ["/O2", "/Zi", "/openmp", "-DENABLE_BF16"]
    else:
        CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-DENABLE_BF16"]"""

if old_flags in content:
    content = content.replace(old_flags, new_flags)
    print("Patched CXX_FLAGS for MSVC compatibility")
else:
    print("WARNING: Could not find CXX_FLAGS block - source may have changed")

# Skip _GLIBCXX_USE_CXX11_ABI on Windows (GCC/libstdc++ only)
old_abi = """    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]"""

new_abi = """    if platform.system() != "Windows":
        ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
        CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
        NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]"""

if old_abi in content:
    content = content.replace(old_abi, new_abi)
    print("Patched _GLIBCXX_USE_CXX11_ABI to skip on Windows")
else:
    print("WARNING: Could not find ABI block - source may have changed")

# Reduce nvcc --threads from 8 to 1 to avoid OOM on GitHub runners.
# Combined with max_jobs=2 in sageattention.yml.
content = content.replace('"--threads=8"', '"--threads=1"')
print("Patched nvcc --threads=8 -> --threads=1")

# Fix _qattn_sm90 extension: it uses Hopper-only wgmma instructions but inherits
# the global NVCC_FLAGS with all arch gencode flags (sm_80, sm_86, etc.), causing
# ptxas to fail with "wgmma.mma_async not supported on .target sm_80".
# Give it a filtered flag list with only sm_90a.
old_sm90_ext = '''                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=['-lcuda'],'''
new_sm90_ext = '''                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": [f for f in NVCC_FLAGS if "gencode" not in f and "arch=" not in f] + ["-gencode", "arch=compute_90a,code=sm_90a"]},
                libraries=["cuda"],'''
if old_sm90_ext in content:
    # Only replace the first occurrence (the _qattn_sm90 block)
    content = content.replace(old_sm90_ext, new_sm90_ext, 1)
    print("Patched _qattn_sm90 to compile only for sm_90a")
else:
    print("WARNING: Could not find _qattn_sm90 compile_args - source may have changed")

setup_file.write_text(content)


# ── Let torch pick the C++ standard (review board 2026-08-24) ───────────
# The block above used to inject /std:c++17 on Windows, which overrides
# torch's own choice (cpp_extension only appends a standard when the caller
# supplied none) and broke every torch >= 2.13 Windows cell: C7555 designated
# initializers in c10/util/StringUtil.h, C7582 bit-field NSDMIs in
# c10/core/AutogradState.h. Upstream setup.py also puts -std=c++17 in
# NVCC_FLAGS; strip that too. Windows torch < 2.7 still gets /std:c++17 from
# the CL env var (93e770d).
import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require as _require, strip_std_flags as _strip

_sp = Path("setup.py")
_t = _sp.read_text()
_t, _n = _strip(_t)
_require(_n > 0,
         "sageattention: no hardcoded C++-standard flag found in setup.py -- "
         "upstream changed; refusing to build against an unverified flag set")
_sp.write_text(_t)
print(f"sageattention patch: dropped {_n} hardcoded std flag(s); "
      f"torch's cpp_extension now selects the standard")
