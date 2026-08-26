"""Patch SageAttention for build compatibility.

1. Fix arch parser to handle space-separated TORCH_CUDA_ARCH_LIST.
2. Replace GCC-specific CXX_FLAGS with MSVC equivalents on Windows.
3. Skip _GLIBCXX_USE_CXX11_ABI on Windows.
4. (REMOVED 2026-08-26.) Used to rewrite nvcc --threads 8 -> 1. Both
   misplaced and dead: the build action appends a TRAILING --threads=
   from the nvcc_threads knob, which beats any setup.py value. The
   docstring also said 8 -> 4 while the code did 8 -> 1.
"""
from pathlib import Path
import sys as _sys_early
import pathlib as _pl_early
_sys_early.path.insert(
    0, str(_pl_early.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require as _require  # used at the first anchor check below

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
    # A miss here is NOT cosmetic: without the rewrite, HAS_SM80/89/90 are
    # never set from a space-separated TORCH_CUDA_ARCH_LIST, every qattn CUDA
    # kernel is skipped, and the build still succeeds -- producing a wheel
    # that raises cudaErrorNoKernelImageForDevice on real hardware.
    _require(False,
             "sageattention: arch parser line not found in setup.py -- the "
             "wheel would ship without its qattn kernels")

# Replace hardcoded GCC CXX_FLAGS with platform-aware version
old_flags = '    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]'

new_flags = """    import platform
    if platform.system() == "Windows":
        # /Zi dropped 2026-08-26, same reasoning as the -g below and from the
        # same review. It was OURS, not upstream's -- this patch introduced it
        # when translating the GCC flags to MSVC, so the "it's upstream's"
        # defence never applied. /Zi is full MSVC debug info: it makes cl.exe
        # write a .pdb per TU, serialises writes through mspdbsrv, and nothing
        # in the wheel consumes it. Windows is already the slowest lane.
        CXX_FLAGS = ["/O2", "/openmp", "-DENABLE_BF16"]
    else:
        # -g dropped 2026-08-26: it is upstream's, it produces debug info
        # nothing here consumes, and auditwheel now runs --strip so it would be
        # generated at full cost and then thrown away. (sageattention's own
        # extensions carried some of the largest .debug_info in the farm.)
        CXX_FLAGS = ["-O3", "-fopenmp", "-lgomp", "-DENABLE_BF16"]"""

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

# nvcc --threads is NOT decided here. It used to be:
#     content.replace('"--threads=8"', '"--threads=1"')
# which was both a policy decision in the wrong place and dead weight. The
# build action appends a TRAILING --threads=${nvcc_threads} to
# NVCC_APPEND_FLAGS precisely so it beats setup.py hardcodes -- see
# action.yml:677 (Linux) / :1124 (Windows), whose input description at :48
# names this exact case. Trailing wins, so upstream's 8 never survived anyway
# and this rewrite changed nothing. Set nvcc_threads: in package.yml instead.
# (Farm default is 1, which is what this line was trying to achieve.)

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

# ── _qattn_sm89: FP8 sources must not be built for pre-Ada archs ────────
# Same defect as _qattn_sm90 above, one arch down and much easier to miss
# because it does NOT fail the build -- it silently ships traps.
#
# The seven sm89_* sources are the FP8 QMMA path. Their mma wrapper is gated:
#   csrc/mma.cuh:44   #if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#                     #define MMA_F8F8F32_M16N8K16_ENABLED
# and when that macro is absent the wrapper body becomes
#   csrc/mma.cuh:56   #define RUNTIME_ASSERT(x) __brkpt()
# So every gencode below sm_89 compiles the whole kernel down to a breakpoint.
# Measured in a shipped wheel: 26,880 BPT.TRAP instructions and zero QMMA.
#
# The extension is built whenever HAS_SM89 or HAS_SM90 or HAS_SM120, so it
# inherits the FULL global gencode list -- sm_80 and sm_86 included. Nothing
# can ever call those cubins: core.py:148 dispatches on `arch == "sm89"`, so a
# pre-Ada GPU never enters this extension at all. They are pure compile cost
# and pure wheel weight.
#
# Filter by capability rather than naming archs, so this keeps working as the
# arch policy moves. compute_100/compute_120 sort correctly against 89 because
# the tag is major*10+minor throughout (89, 90, 100, 120).
helper_anchor = "import warnings"
helper = '''import warnings


def _cuw_min_cc_gencodes(flags, minimum):
    """Drop -gencode pairs whose compute capability is below `minimum`.

    Injected by cuda-wheels (packages/sageattention/patches). Pairs must be
    removed two elements at a time: setup.py appends them as
    ["-gencode", "arch=compute_NN,code=sm_NN"].
    """
    out, i = [], 0
    while i < len(flags):
        if flags[i] == "-gencode" and i + 1 < len(flags):
            spec = flags[i + 1]
            cc = spec.split("compute_")[1].split(",")[0] if "compute_" in spec else ""
            if cc.isdigit() and int(cc) < minimum:
                i += 2
                continue
            out.extend([flags[i], flags[i + 1]])
            i += 2
            continue
        out.append(flags[i])
        i += 1
    return out'''

_require(helper_anchor in content,
         "sageattention: 'import warnings' not found in setup.py -- cannot "
         "inject the gencode filter helper")
content = content.replace(helper_anchor, helper, 1)

old_sm89_ext = '''                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},'''
new_sm89_ext = '''                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": _cuw_min_cc_gencodes(NVCC_FLAGS, 89)},'''

_require(old_sm89_ext in content,
         "sageattention: the _qattn_sm89 extension block was not found in "
         "setup.py. Without the filter its FP8 kernels are compiled for sm_80 "
         "and sm_86, where they are nothing but __brkpt() traps.")
content = content.replace(old_sm89_ext, new_sm89_ext, 1)
print("Patched _qattn_sm89 to skip gencodes below sm_89 (FP8 QMMA floor)")

setup_file.write_text(content)

# Prove both arch filters are on disk, not merely computed.
_final = setup_file.read_text()
for _marker, _what in (
        ("_cuw_min_cc_gencodes(NVCC_FLAGS, 89)", "the _qattn_sm89 FP8 gencode filter"),
        ("arch=compute_90a,code=sm_90a", "the _qattn_sm90 Hopper gencode filter")):
    _require(_marker in _final,
             f"sageattention: {_what} is NOT PRESENT in the setup.py on disk -- "
             "the wheel would ship trap-only kernels for those archs")


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
from patch_lib import strip_std_flags as _strip

_sp = Path("setup.py")
_t = _sp.read_text()
_t, _n = _strip(_t)
_require(_n > 0,
         "sageattention: no hardcoded C++-standard flag found in setup.py -- "
         "upstream changed; refusing to build against an unverified flag set")
_sp.write_text(_t)
print(f"sageattention patch: dropped {_n} hardcoded std flag(s); "
      f"torch's cpp_extension now selects the standard")
