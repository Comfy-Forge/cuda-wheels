"""Patch flash-attn for CI builds.

1. Init only csrc/cutlass submodule (skip composable_kernel — ROCm only, breaks Windows due to long filenames).
2. Bridge TORCH_CUDA_ARCH_LIST → FLASH_ATTN_CUDA_ARCHS.
3. Fix MAX_JOBS auto-detect to account for arch count (upstream assumes 2 archs = 9GB/job,
   but cu128+ has 4 archs = ~18GB/job, causing OOM on 32GB machines).
"""
import sys as _sys_pl
import pathlib as _pl_pl
_sys_pl.path.insert(0, str(_pl_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import require as _require
import subprocess
from pathlib import Path

# Init only the CUDA submodule (composable_kernel is ROCm-only and has
# filenames exceeding Windows' 260-char path limit)
subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
print("Initialized csrc/cutlass submodule")

setup_file = Path("setup.py")
content = setup_file.read_text()

# Remove submodule init calls from setup.py (cutlass already fetched by
# clone_recursive; composable_kernel is ROCm-only and never used here).
#
# REGEX, not str.replace: upstream v2.8.3 spells these WITH `, check=True`
# (setup.py:150-151), so the old literal replacements matched nothing --
# and the success print below fired regardless. The ROCm composable_kernel
# clone therefore ran in every single job: normally 6-9s, but 348s in one
# observed ARM link job (57% of that job's wall time). Same bug class as
# the rest of this farm's silent no-ops, so it is asserted now.
import re as _re
_n_sub = 0
for _mod, _why in (("composable_kernel", "ROCm only, unused"),
                   ("cutlass", "already fetched by clone_recursive")):
    content, _k = _re.subn(
        r'subprocess\.run\(\s*\[\s*"git"\s*,\s*"submodule"\s*,\s*"update"\s*,'
        r'\s*"--init"\s*,\s*"csrc/' + _mod + r'"\s*\][^)]*\)',
        f"None  # cuda-wheels: skipped {_mod} submodule ({_why})",
        content)
    _require(_k > 0,
             f"flash_attn: found no `git submodule update --init csrc/{_mod}` "
             f"call to patch out -- upstream changed its spelling. Fix the "
             f"pattern; do NOT let this silently no-op.")
    _n_sub += _k
print(f"Patched out {_n_sub} submodule init call(s) from setup.py")

# Replace cuda_archs() to also read TORCH_CUDA_ARCH_LIST
old_func = '''def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")'''

new_func = '''def cuda_archs() -> str:
    archs = os.getenv("FLASH_ATTN_CUDA_ARCHS")
    if archs:
        return archs.split(";")
    torch_archs = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    if torch_archs:
        # Convert "8.0 9.0 10.0 12.0+PTX" -> ["80", "90", "100", "120"].
        # The +PTX suffix must be stripped: upstream gates gencode on exact
        # token membership ('"90" in cuda_archs()'), so "90+PTX" would
        # silently drop that arch from the build.
        #
        # Split on SEMICOLONS as well as whitespace. defaults/arch_policy.yml
        # writes ";"-separated lists ("8.0;9.0+PTX;10.0;11.0;12.0+PTX") while
        # this package's arch_override.yml writes space-separated ones -- and
        # flash_attn has no aarch64 override, so the ARM lane gets the policy
        # form. A bare .split() then yields ONE token, no membership test
        # matches, and ZERO cubin gencodes are emitted: the wheel silently
        # builds for nvcc's default arch only (2026-08-26).
        import re as _re
        return [a.split("+")[0].replace(".", "")
                for a in _re.split(r"[;\\s]+", torch_archs.strip()) if a]
    return ["80", "90", "100", "120"]


def cuda_ptx_archs() -> list:
    """Archs the farm asked for PTX on, e.g. ["120"] for "12.0+PTX".

    cuda_archs() has to drop the +PTX marker (see above), and upstream only
    ever emits `code=sm_X` -- never `code=compute_X` -- so flash_attn shipped
    NO PTX at all for any arch. That makes the wheel dead on any GPU newer
    than its newest cubin: there is no JIT path. The farm declares +PTX in
    arch_policy precisely to promise that path, so emit it.
    """
    import re as _re
    out = []
    for a in _re.split(r"[;\\s]+", os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()):
        if a and "+PTX" in a.upper():
            out.append(a.split("+")[0].replace(".", ""))
    return out


# Archs upstream's gencode chain knows about, verbatim from setup.py:179-191
# of the v2.8.3 tag. The chain is a hardcoded if-ladder:
#     if "80"  in cuda_archs(): ... arch=compute_80,code=sm_80
#     if "90"  in cuda_archs(): ... (bare_metal >= 11.8)
#     if "100" in cuda_archs(): ... (bare_metal >= 12.8)
#     if "120" in cuda_archs(): ... (bare_metal >= 12.8)
# Anything else the farm asks for hits NO branch and is SILENTLY DROPPED --
# no cubin, no PTX, no diagnostic. That is how the cu13.0 ARM wheel shipped
# without sm_110 (Thor): arch_policy_aarch64's 13.x row is
# "8.0;9.0+PTX;10.0;11.0;12.0+PTX", 11.0 matched no branch, and only
# verify_wheel's arch_sass check caught it, after a 3h build.
_CUW_UPSTREAM_ARCHS = {"80", "90", "100", "120"}

# FlashAttention-2's hard floor. Upstream README, v2.8.3:
#   "1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100).
#       Support for Turing GPUs (T4, RTX 2080) is coming soon, please use
#       FlashAttention 1.x for Turing GPUs for now.
#    2. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs)."
# Turing is not short of FP16 tensor cores -- it has 2nd-gen ones. The blockers
# are that FA2's CUTLASS pipeline is built on `cp.async` (sm_80+), and that
# Turing has no BF16 tensor cores at all. That is why FA1 ran on Turing and FA2
# does not.
_CUW_ARCH_FLOOR = 80


def cuda_extra_archs() -> list:
    """Archs the farm demands that upstream's if-ladder does not emit.

    Returned as plain `code=sm_X` cubin targets. csrc/flash_attn is one
    CUTLASS-2.x kernel family with zero `__CUDA_ARCH__ <` guards, compiled
    unchanged for every arch AT OR ABOVE THE FLOOR, so such a target is a
    recompile and not a port.

    The floor is load-bearing, and it was missing when this function was first
    written (2026-08-26). Upstream's if-ladder silently drops anything it does
    not recognise; this function's whole purpose is to stop doing that -- which
    means it also stops silently dropping archs FA2 genuinely CANNOT build.
    Restore `7.5` to arch_override.yml and, without this guard, the build would
    emit `arch=compute_75,code=sm_75` and fail on Turing instead of skipping.
    Same for the `5.0;6.0;7.0` on the cu12.4/12.6 POLICY rows, which are only
    out of reach today because this package overrides them away.

    Skipping loudly rather than silently: a sub-floor arch in the list is a
    config mistake worth seeing in the log, but it must not break the build,
    because the farm's preferred encoding for an unbuildable arch is to KEEP it
    in the list and waive it with verify.allow_missing_archs.
    """
    out, below = [], []
    for a in cuda_archs():
        if a in _CUW_UPSTREAM_ARCHS:
            continue
        if int(a) < _CUW_ARCH_FLOOR:
            below.append(a)
            continue
        out.append(a)
    if below:
        print(f"flash_attn: skipping arch(es) {below} -- below FA2's Ampere "
              f"floor (sm_{_CUW_ARCH_FLOOR}); waive them with "
              f"verify.allow_missing_archs, do not expect a cubin")
    return out'''

if old_func in content:
    content = content.replace(old_func, new_func)
    print("Patched cuda_archs() to read TORCH_CUDA_ARCH_LIST")
else:
    # A miss leaves upstream's hardcoded ["80","90","100","120"] in place, so
    # the wheel is built for the WRONG arch set while the build goes green.
    _require(False,
             "flash_attn: cuda_archs() not found in setup.py -- the wheel "
             "would be built for upstream's hardcoded arch list, not the "
             "farm's policy")


# ── Emit PTX for the archs the farm declared with +PTX ──────────────────
# THIS IS A BACKPORT, NOT AN INVENTION. Upstream added the identical line in
# 7bdb426 (PR #1882, 2025-09-12) -- four weeks AFTER the v2.8.3 tag we pin:
#     cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]
#     # PTX for newest requested arch (forward-compat)
# Reviewed by an NVIDIA engineer 2026-08-25. The objection "flash-attention
# has per-arch kernels so PTX is pointless" is true of FA3/FA4 -- but those
# live in hopper/ and the CuTe-DSL path, and setup.py builds NEITHER. What we
# compile (csrc/flash_attn/) is ONE CUTLASS-2.x Ampere kernel family built
# four times unchanged: `grep -c '__CUDA_ARCH__ *<'` over that tree returns 0,
# so a compute_120 image is not a stub. Verified by compiling it: 32 entries,
# 15360 mma.sync, zero FLASH_UNSUPPORTED_ARCH stub strings, and byte-identical
# entry names to compute_80. `.target sm_120` carries no a/f suffix, so it IS
# forward-portable (an `a`-suffix target would NOT be -- see below).
#
# Where this actually earns its keep: PTX JITs FORWARD ONLY (measured --
# compute_120 PTX on an sm_86 device gives cudaErrorNoKernelImageForDevice).
# On cu13.0 it only guards a future major. On the cu12.4/cu12.6 cells
# (8.0 9.0+PTX) it is load-bearing TODAY: those toolkits cannot emit any
# Blackwell cubin, so without compute_90 PTX those wheels are dead on every
# RTX 50-series / B200 / GB10 in existence.
#
# Cost, measured: +1.4% compile, but +25% wheel size (244MB -> ~304MB) --
# nvcc stores PTX pre-compressed so it is ~incompressible in the zip while
# cubins compress ~3.9:1. Budget it deliberately.
#
# DO NOT copy this pattern to packages that compile `a`-suffix targets
# (sageattention compute_90a, natten 90a/100a-real, torchao _C_cutlass_90a,
# sageattn3 sm_100a/120a). `compute_90a` PTX loads ONLY on sm_90 -- emitting
# it would ship dead bytes and a false forward-compat promise. Those packages
# should drop +PTX from their arch_override instead.
# Upstream (setup.py:179-191) appends ONLY `code=sm_X` gencodes, so the
# built wheel carries cubins and no PTX whatsoever. verify_wheel's arch_sass
# check could not see this until it was made PTX-aware (2026-08-25); the
# first Windows wheel this farm ever produced failed it:
#   declared +PTX for ['sm_120'] but shipped NO PTX -- no JIT path onto newer GPUs
# Append a compute_X,code=compute_X gencode for each +PTX arch.
_ptx_anchor = '''        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")'''
_ptx_add = _ptx_anchor + '''

    for _cuw_extra in cuda_extra_archs():
        cc_flag.append("-gencode")
        cc_flag.append(f"arch=compute_{_cuw_extra},code=sm_{_cuw_extra}")
    for _cuw_ptx in cuda_ptx_archs():
        cc_flag.append("-gencode")
        cc_flag.append(f"arch=compute_{_cuw_ptx},code=compute_{_cuw_ptx}")'''
_require(_ptx_anchor in content,
         "flash_attn: gencode block not found -- cannot emit the +PTX the "
         "arch policy promises. Upstream changed setup.py; update the anchor.")
content = content.replace(_ptx_anchor, _ptx_add, 1)
print("Injected PTX gencode emission for +PTX archs")

# The MAX_JOBS auto-detect rewrite that used to live here is GONE. It changed
# upstream's `free_memory_gb / 9` into `free_memory_gb / (4.5 * num_archs)`,
# which is better arithmetic -- and completely dead. Upstream only consults the
# estimator when MAX_JOBS is unset, and this package sets it: package.yml:15 is
# `max_jobs: 1`, and the comment three lines above it says so in as many words
# ("explicit max_jobs bypasses upstream's free-RAM estimator"). So the improved
# formula has never run on a single cell, while looking like live tuning to
# anyone reading the patch. Parallelism is set with the max_jobs knob.

print("flash_attn patch: FORCE_BUILD default -> TRUE (no prebuilt-wheel adoption)")


# ── os.rename -> shutil.move (container cross-device fix) ─────────────────
# FA's bdist override renames its wheel with os.rename (setup.py:499 at
# v2.8.3); in the manylinux container the temp build dir (overlayfs) and
# dist (bind mount) are different filesystems -> EXDEV "Invalid
# cross-device link". shutil.move copies across devices.
_c = content if "content" in dir() else None
from pathlib import Path as _P2
_sp2 = _P2("setup.py")
_s2 = _sp2.read_text()
if "os.rename(wheel_filename, wheel_path)" not in _s2:
    raise SystemExit("flash_attn patch: os.rename call not found -- upstream "
                     "changed; update this patch")
_s2 = _s2.replace("os.rename(wheel_filename, wheel_path)",
                  "shutil.move(wheel_filename, wheel_path)")
_sp2.write_text(_s2)
print("flash_attn patch: os.rename -> shutil.move (shutil already imported upstream)")
