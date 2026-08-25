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
        return [a.split("+")[0].replace(".", "") for a in torch_archs.split()]
    return ["80", "90", "100", "120"]


def cuda_ptx_archs() -> list:
    """Archs the farm asked for PTX on, e.g. ["120"] for "12.0+PTX".

    cuda_archs() has to drop the +PTX marker (see above), and upstream only
    ever emits `code=sm_X` -- never `code=compute_X` -- so flash_attn shipped
    NO PTX at all for any arch. That makes the wheel dead on any GPU newer
    than its newest cubin: there is no JIT path. The farm declares +PTX in
    arch_policy precisely to promise that path, so emit it.
    """
    out = []
    for a in os.getenv("TORCH_CUDA_ARCH_LIST", "").split():
        if "+PTX" in a.upper():
            out.append(a.split("+")[0].replace(".", ""))
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

    for _cuw_ptx in cuda_ptx_archs():
        cc_flag.append("-gencode")
        cc_flag.append(f"arch=compute_{_cuw_ptx},code=compute_{_cuw_ptx}")'''
_require(_ptx_anchor in content,
         "flash_attn: gencode block not found -- cannot emit the +PTX the "
         "arch policy promises. Upstream changed setup.py; update the anchor.")
content = content.replace(_ptx_anchor, _ptx_add, 1)
print("Injected PTX gencode emission for +PTX archs")

# Fix MAX_JOBS auto-detect: use arch count instead of hardcoded /9
# Upstream assumes 2 archs (9GB/job). With 4 archs (cu128+), it's ~18GB/job.
old_memory_estimate = "max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4"
new_memory_estimate = (
    "num_archs = len(cuda_archs())\n"
    "            per_job_gb = 4.5 * num_archs  # ~4.5GB per cicc process x num arch targets\n"
    "            max_num_jobs_memory = int(free_memory_gb / per_job_gb)"
)

if old_memory_estimate in content:
    content = content.replace(old_memory_estimate, new_memory_estimate)
    print(f"Patched MAX_JOBS auto-detect to use arch count (cuda_archs())")
else:
    print("WARNING: Could not find MAX_JOBS memory estimate - source may have changed")

setup_file.write_text(content)


# ── force a real compile: never adopt upstream's prebuilt wheel ──────────
# FA's setup.py looks for a matching wheel on Dao-AILab's GitHub releases
# and DOWNLOADS it instead of compiling (that is what the os.rename below
# moves into place). On linux x86 + cu12 + released torch versions a match
# exists, so the farm would silently ship upstream's binary -- wrong arch
# list, wrong toolchain, no ccache for the shard link job. Flip the
# FORCE_BUILD default to TRUE so the farm always compiles from source.
_force_old = 'FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"'
_force_new = 'FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "TRUE") == "TRUE"'
_s = Path("setup.py").read_text()
if _force_old not in _s:
    raise SystemExit("flash_attn patch: FORCE_BUILD line not found -- "
                     "upstream changed; update this patch")
Path("setup.py").write_text(_s.replace(_force_old, _force_new))
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
