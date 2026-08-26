#!/usr/bin/env python3
"""Pre-upload wheel gate: verify as much as a GPU-less container allows.

Runs INSIDE the build container after repair/rename/metadata-fix and BEFORE
upload; non-zero exit blocks publication of the wheel. All logic lives here --
the CI step is a one-liner. Design: scratchpad verify_wheel_design.md;
mechanisms proven in the verify_hacker research reports.

    python scripts/verify_wheel.py dist/ --package diso --arch-list "..." \
        --cuda 12.8 --torch 2.8.0 --platform linux

Checks (cheap -> expensive; FAIL blocks upload, WARN annotates, SKIP records):
  C1 filename      wheel name matches the cell (pkg/cuda/torch/py/plat tag)
  C2 metadata      METADATA Version == filename; RECORD hashes; zip integrity
  C3 binary_census compiled extension modules exist (unless allow_pure_python)
  C4 elf_sanity    DT_NEEDED allowlist + $ORIGIN RPATH sanity          [linux]
  C5 torch_linkage links_torch declaration is true of the binaries
  C6 glibc_ceiling GLIBC <= 2.28 / GLIBCXX satisfiable by AlmaLinux 8  [linux]
  C7 arch_sass     fatbin SASS/PTX vs the cell's arch list (cuobjdump,
                   byte-scan fallback; empty scan = UNVERIFIED warn, not fail)
  C8 import        subprocess import of facade + every compiled submodule
                   against the exact build torch (stub libcuda lane when the
                   wheel links the driver); allowlist only via package knobs

Exit codes: 0 = pass (warnings allowed), 1 = a wheel failed, 2 = the gate
itself could not run honestly (env mismatch, missing prerequisites). Never
exits 0 on "couldn't check".

Per-package knobs live in an optional `verify:` block in package.yml -- see
the design doc; packages without one get auto-derived defaults.
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from email.parser import Parser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit import (parse_wheel, extract_archs, arch_list_to_sm,  # noqa: E402
                   arch_list_to_sm_ptx, _SKIP_LIBS)
import generate_matrix as _GM  # noqa: E402
from package_loader import iter_packages  # noqa: E402

def _arch_major(sm):
    """sm_90 -> 9; sm_90a -> 9; robust to arch-variant suffixes."""
    digits = re.sub(r"\D", "", sm.replace("sm_", ""))
    n = int(digits) if digits else 0
    return n // 10


DEVICE_ERROR_PATTERNS = (
    r"No CUDA GPUs are available",
    r"Found no NVIDIA driver",
    r"CUDA driver initialization failed",
    r"CUDA_ERROR_STUB_LIBRARY",
    r"CUDA unknown error",
    # triton 3.2/3.3 initialise their driver eagerly at import and raise this
    # on a GPU-less runner; triton >= 3.4 made it lazy. Purely a CI artifact
    # -- the wheel is fine on a real GPU (review board 2026-08-24).
    r"0 active drivers",
)
GLIBC_CEILING = (2, 28)
TORCH_NEEDED_RE = re.compile(r"^(libtorch|libc10|libcaffe2_nvrtc)")
TORCH_SYM_PREFIXES = ("_ZN2at", "_ZN3c10", "_ZN5torch", "THP")
GLIBC_FAMILY = ("libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
                "ld-linux", "libutil.so", "libresolv.so")
# libcuda.so.1 is the DRIVER -- always present on a machine with a GPU, never
# shippable. libstdc++/libgcc_s/libgomp come from the platform toolchain.
ALLOWED_NEEDED_PREFIXES = ("libstdc++", "libgcc_s", "libcuda.so.1")
# cublas-class libs are OK only if the wheel vendors them or torch provides
# them at runtime (torch-linked packages ride torch's copies; a torch-FREE
# package linking these unvendored ships a wheel that cannot load).
#
# libcudart MOVED HERE 2026-08-25. It was in ALLOWED_NEEDED_PREFIXES, i.e.
# unconditionally waved through -- which defeated this rule for precisely the
# package it was written for. cumm declares links_torch: false, DT_NEEDEDs
# libcudart.so.13, vendors only nvrtc, and nothing else in the env preloads
# cudart because it never imports torch. On a box without a system CUDA
# toolkit that wheel cannot dlopen:
#   libcudart.so.13: cannot open shared object file
# Torch-linked packages are unaffected: `import torch` loads cudart from
# site-packages/nvidia/ before their extension is imported, which is exactly
# the distinction this tuple encodes.
TORCH_PROVIDED = ("libcublas", "libcusparse", "libcufft", "libcurand",
                  "libcusolver", "libcudnn", "libnccl", "libnvrtc",
                  "libnvJitLink", "libcudart", "libgomp")
# libgomp ADDED 2026-08-25, same class as libcudart one library over. It was
# in ALLOWED_NEEDED_PREFIXES (unconditional pass) AND excluded from auditwheel
# vendoring unconditionally AND is in no manylinux lib_whitelist at any policy
# level -- so a torch-FREE package linking OpenMP would ship broken and green.
# torch ships torch/lib/libgomp.so.1 with an unmangled SONAME and loads it at
# `import torch`, so torch-linked packages are genuinely covered; torch-free
# ones are not. Currently latent (no wheel in the corpus needs it) -- fixed as
# a class rather than waiting for an instance.


def log(msg):
    print(msg, flush=True)


def annotate(level, msg):
    print(f"::{level}::{msg}", flush=True)


class WheelReport:
    def __init__(self, filename):
        self.filename = filename
        self.checks = []
        self.failed = False

    def add(self, cid, status, summary, data=None):
        self.checks.append({"id": cid, "status": status, "summary": summary,
                            "data": data or {}})
        if status == "fail":
            self.failed = True
            annotate("error", f"[{cid}] {self.filename}: {summary}")
        elif status == "warn":
            annotate("warning", f"[{cid}] {self.filename}: {summary}")
        else:
            log(f"  [{cid}] {status}: {summary}")


def gate_error(msg):
    annotate("error", f"verify_wheel gate broken (not a wheel defect): {msg}")
    sys.exit(2)


PLATFORM_KEYS = ("linux", "linux_aarch64", "windows")


def platform_knob(vknobs, name, platform):
    """Resolve a `verify.<name>` knob that may be per-platform.

    Two accepted shapes:

        skip_arch: "reason"                  # every platform
        skip_arch: {linux: "reason a",       # this platform only
                    windows: "reason b"}

    A waiver written as ONE string is a claim about every lane the farm
    builds, and that is how a platform-specific premise ends up switching a
    check off farm-wide. cumm's `skip_arch` justified itself with "the
    vendored NVRTC (99MB libnvrtc in the wheel)" -- true on Linux, where
    auditwheel grafts libnvrtc.so.13 + libnvrtc-builtins into `cumm.libs/`,
    and false on Windows, where the wheel bundles no DLL at all. The single
    string still disabled the arch gate on both.

    A mapping forces one written-down reason per lane, and a lane with NO
    entry is NOT waived -- the omission fails closed, towards checking.
    Unknown platform keys are a gate error: a typo must not read as a lane
    that simply was not mentioned.
    """
    val = vknobs.get(name)
    if not isinstance(val, dict):
        return val
    unknown = sorted(set(val) - set(PLATFORM_KEYS))
    if unknown:
        gate_error(f"verify.{name}: unknown platform key(s) {unknown}; "
                   f"expected a subset of {list(PLATFORM_KEYS)}")
    return val.get(platform)


def load_pkg_config(package):
    want = package.replace("-", "_").lower()
    for _folder, cfg in iter_packages():
        if cfg["name"].replace("-", "_").lower() == want:
            return cfg
    gate_error(f"no package named {package!r} in packages/")


def wheel_members(zf):
    """(extension_modules, vendored_libs, others) member paths."""
    exts, vendored = [], []
    for name in zf.namelist():
        base = name.rsplit("/", 1)[-1]
        if name.endswith(".dll"):
            # Windows DLLs are never importable modules -- they are dlopen
            # targets (llama_cpp's lib/*.dll) or vendored dependencies.
            vendored.append(name)
            continue
        if not (name.endswith(".so") or name.endswith(".pyd")
                or ".so." in base):
            continue
        if ".libs/" in name or base.startswith(_SKIP_LIBS):
            vendored.append(name)
        else:
            exts.append(name)
    return exts, vendored


def module_name_for(member):
    """pkg/_C.cpython-312-x86_64-linux-gnu.so -> pkg._C  (None if not a module)."""
    base = member.rsplit("/", 1)[-1]
    if base.startswith("lib") and ".cpython-" not in base:
        return None  # plain shared library (libllama.so, libpyg.pyd), not a module
    if base.endswith(".pyd"):
        stem = base[:-4]
    elif base.endswith(".so"):
        stem = base[:-3]
    else:
        return None
    stem = stem.split(".", 1)[0]  # strip cpython-312-... ABI tag
    if not stem.isidentifier():
        return None
    parts = member.split("/")[:-1] + [stem]
    if any(not p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


# ── C1 ─────────────────────────────────────────────────────────────────────

def check_filename(rep, parsed, args, pkg, vknobs=None):
    if parsed is None:
        if (vknobs or {}).get("allow_pure_python"):
            rep.add("filename", "skip",
                    "non-standard name accepted for allow_pure_python package")
        else:
            rep.add("filename", "fail", "filename does not parse as a farm wheel")
        return
    problems = []
    if parsed["package"].replace("-", "_").lower() != args.package.replace("-", "_").lower():
        problems.append(f"package {parsed['package']!r} != {args.package!r}")
    if parsed["cuda"] != args.cuda:
        problems.append(f"cuda {parsed['cuda']} != {args.cuda}")
    want_minor = ".".join(args.torch.split(".")[:2])
    if parsed["torch_short"] != want_minor:
        problems.append(f"torch {parsed['torch_short']} != {want_minor}")
    py_here = f"{sys.version_info.major}{sys.version_info.minor}"
    abi = parsed.get("abi", "cp")
    if abi == "cp":
        if parsed["python"] != py_here:
            problems.append(f"python cp{parsed['python']} != build env cp{py_here}")
    elif abi == "abi3":
        # Stable-ABI wheel: valid for every python >= its cp floor.
        if int(parsed["python"]) > int(py_here):
            problems.append(f"abi3 floor cp{parsed['python']} above build env cp{py_here}")
    # abi == "none" (py3-none): abi-agnostic, any python 3 -- no check.
    plat_tag = parsed["plat_tag"]
    if args.platform == "linux" and "manylinux_2_28_x86_64" not in plat_tag:
        problems.append(f"platform tag {plat_tag!r} lacks manylinux_2_28_x86_64 "
                        f"(auditwheel repair skipped?)")
    if args.platform == "linux_aarch64" and "manylinux_2_28_aarch64" not in plat_tag:
        problems.append(f"platform tag {plat_tag!r} lacks manylinux_2_28_aarch64")
    if args.platform == "windows" and "win_amd64" not in plat_tag:
        problems.append(f"platform tag {plat_tag!r} is not win_amd64")
    pinned = str(pkg.get("version") or "")
    if pinned and parsed["version"] != pinned:
        problems.append(f"version {parsed['version']} != pinned {pinned}")
    if problems:
        rep.add("filename", "fail", "; ".join(problems))
    else:
        rep.add("filename", "pass", "filename matches the cell")


# ── C2 ─────────────────────────────────────────────────────────────────────

def check_metadata(rep, wheel_path, parsed, pkg):
    import base64
    import hashlib
    try:
        zf = zipfile.ZipFile(wheel_path)
    except zipfile.BadZipFile:
        rep.add("metadata", "fail", "not a valid zip archive")
        return
    with zf:
        bad = zf.testzip()
        if bad:
            rep.add("metadata", "fail", f"corrupt member: {bad}")
            return
        names = zf.namelist()
        meta_name = next((n for n in names
                          if n.endswith(".dist-info/METADATA")), None)
        if not meta_name:
            rep.add("metadata", "fail", "no METADATA in wheel")
            return
        meta = Parser().parsestr(zf.read(meta_name).decode("utf-8", "replace"))
        want_version = f"{parsed['version']}+cu{parsed['cuda_short']}torch{parsed['torch_short']}" if parsed else None
        got_version = meta.get("Version", "")
        if want_version and got_version != want_version:
            rep.add("metadata", "fail",
                    f"METADATA Version {got_version!r} != filename {want_version!r} "
                    f"(patch_wheel_version regression)")
            return
        # The farm publishes wheels that declare NO dependencies. Every
        # Requires-Dist and Provides-Extra header is stripped from every wheel
        # by patch_wheel_version.strip_all_requires_dist (owner decision
        # 2026-08-25). A survivor means the patch step was skipped or upstream's
        # header format defeated it.
        #
        # Why empty rather than curated: comfy-env installs these by direct URL
        # and must not have the resolver chase a wheel's own dependency list --
        # pixi has no `--no-deps`, so empty metadata is how that is expressed.
        # It also keeps the farm index from ever needing to be registered, and
        # therefore from shadowing PyPI for the 17 names it shares with it.
        # Runtime deps are declared by the consuming node pack's comfy-env.toml
        # and enforced by `comfy-test run --cuda`.
        #
        # This replaced a check that asserted the wheel matched the config's
        # curated list -- a tautology that proved the patch step ran, never
        # that the list was true. "No dependencies" is falsifiable by the
        # artifact alone.
        leftover_reqs = meta.get_all("Requires-Dist") or []
        leftover_extras = meta.get_all("Provides-Extra") or []
        if leftover_reqs or leftover_extras:
            rep.add("metadata", "fail",
                    f"wheel still declares dependencies: "
                    f"Requires-Dist={leftover_reqs} "
                    f"Provides-Extra={leftover_extras} -- the farm ships no "
                    f"dependency metadata (patch_wheel_version regression)")
            return
        # RECORD re-verify
        rec_name = meta_name.rsplit("/", 1)[0] + "/RECORD"
        if rec_name not in names:
            rep.add("metadata", "fail", "no RECORD in wheel")
            return
        mismatches = []
        for line in zf.read(rec_name).decode().splitlines():
            parts = line.rsplit(",", 2)
            if len(parts) != 3 or not parts[1]:
                continue
            fname, digest, _size = parts
            fname = fname.replace("\\", "/")  # legacy Windows RECORDs
            if fname not in names:
                mismatches.append(f"{fname}: listed but absent")
                continue
            algo, _, b64 = digest.partition("=")
            h = hashlib.new(algo)
            h.update(zf.read(fname))
            got = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
            if got != b64:
                mismatches.append(f"{fname}: {algo} mismatch")
        if mismatches:
            rep.add("metadata", "fail",
                    f"RECORD verification failed: {'; '.join(mismatches[:5])}")
        else:
            rep.add("metadata", "pass",
                    f"version {got_version}; RECORD verified")


# ── C3 ─────────────────────────────────────────────────────────────────────

def check_binary_census(rep, wheel_path, vknobs, exts, vendored):
    if not exts and not vendored and not vknobs.get("allow_pure_python"):
        rep.add("binary_census", "fail",
                "no compiled binaries in wheel at all -- the CUDA compile "
                "silently produced a pure-python wheel (set "
                "verify.allow_pure_python only for JIT-only packages)")
    elif not exts and vendored:
        # dlopen-style package (llama_cpp on Windows: DLLs, no .pyd) --
        # compiled code exists, just nothing directly importable.
        rep.add("binary_census", "pass",
                f"no importable extensions; {len(vendored)} dlopen-style/"
                f"vendored binaries", {"vendored": vendored})
    elif not exts:
        rep.add("binary_census", "skip",
                "pure-python by design (verify.allow_pure_python)",
                {"vendored": len(vendored)})
    else:
        rep.add("binary_census", "pass",
                f"{len(exts)} extension module(s), {len(vendored)} vendored lib(s)",
                {"extensions": exts, "vendored": vendored})


# ── ELF helpers (C4/C5/C6) ────────────────────────────────────────────────

def _elf_info(data):
    """{'needed': [...], 'rpath': [...], 'undef': [...], 'verneed': {lib: [ver,..]}}"""
    from io import BytesIO
    from elftools.elf.elffile import ELFFile
    from elftools.elf.dynamic import DynamicSection
    from elftools.elf.sections import SymbolTableSection
    from elftools.elf.gnuversions import GNUVerNeedSection
    out = {"needed": [], "rpath": [], "undef": [], "verneed": {}}
    elf = ELFFile(BytesIO(data))
    for sec in elf.iter_sections():
        if isinstance(sec, DynamicSection):
            for tag in sec.iter_tags():
                if tag.entry.d_tag == "DT_NEEDED":
                    out["needed"].append(tag.needed)
                elif tag.entry.d_tag in ("DT_RPATH", "DT_RUNPATH"):
                    out["rpath"].extend(str(tag.rpath if tag.entry.d_tag == "DT_RPATH" else tag.runpath).split(":"))
        elif isinstance(sec, GNUVerNeedSection):
            for verneed, vernaux in sec.iter_versions():
                out["verneed"].setdefault(verneed.name, []).extend(
                    aux.name for aux in vernaux)
        elif isinstance(sec, SymbolTableSection) and sec.name == ".dynsym":
            for sym in sec.iter_symbols():
                if (sym["st_shndx"] == "SHN_UNDEF" and sym.name
                        and sym["st_info"]["bind"] != "STB_WEAK"):
                    out["undef"].append(sym.name)
    return out


def elf_infos_for(wheel_path, members):
    infos = {}
    with zipfile.ZipFile(wheel_path) as zf:
        for m in members:
            try:
                infos[m] = _elf_info(zf.read(m))
            except Exception as e:  # noqa: BLE001 -- record, don't crash the gate
                infos[m] = {"error": str(e)}
    return infos


def check_elf_sanity(rep, infos, vendored, vknobs, links_torch=True):
    # Anything shipped in the wheel itself satisfies a DT_NEEDED on its
    # basename: multi-library packages (llama_cpp's GGML family) link
    # their siblings via $ORIGIN rpaths -- self-contained, not a leak.
    vendored_sonames = {v.rsplit("/", 1)[-1] for v in vendored}
    inwheel = vendored_sonames | {m.rsplit("/", 1)[-1] for m in infos}
    # Per-package justified externals (CW-ADR-0009's "or justify"):
    # fnmatch patterns from verify.allowed_needed.
    allowed_pats = list(vknobs.get("allowed_needed") or [])
    problems, driver_linked = [], False
    for m, info in infos.items():
        if "error" in info:
            problems.append(f"{m}: unparseable ELF ({info['error']})")
            continue
        for need in info["needed"]:
            if need == "libcuda.so.1":
                driver_linked = True
                continue
            vendored_ok = (need in inwheel
                           or any(need in v for v in vendored_sonames))
            torch_class = any(need.startswith(p) for p in TORCH_PROVIDED)
            ok = (any(f in need for f in GLIBC_FAMILY)
                  or TORCH_NEEDED_RE.match(need)
                  or any(need.startswith(p) for p in ALLOWED_NEEDED_PREFIXES)
                  or vendored_ok
                  or any(fnmatch.fnmatch(need, p) for p in allowed_pats)
                  or (torch_class and links_torch))
            if not ok:
                why = ("torch-free package links it unvendored -- torch will "
                       "not be there to provide it"
                       if torch_class else
                       "vendor it or justify per CW-ADR-0009")
                problems.append(f"{m}: unexpected DT_NEEDED {need!r} ({why})")
        for rp in info["rpath"]:
            if rp and not rp.startswith("$ORIGIN"):
                problems.append(f"{m}: non-$ORIGIN RPATH entry {rp!r}")
    if problems:
        rep.add("elf_sanity", "fail", "; ".join(problems[:6]))
    else:
        rep.add("elf_sanity", "pass",
                f"NEEDED/RPATH clean across {len(infos)} binaries"
                + (" (driver-linked)" if driver_linked else ""))
    if driver_linked and not vknobs.get("needs_driver"):
        rep.add("elf_sanity", "warn",
                "wheel links libcuda.so.1 but package.yml lacks "
                "verify.needs_driver -- set it to keep the config honest")
    return driver_linked


def check_torch_linkage(rep, infos, pkg, platform):
    links_torch = pkg.get("links_torch") is not False
    bad = [m for m, i in infos.items() if "error" in i]
    if bad:
        rep.add("torch_linkage", "warn",
                f"{len(bad)} binaries unparseable -- linkage evidence incomplete")
    found_needed, found_syms = [], []
    for m, info in infos.items():
        if "error" in info:
            continue
        found_needed += [f"{m}:{n}" for n in info["needed"] if TORCH_NEEDED_RE.match(n)]
        found_syms += [f"{m}:{s}" for s in info["undef"]
                       if s.startswith(TORCH_SYM_PREFIXES)]
    if not links_torch:
        if found_needed or found_syms:
            ev = (found_needed + found_syms)[:4]
            msg = (f"links_torch: false but the binary IS torch-linked "
                   f"({'; '.join(ev)}) -- aliasing this wheel across torch "
                   f"versions would ship ABI breakage (CW-ADR-0011)")
            if platform == "windows":
                rep.add("torch_linkage", "warn", msg + " [PE evidence, warn-only]")
            else:
                rep.add("torch_linkage", "fail", msg)
        else:
            rep.add("torch_linkage", "pass",
                    "torch-free binary confirmed: aliasing-safe by construction")
    else:
        if infos and not (found_needed or found_syms):
            rep.add("torch_linkage", "warn",
                    "links_torch: true but no torch linkage found in any "
                    "binary -- accidental CPU-only build, or should this be "
                    "links_torch: false?")
        else:
            rep.add("torch_linkage", "pass",
                    f"torch linkage present ({len(found_needed)} NEEDED, "
                    f"{len(found_syms)} symbols)")


def check_glibc_ceiling(rep, infos, vendored):
    vendors_libstdcxx = any("libstdc++" in v for v in vendored)
    worst_glibc, worst_owner = (0, 0), None
    cxx_demands = set()
    for m, info in infos.items():
        if "error" in info:
            continue
        for lib, vers in info["verneed"].items():
            for v in vers:
                if v.startswith("GLIBC_"):
                    try:
                        t = tuple(int(x) for x in v[6:].split("."))
                    except ValueError:
                        continue  # GLIBC_PRIVATE / GLIBC_ABI_DT_RELR etc.
                    if t > worst_glibc:
                        worst_glibc, worst_owner = t, m
                elif v.startswith(("GLIBCXX_", "CXXABI_")):
                    cxx_demands.add(v)
    if worst_glibc > GLIBC_CEILING:
        rep.add("glibc_ceiling", "fail",
                f"{worst_owner} requires GLIBC_{'.'.join(map(str, worst_glibc))} "
                f"> 2.28 floor -- wheel will not load on manylinux_2_28 hosts")
        return
    # GLIBCXX/CXXABI: satisfiable by the container's own system libstdc++
    unsat = set()
    sys_libstdcxx = Path("/usr/lib64/libstdc++.so.6")
    if cxx_demands and sys_libstdcxx.exists():
        try:
            defined = set(re.findall(rb"(GLIBCXX_[\d.]+|CXXABI_[\d.]+)",
                                     sys_libstdcxx.resolve().read_bytes()))
            defined = {d.decode().rstrip(".") for d in defined}
            unsat = {d for d in cxx_demands if d.rstrip(".") not in defined}
        except OSError:
            pass
    if unsat and not vendors_libstdcxx:
        rep.add("glibc_ceiling", "fail",
                f"libstdc++ demands beyond the AlmaLinux 8 baseline: "
                f"{sorted(unsat)[:4]}")
    elif unsat:
        rep.add("glibc_ceiling", "warn",
                f"libstdc++ demands {sorted(unsat)[:4]} satisfied only by the "
                f"wheel's vendored libstdc++")
    else:
        rep.add("glibc_ceiling", "pass",
                f"max GLIBC_{'.'.join(map(str, worst_glibc)) if worst_owner else 'n/a'}"
                f" <= 2.28; libstdc++ demands within baseline")


# ── C7 ─────────────────────────────────────────────────────────────────────

def _cuobjdump_archs(binary_path, cuobjdump, timeout):
    sass, ptx = set(), set()
    for flag, target in (("--list-elf", sass), ("--list-ptx", ptx)):
        r = subprocess.run([cuobjdump, flag, str(binary_path)],
                           capture_output=True, text=True, timeout=timeout)
        for m in re.finditer(r"sm_(\d+)", r.stdout):
            target.add(f"sm_{m.group(1)}")
    return sass, ptx


def check_arch_sass(rep, wheel_path, exts, args, vknobs):
    # Per-platform waiver: see platform_knob(). A lane the mapping does not
    # mention is checked for real.
    waiver = platform_knob(vknobs, "skip_arch", args.platform)
    if waiver:
        rep.add("arch_sass", "skip",
                f"verify.skip_arch[{args.platform}]: {waiver}")
        return
    if not exts:
        rep.add("arch_sass", "skip", "no compiled members")
        return
    expected = arch_list_to_sm(args.arch_list) if args.arch_list else set()
    if not expected:
        rep.add("arch_sass", "skip", "no --arch-list provided")
        return
    cuobjdump = None
    cuda_home = args.cuda_home or os.environ.get("CUDA_HOME") or f"/usr/local/cuda-{args.cuda}"
    for cand in (Path(cuda_home) / "bin" / "cuobjdump",
                 Path(cuda_home) / "bin" / "cuobjdump.exe"):
        if cand.exists():
            cuobjdump = str(cand)
            break
    sass, ptx = set(), set()
    source = "none"
    timeout = int(vknobs.get("cuobjdump_timeout", 300))
    if cuobjdump:
        try:
            with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(wheel_path) as zf:
                for m in exts:
                    p = Path(td) / m.rsplit("/", 1)[-1]
                    p.write_bytes(zf.read(m))
                    s, x = _cuobjdump_archs(p, cuobjdump, timeout)
                    sass |= s
                    ptx |= x
            source = "cuobjdump" if (sass or ptx) else "none"
        except (subprocess.TimeoutExpired, OSError):
            source = "none"
    if source == "none":
        got = extract_archs(str(wheel_path))
        sass, ptx = got["sass"], got["ptx"]
        source = "byte-scan" if (sass or ptx) else "none"
    data = {"expected": sorted(expected), "sass": sorted(sass),
            "ptx": sorted(ptx), "source": source}
    if not sass and not ptx:
        rep.add("arch_sass", "warn",
                "UNVERIFIED: no SASS/PTX visible to any scanner (compressed "
                "fatbin without cuobjdump?) -- confirm by hand", data)
        return
    # cuobjdump reports arch-VARIANT names: Hopper cubins come back as `sm_90a`
    # (arch-specific features), Blackwell can be `sm_100f`. The arch list spells
    # them `9.0`/`10.0`. Fold the suffix before comparing, or every Hopper wheel
    # reads as "missing sm_90" the moment this check fails instead of warns.
    def _norm(a):
        m = re.match(r"^(sm_\d+)[a-z]*$", a)
        return m.group(1) if m else a
    sass = {_norm(a) for a in sass}
    ptx = {_norm(a) for a in ptx}
    actual = sass | ptx
    data["sass"], data["ptx"] = sorted(sass), sorted(ptx)
    exp_sass, exp_ptx = arch_list_to_sm_ptx(args.arch_list)
    # Documented upstream gaps only. A package whose UPSTREAM has no kernel for
    # an arch declares it here with a reason; anything else is a defect.
    waived = set(vknobs.get("allow_missing_archs") or [])
    data["expected_ptx"] = sorted(exp_ptx)
    data["waived"] = sorted(waived)

    # `expected - waived`, not `expected`. allow_missing_archs was subtracted
    # only in the EXACT-arch check below, which made it useless for the case it
    # exists for: an arch upstream cannot build at all is usually the only
    # member of its family, so the family check fired first and failed the
    # wheel before the waiver was ever consulted. That left narrowing the arch
    # list as the only way to get such a package green -- and narrowing moves
    # both sides of this comparison, so the gap vanishes from the record
    # instead of being recorded as waived. (sageattention: upstream's
    # SUPPORTED_ARCHS has no 10.0, so sm_100 is the whole sm_10x family.)
    missing_majors = ({_arch_major(a) for a in (expected - waived)}
                      - {_arch_major(a) for a in actual})
    if missing_majors:
        rep.add("arch_sass", "fail",
                f"missing arch families sm_{sorted(missing_majors)} -- "
                f"expected {sorted(expected)}, found {sorted(actual)} "
                f"[{source}]", data)
        return

    problems = []
    # (1) Exact archs. Was a WARN, which meant you could drop every consumer
    # Ampere and Ada cubin and still exit 0 as long as one sm_80 survived --
    # the family check compares majors, so {8} == {8}. Promoted to fail.
    missing_exact = expected - actual - waived
    if missing_exact:
        problems.append(
            f"missing cubin/PTX for {sorted(missing_exact)} -- the wheel has no "
            f"code path on those GPUs")
    # (2) PTX. The `+PTX` marker was discarded before comparison, so a wheel
    # that declared forward-compat and shipped none looked identical to one
    # that shipped it. A cubin-only wheel is dead on every future arch.
    missing_ptx = exp_ptx - ptx - waived
    if missing_ptx:
        problems.append(
            f"declared +PTX for {sorted(missing_ptx)} but shipped NO PTX for them "
            f"(have PTX: {sorted(ptx) or 'none'}) -- no JIT path onto newer GPUs")
    if problems:
        rep.add("arch_sass", "fail", "; ".join(problems) + f" [{source}]", data)
        return

    notes = []
    if waived & (expected - actual):
        notes.append(f"waived (documented upstream gap): {sorted(waived & (expected - actual))}")
    extra = actual - expected
    if extra:
        notes.append(f"extra archs {sorted(extra)}")
    if notes:
        rep.add("arch_sass", "warn", "; ".join(notes) + f" [{source}]", data)
    else:
        rep.add("arch_sass", "pass",
                f"all {len(expected)} expected archs + {len(exp_ptx)} PTX present "
                f"[{source}]", data)


# ── C8 ─────────────────────────────────────────────────────────────────────

CHILD_SCRIPT = r"""
import importlib, json, os, sys, traceback
spec = json.loads(sys.argv[1])
out = {"phases": [], "torch_ops": {}}
def phase(name, module, fn):
    entry = {"phase": name, "module": module, "ok": False, "error": None}
    print(f"VERIFY_PROG:{name}:{module}", flush=True)
    try:
        fn(); entry["ok"] = True
    except BaseException as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc()[-2000:]
    out["phases"].append(entry)
    return entry["ok"], entry
ok, _ = phase("torch", "torch", lambda: __import__("torch"))
if ok:
    import torch
    v_ok = torch.__version__.startswith(spec["torch"])
    cu_ok = (torch.version.cuda or "").replace(".", "").startswith(spec["cuda_short"])
    out["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                  "env_ok": bool(v_ok and cu_ok),
                  "cuda_available": torch.cuda.is_available()}
    for mod in spec["facade"]:
        f_ok, f_entry = phase("facade", mod, lambda m=mod: importlib.import_module(m))
        # Windows WinError 126 is opaque ("...or one of its dependencies").
        # Name the culprit: probe every DLL the package ships individually.
        if not f_ok and os.name == "nt" and "Could not find module" in (f_entry["error"] or ""):
            import ctypes, glob
            probes = {}
            sites = [p for p in sys.path if "verify-import" in p] or sys.path[:1]
            for site in sites:
                for dll in glob.glob(os.path.join(site, "**", "*.dll"), recursive=True):
                    rel = os.path.basename(dll)
                    if rel in probes:
                        continue
                    try:
                        ctypes.WinDLL(dll)
                        probes[rel] = "loads"
                    except OSError as pe:
                        probes[rel] = f"FAILS: {pe}"
            f_entry["dll_probe"] = probes
            bad = {k: v for k, v in probes.items() if v != "loads"}
            print("VERIFY_PROG:dll_probe:" + json.dumps(bad)[:1500], flush=True)
    for mod in spec["compiled"]:
        phase("compiled", mod, lambda m=mod: importlib.import_module(m))
    for op in spec.get("expect_ops", []):
        ns, _, name = op.partition("::")
        phase("op", op, lambda ns=ns, name=name: getattr(getattr(torch.ops, ns), name))
    try:
        import torch  # snapshot registered ops per facade namespace
        for op in spec.get("expect_ops", []):
            ns = op.partition("::")[0]
            out["torch_ops"][ns] = sorted(dir(getattr(torch.ops, ns)))[:200]
    except Exception:
        pass
    out["cuda_available_after"] = torch.cuda.is_available()
print("VERIFY_JSON:" + json.dumps(out))
"""


def check_import(rep, wheel_path, parsed, exts, args, vknobs, driver_linked):
    if vknobs.get("skip_import"):
        rep.add("import", "skip", f"verify.skip_import: {vknobs['skip_import']}")
        return
    needs_driver = driver_linked or bool(vknobs.get("needs_driver"))
    if args.platform == "windows" and needs_driver:
        rep.add("import", "skip",
                "driver-linked wheel on Windows: no nvcuda stub exists; "
                "static checks still gate (Linux cell runs the full lane)")
        return

    # facade modules
    facade = vknobs.get("import_name")
    if facade is None:
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            tl = next((n for n in names if n.endswith(".dist-info/top_level.txt")), None)
            if tl:
                facade = [l.strip() for l in zf.read(tl).decode().splitlines() if l.strip()]
            else:
                facade = sorted({n.split("/")[0] for n in names
                                 if n.endswith("/__init__.py") and n.count("/") == 1})
    elif isinstance(facade, str):
        facade = [facade]
    facade = list(facade) + list(vknobs.get("extra_imports") or [])

    # compiled submodules (the hard assert)
    no_direct = vknobs.get("no_direct_import") or []
    compiled = []
    for m in exts:
        if any(fnmatch.fnmatch(m, pat) for pat in no_direct):
            continue
        mod = module_name_for(m)
        if mod:
            compiled.append(mod)

    spec = {"torch": ".".join(args.torch.split(".")[:2]),
            "cuda_short": args.cuda.replace(".", ""),
            "facade": facade, "compiled": sorted(set(compiled)),
            "expect_ops": vknobs.get("expect_ops") or []}

    with tempfile.TemporaryDirectory(prefix="verify-import-") as td:
        target = Path(td) / "site"
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                            "--no-index", "--quiet", "--target", str(target),
                            str(wheel_path)], capture_output=True, text=True)
        if r.returncode != 0:
            rep.add("import", "fail",
                    f"pip install --target failed: {r.stderr.strip()[-400:]}")
            return
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Force eager symbol binding. CPython already dlopens extensions with
        # RTLD_NOW, but that is an implicit guarantee: one upstream
        # setdlopenflags change, or one package setting verify.skip_import, and
        # a wheel whose kernels failed to link would import clean. Belt and
        # braces, and it makes the failure name the symbol.
        env["LD_BIND_NOW"] = "1"
        env.pop("TORCH_CUDA_ARCH_LIST", None)
        env["PYTHONPATH"] = str(target)
        if needs_driver and args.platform != "windows":
            cuda_home = args.cuda_home or env.get("CUDA_HOME") or f"/usr/local/cuda-{args.cuda}"
            stub_src = Path(cuda_home) / "lib64" / "stubs" / "libcuda.so"
            if stub_src.exists():
                stubdir = Path(td) / "stubs"
                stubdir.mkdir()
                (stubdir / "libcuda.so.1").symlink_to(stub_src)
                env["LD_LIBRARY_PATH"] = f"{stubdir}:{env.get('LD_LIBRARY_PATH', '')}"
            else:
                rep.add("import", "warn",
                        f"driver-linked but no stub at {stub_src}; import may fail")
        empty_cwd = Path(td) / "cwd"
        empty_cwd.mkdir()
        try:
            r = subprocess.run([sys.executable, "-c", CHILD_SCRIPT, json.dumps(spec)],
                               capture_output=True, text=True, timeout=600,
                               env=env, cwd=empty_cwd)
        except subprocess.TimeoutExpired:
            rep.add("import", "fail", "import child timed out (600s)")
            return
    try:
        marker = [l for l in r.stdout.splitlines() if l.startswith("VERIFY_JSON:")]
        result = json.loads(marker[-1][len("VERIFY_JSON:"):])
    except (ValueError, IndexError):
        prog = [l for l in r.stdout.splitlines() if l.startswith("VERIFY_PROG:")]
        last = prog[-1][len("VERIFY_PROG:"):] if prog else "before any import"
        rep.add("import", "fail",
                f"import child crashed (rc={r.returncode}) while importing "
                f"[{last}]: {(r.stderr or r.stdout).strip()[-400:]}")
        return

    envd = result.get("env") or {}
    if not envd:
        rep.add("import", "fail", "child could not import torch itself")
        return
    if not envd.get("env_ok"):
        gate_error(f"build env has torch {envd.get('torch')} cuda {envd.get('cuda')}, "
                   f"cell wants torch {spec['torch']} cu{spec['cuda_short']}")
    if result.get("cuda_available_after"):
        rep.add("import", "fail",
                "torch.cuda.is_available() became True on a GPU-less runner "
                "-- something initialized CUDA against the stub")
        return

    with zipfile.ZipFile(wheel_path) as zf:
        own_toplevels = {n.split("/")[0].split(".")[0] for n in zf.namelist()}
    dev_patterns = list(DEVICE_ERROR_PATTERNS)
    ede = vknobs.get("expect_device_error")
    if isinstance(ede, str):
        dev_patterns.append(ede)
    forgivable = bool(ede)
    failures, forgiven = [], []
    undeclared = set()
    # top_level.txt entries that resolve to no shipped path at all.
    try:
        with zipfile.ZipFile(wheel_path) as _zf:
            _names = _zf.namelist()
            _tl = [n for n in _names if n.endswith(".dist-info/top_level.txt")]
            _declared = set()
            if _tl:
                _declared = {l.strip() for l in
                             _zf.read(_tl[0]).decode().splitlines() if l.strip()}
        def _provides_toplevel(d, names):
            # A top_level entry means `import <d>` works, so the member must be
            # at the ROOT of the wheel: d.py, d/..., or d<ext>.so. Matching by
            # BASENAME anywhere in the tree is wrong -- cumm/core_cc*.so is
            # importable as cumm.core_cc, NOT as core_cc, and a basename match
            # would call that satisfied and miss the very defect this detects.
            for n in names:
                if "/" in n.rstrip("/"):
                    continue                      # nested: cannot be top-level
                if n == f"{d}.py" or n.split(".")[0] == d:
                    return True
            return any(n.startswith(f"{d}/") for n in names)
        declared_phantoms = {d for d in _declared
                             if not _provides_toplevel(d, _names)}
    except Exception:
        declared_phantoms = set()
    for ph in result["phases"]:
        if ph["ok"] or ph["phase"] == "torch":
            continue
        err = ph.get("error") or ""
        is_device = any(re.search(p, err) for p in dev_patterns)
        is_load = err.startswith(("ImportError", "OSError", "ModuleNotFoundError"))
        mnfe = re.search(r"No module named '([^.']+)", err)
        missing_foreign = bool(mnfe) and mnfe.group(1) not in own_toplevels
        if ph["phase"] == "compiled" and is_load and not missing_foreign:
            failures.append(f"{ph['module']}: {err}")        # dlopen never forgiven
        elif missing_foreign:
            # The farm ships NO Requires-Dist by design -- runtime deps are
            # declared by the consuming node pack's comfy-env.toml. So this is
            # not a wheel defect and must not fail the gate.
            #
            # But it was being swallowed into a generic "forgiven" bucket, and
            # once the metadata strip landed that turned this branch into the
            # thing hiding the blast radius: an adversarial audit found 137 of
            # 362 published wheels fail a bare `import`, every one of them seen
            # by this check and reported as pass (2026-08-25). Name the module
            # so the undeclared-dependency surface is a published fact rather
            # than a green tick.
            # A top_level.txt entry that matches NO shipped path is a wheel
            # defect, not a missing dependency -- but both surface as the same
            # ModuleNotFoundError, so this branch was filing the former as the
            # latter. cumm declares top_level `core_cc` while the real module
            # is `cumm.core_cc`, and it was being published as "cumm has an
            # undeclared third-party dependency named core_cc". It does not.
            if mnfe.group(1) in declared_phantoms:
                failures.append(
                    f"{ph['module']}: top_level.txt declares {mnfe.group(1)!r} "
                    f"but no member of the wheel provides it -- phantom "
                    f"top-level entry, not a missing dependency")
            else:
                undeclared.add(mnfe.group(1))
                forgiven.append(f"{ph['module']}: missing runtime dep ({err})")
        elif is_device and forgivable:
            forgiven.append(f"{ph['module']}: {err}")
        else:
            # Surface the per-DLL probe in the failure text. It is computed in
            # the child (see the WinError-126 probe) and stored on the phase,
            # but the child's stdout is only echoed when it CRASHES -- so on a
            # clean non-zero import the one diagnostic that names the missing
            # DLL was being discarded, leaving WinError 126 unattributable
            # (llama_cpp_python Windows, open since 2026-08-22).
            probe = ph.get("dll_probe")
            detail = f" [dll_probe: {probe}]" if probe else ""
            failures.append(f"{ph['module']} [{ph['phase']}]: {err}{detail}")
    data = {"stub_lane": needs_driver, "facade": facade,
            "compiled": spec["compiled"], "forgiven": forgiven,
            "undeclared_deps": sorted(undeclared),
            "phases": result["phases"]}
    if undeclared:
        # Machine-readable and loud. This is the list a node pack's
        # comfy-env.toml must carry for the wheel to actually import.
        annotate("warning",
                 f"{Path(wheel_path).name}: imports {len(undeclared)} module(s) "
                 f"it does not declare and nothing else provides: "
                 f"{', '.join(sorted(undeclared))} -- the consuming "
                 f"comfy-env.toml must declare these or `import` will fail")
    if failures:
        rep.add("import", "fail", "; ".join(failures[:4]), data)
    elif forgiven:
        rep.add("import", "warn",
                f"passed with {len(forgiven)} forgiven device error(s) "
                f"(verify.expect_device_error): {forgiven[0]}", data)
    else:
        rep.add("import", "pass",
                f"imported {len(facade)} facade + {len(spec['compiled'])} "
                f"compiled module(s) against torch {envd['torch']}", data)


# ── driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wheels", nargs="+", help="wheel file(s) or a directory")
    ap.add_argument("--package", required=True)
    ap.add_argument("--arch-list", default="")
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--torch", required=True)
    ap.add_argument("--platform", required=True,
                    choices=["linux", "linux_aarch64", "windows"])
    ap.add_argument("--report", default=None)
    ap.add_argument("--skip", default="", help="comma list of check ids")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--cuda-home", default=None)
    args = ap.parse_args()
    t0 = time.time()

    # The aarch64 job passes platform=linux to the action (the action's own
    # build-step gating needs that); the machine is the truth for the wheel
    # tag we must expect.
    import platform as _plat
    if args.platform == "linux" and _plat.machine() == "aarch64":
        args.platform = "linux_aarch64"

    paths = []
    for w in args.wheels:
        p = Path(w)
        if p.is_dir():
            paths += sorted(p.glob("*.whl"))
        elif p.exists():
            paths.append(p)
    if not paths:
        gate_error(f"no wheels found in {args.wheels}")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if not args.package.strip():
        gate_error("--package is empty")

    pkg = load_pkg_config(args.package)
    vknobs = pkg.get("verify") or {}
    is_linux = args.platform in ("linux", "linux_aarch64")
    if is_linux and "elf_sanity" not in skip:
        try:
            import elftools  # noqa: F401
        except ImportError:
            gate_error("pyelftools missing on Linux -- the Repair step "
                       "(auditwheel) should have installed it; step order broken")

    # cross-check the arch list against the resolver (warn-only drift alarm)
    if args.arch_list and is_linux:
        try:
            # The ARM lane has its own resolver and its own override fields;
            # calling the x86 one made this alarm fire on EVERY aarch64 job
            # ("--arch-list '8.0 9.0 12.0+PTX' differs from resolver output
            # '8.0 8.6 8.9 9.0 12.0+PTX'"), so the one cross-check meant to
            # catch real matrix/action drift was permanent noise on ARM.
            if args.platform == "linux_aarch64":
                resolved = _GM.resolve_aarch64_arch_list(
                    pkg, args.cuda, args.torch)
            else:
                resolved = _GM.resolve_arch_list(
                    pkg, args.cuda, pytorch_version=args.torch,
                    default_arch_list=_GM.policy_arch_list(
                        args.cuda, args.torch, platform=args.platform))
            if arch_list_to_sm(resolved) != arch_list_to_sm(args.arch_list):
                annotate("warning",
                         f"--arch-list {args.arch_list!r} differs from resolver "
                         f"output {resolved!r} -- matrix/action drift?")
        except Exception:  # noqa: BLE001 -- advisory only
            pass

    reports, any_fail = [], False
    for wheel_path in paths:
        log(f"verifying {wheel_path.name}")
        rep = WheelReport(wheel_path.name)
        parsed = parse_wheel(wheel_path.name)
        if "filename" not in skip:
            check_filename(rep, parsed, args, pkg, vknobs)
        if "metadata" not in skip and parsed:
            check_metadata(rep, wheel_path, parsed, pkg)
        with zipfile.ZipFile(wheel_path) as _zf:
            exts, vendored = wheel_members(_zf)
        if "binary_census" not in skip:
            check_binary_census(rep, wheel_path, vknobs, exts, vendored)
        driver_linked = False
        if is_linux and exts:
            infos = elf_infos_for(wheel_path, exts)
            if "elf_sanity" not in skip:
                driver_linked = check_elf_sanity(
                    rep, infos, vendored, vknobs,
                    links_torch=pkg.get("links_torch") is not False)
            if "torch_linkage" not in skip:
                check_torch_linkage(rep, infos, pkg, args.platform)
            if "glibc_ceiling" not in skip:
                check_glibc_ceiling(rep, infos, vendored)
        elif not is_linux and exts and "torch_linkage" not in skip:
            # Windows: no ELF; warn-quality evidence only via byte scan
            with zipfile.ZipFile(wheel_path) as zf:
                blob = b"".join(zf.read(m) for m in exts)
            if pkg.get("links_torch") is False and (b"torch_cpu.dll" in blob
                                                    or b"c10.dll" in blob):
                rep.add("torch_linkage", "warn",
                        "links_torch: false but .pyd references torch DLLs "
                        "[byte-scan evidence]")
        if "arch_sass" not in skip:
            check_arch_sass(rep, wheel_path, exts, args, vknobs)
        if "import" not in skip:
            check_import(rep, wheel_path, parsed, exts, args, vknobs, driver_linked)
        if args.strict:
            for c in rep.checks:
                if c["status"] == "warn":
                    rep.failed = True
        any_fail |= rep.failed
        reports.append(rep)

    report = {
        "schema": 1,
        "invocation": {"package": args.package, "cuda": args.cuda,
                       "torch": args.torch, "platform": args.platform,
                       "arch_list": args.arch_list, "argv": sys.argv[1:]},
        "verdict": "fail" if any_fail else "pass",
        "wheels": [{"filename": r.filename,
                    "verdict": "fail" if r.failed else "pass",
                    "checks": r.checks} for r in reports],
        "duration_s": round(time.time() - t0, 1),
    }
    out = Path(args.report) if args.report else paths[0].parent / "verify_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    log(f"report: {out}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        glyph = {"pass": "OK", "warn": "WARN", "fail": "FAIL", "skip": "skip"}
        with open(summary_path, "a") as f:
            f.write(f"\n### verify_wheel: {args.package} "
                    f"cu{args.cuda} torch{args.torch} {args.platform}\n\n")
            f.write("| wheel | " + " | ".join(c["id"] for c in reports[0].checks) + " |\n")
            f.write("|---" * (len(reports[0].checks) + 1) + "|\n")
            for r in reports:
                f.write(f"| {r.filename} | "
                        + " | ".join(glyph[c["status"]] for c in r.checks) + " |\n")

    if any_fail:
        annotate("error", f"verify_wheel: {sum(r.failed for r in reports)} of "
                          f"{len(reports)} wheel(s) FAILED -- upload blocked")
        sys.exit(1)
    log(f"verify_wheel: all {len(reports)} wheel(s) passed "
        f"({report['duration_s']}s)")


if __name__ == "__main__":
    main()
