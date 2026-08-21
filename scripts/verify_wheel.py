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
from audit import parse_wheel, extract_archs, arch_list_to_sm, _arch_major, _SKIP_LIBS  # noqa: E402
import generate_matrix as _GM  # noqa: E402
from package_loader import iter_packages  # noqa: E402

DEVICE_ERROR_PATTERNS = (
    r"No CUDA GPUs are available",
    r"Found no NVIDIA driver",
    r"CUDA driver initialization failed",
    r"CUDA_ERROR_STUB_LIBRARY",
    r"CUDA unknown error",
)
GLIBC_CEILING = (2, 28)
TORCH_NEEDED_RE = re.compile(r"^(libtorch|libc10|libcaffe2_nvrtc)")
TORCH_SYM_PREFIXES = ("_ZN2at", "_ZN3c10", "_ZN5torch", "THP")
GLIBC_FAMILY = ("libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
                "ld-linux", "libutil.so", "libresolv.so")
ALLOWED_NEEDED_PREFIXES = ("libstdc++", "libgcc_s", "libcudart.so", "libgomp",
                           "libcuda.so.1", "libnvrtc", "libnvJitLink",
                           "libcublas", "libcusparse", "libcufft", "libcurand",
                           "libcusolver", "libcudnn", "libnccl")
# NOTE on the CUDA-lib prefixes above: NEEDED entries for cublas-class libs
# are allowed only when the wheel VENDORS them or torch provides them (the
# repair step's exclude set); C4 checks vendored-or-torch-provided explicitly.
TORCH_PROVIDED = ("libcublas", "libcusparse", "libcufft", "libcurand",
                  "libcusolver", "libcudnn", "libnccl", "libnvrtc",
                  "libnvJitLink", "libcudart.so")


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

def check_filename(rep, parsed, args, pkg):
    if parsed is None:
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
    if parsed["python"] != py_here:
        problems.append(f"python cp{parsed['python']} != build env cp{py_here}")
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

def check_metadata(rep, wheel_path, parsed):
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

def check_binary_census(rep, wheel_path, vknobs):
    with zipfile.ZipFile(wheel_path) as zf:
        exts, vendored = wheel_members(zf)
    if not exts and not vknobs.get("allow_pure_python"):
        rep.add("binary_census", "fail",
                "no compiled extension modules in wheel -- the CUDA compile "
                "silently produced a pure-python wheel (set "
                "verify.allow_pure_python only for JIT-only packages)")
    elif not exts:
        rep.add("binary_census", "skip",
                "pure-python by design (verify.allow_pure_python)",
                {"vendored": len(vendored)})
    else:
        rep.add("binary_census", "pass",
                f"{len(exts)} extension module(s), {len(vendored)} vendored lib(s)",
                {"extensions": exts, "vendored": vendored})
    return exts, vendored


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


def check_elf_sanity(rep, infos, vendored, vknobs):
    vendored_sonames = {v.rsplit("/", 1)[-1] for v in vendored}
    problems, driver_linked = [], False
    for m, info in infos.items():
        if "error" in info:
            problems.append(f"{m}: unparseable ELF ({info['error']})")
            continue
        for need in info["needed"]:
            if need == "libcuda.so.1":
                driver_linked = True
                continue
            ok = (any(f in need for f in GLIBC_FAMILY)
                  or TORCH_NEEDED_RE.match(need)
                  or any(need.startswith(p) for p in ALLOWED_NEEDED_PREFIXES)
                  or need in vendored_sonames
                  or any(need in v for v in vendored_sonames))
            if not ok:
                problems.append(f"{m}: unexpected DT_NEEDED {need!r} "
                                f"(vendor it or justify per CW-ADR-0009)")
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
                    t = tuple(int(x) for x in v[6:].split("."))
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
            source = "cuobjdump"
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
    actual = sass | ptx
    missing_majors = {_arch_major(a) for a in expected} - {_arch_major(a) for a in actual}
    if missing_majors:
        rep.add("arch_sass", "fail",
                f"missing arch families sm_{sorted(missing_majors)} -- "
                f"expected {sorted(expected)}, found {sorted(actual)} "
                f"[{source}]", data)
        return
    missing_exact = expected - actual
    extra = actual - expected
    notes = []
    if missing_exact:
        notes.append(f"sub-arch diff: missing exact {sorted(missing_exact)}")
    if extra:
        notes.append(f"extra archs {sorted(extra)}")
    if notes:
        rep.add("arch_sass", "warn", "; ".join(notes) + f" [{source}]", data)
    else:
        rep.add("arch_sass", "pass",
                f"all {len(expected)} expected archs present [{source}]", data)


# ── C8 ─────────────────────────────────────────────────────────────────────

CHILD_SCRIPT = r"""
import importlib, json, os, sys, traceback
spec = json.loads(sys.argv[1])
out = {"phases": [], "torch_ops": {}}
def phase(name, module, fn):
    entry = {"phase": name, "module": module, "ok": False, "error": None}
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
        phase("facade", mod, lambda m=mod: importlib.import_module(m))
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
print(json.dumps(out))
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
        result = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        rep.add("import", "fail",
                f"import child crashed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout).strip()[-500:]}")
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

    dev_patterns = list(DEVICE_ERROR_PATTERNS)
    ede = vknobs.get("expect_device_error")
    if isinstance(ede, str):
        dev_patterns.append(ede)
    forgivable = bool(ede)
    failures, forgiven = [], []
    for ph in result["phases"]:
        if ph["ok"] or ph["phase"] == "torch":
            continue
        err = ph.get("error") or ""
        is_device = any(re.search(p, err) for p in dev_patterns)
        is_load = err.startswith(("ImportError", "OSError", "ModuleNotFoundError"))
        if ph["phase"] == "compiled" and is_load:
            failures.append(f"{ph['module']}: {err}")        # dlopen never forgiven
        elif is_device and forgivable:
            forgiven.append(f"{ph['module']}: {err}")
        else:
            failures.append(f"{ph['module']} [{ph['phase']}]: {err}")
    data = {"stub_lane": needs_driver, "facade": facade,
            "compiled": spec["compiled"], "forgiven": forgiven,
            "phases": result["phases"]}
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
            resolved = _GM.resolve_arch_list(
                pkg, args.cuda, pytorch_version=args.torch,
                default_arch_list=_GM.policy_arch_list(args.cuda, args.torch,
                                                       platform=args.platform))
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
            check_filename(rep, parsed, args, pkg)
        if "metadata" not in skip and parsed:
            check_metadata(rep, wheel_path, parsed)
        exts, vendored = ([], [])
        if "binary_census" not in skip:
            exts, vendored = check_binary_census(rep, wheel_path, vknobs)
        driver_linked = False
        if is_linux and exts:
            infos = elf_infos_for(wheel_path, exts)
            if "elf_sanity" not in skip:
                driver_linked = check_elf_sanity(rep, infos, vendored, vknobs)
            if "torch_linkage" not in skip:
                check_torch_linkage(rep, infos, pkg, args.platform)
            if "glibc_ceiling" not in skip:
                check_glibc_ceiling(rep, infos, vendored)
        elif not is_linux and exts and "torch_linkage" not in skip:
            # Windows: no ELF; warn-quality evidence only via byte scan
            with zipfile.ZipFile(wheel_path) as zf:
                blob = b"".join(zf.read(m) for m in exts[:4])
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
