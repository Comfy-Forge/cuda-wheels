#!/usr/bin/env python3
"""Audit the farm: one command, three lenses.

    python scripts/audit.py --gaps            declared vs published: what is missing
    python scripts/audit.py --naming          filename syntax + version consistency
    python scripts/audit.py --archs           compiled SASS/PTX vs the arch policy
    python scripts/audit.py --all             everything

Replaces the former gap_analysis.py, check_wheels.py and audit_wheel_archs.py,
which were three half-overlapping definitions of "correct" with three copies
of the wheel-name parser and the release fetcher. One parser, one fetcher,
one command.

Common filters: --package NAME, --cuda 12.8, --repo OWNER/REPO.

--archs notes (inherited from audit_wheel_archs.py):
nvcc compresses device code by default from CUDA 12.8 (LZ4), so the byte
scanner may find no SASS markers at all. Such wheels are reported UNVERIFIED,
not MISMATCH -- a scan that found nothing has failed to look, not detected a
defect. Confirm by hand with:
    cuobjdump --list-elf <extracted .so or .pyd> | grep -o 'sm_[0-9]*' | sort -u
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib import request as _urlreq

sys.path.insert(0, str(Path(__file__).parent))
import generate_matrix as _GM  # noqa: E402  single source of truth for arch resolution
from package_loader import iter_packages, load_pcto  # noqa: E402

DEFAULT_REPO = "Comfy-Forge/cuda-wheels"

# One wheel-name parser for all three lenses. Handles v2 (+cu124torch2.4) and
# v1 (+cu124torch24) local tags, manylinux single- and dual-tag platforms,
# plain linux, aarch64 and Windows.
# Three python/abi tag shapes: ordinary CPython (cp312-cp312), stable-ABI
# (cp310-abi3 -- torchao's limited-API wheel, one build serves cp310+),
# and abi-agnostic (py3-none -- llama_cpp_python's ctypes-only binding).
WHEEL_RE = re.compile(
    r"^(?P<pkg>[A-Za-z0-9_]+)"
    r"-(?P<ver>[0-9][^+-]*)"
    r"\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)"
    r"-(?:cp(?P<py>\d+)-cp\d+[a-z]*|cp(?P<pyabi>\d+)-abi3|py(?P<pynone>\d)-none)-"
    r"(?P<plat>[\w.]+)\.whl$"
)
_VALID_PLAT = ("manylinux", "linux_x86_64", "linux_aarch64", "win_amd64")

def parse_wheel(name: str) -> Optional[dict]:
    m = WHEEL_RE.match(name)
    if not m:
        return None
    cuda_short = m.group("cuda")
    torch_short = m.group("torch")
    if "." not in torch_short:  # v1 naming: "24" -> "2.4"
        torch_short = f"{torch_short[0]}.{torch_short[1:]}"
    plat = m.group("plat")
    if "aarch64" in plat:
        platform = "linux_aarch64"
    elif "linux" in plat:
        platform = "linux"
    else:
        platform = "windows"
    if m.group("py"):
        python, abi = m.group("py"), "cp"
    elif m.group("pyabi"):
        python, abi = m.group("pyabi"), "abi3"   # python = the cp floor
    else:
        python, abi = None, "none"               # any python 3
    return {
        "package": m.group("pkg"),
        "version": m.group("ver"),
        "cuda": f"{cuda_short[:2]}.{cuda_short[2:]}" if len(cuda_short) == 3 else cuda_short,
        "cuda_short": cuda_short,
        "torch_short": torch_short,
        "python": python,
        "abi": abi,
        "platform": platform,
        "plat_tag": plat,
        "filename": name,
    }


def release_wheels(repo: str, tag: str) -> list:
    """Wheel filenames attached to one release (empty if release absent).

    A missing release is normal (package not built yet) and returns [].
    Any OTHER gh failure (auth, network, rate limit) raises -- an audit
    that silently reads a fetch error as "no wheels" reports every
    package as 100% missing and trains you to ignore it.
    """
    r = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets",
         "-q", ".assets[].name"],
        capture_output=True, text=True)
    if r.returncode != 0:
        if "release not found" in r.stderr.lower() or "not found" in r.stderr.lower():
            return []
        raise RuntimeError(f"gh release view {tag} failed: {r.stderr.strip()}")
    return [w for w in r.stdout.strip().split("\n") if w.endswith(".whl")]


def torch_minor(v: str) -> str:
    p = str(v).split(".")
    return f"{p[0]}.{p[1]}"


def vtuple(v: str) -> tuple:
    return tuple(int(x) for x in str(v).split("."))


def expected_cells(pkg: dict, defaults: dict, exclude_torch: set) -> set:
    """Expected (cuda_short, torch_minor, py_short, platform) set for one package.

    links_torch: false collapses the torch axis: the cell key uses "*" for
    torch, matching generate_matrix's wildcard wheel_exists (CW-ADR-0011).
    """
    build = pkg.get("build_matrix") or {}
    combos = build.get("combinations") or defaults.get("combinations", [])
    platforms = build.get("platforms") or defaults.get("platforms", ["linux"])
    min_pt = pkg.get("min_pytorch")
    torch_free = pkg.get("links_torch") is False

    cells = set()
    seen_torch_free = set()
    for c in combos:
        # Cells are keyed by torch MINOR: the grid pins one patch release per
        # (cuda, minor) row and wheels are tagged with the minor, so no
        # patch-release filtering is needed (or correct) here.
        pt = c["pytorch"]
        if min_pt and vtuple(torch_minor(pt)) < vtuple(torch_minor(str(min_pt))):
            continue
        if torch_minor(pt) in exclude_torch:
            continue
        cu = str(c["cuda"]).replace(".", "")
        for py in c["python_versions"]:
            for plat in platforms:
                if torch_free:
                    key = (cu, "*", py.replace(".", ""), plat)
                    if key in seen_torch_free:
                        continue
                    seen_torch_free.add(key)
                    cells.add(key)
                else:
                    cells.add((cu, torch_minor(pt), py.replace(".", ""), plat))
    if not torch_free:
        cells -= {tuple(x) for x in _GM.PHANTOM_COMBOS}
    return cells


# ── lens 1: --gaps ─────────────────────────────────────────────────────────

def run_gaps(args, pkgs, defaults) -> int:
    exclude = set(args.exclude_torch.split(",")) if args.exclude_torch else set()
    print(f"{'Package':<26} {'Expected':>8} {'Actual':>8} {'Missing':>8} {'%':>6}")
    print("-" * 62)
    tot_exp = tot_hit = 0
    incomplete = []
    for _pname, pkg in pkgs:
        name = pkg["name"]
        torch_free = pkg.get("links_torch") is False
        expected = expected_cells(pkg, defaults, exclude)
        actual = set()
        for w in release_wheels(args.repo, f"{name.replace('-', '_')}-latest"):
            p = parse_wheel(w)
            if p:
                actual.add((p["cuda_short"],
                            "*" if torch_free else p["torch_short"],
                            p["python"], p["platform"]))
        missing = expected - actual
        hit = len(expected) - len(missing)
        pct = hit / len(expected) * 100 if expected else 100
        print(f"{name:<26} {len(expected):>8} {len(actual):>8} {len(missing):>8} {pct:>5.0f}%")
        tot_exp += len(expected)
        tot_hit += hit
        if missing:
            incomplete.append((name, missing))
    print("-" * 62)
    pct = tot_hit / tot_exp * 100 if tot_exp else 100
    print(f"{'TOTAL':<26} {tot_exp:>8} {tot_hit:>8} {tot_exp - tot_hit:>8} {pct:>5.0f}%")
    if args.verbose and incomplete:
        print("\n=== MISSING CELLS ===")
        for name, missing in incomplete:
            print(f"\n{name} ({len(missing)}):")
            by_ct = {}
            for cu, tv, py, plat in sorted(missing):
                by_ct.setdefault(f"cu{cu}/torch{tv}", []).append(f"cp{py}-{plat}")
            for key in sorted(by_ct):
                print(f"  {key}: {', '.join(sorted(by_ct[key]))}")
    return 0 if tot_hit == tot_exp else 1


# ── lens 2: --naming ───────────────────────────────────────────────────────

def _fetch(url: str) -> Optional[str]:
    try:
        with _urlreq.urlopen(url, timeout=5) as resp:
            return resp.read().decode()
    except Exception:
        return None


def source_version(pkg: dict, wheels: list) -> Optional[str]:
    """Best-effort upstream version: pyproject/version.py/__init__/setup.py,
    then infer from existing wheel names."""
    ref = pkg.get("source_tag") or "main"
    sub = f"{pkg['build_subdir']}/" if pkg.get("build_subdir") else ""
    base = f"https://raw.githubusercontent.com/{pkg['source_repo']}/{ref}/{sub}"
    name = pkg["name"]
    probes = [
        (f"{base}pyproject.toml", r'^version\s*=\s*["\']([^"\']+)["\']', re.M),
        (f"{base}{name}/version.py", r'__version__\s*=\s*["\']([^"\']+)["\']', 0),
        (f"{base}{name}/__init__.py", r'__version__\s*=\s*["\']([^"\']+)["\']', 0),
        (f"{base}setup.py", r'version\s*=\s*["\']([^"\']+)["\']', 0),
    ]
    for url, pat, flags in probes:
        content = _fetch(url)
        if content:
            m = re.search(pat, content, flags)
            if m:
                return m.group(1)
    for w in wheels:
        p = parse_wheel(w)
        if p:
            return p["version"]
    return None


def run_naming(args, pkgs, defaults) -> int:
    print("Wheel naming / version report")
    print(f"{'Package':<26} {'Wheels':>7} {'Version':<14} {'Status'}")
    print("-" * 62)
    errors = []
    for _pname, pkg in pkgs:
        name = pkg["name"]
        wheels = release_wheels(args.repo, f"{name.replace('-', '_')}-latest")
        ver = pkg.get("version") or source_version(pkg, wheels) or "?"
        bad_syntax = [w for w in wheels
                      if not parse_wheel(w)
                      or not any(v in parse_wheel(w)["plat_tag"] for v in _VALID_PLAT)]
        bad_version = [w for w in wheels
                       if ver != "?" and (p := parse_wheel(w)) and p["version"] != ver]
        status = "OK" if not bad_syntax and not bad_version else "ERRORS"
        print(f"{name:<26} {len(wheels):>7} {ver:<14} {status}")
        errors += [f"[{name}] bad syntax: {w}" for w in bad_syntax]
        errors += [f"[{name}] version mismatch (expect {ver}): {w}" for w in bad_version]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:30]:
            print(f"  {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        return 1
    print("\nAll wheel names and versions OK.")
    return 0


# ── lens 3: --archs ────────────────────────────────────────────────────────

def arch_list_to_sm(arch_list: str) -> set:
    out = set()
    for a in arch_list.replace(";", " ").split():
        a = a.split("+")[0].strip()
        if a:
            major, minor = a.split(".")
            out.add(f"sm_{major}{minor}")
    return out


def arch_list_to_sm_ptx(arch_list: str) -> tuple:
    """(expected_sass, expected_ptx) -- the `+PTX` marker is MEANINGFUL.

    `9.0+PTX` asks nvcc for BOTH sm_90 cubins and compute_90 PTX. The PTX is
    the forward-compatibility path: it is what lets the wheel JIT onto a GPU
    newer than any cubin it ships. arch_list_to_sm() throws the marker away
    (`a.split("+")[0]`), so a wheel that declared `9.0+PTX` and shipped SASS
    with NO PTX compared equal to one that shipped both -- i.e. the check
    could not see a wheel that is dead on the next GPU generation.
    """
    sass, ptx = set(), set()
    for a in arch_list.replace(";", " ").split():
        a = a.strip()
        if not a:
            continue
        want_ptx = "+PTX" in a.upper()
        base = a.split("+")[0].strip()
        if not base:
            continue
        major, minor = base.split(".")
        sass.add(f"sm_{major}{minor}")
        if want_ptx:
            ptx.add(f"sm_{major}{minor}")
    return sass, ptx


def expected_archs(pkg: dict, cuda: str, pytorch: str) -> set:
    """Resolved exactly as the build resolves it (same code path)."""
    build = pkg.get("build_matrix") or {}
    combo_arch = None
    for c in build.get("combinations") or []:
        if str(c.get("cuda")) == str(cuda) and str(c.get("pytorch")) == str(pytorch):
            combo_arch = c.get("arch_list")
            break
    try:
        default_arch = _GM.policy_arch_list(str(cuda), str(pytorch))
    except KeyError:
        default_arch = None
    return arch_list_to_sm(_GM.resolve_arch_list(
        pkg, str(cuda), combo_arch_list=combo_arch,
        pytorch_version=str(pytorch), default_arch_list=default_arch))


_SKIP_LIBS = ("libtorch", "libc10", "libcudart", "libcuda.", "libcublas",
              "libcusparse", "libcufft", "libcurand", "libcusolver",
              "libnvrtc", "libnvJitLink", "libcudnn", "libcaffe2_nvrtc")


def extract_archs(wheel_path: str) -> dict:
    """{"sass": {sm_XX}, "ptx": {sm_XX}} from cubin ELF headers (EM_CUDA=190)
    and PTX .target directives inside the wheel's .so/.pyd files."""
    sass, ptx, thrust = set(), set(), set()
    try:
        with zipfile.ZipFile(wheel_path) as zf:
            for entry in zf.namelist():
                if not entry.endswith((".so", ".pyd")) or "_cpu" in entry:
                    continue
                if entry.rsplit("/", 1)[-1].startswith(_SKIP_LIBS):
                    continue
                try:
                    data = zf.read(entry)
                except Exception:
                    continue
                pos = 0
                while True:
                    idx = data.find(b"\x7fELF", pos)
                    if idx == -1 or idx + 64 >= len(data):
                        break
                    if struct.unpack_from("<H", data, idx + 18)[0] == 190:
                        ei_class = data[idx + 4]
                        off = 48 if ei_class == 2 else 36 if ei_class == 1 else None
                        if off is not None:
                            e_flags = struct.unpack_from("<I", data, idx + off)[0]
                            byte3 = (e_flags >> 24) & 0xFF
                            sm = ((e_flags >> 8) & 0xFF) if byte3 in (0x06, 0x0a) \
                                else (e_flags & 0xFF)
                            if 50 <= sm <= 130:
                                sass.add(f"sm_{sm}")
                    pos = idx + 1
                # A bare `.target sm_NN` match is NOT proof of a PTX image:
                # the same string lives in a cubin's DWARF string table, so
                # this reported PTX for wheels that provably have none
                # (sageattn3: cuobjdump --list-ptx says "No PTX file found to
                # extract", the byte scan claimed sm_100a/sm_120a). Require
                # the surrounding PTX header framing -- a real module carries
                # `.version X.Y` before and `.address_size NN` after -- so the
                # fallback stops inventing forward-compat coverage that is not
                # there. Windows wheels take this path exclusively (cuobjdump
                # cannot read PE .pyd), so the false positives were farm-wide.
                for m in re.finditer(
                        rb"\.version\s+[\d.]+\s+\.target\s+sm_(\d+)a?"
                        rb"\s+\.address_size\s+\d+", data):
                    sm = int(m.group(1).decode())
                    if 50 <= sm <= 130:
                        ptx.add(f"sm_{sm}")
                for m in re.finditer(rb"THRUST_\d+_([\d_]+)_NS", data):
                    for n in m.group(1).decode().split("_"):
                        if n and 500 <= int(n) <= 13000:
                            thrust.add(f"sm_{int(n) // 10}")
    except zipfile.BadZipFile:
        pass
    if not sass and not ptx and thrust:
        sass = thrust
    return {"sass": sass, "ptx": ptx}


def _arch_major(sm: str) -> int:
    n = int(sm.replace("sm_", ""))
    return n // 10


def run_archs(args, pkgs, defaults) -> int:
    configs = {pkg["name"].replace("-", "_"): pkg for _p, pkg in pkgs}
    r = subprocess.run(
        ["gh", "api", f"repos/{args.repo}/releases", "--paginate", "--jq",
         '.[].assets[] | {name: .name, url: .url, size: .size}'],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"ERROR: gh api failed: {r.stderr}", file=sys.stderr)
        return 2
    wheels = []
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        asset = json.loads(line)
        p = parse_wheel(asset["name"])
        if not p or p["package"] not in configs:
            continue
        if args.package and p["package"] != args.package:
            continue
        if args.cuda and p["cuda"] != args.cuda:
            continue
        p.update(url=asset["url"], size=asset["size"])
        wheels.append(p)
    print(f"{len(wheels)} wheels to audit")
    if args.dry_run:
        for w in wheels:
            print(f"  {w['filename']} ({w['size'] >> 20}MB)")
        print(f"Total download: {sum(w['size'] for w in wheels) >> 20} MB")
        return 0

    token = subprocess.run(["gh", "auth", "token"], capture_output=True,
                           text=True).stdout.strip()
    results, mismatches, unverified = [], [], []
    with tempfile.TemporaryDirectory(prefix="wheel-audit-") as tmp:
        for i, w in enumerate(wheels, 1):
            pkg = configs[w["package"]]
            pytorch_full = next(
                (c["pytorch"] for c in
                 ((pkg.get("build_matrix") or {}).get("combinations")
                  or _GM.DEFAULTS.get("combinations") or [])
                 if str(c["cuda"]) == w["cuda"]
                 and torch_minor(c["pytorch"]) == w["torch_short"]),
                w["torch_short"] + ".0")
            expected = expected_archs(pkg, w["cuda"], pytorch_full)
            dest = os.path.join(tmp, w["filename"])
            print(f"[{i}/{len(wheels)}] {w['filename']} ({w['size'] >> 20}MB) ...",
                  end=" ", flush=True)
            dl = subprocess.run(
                ["curl", "-sL", "-H", f"Authorization: token {token}",
                 "-H", "Accept: application/octet-stream", w["url"], "-o", dest],
                capture_output=True, timeout=600)
            if dl.returncode != 0 or not os.path.getsize(dest):
                print("DOWNLOAD FAILED")
                results.append({"wheel": w["filename"], "status": "download_failed"})
                continue
            got = extract_archs(dest)
            os.unlink(dest)
            actual = got["sass"] | got["ptx"]
            if not actual:
                print("UNVERIFIED (no SASS visible -- compressed fatbin?)")
                unverified.append({"wheel": w["filename"],
                                   "expected": sorted(expected)})
                continue
            missing_majors = {_arch_major(a) for a in expected} \
                - {_arch_major(a) for a in actual}
            missing_exact = expected - actual
            match = not missing_majors
            if match and not missing_exact:
                print(f"OK {sorted(actual)}")
            elif match:
                print(f"OK (sub-arch diff) expected {sorted(expected)}, got {sorted(actual)}")
            else:
                print(f"MISMATCH expected {sorted(expected)}, got {sorted(actual)}")
                mismatches.append({"wheel": w["filename"],
                                   "expected": sorted(expected),
                                   "actual": sorted(actual),
                                   "missing_exact": sorted(missing_exact)})
            results.append({
                "wheel": w["filename"], "package": w["package"],
                "cuda": w["cuda"], "pytorch": pytorch_full,
                "expected": sorted(expected), "actual": sorted(actual),
                "actual_sass": sorted(got["sass"]), "actual_ptx": sorted(got["ptx"]),
                "match": match,
            })
    ok = sum(1 for r_ in results if r_.get("match"))
    print(f"\nARCH AUDIT: {len(results)} checked, {ok} OK, "
          f"{len(mismatches)} MISMATCH, {len(unverified)} UNVERIFIED")
    for m in mismatches:
        print(f"  {m['wheel']}: missing {m['missing_exact']}")
    if args.output:
        Path(args.output).write_text(json.dumps({
            "ok": ok, "mismatches": mismatches, "unverified": unverified,
            "results": results}, indent=2) + "\n")
        print(f"Report written to {args.output}")
    return 1 if mismatches else 0


# ── entry ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gaps", action="store_true", help="declared vs published")
    ap.add_argument("--naming", action="store_true", help="filename/version checks")
    ap.add_argument("--archs", action="store_true", help="compiled SASS vs policy")
    ap.add_argument("--all", action="store_true", help="run every lens")
    ap.add_argument("--package", help="filter by package name")
    ap.add_argument("--cuda", help="filter by CUDA version, e.g. 12.8")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--exclude-torch", help="comma-separated torch minors to skip in --gaps")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="--archs: list, no download")
    ap.add_argument("--output", default="audit_report.json", help="--archs: json report")
    args = ap.parse_args()

    lenses = []
    if args.gaps or args.all:
        lenses.append(run_gaps)
    if args.naming or args.all:
        lenses.append(run_naming)
    if args.archs or args.all:
        lenses.append(run_archs)
    if not lenses:
        ap.error("pick at least one of --gaps / --naming / --archs / --all")

    pkgs = list(iter_packages())
    if args.package:
        pkgs = [(n, p) for n, p in pkgs
                if p["name"].replace("-", "_") == args.package]
    defaults = load_pcto()

    rc = 0
    for lens in lenses:
        print()
        rc = max(rc, lens(args, pkgs, defaults))
    sys.exit(rc)


if __name__ == "__main__":
    main()
