#!/usr/bin/env python3
"""Generate build matrix from package YAML configs, excluding existing wheels."""
import argparse
import json
import re
import subprocess
import urllib.request
import yaml
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11 fallback
    except ImportError:
        tomllib = None  # only fetch_package_info needs it (matrix time, py3.12);
                        # importers like verify_wheel on cp310 cells do not

# Combos where upstream PyTorch ships no wheel — skipped in matrix generation
# (CW-ADR-0007). Generated from PCWM by scripts/derive_defaults.py; the file
# is committed, so builds stay a function of the git SHA. Missing file is a
# hard error: silently building phantom cells fails at torch-install time.
_PHANTOMS_FILE = Path(__file__).parent.parent / "defaults" / "phantom_combos.json"
PHANTOM_COMBOS = {tuple(c) for c in json.loads(_PHANTOMS_FILE.read_text())["combos"]}


def fetch_package_info(repo: str, tag: str, subdir: str = "") -> tuple[Optional[str], Optional[str]]:
    """
    Fetch package name and version from source repo.

    Tries in order:
    1. pyproject.toml [project] section
    2. version.txt
    3. Returns (None, None) if not found
    """
    ref = tag or "main"
    base = f"https://raw.githubusercontent.com/{repo}/{ref}"
    if subdir:
        base = f"{base}/{subdir}"

    name, version = None, None

    # 1. Try pyproject.toml
    try:
        if tomllib is None:
            raise ImportError("tomllib unavailable")
        with urllib.request.urlopen(f"{base}/pyproject.toml", timeout=10) as r:
            data = tomllib.loads(r.read().decode())
            project = data.get("project", {})
            name = project.get("name", "").replace("-", "_") or None
            version = project.get("version")
    except Exception:
        pass

    # 2. Try version.txt if version not found
    if not version:
        try:
            with urllib.request.urlopen(f"{base}/version.txt", timeout=10) as r:
                version = r.read().decode().strip() or None
        except Exception:
            pass

    return name, version


# Loaded once; provides standard combinations + arch_list_by_cuda + per-build defaults
# inherited by package YAMLs that don't override them.
import sys as _sys0
_sys0.path.insert(0, str(Path(__file__).parent))
from package_loader import load_pcto as _load_pcto, load_arch_policy as _load_arch_policy, iter_packages as _iter_packages  # noqa: E402
DEFAULTS = _load_pcto()

# Per-platform patch overrides (pytorch_windows / pytorch_linux_aarch64 on a
# policy row): upstream sometimes ships a platform only at a different patch
# of the same minor (cu129 Windows: 2.9.0, never 2.9.1). Keyed by the row's
# (cuda, pytorch) anchor; applied at cell-emission time.
_PT_OVERRIDES = {
    (c["cuda"], c["pytorch"]): {
        k[len("pytorch_"):]: v for k, v in c.items()
        if k.startswith("pytorch_") and k != "pytorch"
    }
    for c in DEFAULTS["combinations"]
}

# The owned arch policy (CW-ADR-0012): per-CUDA rows plus hand-maintained
# per-(cuda, torch-minor) exceptions. Read at BUILD time -- _defaults.yml
# carries cells only, never arch data, so there is exactly one arch source.
_ARCH_POLICY = _load_arch_policy()


def policy_arch_list(cuda_version: str, pytorch_version: str,
                     platform: str = "linux") -> str:
    """The farm's arch list for a (cuda, torch) pairing, from _arch_policy.yml.

    Exceptions win over the per-CUDA row (they encode combos whose torch
    ships no SASS for an arch the row includes -- e.g. no sm_70 on
    cu128/torch2.7). A missing CUDA key is a hard error: the policy file is
    the single arch source, and silence here would rebuild history's
    mirror-PyTorch guesswork.
    """
    minor = ".".join(str(pytorch_version).split(".")[:2])
    if platform == "linux_aarch64":
        # Separate table: SBSA/Jetson GPUs share no history with the x86 rows.
        try:
            return _ARCH_POLICY["arch_policy_aarch64"][cuda_version]
        except KeyError:
            raise KeyError(
                f"No aarch64 arch policy for cuda={cuda_version}; add a row to "
                f"defaults/arch_policy.yml's arch_policy_aarch64."
            ) from None
    exc = (_ARCH_POLICY.get("arch_exceptions") or {}).get(f"{cuda_version}/{minor}")
    if exc:
        return exc
    try:
        return _ARCH_POLICY["arch_policy"][cuda_version]
    except KeyError:
        raise KeyError(
            f"No arch policy for cuda={cuda_version} "
            f"(pytorch={pytorch_version}); add a row to "
            f"defaults/arch_policy.yml's arch_policy."
        ) from None


def _ensure_ptx_on_highest_base(arch_list_str: str) -> str:
    """Ensure the highest non-`a` token in the arch list has a `+PTX` suffix.

    Our build policy: every wheel always ships a PTX tail at the highest
    base arch, so it stays JIT-compatible with future GPUs even after PyTorch
    rotates `+PTX` away from a maturing toolchain. Belt-and-suspenders to the
    YAMLs — if a future combo is added without `+PTX`, this normalizes it.

    Tokens are space- or semicolon-separated (both formats observed).
    Preserves the existing separator. Idempotent.
    """
    if not arch_list_str or not arch_list_str.strip():
        return arch_list_str
    sep = ";" if ";" in arch_list_str else " "
    tokens = [t for t in re.split(r"[;\s]+", arch_list_str.strip()) if t]
    best_idx, best_val = -1, -1.0
    for i, tok in enumerate(tokens):
        bare = tok.replace("+PTX", "")
        if bare.endswith("a"):
            continue
        try:
            val = float(bare)
        except ValueError:
            continue
        if val > best_val:
            best_val, best_idx = val, i
    if best_idx >= 0 and "+PTX" not in tokens[best_idx]:
        tokens[best_idx] = tokens[best_idx] + "+PTX"
    return sep.join(tokens)


def resolve_aarch64_arch_list(pkg: dict, cuda_version: str,
                              pytorch_version: str) -> str:
    """ARM arch list: package's dedicated aarch64 override, else the
    aarch64 policy table. The x86 fields (arch_list/arch_list_by_cuda)
    are deliberately NOT consulted -- their sm_86/sm_89 floors describe
    consumer x86 GPUs and mean nothing on SBSA. A package uses these
    fields when its kernel gap is platform-independent (e.g. upstream
    ships no sm_100 kernel at all)."""
    by_cuda = pkg.get("arch_list_by_cuda_aarch64") or {}
    raw = by_cuda.get(cuda_version) or pkg.get("arch_list_aarch64")
    if raw:
        return _ensure_ptx_on_highest_base(raw)
    return policy_arch_list(cuda_version, pytorch_version, "linux_aarch64")


def resolve_arch_list(pkg: dict, cuda_version: str,
                      combo_arch_list: Optional[str] = None,
                      pytorch_version: Optional[str] = None,
                      default_arch_list: Optional[str] = None) -> str:
    """
    Resolve the TORCH_CUDA_ARCH_LIST for a (package, cuda, torch) tuple.

    Priority (highest first):
      1. per-combo arch_list from the package's OWN build_matrix.combinations
      2. pkg.arch_list_by_cuda[cuda] (per-CUDA package override)
      3. pkg.arch_list (static package-wide override)
      4. default_arch_list — the policy row from packages/_arch_policy.yml
         (passed in by the caller via policy_arch_list)

    Final post-processing: every result is normalized to ensure the highest
    non-`a` token has a `+PTX` suffix (forward-compat tail for future GPUs).
    """
    raw = None
    if combo_arch_list:
        raw = combo_arch_list
    elif pkg.get("arch_list_by_cuda") and cuda_version in pkg["arch_list_by_cuda"]:
        raw = pkg["arch_list_by_cuda"][cuda_version]
    elif pkg.get("arch_list"):
        raw = pkg["arch_list"]
    elif default_arch_list:
        raw = default_arch_list

    if raw is not None:
        return _ensure_ptx_on_highest_base(raw)

    raise KeyError(
        f"No arch_list resolved for cuda={cuda_version} pytorch={pytorch_version} "
        f"pkg={pkg.get('name')}; add an entry to defaults/arch_policy.yml "
        f"or to the package YAML."
    )


def get_existing_wheels(package_name: str) -> set:
    """Fetch existing wheel filenames from a package's rolling release.

    An empty set means "this release has no wheels yet, build everything", so it
    must NEVER be the answer to "the API call failed". Returning set() on a
    timeout or an auth error makes one blip look like a virgin release and plans
    a full rebuild of wheels that already exist -- which then round-trips them
    back through `gh release upload --clobber`. Only a genuine 'release not
    found' is allowed to produce an empty set; anything else aborts the run.
    """
    try:
        result = subprocess.run(
            ["gh", "release", "view", f"{package_name}-latest",
             "--json", "assets", "-q", ".assets[].name"],
            capture_output=True, text=True, timeout=30
        )
    except Exception as e:
        raise SystemExit(
            f"ERROR: could not query the {package_name}-latest release: "
            f"{type(e).__name__}: {e}\n"
            "Refusing to continue -- treating this as 'no wheels exist' would "
            "plan a full rebuild and re-upload the existing assets."
        )

    if result.returncode == 0:
        return set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

    stderr = (result.stderr or "").lower()
    if "release not found" in stderr or "not found" in stderr:
        return set()  # legitimately new package

    raise SystemExit(
        f"ERROR: `gh release view {package_name}-latest` failed "
        f"(exit {result.returncode}): {result.stderr.strip()}\n"
        "Refusing to continue -- see above."
    )


def wheel_exists(existing_wheels: set, package: str, cuda_short: str,
                 torch_short: str, python_short: str, platform: str,
                 version: str = "") -> bool:
    """Check if a wheel matching this combo AND version exists in our releases.

    The version is part of the match on purpose. Without it, bumping a package's
    `source_tag` to a new upstream release leaves every cell matching the *old*
    version's wheel, the matrix comes back empty, and the run goes green having
    rebuilt nothing -- the release still holds the previous version. The
    per-job check at build.yml already includes the version, but never runs,
    because the cell was dropped here first.

    `version` is optional so a caller that genuinely cannot resolve one still
    gets the old combo-only behaviour rather than rebuilding the world.
    """
    # Check both v2 naming (torch2.9) and v1 naming (torch29)
    torch_short_v1 = torch_short.replace(".", "")
    if torch_short == "*":
        # links_torch: false (CW-ADR-0011): the binary is torch-agnostic, so a
        # wheel under ANY torch tag satisfies this (cuda, python, platform).
        import re as _re
        # cp312-cp312 exact-python wheels, plus the abi-agnostic shapes:
        # py3-none (llama_cpp_python) and cpXY-abi3 (torchao) satisfy every
        # python cell, so their presence must skip the rebuild too.
        pat = _re.compile(
            (_re.escape(f"-{version}") if version else "") +
            _re.escape(f"+cu{cuda_short}torch") + r"[\d.]+" +
            rf"-(?:cp{python_short}-cp{python_short}[a-z]*|py3-none|cp\d+-abi3)-")
        if platform == "linux":
            return any(pat.search(w) and "aarch64" not in w
                       and ("manylinux" in w or "linux_x86_64" in w)
                       for w in existing_wheels)
        elif platform == "linux_aarch64":
            return any(pat.search(w) and "aarch64" in w for w in existing_wheels)
        return any(pat.search(w) and "win_amd64" in w for w in existing_wheels)
    combos = [
        f"+cu{cuda_short}torch{t}-{tag}"
        for t in (torch_short, torch_short_v1)
        for tag in (f"cp{python_short}-cp{python_short}-", "py3-none-")
    ]
    if version:
        # Wheel names are <dist>-<version>+<combo>-... so anchor the version to
        # the '+' that begins the local segment.
        patterns = [f"-{version}{c}" for c in combos]
    else:
        patterns = combos

    def _matches(w):
        # abi3 wheels carry an arbitrary cp floor (cp310-abi3), so the tag
        # can't be a fixed substring: pair the combo prefix with "-abi3-".
        if any(p in w for p in patterns):
            return True
        return "-abi3-" in w and any(
            (f"-{version}+cu{cuda_short}torch{t}-" if version
             else f"+cu{cuda_short}torch{t}-") in w
            for t in (torch_short, torch_short_v1))

    if platform == "linux":
        return any(_matches(w) and "aarch64" not in w
                   and ("manylinux" in w or "linux_x86_64" in w)
                   for w in existing_wheels)
    elif platform == "linux_aarch64":
        return any(_matches(w) and "aarch64" in w for w in existing_wheels)
    else:
        return any(_matches(w) and "win_amd64" in w for w in existing_wheels)


def _validate_filters(package_filter, platform_filter,
                      cuda_filter, pytorch_filter, python_filter) -> None:
    """Reject filters that match nothing, before an empty matrix goes green.

    A filter matching zero rows yields an empty matrix; every build job is then
    skipped by its `include[0] != null` guard, `release` still runs because a
    skipped job is not a cancelled one, finds no artifacts and exits 0. The run
    is green and built nothing, and says so nowhere. `pytorch` and `python` are
    free-text dispatch inputs and the grid trails upstream by a release or two,
    so `-f pytorch=2.12` is a plausible input, not a contrived typo.
    """
    def fail(what, value, known):
        raise SystemExit(
            f"ERROR: unknown {what} {value!r}.\n"
            f"  known {what}s: {', '.join(sorted(known))}\n"
            "Refusing to continue -- an unmatched filter builds nothing and "
            "reports success."
        )

    names = {cfg["name"] for _d, cfg in _iter_packages() if cfg.get("name")}
    if package_filter != "all" and package_filter not in names:
        fail("package", package_filter, names)

    if platform_filter != "all" and platform_filter not in {"linux", "windows", "linux_aarch64"}:
        fail("platform", platform_filter, {"linux", "windows", "linux_aarch64"})

    combos = DEFAULTS.get("combinations", [])
    cudas = {str(c["cuda"]) for c in combos if "cuda" in c}
    torches = {str(c["pytorch"]) for c in combos if "pytorch" in c}
    torches |= {".".join(str(c["pytorch"]).split(".")[:2]) for c in combos if "pytorch" in c}
    pythons = {str(v) for c in combos for v in c.get("python_versions", [])}

    if cuda_filter != "all" and cuda_filter not in cudas:
        fail("cuda version", cuda_filter, cudas)
    if pytorch_filter not in ("all", "") and pytorch_filter not in torches:
        fail("pytorch version", pytorch_filter, torches)
    if python_filter not in ("all", "") and python_filter not in pythons:
        fail("python version", python_filter, pythons)

    # A --pytorch filter against a links_torch:false package matches NOTHING.
    # Such a package has no torch axis at all (CW-ADR-0011): it is built once
    # per (cuda, python, platform), and the torch shown in its tag is just the
    # environment the build happened in. So `--package cumm --pytorch 2.11`
    # yields zero cells, every build job skips, `release` finds no artifacts
    # and exits 0 -- green, having built nothing. That is exactly what happened
    # on 2026-08-25: cumm reported success with an empty matrix, and spconv
    # would then have failed all 125 cells in pick_cumm because the sibling
    # wheels it fetches never existed.
    torch_free = sorted(cfg["name"] for _d, cfg in _iter_packages()
                        if cfg.get("links_torch") is False and cfg.get("name"))
    if pytorch_filter not in ("all", ""):
        if package_filter in torch_free:
            raise SystemExit(
                f"ERROR: --pytorch {pytorch_filter!r} was given for "
                f"{package_filter!r}, which declares links_torch: false.\n"
                f"  That package has no torch axis -- it is built once per "
                f"(cuda, python, platform) -- so this filter matches zero "
                f"cells.\n"
                f"  Use --pytorch all (optionally with --cuda) instead.\n"
                "Refusing to continue -- an unmatched filter builds nothing "
                "and reports success.")
        if package_filter == "all" and torch_free:
            # Not an error: "build everything for torch X" legitimately has
            # nothing to say about torch-free packages. But it must not be
            # silent, or their absence reads as success.
            print(f"NOTE: --pytorch {pytorch_filter} excludes torch-free "
                  f"package(s) {', '.join(torch_free)} (links_torch: false, "
                  f"no torch axis). Dispatch them separately with "
                  f"--pytorch all.")


def generate_matrix(package_filter: str, overwrite: bool = False,
                    platform_filter: str = "all", cuda_filter: str = "all",
                    pytorch_filter: str = "all", python_filter: str = "all") -> list:
    """Generate build matrix from package configs, excluding existing wheels."""
    matrix = []
    skipped = 0

    _validate_filters(package_filter, platform_filter,
                      cuda_filter, pytorch_filter, python_filter)

    for pkg_name_dir, pkg in _iter_packages():

        if package_filter != "all" and pkg["name"] != package_filter:
            continue

        # Fetch package info from source repo
        detected_name, detected_version = fetch_package_info(
            pkg["source_repo"],
            pkg.get("source_tag", ""),
            pkg.get("build_subdir", "")
        )
        # The declared name is authoritative -- it keys the release tag, the
        # wheel-exists prefix and the index entry. But the wheel FILENAME
        # comes from upstream's setup.py, so if upstream's dist name differs
        # and no patch renames it, every artifact misses every check forever
        # (perpetual rebuilds, wrong-tag uploads). Renamed forks fix this in
        # their patch (flexgemm_vb etc.), so only warn when there is no
        # patch_script to do the renaming.
        pkg_name = pkg["name"].replace("-", "_")
        if (detected_name and not pkg.get("patch_script")
                and detected_name.replace("-", "_").lower() != pkg_name.lower()):
            print(f"WARNING: {pkg['name']}: upstream dist name is "
                  f"'{detected_name}' and there is no patch_script to rename "
                  f"it -- built wheels will never match the release prefix "
                  f"checks (perpetual rebuild).")
        pkg_version = detected_version or pkg.get("version", "")

        if pkg_version:
            print(f"Detected version {pkg_version} for {pkg['name']}")
        else:
            print(f"WARNING: No version found for {pkg['name']}")

        # Fetch existing wheels for this package (skip when overwriting)
        existing_wheels = set()
        if not overwrite:
            # Normalize name: wheels/releases use underscores (PEP 427), configs may use hyphens
            wheel_pkg_name = pkg["name"].replace("-", "_")
            existing_wheels = get_existing_wheels(wheel_pkg_name)
            if existing_wheels:
                print(f"Found {len(existing_wheels)} existing wheels for {pkg['name']}")
        else:
            print(f"Overwrite enabled, skipping existing wheel check for {pkg['name']}")

        build = pkg.get("build_matrix") or {}

        # Resolve combinations: package's own override > _defaults.combinations.
        # Combos carry `combo_arch_list` (highest priority — per-combo override
        # in the package's OWN build_matrix) and `default_arch_list` (lower
        # priority — the corresponding entry in _defaults.yml, used as fallback
        # only when the package has no other override).
        if "combinations" in build:
            combos_src = build["combinations"]
            default_python_vers = build.get("python_versions", [])
            combos = []
            for c in combos_src:
                python_vers = c.get("python_versions", default_python_vers)
                combo_arch_list = c.get("arch_list")  # explicit per-combo override in this YAML
                default_arch_list = policy_arch_list(c["cuda"], c["pytorch"])
                combo_source_tag = c.get("source_tag")
                combos.append((c["cuda"], c["pytorch"], python_vers,
                               combo_arch_list, combo_source_tag, default_arch_list))
        elif "cuda_versions" in build and "pytorch_versions" in build:
            # Legacy cartesian-product form
            python_vers = build["python_versions"]
            combos = [(cuda, pytorch, python_vers, None, None,
                       policy_arch_list(cuda, pytorch))
                      for cuda in build["cuda_versions"]
                      for pytorch in build["pytorch_versions"]]
        elif pkg.get("links_torch") is False:
            # CW-ADR-0011, build half: a package that never links libtorch is
            # built ONCE per (cuda, python, platform) -- the torch axis is not
            # a dimension for it. Pick the newest torch row per CUDA line as
            # the build environment (the wheel's tag records it; the index's
            # alias expansion lists the asset under every other torch).
            newest = {}
            for c in DEFAULTS["combinations"]:
                cur = newest.get(c["cuda"])
                if cur is None or [int(x) for x in c["pytorch"].split(".")] > \
                        [int(x) for x in cur["pytorch"].split(".")]:
                    newest[c["cuda"]] = c
            combos = []
            for c in newest.values():
                combos.append((c["cuda"], c["pytorch"], c["python_versions"],
                               None, c.get("source_tag"),
                               policy_arch_list(c["cuda"], c["pytorch"])))
        else:
            # Inherit standard combinations from packages/_defaults.yml
            combos = []
            for c in DEFAULTS["combinations"]:
                combos.append((c["cuda"], c["pytorch"], c["python_versions"],
                               None, c.get("source_tag"),
                               policy_arch_list(c["cuda"], c["pytorch"])))

        platforms = build.get("platforms") or DEFAULTS.get("platforms", ["linux"])
        # Inject platforms back into build dict so existing code below reads it uniformly
        build = dict(build)
        build["platforms"] = platforms

        # Optional per-package floor: skip any combo with pytorch < min_pytorch.
        # Used when upstream source has a hard-coded torch version assert
        # (e.g. NATTEN v0.21.6 setup.py: `assert torch_ver >= [2, 5]`).
        min_pytorch_parts = None
        if pkg.get("min_pytorch"):
            min_pytorch_parts = [int(x) for x in str(pkg["min_pytorch"]).split(".")[:2]]
        # Optional per-package CUDA ceiling: skip combos with cuda > max_cuda.
        # For sources broken by a frontier toolkit (CCCL 3.2 in CUDA 13.2
        # removed CUB's classic two-phase API; cumesh's DeviceScan calls
        # mis-resolve there). Lift when upstream adapts.
        max_cuda_parts = None
        if pkg.get("max_cuda"):
            max_cuda_parts = [int(x) for x in str(pkg["max_cuda"]).split(".")[:2]]

        for cuda, pytorch, python_versions, combo_arch_list, combo_source_tag, default_arch_list in combos:
            if cuda_filter != "all" and cuda != cuda_filter:
                continue
            # pytorch_filter accepts either full ("2.11.0") or major.minor ("2.11")
            if pytorch_filter != "all":
                torch_mm = ".".join(pytorch.split(".")[:2])
                if pytorch != pytorch_filter and torch_mm != pytorch_filter:
                    continue
            # Enforce per-package torch floor
            if min_pytorch_parts is not None:
                pt_parts = [int(x) for x in pytorch.split(".")[:2]]
                if pt_parts < min_pytorch_parts:
                    continue
            # Enforce per-package CUDA ceiling
            if max_cuda_parts is not None:
                cu_parts = [int(x) for x in cuda.split(".")[:2]]
                if cu_parts > max_cuda_parts:
                    continue

            cuda_short = cuda.replace(".", "")
            torch_short = ".".join(pytorch.split(".")[:2])  # 2.9.1 -> 2.9

            for python_ver in python_versions:
                python_short = python_ver.replace(".", "")
                if python_filter != "all" and python_ver != python_filter:
                    continue

                for platform in build["platforms"]:
                    if platform_filter != "all" and platform != platform_filter:
                        continue
                    # A CUDA line absent from arch_policy_aarch64 is declared
                    # not-built-for-ARM (e.g. 12.4: the ubuntu2404/sbsa repo
                    # has no 12.4 toolkit). Deliberate skip, not an error.
                    if (platform == "linux_aarch64"
                            and cuda not in (_ARCH_POLICY.get("arch_policy_aarch64") or {})):
                        continue
                    # Skip phantom combos (no upstream torch wheel)
                    if (cuda_short, torch_short, python_short, platform) in PHANTOM_COMBOS:
                        continue
                    # Skip if wheel already exists
                    if wheel_exists(existing_wheels, pkg["name"], cuda_short,
                                    "*" if pkg.get("links_torch") is False else torch_short,
                                    python_short, platform,
                                    pkg_version):
                        skipped += 1
                        continue

                    defaults = DEFAULTS.get("defaults", {})
                    base_entry = {
                        "package": pkg_name,
                        "version": pkg_version,
                        "source_repo": pkg["source_repo"],
                        "source_tag": combo_source_tag or pkg.get("source_tag", ""),
                        "cuda": cuda,
                        "cuda_short": cuda_short,
                        # Same minor, possibly a different patch on this
                        # platform (see _PT_OVERRIDES).
                        "pytorch": _PT_OVERRIDES.get((cuda, pytorch), {})
                                   .get(platform, pytorch),
                        "python": python_ver,
                        "platform": platform,
                        # aarch64 resolves from its own policy table, NOT the
                        # x86 overrides (sm_86/sm_89 floors mean nothing on
                        # SBSA). Packages whose kernel gaps are platform-
                        # independent (sageattention ships no sm_100 anywhere)
                        # declare the dedicated aarch64 fields instead.
                        "arch_list": (resolve_aarch64_arch_list(pkg, cuda, pytorch)
                                      if platform == "linux_aarch64"
                                      else resolve_arch_list(pkg, cuda, combo_arch_list, pytorch, default_arch_list)),
                        "extra_deps": pkg.get("extra_deps", ""),
                        "pre_build_script": pkg.get("pre_build_script", ""),
                        "free_disk_space": pkg.get("free_disk_space", defaults.get("free_disk_space", False)),
                        "max_jobs": pkg.get("max_jobs", defaults.get("max_jobs", 0)),
                        # nvcc --threads (per-arch device compiles within one
                        # TU). Farm default 1: same-TU per-arch peaks land
                        # simultaneously; RAM is better spent on max_jobs.
                        "nvcc_threads": int(pkg.get("nvcc_threads", defaults.get("nvcc_threads", 1))),
                        "clone_recursive": pkg.get("clone_recursive", defaults.get("clone_recursive", False)),
                        "patch_script": pkg.get("patch_script", ""),
                        "build_subdir": pkg.get("build_subdir", ""),
                        "cuda_installer": pkg.get("cuda_installer", "network"),
                        "extra_cuda_components": pkg.get("extra_cuda_components", ""),
                        "nvcc_flags": pkg.get("nvcc_flags", ""),
                    }

                    # Sharding: when pkg.sharding > 0, emit N compile-shard entries
                    # (one per shard, distinguished by shard_index) into the matrix.
                    # The downstream link job is fanned out by a separate matrix
                    # produced via _link_matrix_from() below.
                    sharding = int(pkg.get("sharding", 0))
                    # Optional platform restriction for sharding: a package
                    # whose partition mechanism only exists on some lanes
                    # (spconv: seat wrapper on linux; the Windows
                    # CUDAExtension filter never fires for pccm builds)
                    # builds unsharded elsewhere.
                    shard_platforms = pkg.get("sharding_platforms")
                    if shard_platforms is not None and platform not in shard_platforms:
                        sharding = 0
                    # "seat" (default): the nvcc-seat wrapper / CUDAExtension
                    # monkeypatch partitions. "source": the package's patch
                    # partitions its own source list (natten) and the seat
                    # must not double-partition.
                    shard_filter = pkg.get("shard_filter", "seat")
                    if shard_filter not in ("seat", "source"):
                        print(f"ERROR: {pkg_name}: shard_filter must be "
                              f"'seat' or 'source', got {shard_filter!r}")
                        # the module binds `import sys as _sys0`; plain `sys`
                        # was a NameError, so the one existing key-value check
                        # crashed instead of failing cleanly.
                        _sys0.exit(1)
                    if sharding > 0:
                        for shard_index in range(1, sharding + 1):
                            entry = dict(base_entry)
                            entry["shard_index"] = shard_index
                            entry["shard_count"] = sharding
                            entry["shard_filter"] = shard_filter
                            matrix.append(entry)
                    else:
                        # Unsharded: emit defaults so action.yml inputs always get sane values.
                        base_entry["shard_index"] = 0
                        base_entry["shard_count"] = 0
                        base_entry["shard_filter"] = shard_filter
                        matrix.append(base_entry)

    if skipped > 0:
        print(f"Skipped {skipped} existing wheels")

    return matrix


def link_matrix_from(matrix: list) -> list:
    """Derive the link-job matrix from the compile matrix.

    For each unique (package, cuda, pytorch, python, platform) tuple that has
    shard_count > 0, emit ONE link entry. The link job downloads all N shards'
    .o artifacts, runs link-only, and produces the final wheel.

    Unsharded entries (shard_count == 0) are filtered out — the regular build
    job handles those end-to-end and produces the wheel directly.
    """
    seen = set()
    link_jobs = []
    for entry in matrix:
        if int(entry.get("shard_count", 0)) <= 0:
            continue
        key = (entry["package"], entry["cuda"], entry["pytorch"],
               entry["python"], entry["platform"])
        if key in seen:
            continue
        seen.add(key)
        link_entry = dict(entry)
        link_entry.pop("shard_index", None)  # link job is not per-shard
        link_jobs.append(link_entry)
    return link_jobs


def main():
    parser = argparse.ArgumentParser(description="Generate build matrix from package configs")
    parser.add_argument("--package", default="all", help="Package to build (or 'all')")
    parser.add_argument("--output", default="matrix.json", help="Output file path")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing wheels and rebuild all")
    parser.add_argument("--platform", default="all", help="Platform filter: all, linux, linux_aarch64, windows")
    parser.add_argument("--cuda", default="all", help="CUDA version filter: all, or any CUDA line in the grid (validated)")
    parser.add_argument("--pytorch", default="all", help="PyTorch version filter (full like 2.11.0 or major.minor like 2.11), or 'all'")
    parser.add_argument("--python", default="all", help="Python version filter like 3.12, or 'all'")
    args = parser.parse_args()

    matrix = generate_matrix(args.package, overwrite=args.overwrite,
                            platform_filter=args.platform, cuda_filter=args.cuda,
                            pytorch_filter=args.pytorch, python_filter=args.python)

    # Split by platform; for sharded packages, also produce a separate
    # link-job matrix per platform (one link job per unique pkg/cuda/torch/py).
    linux_jobs_all = [j for j in matrix if j["platform"] == "linux"]
    windows_jobs_all = [j for j in matrix if j["platform"] == "windows"]
    aarch64_jobs = [j for j in matrix if j["platform"] == "linux_aarch64"]

    linux_jobs = linux_jobs_all
    windows_jobs = windows_jobs_all

    linux_link_jobs = link_matrix_from(linux_jobs)
    windows_link_jobs = link_matrix_from(windows_jobs)
    aarch64_link_jobs = link_matrix_from(aarch64_jobs)

    output = {
        "linux": {"include": linux_jobs},
        "linux_aarch64": {"include": aarch64_jobs},
        "windows": {"include": windows_jobs},
        "linux_link": {"include": linux_link_jobs},
        "windows_link": {"include": windows_link_jobs},
        "linux_aarch64_link": {"include": aarch64_link_jobs},
    }

    with open(args.output, "w") as f:
        # No indent - GitHub Actions needs single-line JSON for GITHUB_OUTPUT
        json.dump(output, f, separators=(',', ':'))

    if not matrix and args.package != "all":
        raise SystemExit(
            f"ERROR: {args.package!r} produced ZERO build jobs under "
            f"platform={args.platform} cuda={args.cuda} "
            f"pytorch={args.pytorch} python={args.python}.\n"
            "Refusing to emit an empty matrix -- every build job would skip, "
            "release would find no artifacts, and the run would report "
            "success having built nothing.")

    print(f"Generated {len(matrix)} build jobs "
          f"({len(linux_jobs)} Linux, {len(aarch64_jobs)} aarch64, "
          f"{len(windows_jobs)} Windows, " f")")
    if linux_link_jobs or windows_link_jobs or aarch64_link_jobs:
        print(f"  + {len(linux_link_jobs)} Linux link jobs, "
              f"{len(windows_link_jobs)} Windows link jobs (sharded packages)")

    # Also print to stdout for debugging
    for job in matrix:
        shard_info = (f" shard={job['shard_index']}/{job['shard_count']}"
                      if int(job.get("shard_count", 0)) > 0 else "")
        print(f"  - {job['package']} py{job['python']} cu{job['cuda_short']} {job['platform']}{shard_info}")
    for job in linux_link_jobs + windows_link_jobs:
        print(f"  - LINK {job['package']} py{job['python']} cu{job['cuda_short']} {job['platform']} (collects {job['shard_count']} shards)")


if __name__ == "__main__":
    main()
