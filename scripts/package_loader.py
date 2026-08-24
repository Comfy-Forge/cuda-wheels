"""Single loader for the farm's layout (Comfy-Forge line).

Layout:
    defaults/python_cuda_torch_os_policy.yml   the PCTO axes: owned policy
                                               (platforms, python bounds,
                                               supported_cudas, defaults)
                                               + the GENERATED combinations
    defaults/arch_policy.yml                   the owned arch policy
    packages/<name>/package.yml                source, build knobs
    packages/<name>/pcto_override.yml          build_matrix / min_pytorch /
                                               links_torch (optional)
    packages/<name>/arch_override.yml          arch_list / arch_list_by_cuda
                                               (optional)
    packages/<name>/patches/*.py               source patches (optional)

Every consumer assembles the SAME flat package dict the old single-file
layout produced, so downstream logic is unchanged -- only discovery and
paths live here.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PCTO_FILE = ROOT / "defaults" / "python_cuda_torch_os_policy.yml"
ARCH_POLICY_FILE = ROOT / "defaults" / "arch_policy.yml"
PACKAGES_DIR = ROOT / "packages"


def load_pcto() -> dict:
    """The shared axes: combinations, platforms, python bounds, defaults."""
    return yaml.safe_load(PCTO_FILE.read_text())


def load_arch_policy() -> dict:
    """The owned arch policy: arch_policy[_aarch64], arch_exceptions."""
    return yaml.safe_load(ARCH_POLICY_FILE.read_text())


# GitHub Actions REDACTS a job output that looks credential-bearing, and
# generate_matrix embeds pre_build_script verbatim into the matrix output. A
# package whose inline pre-build mentions a token therefore vanishes from its
# own build: every job sees a null matrix, skips, and the run reports success
# having produced nothing (spconv, 2026-08-24). Keep such scripts in a file.
_CREDENTIAL_WORDS = ("Authorization", "Bearer", "GH_TOKEN", "GITHUB_TOKEN",
                     "authorization", "api_key", "apikey", "password")


def _check_pre_build_not_redactable(cfg: dict, pkg_dir: Path) -> None:
    script = cfg.get("pre_build_script") or ""
    if "\n" not in script.strip():
        return  # a one-line `bash packages/x/pre_build.sh` is always fine
    hits = sorted({w for w in _CREDENTIAL_WORDS if w in script})
    if hits:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}/package.yml: inline pre_build_script "
            f"mentions {hits} -- GitHub will redact the matrix output and the "
            f"package will silently build NOTHING. Move it to "
            f"{pkg_dir.name}/pre_build.sh and use "
            f"'pre_build_script: bash packages/{pkg_dir.name}/pre_build.sh'.")


def load_package(pkg_dir: Path) -> dict:
    """One package's flat config dict, overrides merged in."""
    cfg = yaml.safe_load((pkg_dir / "package.yml").read_text()) or {}
    _check_pre_build_not_redactable(cfg, pkg_dir)
    for extra in ("pcto_override.yml", "arch_override.yml"):
        p = pkg_dir / extra
        if p.exists():
            cfg.update(yaml.safe_load(p.read_text()) or {})
    overrides = [e for e in ("pcto_override.yml", "arch_override.yml")
                 if (pkg_dir / e).exists()]
    if overrides:
        readme = pkg_dir / "README.md"
        if not readme.exists() or "verride" not in readme.read_text():
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: has {', '.join(overrides)} but no "
                f"README.md explaining the override -- every deviation from "
                f"defaults/ must say why (add an '## Overrides' section).")
    for req in ("name", "source_repo", "source_tag"):
        if not str(cfg.get(req) or "").strip():
            raise SystemExit(
                f"ERROR: {pkg_dir.name}: '{req}' is required in package.yml.")
    if str(cfg["source_tag"]).strip().lower() in ("main", "master", "head"):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}: source_tag is a floating ref "
            f"({cfg['source_tag']!r}) -- pin a tag or commit SHA, or the "
            f"wheels in one release need not come from the same source.")
    if "links_torch" not in cfg:
        raise SystemExit(
            f"ERROR: {pkg_dir.name}: no links_torch declared -- state it "
            f"explicitly (true: wheel per (cuda x torch); false: torch-free, "
            f"one wheel per cuda, CW-ADR-0011).")
    rd = cfg.get("requires_dist")
    if rd is not None and (
            not isinstance(rd, list) or not rd
            or not all(isinstance(e, str) and e.strip() for e in rd)):
        raise SystemExit(
            f"ERROR: {pkg_dir.name}: requires_dist must be a non-empty "
            f"list of PEP 508 strings (it REPLACES the wheel's upstream "
            f"Requires-Dist wholesale).")
    return cfg


def expand_requires_dist(entries, local_tag):
    """Expand a curated requires_dist list for one concrete wheel.

    Placeholders:
      {LOCAL}        -> the wheel's own local version tag (cu128torch2.8),
                        so sibling pins land on the same build cell
      {VER:<folder>} -> the pinned `version` of packages/<folder> -- hard
                        error if that package declares none, because a
                        sibling pin must be exact+local to be un-spoofable
                        (PyPI forbids local versions, so the pin resolves
                        from our index or fails loudly, never to a
                        stranger's package)
    """
    out = []
    for entry in entries:
        for folder in re.findall(r"\{VER:([A-Za-z0-9_-]+)\}", entry):
            sib_yml = PACKAGES_DIR / folder / "package.yml"
            if not sib_yml.exists():
                raise SystemExit(
                    f"ERROR: requires_dist entry {entry!r} references "
                    f"unknown package folder {folder!r}.")
            ver = str((yaml.safe_load(sib_yml.read_text()) or {})
                      .get("version") or "").strip()
            if not ver:
                raise SystemExit(
                    f"ERROR: requires_dist entry {entry!r} pins sibling "
                    f"{folder!r}, but packages/{folder}/package.yml has no "
                    f"'version' -- exact sibling pins require one.")
            entry = entry.replace("{VER:%s}" % folder, ver)
        out.append(entry.replace("{LOCAL}", local_tag))
    return out


def iter_packages():
    """Yield (name, config) for every package folder, sorted."""
    for d in sorted(PACKAGES_DIR.iterdir()):
        if d.is_dir() and (d / "package.yml").exists():
            yield d.name, load_package(d)
