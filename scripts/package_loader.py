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


def load_package(pkg_dir: Path) -> dict:
    """One package's flat config dict, overrides merged in."""
    cfg = yaml.safe_load((pkg_dir / "package.yml").read_text()) or {}
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
    return cfg


def iter_packages():
    """Yield (name, config) for every package folder, sorted."""
    for d in sorted(PACKAGES_DIR.iterdir()):
        if d.is_dir() and (d / "package.yml").exists():
            yield d.name, load_package(d)
