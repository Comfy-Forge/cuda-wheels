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
    return cfg


def iter_packages():
    """Yield (name, config) for every package folder, sorted."""
    for d in sorted(PACKAGES_DIR.iterdir()):
        if d.is_dir() and (d / "package.yml").exists():
            yield d.name, load_package(d)
