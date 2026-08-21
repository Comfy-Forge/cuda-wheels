#!/usr/bin/env python3
"""
Patch wheel METADATA in-place after the local-version rename.

Two fixes, one repack:
1. Version: the filename carries the local tag (+cu130torch29) but the
   internal METADATA still has the base version -- sync them so pip/uv
   see consistent versions.
2. Requires-Dist curation: when the package's config declares a
   `requires_dist` list, it REPLACES the upstream Requires-Dist (and
   Provides-Extra) wholesale -- upstream lists leak build tools and
   mis-pin siblings (CW-ADR-0004). Placeholders {LOCAL}/{VER:<folder>}
   expand via package_loader.expand_requires_dist.

Usage:
    python patch_wheel_version.py [--package <name>] <wheel_or_directory> [...]
"""

import base64
import csv
import hashlib
import io
import re
import sys
import tempfile
import zipfile
from pathlib import Path


def extract_version_from_filename(filename: str) -> tuple[str, str]:
    """Extract package name and full version from wheel filename.

    Returns (package_name, version) e.g. ('sageattention', '0.2+cu130torch29')
    """
    m = re.match(r"^([A-Za-z0-9_]+)-([^-]+)-(cp|py)", filename)
    if not m:
        raise ValueError(f"Could not parse wheel filename: {filename}")
    return m.group(1), m.group(2)


def hash_content(data: bytes) -> tuple[str, int]:
    """Return (sha256=urlsafe_b64_hash, size) for RECORD."""
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={b64}", len(data)


def rebuild_record(tmpdir: Path, dist_info_name: str) -> None:
    """Regenerate the RECORD file with correct hashes for all files."""
    record_path = tmpdir / dist_info_name / "RECORD"
    record_rel = f"{dist_info_name}/RECORD"

    rows = []
    for file in sorted(tmpdir.rglob("*")):
        if not file.is_file():
            continue
        # RECORD paths MUST be forward-slash (PEP 376/427); str() of a
        # relative Path yields backslashes on Windows, which also broke the
        # RECORD-self-exclusion compare below (duplicate RECORD rows).
        rel = file.relative_to(tmpdir).as_posix()
        if rel == record_rel:
            # RECORD itself gets an empty hash
            continue
        digest, size = hash_content(file.read_bytes())
        rows.append((rel, digest, str(size)))

    # RECORD entry for itself: empty hash and size
    rows.append((record_rel, "", ""))

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(rows)
    record_path.write_text(buf.getvalue(), encoding="utf-8")


def curate_requires_dist(content: str, curated: list[str]) -> str:
    """Replace every Requires-Dist/Provides-Extra header with the curated
    list. Folded continuation lines of dropped headers are dropped too."""
    head, sep, body = content.partition("\n\n")
    kept, dropping = [], False
    for line in head.splitlines():
        if line.startswith(("Requires-Dist:", "Provides-Extra:")):
            dropping = True
            continue
        if dropping and line[:1] in (" ", "\t"):
            continue
        dropping = False
        kept.append(line)
    kept += [f"Requires-Dist: {req}" for req in curated]
    return "\n".join(kept) + (sep + body if sep else "\n")


def fix_wheel(wheel_path: Path, curated_template: list[str] | None = None) -> bool:
    """Fix METADATA (version + curated Requires-Dist) in-place.
    Returns True if modified."""
    filename = wheel_path.name
    pkg_name, version = extract_version_from_filename(filename)

    if "+" not in version:
        return False
    local_tag = version.split("+", 1)[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(tmpdir)

        dist_info_dirs = list(tmpdir.glob("*.dist-info"))
        if not dist_info_dirs:
            print(f"  WARNING: No .dist-info found in {filename}, skipping")
            return False
        dist_info = dist_info_dirs[0]

        metadata_path = dist_info / "METADATA"
        if not metadata_path.exists():
            print(f"  WARNING: No METADATA found in {filename}, skipping")
            return False

        content = metadata_path.read_text(encoding="utf-8")
        modified = False

        m = re.search(r"^Version: (.+)$", content, re.MULTILINE)
        if m and m.group(1) == version:
            print(f"  {filename}: version already correct ({version})")
        else:
            current_version = m.group(1) if m else "unknown"
            print(f"  {filename}: {current_version} -> {version}")
            content = re.sub(
                r"^Version: .+$",
                f"Version: {version}",
                content,
                flags=re.MULTILINE,
            )
            modified = True

        if curated_template is not None:
            from package_loader import expand_requires_dist
            curated = expand_requires_dist(curated_template, local_tag)
            new_content = curate_requires_dist(content, curated)
            if new_content != content:
                content = new_content
                modified = True
            print(f"  {filename}: Requires-Dist curated "
                  f"({len(curated)} entries)")

        if not modified:
            return False
        metadata_path.write_text(content, encoding="utf-8")

        # Rename dist-info directory to match new version
        old_name = dist_info.name
        new_name = f"{pkg_name}-{version}.dist-info"
        if old_name != new_name:
            new_dist_info = dist_info.parent / new_name
            dist_info.rename(new_dist_info)
            dist_info = new_dist_info

        # Rebuild RECORD with correct hashes
        rebuild_record(tmpdir, dist_info.name)

        # Repack wheel
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(tmpdir.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(tmpdir))

    return True


def load_curated_template(package: str) -> list[str] | None:
    """The package's curated requires_dist from packages/<folder>/, or None.
    The action runs from the farm repo checkout, so the config is three
    levels up from this co-located script."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from package_loader import iter_packages
    want = package.replace("-", "_").lower()
    for _folder, cfg in iter_packages():
        if cfg["name"].replace("-", "_").lower() == want:
            return cfg.get("requires_dist")
    raise SystemExit(f"patch_wheel_version: no package named {package!r} "
                     f"in packages/")


def main():
    argv = sys.argv[1:]
    package = None
    if "--package" in argv:
        i = argv.index("--package")
        package = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        print("Usage: python patch_wheel_version.py [--package <name>] "
              "<wheel_or_directory> [...]")
        sys.exit(1)
    curated = load_curated_template(package) if package else None

    paths = [Path(p) for p in argv]
    fixed = 0

    for path in paths:
        if path.is_dir():
            wheels = sorted(path.glob("*.whl"))
        elif path.suffix == ".whl":
            wheels = [path]
        else:
            print(f"Skipping non-wheel: {path}")
            continue

        for whl in wheels:
            if fix_wheel(whl, curated):
                fixed += 1

    print(f"Fixed {fixed} wheel(s)")


if __name__ == "__main__":
    main()
