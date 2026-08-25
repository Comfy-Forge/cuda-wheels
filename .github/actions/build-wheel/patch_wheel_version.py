#!/usr/bin/env python3
"""
Patch wheel METADATA in-place after the local-version rename.

Two fixes, one repack:
1. Version: the filename carries the local tag (+cu130torch29) but the
   internal METADATA still has the base version -- sync them so pip/uv
   see consistent versions.
2. Dependency strip: EVERY Requires-Dist and Provides-Extra header is
   removed, from every wheel, unconditionally. The farm publishes wheels
   that declare no dependencies -- see strip_all_requires_dist() for why.
   `--package` is accepted for call-site compatibility and ignored.

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


def strip_all_requires_dist(content: str) -> tuple[str, int]:
    """Remove EVERY Requires-Dist and Provides-Extra header.

    The farm publishes wheels with NO declared dependencies. This is not
    neglect, it is the contract (owner decision 2026-08-25):

      * comfy-env, the consumer these wheels exist for, installs them by
        direct URL. Today that is `uv pip install --no-deps`; the single-phase
        successor inlines them as pixi `[pypi-dependencies]` with `{url=...}`.
        Either way the resolver must not chase a wheel's own dependency list --
        pixi has no `--no-deps`, so an EMPTY Requires-Dist is how that is
        expressed. Zero declared deps means zero resolver surface, which also
        means the farm index never has to be registered and therefore never
        shadows PyPI for the 17 names it shares with it.
      * The field was never load-bearing anyway. Nothing has ever read it
        (--no-deps), and it was measurably wrong: 22 of 39 published wheels
        failed a bare `import <pkg>` from their own metadata, ~17 of those
        because upstream itself never declared the dep.
      * Runtime deps are declared by the consuming node pack's comfy-env.toml
        and enforced by `comfy-test run --cuda` (install + node instantiation
        on real GPUs), which is a test that actually executes the code, rather
        than a static list nobody validates.

    Folded continuation lines go with their header. Returns (new, n_removed).
    """
    head, sep, body = content.partition("\n\n")
    kept, removed, dropping = [], 0, False
    for line in head.splitlines():
        if line.startswith(("Requires-Dist:", "Provides-Extra:")):
            removed += 1
            dropping = True
            continue
        if dropping and line[:1] in (" ", "\t"):
            continue
        dropping = False
        kept.append(line)
    return "\n".join(kept) + (sep + body if sep else "\n"), removed


def fix_wheel(wheel_path: Path) -> bool:
    """Fix METADATA (version) and strip all dependency headers, in-place.
    Returns True if modified."""
    filename = wheel_path.name
    pkg_name, version = extract_version_from_filename(filename)

    if "+" not in version:
        return False

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

        # Unconditional, every wheel, every package: ship no dependencies.
        new_content, removed = strip_all_requires_dist(content)
        if new_content != content:
            content = new_content
            modified = True
        print(f"  {filename}: stripped {removed} Requires-Dist/Provides-Extra "
              f"header(s) -- the farm declares no dependencies")

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
            if fix_wheel(whl):
                fixed += 1

    print(f"Fixed {fixed} wheel(s)")


if __name__ == "__main__":
    main()
