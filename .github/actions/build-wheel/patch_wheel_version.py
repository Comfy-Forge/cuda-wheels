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

# The arch list this wheel was built for, stamped into METADATA so the wheel
# CARRIES the fact rather than it living only in a workflow input that vanished
# when the run's logs were deleted. See _stamp_arch_list().
ARCH_LIST = ""


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

        # Prune phantom top_level.txt entries: a declared top-level name that
        # NO member of the wheel provides.
        #
        # spconv (upstream v2.3.8) declares:
        #     core_cc
        #     spconv
        # but its only top-level member is `spconv/`. `core_cc` is a SUBMODULE
        # -- the extension is spconv/core_cc.<abi>.so and spconv/core_cc/ holds
        # nothing but type stubs (__init__.pyi). Every real reference in the
        # package is `import spconv.core_cc as _ext` /
        # `from spconv.core_cc.csrc.sparse.all import SpconvOps`; nothing
        # imports `core_cc` top-level and nothing could. Upstream's
        # find_packages() swept up the generated stub directory as if it were a
        # distributable package.
        #
        # verify_wheel's C8 builds its import list from top_level.txt, so it
        # dutifully tried `import core_cc` and blocked the upload:
        #   [import] core_cc: top_level.txt declares 'core_cc' but no member of
        #   the wheel provides it -- phantom top-level entry, not a missing
        #   dependency
        # The gate is right and the metadata is wrong. Fixed farm-wide rather
        # than per-package: the gate already distinguishes this case generically,
        # which says whoever wrote it expected recurrence. Dropping a name that
        # points at nothing cannot break an import that never worked.
        top_level_path = dist_info / "top_level.txt"
        if top_level_path.exists():
            declared = [ln.strip() for ln in
                        top_level_path.read_text(encoding="utf-8").splitlines()
                        if ln.strip()]
            provided = set()
            for entry in tmpdir.iterdir():
                if entry.name.endswith((".dist-info", ".data")):
                    continue
                # a package dir provides its own name; a module file provides
                # its name up to the first dot (strips .py/.so/.pyd + ABI tag)
                provided.add(entry.name if entry.is_dir()
                             else entry.name.split(".", 1)[0])
            kept = [n for n in declared if n in provided]
            phantom = [n for n in declared if n not in provided]
            if phantom and kept:
                top_level_path.write_text("\n".join(kept) + "\n",
                                          encoding="utf-8")
                modified = True
                print(f"  {filename}: dropped phantom top_level.txt "
                      f"entr{'y' if len(phantom) == 1 else 'ies'} "
                      f"{phantom} -- no wheel member provides "
                      f"{'it' if len(phantom) == 1 else 'them'}")
            elif phantom and not kept:
                # Every declared name is phantom. That is not a metadata typo,
                # it means the wheel has no top-level content at all -- refuse
                # to paper over it with an empty file.
                print(f"  WARNING: {filename}: ALL top_level.txt entries "
                      f"{phantom} are phantom; leaving the file alone -- "
                      f"this wheel looks structurally wrong, not mislabelled")

        # Stamp the arch list into METADATA.
        #
        # This is the missing half of the farm's worst repeat failure. Rebuilds
        # are decided by wheel_exists(), which matches on the FILENAME -- and
        # changing a package's arch list does not change any wheel's name. So a
        # coverage fix lands, the next run looks at the release, sees a wheel
        # with the right name, and skips the cell. That is how 24 sageattention
        # aarch64 wheels kept shipping an Ada trap for fourteen hours after the
        # fix was committed: the cells that already had a wheel were skipped,
        # and only the cells that happened to be empty picked the fix up.
        #
        # A wheel cannot record this in its name (PEP 427 fixes the fields, and
        # the index parses them), so it records it here instead. Nothing reads
        # it during a build -- C7 already checks the binary against the resolved
        # list. It exists so staleness is DETECTABLE after the fact:
        # scripts/check_stale.py reads this header from published wheels and
        # reports every cell whose wheel was built for a different arch list
        # than the config now resolves to. Before this, nothing anywhere could
        # answer that question.
        if ARCH_LIST:
            # Strip any previous copy, INCLUDING its folded continuation lines.
            # A bare `^Comfy-Forge-Arch-List:.*\n` leaves the continuations
            # behind, and re-adding the header would then orphan them onto
            # whatever field follows.
            content = re.sub(r"^Comfy-Forge-Arch-List:.*(?:\n[ \t].*)*\n",
                             "", content, flags=re.MULTILINE)
            lines = content.split("\n")

            # Insert immediately after Metadata-Version, NOT at the first
            # blank-looking line.
            #
            # The old code walked to the first `ln.strip() == ""` and called
            # that the end of the header block. It is not: RFC 822 folding puts
            # a field's continuation on INDENTED lines, and a multi-line
            # License: body contains whitespace-only lines ("        ") which
            # strip() to "". So the header landed in the MIDDLE of the licence,
            # and every continuation line below it was reassigned to us:
            #
            #     License: MIT License
            #     Comfy-Forge-Arch-List: 7.5 8.0 8.6 9.0+PTX 10.0 12.0+PTX
            #             <- this and everything under it now belongs to US
            #             Copyright (c) 2025 ...
            #
            # Measured across all 41 published packages: 7 corrupted (every one
            # with a multi-line License:), License truncated to its first line
            # and this header holding ~1.3KB of licence prose. It also made
            # check_stale.py -- the only consumer of this header -- unable to
            # read the very packages it was written for.
            #
            # Metadata-Version is the first line and is never folded, so
            # inserting after it cannot land inside another field's body.
            insert_at = 0
            for i, ln in enumerate(lines):
                if ln.lower().startswith("metadata-version:"):
                    insert_at = i + 1
                    break
            lines.insert(insert_at, f"Comfy-Forge-Arch-List: {ARCH_LIST}")
            content = "\n".join(lines)

            # Prove it round-trips rather than trusting the insertion point.
            import email as _email
            _m = _email.message_from_string(content)
            _got = (_m.get("Comfy-Forge-Arch-List") or "").strip()
            if _got != ARCH_LIST.strip():
                raise SystemExit(
                    f"patch_wheel_version: the arch-list header does not parse "
                    f"back cleanly (wrote {ARCH_LIST!r}, reads {_got[:120]!r}). "
                    f"Refusing to ship METADATA that misreports itself.")
            modified = True

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
    if "--arch-list" in argv:
        global ARCH_LIST
        i = argv.index("--arch-list")
        ARCH_LIST = argv[i + 1].strip()
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
