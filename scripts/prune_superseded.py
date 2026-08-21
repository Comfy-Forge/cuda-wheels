#!/usr/bin/env python3
"""Delete superseded wheel assets from a package's rolling release.

The -latest releases are mutable: a version bump re-fills every cell with
new-version wheels, but nothing removed the old ones -- releases rotted
into version mixtures (the legacy farm's cumesh_vb-latest holds 0.0.1 AND
1.0 side by side, 180 assets). Per CW-ADR-0002 the rolling release should
hold exactly one wheel per cell: for every (cuda, torch, python, platform)
cell that has wheels from multiple versions, keep the highest version and
delete the rest.

Deletion is CELL-SCOPED on purpose: after a partial-grid rebuild, cells not
yet rebuilt keep their old-version wheel (coverage stays intact); the old
wheel disappears only when its replacement exists.

    python scripts/prune_superseded.py --repo OWNER/REPO --package diso
    python scripts/prune_superseded.py --repo OWNER/REPO --package diso --dry-run
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit import parse_wheel  # noqa: E402  the one wheel-name grammar


def vkey(version: str):
    """Sortable key for a base version: numeric tuple where possible."""
    base = version.split("+")[0]
    parts = []
    for tok in base.replace("-", ".").split("."):
        parts.append((0, int(tok)) if tok.isdigit() else (1, tok))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tag = f"{args.package.replace('-', '_')}-latest"

    r = subprocess.run(["gh", "release", "view", tag, "--repo", args.repo,
                        "--json", "assets"], capture_output=True, text=True)
    if r.returncode != 0:
        if "not found" in r.stderr.lower():
            print(f"{tag}: no release, nothing to prune")
            return
        raise RuntimeError(f"gh release view failed: {r.stderr.strip()}")
    assets = json.loads(r.stdout)["assets"]

    cells = {}
    for a in assets:
        p = parse_wheel(a["name"])
        if not p:
            continue  # non-wheel or foreign name: never touch it
        cell = (p["cuda_short"], p["torch_short"], p["python"], p["platform"])
        cells.setdefault(cell, []).append((p["version"], a["name"]))

    doomed = []
    for cell, entries in cells.items():
        if len({v for v, _ in entries}) < 2:
            continue
        entries.sort(key=lambda e: vkey(e[0]))
        keep = entries[-1]
        doomed += [name for v, name in entries[:-1] if v != keep[0]]

    if not doomed:
        print(f"{tag}: {len(assets)} assets, no superseded versions")
        return
    print(f"{tag}: deleting {len(doomed)} superseded asset(s):")
    for name in doomed:
        print(f"  - {name}")
        if not args.dry_run:
            subprocess.run(["gh", "release", "delete-asset", tag, name,
                            "--repo", args.repo, "--yes"],
                           check=True, capture_output=True, text=True)
    print("dry run -- nothing deleted" if args.dry_run else "pruned")


if __name__ == "__main__":
    main()
