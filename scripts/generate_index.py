#!/usr/bin/env python3
"""Generate PEP 503 compliant package index from GitHub releases."""
import os
import datetime as _dt
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import yaml

# Pulls the combo out of a wheel filename: +cu128torch2.8 -> ("cu128", "torch2.8")
_COMBO_RE = re.compile(r'\+(cu\d+)(torch[\d.]+)')


def _next_link(link_header):
    """Return the rel="next" URL from a GitHub Link header, or None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) >= 2 and 'rel="next"' in section[1].strip():
            return section[0].strip().strip("<>")
    return None


# Pulls the Python tag out of a wheel filename: -cp313-cp313t- -> "cp313"
_PYTAG_RE = re.compile(r"-(cp\d+)-cp\d+t?-")


def load_torch_free_packages() -> set:
    """Index-normalised names of packages declaring `links_torch: false`.

    These do not link libtorch, so one built wheel is valid for every torch in
    its CUDA line. See CW-ADR-0011.
    """
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from package_loader import iter_packages
    names = set()
    for stem, cfg in iter_packages():
        if cfg.get("links_torch") is False:
            names.add((cfg.get("name") or stem).lower().replace("_", "-"))
    return names


def load_grid(defaults_path=None) -> dict:
    """{"cu128": {"torch2.9": {"cp310", ...}, ...}, ...} from the shared grid.

    The alias set is derived from the grid rather than hardcoded so that adding
    a CUDA or torch line to _defaults.yml automatically widens the aliases.
    """
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from package_loader import load_pcto
    grid = {}
    cfg = load_pcto() or {}
    for combo in cfg.get("combinations", []):
        cuda = "cu" + str(combo["cuda"]).replace(".", "")
        torch = "torch" + ".".join(str(combo["pytorch"]).split(".")[:2])
        pys = {"cp" + str(v).replace(".", "") for v in combo.get("python_versions", [])}
        grid.setdefault(cuda, {}).setdefault(torch, set()).update(pys)
    return grid


def expand_torch_free_aliases(packages: dict, torch_free: set, grid: dict) -> int:
    """List one torch-free asset under every torch in its CUDA line.

    The wheel is built once and uploaded once. comfy-env's index resolver
    filters on the anchor *text* and downloads from the *href*
    (`packages/cuda_wheels.py`), and the two are independent -- so emitting the
    same href under several display names makes a single asset resolvable for
    every torch, at the cost of nothing but anchor tags.

    Known limits, recorded here because they are invisible at the call site:

    - comfy-env's tier-2 fallback walks the Releases API and matches on the
      real asset name, which carries exactly one torch tag. An aliased wheel is
      therefore findable under every torch via the index but only under its
      built torch via the fallback -- a gap that opens only when GH Pages is
      unreachable.
    - pip takes the filename from the URL, not the anchor text, so a user
      resolving for torch 2.11 installs a distribution whose version reads
      `+cu128torch2.8`. `pip freeze` will disagree with the environment.

    Both are accepted for the transition. The durable fix is a torch-less local
    tag understood by both resolvers (CW-ADR-0011).
    """
    aliased = 0
    for pkg in sorted(torch_free & set(packages)):
        wheels = packages[pkg]
        seen = {w["filename"] for w in wheels}
        for wheel in list(wheels):
            m = _COMBO_RE.search(wheel["filename"])
            pm = _PYTAG_RE.search(wheel["filename"])
            if not m or not pm:
                continue
            cuda, built_torch, py_tag = m.group(1), m.group(2), pm.group(1)
            for torch, pys in sorted(grid.get(cuda, {}).items()):
                if torch == built_torch:
                    continue
                # Don't advertise a (torch, python) pairing upstream never
                # shipped -- nothing would ever ask for it, and it is noise.
                if py_tag not in pys:
                    continue
                alias = wheel["filename"].replace(
                    f"+{cuda}{built_torch}", f"+{cuda}{torch}", 1
                )
                if alias in seen:
                    continue
                seen.add(alias)
                wheels.append({
                    "filename": alias,
                    "url": wheel["url"],          # same asset, no second upload
                    "alias_of": wheel["filename"],
                })
                aliased += 1
    return aliased


def published_branch_exists(repo: str, branch: str, token: str = None) -> bool:
    """Does `branch` exist on `repo`? Raises if the answer cannot be determined.

    Used to tell "nothing was ever published" apart from "the baseline checkout
    failed". Never returns a guess: an inconclusive probe propagates, because
    the whole point is that the shrinkage guard must not be skipped on an
    ambiguity.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/branches/{branch}", headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def get_releases(repo: str, token: str = None) -> list:
    """Fetch ALL releases from a GitHub repository.

    This endpoint is paginated -- 30 per page by default. A single unpaginated
    fetch silently truncates once the repo passes that many releases, and the
    resulting short index looks exactly like a healthy one: the packages that
    fall off simply stop existing as far as consumers are concerned. Ask for the
    maximum page size and follow the Link: rel="next" chain to the end.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    releases = []
    pages = 0
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            releases.extend(json.loads(response.read().decode()))
            url = _next_link(response.headers.get("Link"))
        pages += 1
        if pages > 50:  # 5000 releases; a runaway Link chain is a bug, not a repo
            raise RuntimeError("release pagination did not terminate")
    print(f"Fetched {len(releases)} releases across {pages} page(s)")
    # A 200 returning [] is indistinguishable from a total auth failure, and
    # there is no legitimate reason to publish a zero-package index. Fail
    # rather than deploy one -- an empty index is worse than a stale one,
    # because pip hard-fails on a 404 instead of falling through.
    if not releases:
        raise SystemExit(
            "ERROR: the releases API returned ZERO releases. That is either an "
            "auth failure or a repo with nothing published; either way, refusing "
            "to deploy an empty index. To publish an empty index deliberately, "
            "delete the gh-pages branch by hand.")
    return releases


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site",
                    help="Directory to write the generated site into (never committed; "
                         "deployed wholesale to gh-pages)")
    ap.add_argument("--previous", default="previous-site",
                    help="Checkout of the CURRENT gh-pages branch. Its "
                         "packages.json is the baseline for the shrinkage guard. "
                         "A missing baseline is NOT a bypass: it is an error "
                         "whenever the gh-pages branch exists.")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "Comfy-Forge/cuda-wheels")

    print(f"Generating index for {repo}")

    # Fetch releases
    releases = get_releases(repo, token)

    # Collect all wheels from releases
    packages = {}
    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue

            # Extract package name (first part before -)
            pkg_name = name.split("-")[0].lower().replace("_", "-")

            url = asset["browser_download_url"]

            packages.setdefault(pkg_name, []).append({
                "filename": name,      # the actual asset name
                "url": url,
                # GitHub's releases API serves a sha256 digest per asset --
                # free hash verification for the manifest, no downloads.
                "sha256": (asset.get("digest") or "").removeprefix("sha256:"),
            })

    # One built wheel, many display names, for packages that never link
    # libtorch. Must run before the guard so aliases are counted in the index.
    torch_free = load_torch_free_packages()
    n_aliases = expand_torch_free_aliases(packages, torch_free, load_grid())
    if torch_free:
        print(f"torch-independent packages: {', '.join(sorted(torch_free))}")
        print(f"  emitted {n_aliases} alias listing(s) (0 extra wheels built or stored)")

    # ── Shrinkage guard ────────────────────────────────────────────────
    # The old guard diffed DIRECTORY NAMES in the previous gh-pages checkout
    # and refused to publish if any were missing. Three things were wrong with
    # it, and all three fired in the same week:
    #
    #  * It measured names, ~20x coarser than the content. A package whose
    #    every link had gone 404 still "passed" as long as its directory
    #    existed -- which is exactly the state the live index is in now.
    #  * Its baseline was the live gh-pages content, which the deploy itself
    #    overwrites under force_orphan. One bad deploy anchored the guard to
    #    the bad state permanently, with no history to revert to.
    #  * It was UNFALSIFIABLE after a wipe. Its remedy text says "delete its
    #    release and re-run" -- but the releases being gone is precisely why
    #    the packages are missing. There was no action that satisfied it.
    #
    # The replacement discriminates by CAUSE rather than by magnitude, which
    # needs no flag (a bare --force becomes muscle memory and the guard stops
    # existing) and no hand-maintained list:
    #
    #    a truncated fetch / bad prune loses ASSETS under releases that STILL
    #    EXIST.  an operator deleting a package loses the RELEASE OBJECT.
    #
    # So: lost package whose release tag is still live -> HARD FAIL.
    #     lost package whose release tag is gone       -> loud WARN, proceed.
    #
    # A full farm wipe (which has happened twice this week and will happen
    # again) therefore self-clears in one run, while the truncation case the
    # guard was actually written for still stops the deploy.
    live_tags = {r.get("tag_name", "") for r in releases}

    def _tag_alive(pkg_name):
        # Release tags are "<pkg>-latest"; index names are normalised
        # (lowercase, "_" -> "-"), so compare normalised on both sides.
        for t in live_tags:
            base = t[:-len("-latest")] if t.endswith("-latest") else t
            if base.lower().replace("_", "-") == pkg_name:
                return True
        return False

    prev_root = Path(args.previous)
    baseline = None
    manifest_path = prev_root / "packages.json"
    if manifest_path.is_file():
        # Baseline from the machine-readable manifest this script already
        # writes, NOT from a directory listing. That also deletes the
        # hand-maintained exclusion list ("matrix", "dashboard", "find", the
        # cu\d+ channel regex...) which had already broken once.
        try:
            _prev = json.loads(manifest_path.read_text())
            if int(_prev.get("schema", 0)) >= 1:
                baseline = set(_prev.get("packages", {}))
        except Exception as exc:
            print(f"WARNING: could not parse {manifest_path}: {exc}")
    if baseline is None:
        # An absent baseline is ambiguous in exactly the same shape as an
        # absent package, so discriminate it the same way -- by cause, not by
        # shrugging. Either nothing was ever published (genuine first deploy)
        # or something ate the checkout: a network blip on the gh-pages
        # checkout step, which `continue-on-error: true` in the workflow turns
        # into a silent skip, or `--previous` aimed at a path that does not
        # exist. Skipping on ambiguity is fail-OPEN, and fail-open is how the
        # index came to be 58% dead links in the first place.
        if published_branch_exists(repo, "gh-pages", token):
            raise SystemExit(
                f"ERROR: no usable baseline manifest at {manifest_path}, but the "
                f"gh-pages branch EXISTS on {repo}. The baseline was lost, not "
                f"absent -- most likely the gh-pages checkout step failed (it "
                f"runs with continue-on-error) or --previous points somewhere "
                f"wrong. Refusing to publish with the shrinkage guard disabled; "
                f"fix the checkout and re-run. There is deliberately no flag to "
                f"bypass this.")
        print(f"WARNING: no usable baseline manifest at {manifest_path} and no "
              f"gh-pages branch on {repo} -- nothing has ever been published, so "
              f"the shrinkage guard has nothing to compare against and is "
              f"SKIPPED. Expected on the very first deploy only.")
        lost_gone, lost_live = [], []
    else:
        lost = sorted(baseline - set(packages))
        lost_live = [p for p in lost if _tag_alive(p)]
        lost_gone = [p for p in lost if not _tag_alive(p)]

    if lost_live:
        print(f"ERROR: {len(lost_live)} package(s) vanished from the index while "
              f"their release still exists:")
        for name in lost_live:
            print(f"  - {name}  (release tag is LIVE -- assets went missing)")
        print("That is the signature of a truncated fetch, a partial API page, "
              "or a prune running against this repo mid-generation -- NOT of a "
              "deliberate removal. Refusing to publish. Re-run; if it persists, "
              "the release genuinely lost its assets and needs a rebuild.")
        raise SystemExit(1)

    if lost_gone:
        print(f"NOTE: {len(lost_gone)} package(s) are absent because their "
              f"release was deleted -- publishing without them:")
        for name in lost_gone:
            print(f"  - {name}")

    print(f"{len(packages)} packages, {sum(len(v) for v in packages.values())} wheels")

    # Create docs directory
    docs = Path(args.out)
    docs.mkdir(exist_ok=True)

    all_packages = sorted(packages.keys())

    # Generate root index. Still a valid PEP 503 root (pip constructs
    # /<package>/ URLs directly and never scrapes this page), so the human
    # nav on top costs nothing.
    with open(docs / "index.html", "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head><title>CUDA Wheels Index</title>\n")
        f.write('<style>body{font-family:sans-serif;max-width:900px;margin:2rem auto;'
                'padding:0 1rem;line-height:1.5}nav{padding:.8rem 1rem;background:#f4f6f8;'
                'border-radius:8px;margin-bottom:1.2rem}nav a{margin-right:1.2rem}'
                'p.hint{color:#555;font-size:.92rem}</style></head>\n')
        f.write("<body>\n")
        f.write("<h1>CUDA Wheels</h1>\n")
        f.write('<nav><a href="find/"><b>Find your wheel</b></a> '
                '<a href="matrix/">Upstream PyTorch matrix</a> '
                '<a href="archs/">GPU architectures</a></nav>\n')
        f.write('<p class="hint">This page is the PEP 503 simple index '
                '(what pip and comfy-env resolve against; per-combo channels '
                'live at <code>/cu&lt;ver&gt;/&lt;torch&gt;/</code>). Humans '
                'wanting an install command: use <a href="find/">Find your '
                'wheel</a>.</p>\n')
        for pkg in all_packages:
            f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
        f.write("</body>\n</html>\n")

    # Per-package pages: THE index. One flat PEP 503 tree at the root,
    # actual asset filenames -- the v1 naming shim and the /v2/ mirror are
    # gone (this repo has no legacy consumers; the old farm keeps serving
    # the old names).
    for pkg, wheels in packages.items():
        pkg_dir = docs / pkg
        pkg_dir.mkdir(exist_ok=True)

        with open(pkg_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>{pkg}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>{pkg}</h1>\n")
            for wheel in sorted(wheels, key=lambda w: w["filename"]):
                f.write(f'<a href="{wheel["url"]}">{wheel["filename"]}</a><br>\n')
            f.write("</body>\n</html>\n")

    print(f"Generated index for {len(packages)} built packages:")
    for pkg, wheels in packages.items():
        print(f"  - {pkg}: {len(wheels)} wheels")
    print(f"Total: {len(all_packages)} packages in index")

    # Per-combo indexes: docs/<cuda>/<torch>/<pkg>/
    #
    # The flat index cannot be resolved. A wheel's CUDA and torch versions live
    # only in its local version tag, and pip matches neither -- so an unpinned
    # install against the flat index picks the highest combo present, not the one the
    # machine can load. GPU architecture is not expressible at all. Putting the
    # combo in the URL is the only way to make selection unambiguous, and it is
    # what download.pytorch.org does (/whl/cu128/).
    #
    # Additive next to the flat root index.
    combos = {}
    for pkg, wheels in packages.items():
        for wheel in wheels:
            m = _COMBO_RE.search(wheel["filename"])
            if not m:
                continue
            combos.setdefault((m.group(1), m.group(2)), {}).setdefault(pkg, []).append(wheel)

    for (cuda, torch), pkgs in sorted(combos.items()):
        combo_dir = docs / cuda / torch
        combo_dir.mkdir(parents=True, exist_ok=True)
        with open(combo_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>CUDA Wheels {cuda}/{torch}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>CUDA Wheels -- {cuda} / {torch}</h1>\n")
            for pkg in sorted(pkgs):
                f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
            f.write("</body>\n</html>\n")

        for pkg, wheels in pkgs.items():
            pkg_dir = combo_dir / pkg
            pkg_dir.mkdir(exist_ok=True)
            with open(pkg_dir / "index.html", "w") as f:
                f.write("<!DOCTYPE html>\n")
                f.write(f"<html>\n<head><title>{pkg} {cuda}/{torch}</title></head>\n")
                f.write("<body>\n")
                f.write(f"<h1>{pkg} -- {cuda} / {torch}</h1>\n")
                for wheel in sorted(wheels, key=lambda w: w["filename"]):
                    f.write(f'<a href="{wheel["url"]}">{wheel["filename"]}</a><br>\n')
                f.write("</body>\n</html>\n")

    # Machine-readable manifest: consumers (comfy-env first) stop
    # regex-scraping the PEP 503 HTML and gain sha256 verification. Real
    # assets only -- aliases are derivable from torch_free + the grid.
    import sys as _s2
    _s2.path.insert(0, str(Path(__file__).resolve().parent))
    from audit import parse_wheel as _parse_wheel
    manifest = {
        "schema": 1, "repo": repo,
        # Provenance: makes "how stale is the live index" answerable with a
        # single curl, and lets a future concurrency check detect another
        # writer landing on gh-pages underneath this run.
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
                            .replace(microsecond=0).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA", ""),
        "release_count": len(releases),
        "asset_count": sum(len(v) for v in packages.values()),
        "removed_since_previous": lost_gone,
        "packages": {}}
    for pkg, wheels in packages.items():
        entries = []
        for w in wheels:
            if w.get("alias_of"):
                continue
            parsed = _parse_wheel(w["filename"])
            if not parsed:
                continue
            entries.append({
                "filename": w["filename"],
                "version": parsed["version"],
                "cuda": parsed["cuda"],
                "torch": parsed["torch_short"],
                # None for abi-agnostic wheels (py3-none / cpXY-abi3):
                # they satisfy every python; "abi" says which shape.
                "python": parsed["python"],
                "abi": parsed.get("abi", "cp"),
                "platform": parsed["platform"],
                "url": w["url"],
                "sha256": w.get("sha256", ""),
            })
        manifest["packages"][pkg] = {
            "torch_free": pkg in torch_free,
            "wheels": sorted(entries, key=lambda e: e["filename"]),
        }
    (docs / "packages.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"Wrote packages.json manifest: "
          f"{sum(len(v['wheels']) for v in manifest['packages'].values())} wheels, "
          f"{len(manifest['packages'])} packages")

    print(f"Generated {len(combos)} per-combo indexes "
          f"({sum(len(p) for p in combos.values())} package entries)")


if __name__ == "__main__":
    main()
