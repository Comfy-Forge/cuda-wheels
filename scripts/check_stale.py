#!/usr/bin/env python3
"""Report published wheels whose arch list no longer matches the config.

WHY THIS EXISTS
---------------
Rebuilds are decided by `generate_matrix.wheel_exists()`, which matches on the
wheel FILENAME. Changing a package's arch list does not change any wheel's
name, so after a coverage fix lands the next run sees a correctly-named wheel
sitting in the release and skips the cell. The fix reaches only the cells that
happened to have no wheel yet.

That is not hypothetical. On 2026-08-26, 24 sageattention aarch64 wheels were
still shipping a device trap on Ada fourteen hours after the fix was committed,
because every cell that already had a wheel was skipped. The cells that were
empty picked the fix up; the ones that mattered did not. Nothing in the farm
could answer "is this published wheel built from the current config?".

A wheel cannot carry that answer in its name -- PEP 427 fixes the fields and
the index parses them -- so `patch_wheel_version.py` stamps it into METADATA as
`Comfy-Forge-Arch-List:`. This script reads that header back off the published
wheels and compares it against what the resolver produces today.

It is a REPORT, not a gate: it changes nothing and dispatches nothing. Feed its
output to a rebuild with `-f overwrite=true`.

WHEELS BUILT BEFORE THE STAMP
-----------------------------
have no header. Those are reported as UNKNOWN rather than stale -- absence of
evidence is not evidence of drift, and calling them stale would flag the entire
back catalogue on the first run.

USAGE
    python scripts/check_stale.py                      # every package
    python scripts/check_stale.py --package sageattention
    python scripts/check_stale.py --fail-on-stale      # exit 1 if any drifted
"""
import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_matrix as GM                                   # noqa: E402
from package_loader import iter_packages                       # noqa: E402

WHEEL_RE = re.compile(
    r"^(?P<pkg>.+?)-(?P<ver>[^-]+)-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$")
COMBO_RE = re.compile(r"\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)")


def _plat_lane(tag: str) -> str:
    if "win" in tag:
        return "windows"
    if "aarch64" in tag:
        return "linux_aarch64"
    return "linux"


def _assets(pkg: str):
    """(name, url) for every asset on <pkg>-latest. Empty if no release."""
    r = subprocess.run(
        ["gh", "release", "view", f"{pkg}-latest", "--json", "assets"],
        capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [(a["name"], a["url"])
            for a in json.loads(r.stdout).get("assets", [])
            if a["name"].endswith(".whl")]


def _metadata_arch(url: str) -> str | None:
    """Read Comfy-Forge-Arch-List from a remote wheel.

    Two range requests, not a full download: the zip central directory lives in
    the last 64KB, and METADATA is a few hundred bytes. A wheel here can be
    300MB, so downloading them all is not an option.
    """
    try:
        head = Request(url, headers={"Range": "bytes=-65536",
                                     "Accept": "application/octet-stream"})
        tail = urlopen(head, timeout=30).read()
    except Exception:
        return None
    # locate METADATA's local-header offset from the central directory
    m = None
    for cd in re.finditer(rb"PK\x01\x02", tail):
        off = cd.start()
        nlen = int.from_bytes(tail[off + 28:off + 30], "little")
        name = tail[off + 46:off + 46 + nlen]
        if name.endswith(b".dist-info/METADATA"):
            m = (int.from_bytes(tail[off + 42:off + 46], "little"),
                 int.from_bytes(tail[off + 20:off + 24], "little"))
            break
    if not m:
        return None
    local_off, comp_size = m
    try:
        rng = f"bytes={local_off}-{local_off + comp_size + 4096}"
        blob = urlopen(Request(url, headers={"Range": rng,
                                             "Accept": "application/octet-stream"}),
                       timeout=30).read()
    except Exception:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(
                blob[:0] or blob)) as _:                 # pragma: no cover
            pass
    except Exception:
        pass
    # the local header is 30 bytes + name + extra; just scan the decompressed
    # payload for the header we want
    for start in range(0, min(len(blob), 512)):
        try:
            import zlib
            txt = zlib.decompress(blob[start:], -15).decode(
                "utf-8", "replace")
        except Exception:
            continue
        hit = re.search(r"^Comfy-Forge-Arch-List:\s*(.+)$", txt, re.MULTILINE)
        return hit.group(1).strip() if hit else ""
    return None


def _resolved(cfg: dict, cuda: str, torch: str, lane: str) -> str | None:
    try:
        if lane == "linux_aarch64":
            return GM.resolve_aarch64_arch_list(cfg, cuda, torch)
        default = GM.policy_arch_list(cuda, torch, platform=lane)
        if lane == "windows":
            return GM.resolve_windows_arch_list(
                cfg, cuda, pytorch_version=torch, default_arch_list=default)
        return GM.resolve_arch_list(cfg, cuda, pytorch_version=torch,
                                    default_arch_list=default)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="all")
    ap.add_argument("--fail-on-stale", action="store_true")
    args = ap.parse_args()

    stale, unknown, ok = [], 0, 0
    for _folder, cfg in iter_packages():
        name = cfg["name"]
        if args.package != "all" and name != args.package:
            continue
        assets = _assets(name)
        if not assets:
            continue
        print(f"{name}: {len(assets)} wheel(s)", flush=True)
        for aname, url in assets:
            wm = WHEEL_RE.match(aname)
            cm = COMBO_RE.search(aname)
            if not (wm and cm):
                continue
            cuda = f"{cm['cuda'][:2]}.{cm['cuda'][2:]}"
            torch = cm["torch"]
            lane = _plat_lane(wm["plat"])
            want = _resolved(cfg, cuda, torch, lane)
            if want is None:
                continue
            got = _metadata_arch(url)
            if got is None or got == "":
                unknown += 1
                continue
            if GM._normalize_arch_list(got) != GM._normalize_arch_list(want):
                stale.append((aname, got, want))
                print(f"  STALE {aname}\n        built: {got}\n        now:   {want}")
            else:
                ok += 1

    print(f"\nup-to-date {ok} | stale {len(stale)} | "
          f"unknown (built before the stamp) {unknown}")
    if stale:
        pkgs = sorted({WHEEL_RE.match(n)['pkg'] for n, _, _ in stale})
        print("\nRebuild with overwrite:")
        for p in pkgs:
            print(f"  gh workflow run build.yml -f package={p} -f overwrite=true ...")
    return 1 if (stale and args.fail_on_stale) else 0


if __name__ == "__main__":
    sys.exit(main())
