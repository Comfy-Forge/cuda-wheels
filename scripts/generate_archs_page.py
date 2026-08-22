#!/usr/bin/env python3
"""Build the GPU-architecture page into <out>/archs/.

Three truths on one static page, all baked at build time:

1. The farm's OWNED arch policy (defaults/arch_policy.yml): which SM
   archs every wheel is compiled for, per CUDA line, x86 and aarch64,
   plus the per-(cuda, torch) exceptions.
2. What PyTorch ITSELF bakes into libtorch per (cuda, torch) cell --
   evaluated from pytorch's own .ci/manywheel/build_cuda.sh at each
   release tag (fetch_pytorch_arch_lists.py), both lanes. This is the
   derivation input for 1 (CW-ADR-0012): a farm wheel never targets an
   arch the underlying torch cannot reach.
3. Per-package overrides (packages/*/arch_override.yml) for kernels
   that cannot span the whole policy row.

Usage:
    python scripts/generate_archs_page.py --out _site
"""
import argparse
import html
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from package_loader import load_pcto, load_arch_policy  # noqa: E402
import fetch_pytorch_arch_lists as fpal                 # noqa: E402

SM_LEGEND = [
    ("7.0", "Volta (V100)"), ("7.5", "Turing (RTX 20xx, T4)"),
    ("8.0", "Ampere DC (A100)"), ("8.6", "Ampere consumer (RTX 30xx)"),
    ("8.7", "Jetson Orin"), ("8.9", "Ada (RTX 40xx, L4/L40)"),
    ("9.0", "Hopper (H100/H200, GH200)"), ("10.0", "Blackwell DC (B200/GB200)"),
    ("11.0", "Thor"), ("12.0", "Blackwell consumer (RTX 50xx)"),
    ("12.1", "DGX Spark (GB10)"),
]

CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:#0d1117; color:#c9d1d9; padding:2rem; max-width:1100px; margin:0 auto; }
h1 { color:#f0f6fc; font-size:1.9rem; margin-bottom:.35rem; }
h2 { color:#f0f6fc; font-size:1.15rem; margin:1.8rem 0 .6rem; }
.subtitle { color:#8b949e; margin-bottom:1.4rem; }
a { color:#58a6ff; text-decoration:none; } a:hover { text-decoration:underline; }
nav { margin-bottom:1.6rem; font-size:.92rem; } nav a { margin-right:1.1rem; }
table { border-collapse:collapse; font-size:.88rem; margin-bottom:.4rem; }
th,td { border:1px solid #30363d; padding:.35rem .7rem; text-align:left; vertical-align:top; }
th { background:#161b22; color:#8b949e; font-weight:600; }
code, .mono { font-family:"SF Mono","Fira Code",Consolas,monospace; font-size:.85em; }
.note { color:#8b949e; font-size:.85rem; margin:.3rem 0 .8rem; }
.wrap { overflow-x:auto; }
.ptx { color:#d2a8ff; }
footer { margin-top:2.5rem; color:#484f58; font-size:.88rem;
         border-top:1px solid #21262d; padding-top:1rem; }
"""


def fmt_archs(s):
    out = []
    for tok in str(s).replace(";", " ").split():
        if tok.endswith("+PTX"):
            out.append(f'{html.escape(tok[:-4])}<span class="ptx">+PTX</span>')
        else:
            out.append(html.escape(tok))
    return " ".join(out)


def upstream_rows():
    """[(cuda, torch, x86_str, arm_str)] for every grid combo, newest first."""
    pcto = load_pcto()
    rows = []
    for combo in pcto.get("combinations", []):
        cuda, torch = str(combo["cuda"]), str(combo["pytorch"])
        tag = f"v{torch}"
        cells = {}
        for arch in ("x86_64", "aarch64"):
            try:
                r = fpal.fetch(tag, cuda, arch=arch)
                if not r:
                    cells[arch] = "(not in this tag's build script)"
                    continue
                sass = [s.replace("sm_", "") for s in r.get("sass", [])]
                ptx = {s.replace("sm_", "") for s in r.get("ptx", [])}
                cells[arch] = " ".join(
                    (f"{s[0]}.{s[1:]}" if s.isdigit() else s) +
                    ("+PTX" if s in ptx else "")
                    for s in sass) or "(empty)"
            except Exception as e:
                cells[arch] = f"(unavailable: {e})"
        # Older tags have no aarch64 branch in build_cuda.sh at all (their
        # ARM wheels came from a separate script); an ARM eval that equals
        # x86 verbatim there would be a fabrication, not a fact.
        if cells["aarch64"] == cells["x86_64"]:
            cells["aarch64"] = "(no aarch64 branch in this tag's script)"
        rows.append((cuda, torch, cells["x86_64"], cells["aarch64"]))
    return rows


def override_rows():
    out = []
    for d in sorted((ROOT / "packages").iterdir()):
        f = d / "arch_override.yml"
        if not f.exists():
            continue
        cfg = yaml.safe_load(f.read_text()) or {}
        for field in ("arch_list", "arch_list_by_cuda",
                      "arch_list_aarch64", "arch_list_by_cuda_aarch64"):
            if field not in cfg:
                continue
            val = cfg[field]
            if isinstance(val, dict):
                rendered = "<br>".join(
                    f"cu{str(k).replace('.', '')}: {fmt_archs(v)}"
                    for k, v in sorted(val.items()))
            else:
                rendered = fmt_archs(val)
            out.append((d.name, field, rendered))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    args = ap.parse_args()

    policy = load_arch_policy()
    up = upstream_rows()
    ov = override_rows()

    p = []
    p.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU architectures — cuda-wheels</title>
<style>{CSS}</style></head><body>
<h1>GPU architectures</h1>
<div class="subtitle">Which SM targets every wheel carries, where that policy
comes from, and which packages deviate.</div>
<nav><a href="../">PEP 503 index</a> <a href="../find/">Find your wheel</a>
<a href="../matrix/">Upstream PyTorch matrix</a></nav>

<h2>Legend</h2>
<div class="wrap"><table><tr><th>SM</th><th>Silicon</th></tr>""")
    for sm, name in SM_LEGEND:
        p.append(f"<tr><td class=mono>{sm}</td><td>{name}</td></tr>")
    p.append("""</table></div>
<div class="note"><span class="ptx">+PTX</span> = forward-compatible IR
embedded at that arch, so future GPUs can JIT it.</div>

<h2>Farm policy — x86_64 (linux + windows)</h2>
<div class="wrap"><table><tr><th>CUDA</th><th>TORCH_CUDA_ARCH_LIST</th></tr>""")
    for cuda, archs in sorted(policy.get("arch_policy", {}).items()):
        p.append(f"<tr><td class=mono>{cuda}</td><td class=mono>{fmt_archs(archs)}</td></tr>")
    p.append("""</table></div>

<h2>Farm policy — aarch64 (SBSA / Jetson)</h2>
<div class="note">A CUDA line absent here is deliberately not built for ARM.</div>
<div class="wrap"><table><tr><th>CUDA</th><th>TORCH_CUDA_ARCH_LIST</th></tr>""")
    for cuda, archs in sorted(policy.get("arch_policy_aarch64", {}).items()):
        p.append(f"<tr><td class=mono>{cuda}</td><td class=mono>{fmt_archs(archs)}</td></tr>")
    p.append("""</table></div>

<h2>Exceptions (per CUDA / torch pairing)</h2>
<div class="note">Old torch minors that still shipped archs the current policy
floor dropped -- a farm wheel never targets an arch its torch cannot reach.</div>
<div class="wrap"><table><tr><th>CUDA / torch</th><th>Arch list</th></tr>""")
    for key, archs in sorted((policy.get("arch_exceptions") or {}).items()):
        p.append(f"<tr><td class=mono>{key}</td><td class=mono>{fmt_archs(archs)}</td></tr>")
    p.append("""</table></div>

<h2>What PyTorch itself bakes into libtorch</h2>
<div class="note">Evaluated from pytorch's own
<code>.ci/manywheel/build_cuda.sh</code> at each release tag -- the derivation
input for the farm policy above (CW-ADR-0012).</div>
<div class="wrap"><table><tr><th>CUDA</th><th>torch</th>
<th>x86_64</th><th>aarch64</th></tr>""")
    for cuda, torch, x86, arm in up:
        p.append(f"<tr><td class=mono>{cuda}</td><td class=mono>{torch}</td>"
                 f"<td class=mono>{fmt_archs(x86)}</td>"
                 f"<td class=mono>{fmt_archs(arm)}</td></tr>")
    p.append("""</table></div>

<h2>Per-package overrides</h2>
<div class="note">Packages whose kernels cannot span the whole policy row --
each override is justified in the package folder's README.</div>
<div class="wrap"><table><tr><th>Package</th><th>Field</th><th>Value</th></tr>""")
    for pkg, field, rendered in ov:
        p.append(f"<tr><td class=mono>{pkg}</td><td class=mono>{field}</td>"
                 f"<td class=mono>{rendered}</td></tr>")
    p.append("""</table></div>

<footer>Built by
<a href="https://github.com/Comfy-Forge/cuda-wheels">Comfy-Forge/cuda-wheels</a>
from <code>defaults/arch_policy.yml</code>, pytorch's build scripts, and
<code>packages/*/arch_override.yml</code>.</footer>
</body></html>""")

    out = Path(args.out) / "archs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text("\n".join(p))
    print(f"Wrote {out / 'index.html'}: {len(up)} upstream combos, "
          f"{len(ov)} package override rows")


if __name__ == "__main__":
    main()
