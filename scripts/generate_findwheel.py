#!/usr/bin/env python3
"""Build the "Find your wheel" page into <out>/find/.

A static, self-contained HTML page: pick OS / CUDA / PyTorch / Python,
tick packages, get a copy-pasteable install command against the
per-combo channel of THIS site's PEP 503 index, plus direct wheel
links. Wheel availability comes from ../packages.json (the manifest
generate_index.py writes) fetched at page load, so the page never goes
stale between deploys of the same site. The "what PyTorch upstream
ships" panel is baked in at build time from the committed PCWM snapshot
(defaults/scraped_torch_matrix.json) -- same source the /matrix/ page
renders.

Usage:
    python scripts/generate_findwheel.py --out _site
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    args = ap.parse_args()

    snapshot = json.loads(
        (ROOT / "defaults" / "scraped_torch_matrix.json").read_text())
    upstream = snapshot.get("summary", [])

    out = Path(args.out) / "find"
    out.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("/*__UPSTREAM__*/[]",
                            json.dumps(upstream, separators=(",", ":")))
    (out / "index.html").write_text(html)
    print(f"Wrote {out / 'index.html'} "
          f"({len(upstream)} upstream combo rows embedded)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Find your wheel — cuda-wheels</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 2rem; font-size: 1rem; max-width: 1100px; margin: 0 auto; }
h1 { color: #f0f6fc; margin-bottom: 0.35rem; font-size: 1.9rem; }
h2 { color: #f0f6fc; font-size: 1.15rem; margin: 1.6rem 0 0.7rem; }
.subtitle { color: #8b949e; margin-bottom: 1.6rem; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
nav { margin-bottom: 1.8rem; font-size: 0.92rem; color: #484f58; }
nav a { margin-right: 1.1rem; }
.selectors { display: flex; flex-wrap: wrap; gap: 1.2rem; margin-bottom: 1.4rem; }
.selector { display: flex; flex-direction: column; gap: 0.4rem; }
.selector label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.selector select { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
                   padding: 0.5rem 0.75rem; font-size: 1rem; font-family: inherit; cursor: pointer; min-width: 150px; }
.selector select:focus { outline: 2px solid #58a6ff; outline-offset: 1px; }
.pkg-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pkg-chip { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 0.35rem 0.75rem;
            font-size: 0.9rem; cursor: pointer; user-select: none; }
.pkg-chip:hover { border-color: #58a6ff; }
.pkg-chip.selected { background: #1f3a5f; border-color: #58a6ff; color: #f0f6fc; }
.pkg-chip.unavailable { opacity: 0.35; cursor: not-allowed; text-decoration: line-through; }
.command-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.2rem;
               font-family: "SF Mono", "Fira Code", Consolas, monospace; font-size: 0.88rem; line-height: 1.6;
               white-space: pre-wrap; word-break: break-all; position: relative; min-height: 3rem; }
.command-box .placeholder { color: #484f58; }
.copy-btn { position: absolute; top: 0.5rem; right: 0.5rem; background: #21262d; border: 1px solid #30363d;
            color: #8b949e; padding: 0.3rem 0.7rem; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }
.copy-btn.copied { color: #3fb950; border-color: #3fb950; }
table { border-collapse: collapse; font-size: 0.88rem; }
th, td { border: 1px solid #30363d; padding: 0.35rem 0.7rem; text-align: left; }
th { background: #161b22; color: #8b949e; font-weight: 600; }
.ok { color: #3fb950; } .no { color: #f85149; }
.wheel-list { font-family: "SF Mono", "Fira Code", Consolas, monospace; font-size: 0.82rem; line-height: 1.7; }
.note { color: #8b949e; font-size: 0.85rem; margin-top: 0.5rem; }
footer { margin-top: 2.5rem; color: #484f58; font-size: 0.88rem; border-top: 1px solid #21262d; padding-top: 1rem; }
</style>
</head>
<body>
<h1>Find your wheel</h1>
<div class="subtitle">Pick your system, tick your packages, copy the command.</div>
<nav>
  <a href="../">PEP 503 index (for pip / comfy-env)</a>
  <a href="../matrix/">Upstream PyTorch matrix</a>
  <a href="../archs/">GPU architectures</a>
</nav>

<div class="selectors">
  <div class="selector"><label for="sel-os">Operating system</label>
    <select id="sel-os"></select></div>
  <div class="selector"><label for="sel-cuda">CUDA</label>
    <select id="sel-cuda"></select></div>
  <div class="selector"><label for="sel-torch">PyTorch</label>
    <select id="sel-torch"></select></div>
  <div class="selector"><label for="sel-py">Python</label>
    <select id="sel-py"></select></div>
</div>

<div id="upstream-verdict" class="note"></div>

<h2>Packages</h2>
<div id="chips" class="pkg-chips"><span class="placeholder">loading packages.json…</span></div>

<h2>Install</h2>
<div class="command-box"><button class="copy-btn" id="copy">copy</button><span id="cmd" class="placeholder">select a system and at least one package</span></div>
<div class="note">Torch itself installs from PyTorch's own index; our wheels ride the matching per-combo channel of this site.</div>

<h2>Matching wheels</h2>
<div id="wheels" class="wheel-list"><span class="placeholder">—</span></div>

<h2>What PyTorch upstream ships</h2>
<div class="note" style="margin-bottom:0.6rem">The farm only builds cells PyTorch itself publishes wheels for. Full page: <a href="../matrix/">upstream matrix</a>.</div>
<div style="overflow-x:auto"><table id="upstream"><thead><tr><th>CUDA</th><th>PyTorch</th><th>Pythons</th><th>Platforms</th></tr></thead><tbody></tbody></table></div>

<footer>
  Served from the <a href="../">cuda-wheels index</a> · built by the
  <a href="https://github.com/Comfy-Forge/cuda-wheels">Comfy-Forge/cuda-wheels</a> farm ·
  wheel data: <a href="../packages.json">packages.json</a>
</footer>

<script>
var UPSTREAM = /*__UPSTREAM__*/[];
var OS_LABELS = { linux: "Linux x86_64", windows: "Windows", linux_aarch64: "Linux ARM64" };
var state = { os: "", cuda: "", torch: "", py: "" };
var manifest = null, selected = {};

function el(id) { return document.getElementById(id); }
function uniq(a) { return Array.from(new Set(a)); }
function vercmp(a, b) {
  var pa = String(a).split(".").map(Number), pb = String(b).split(".").map(Number);
  for (var i = 0; i < Math.max(pa.length, pb.length); i++) {
    var d = (pa[i] || 0) - (pb[i] || 0); if (d) return d;
  } return 0;
}

function allWheels() {
  var out = [];
  Object.keys(manifest.packages).forEach(function(pkg) {
    manifest.packages[pkg].wheels.forEach(function(w) {
      out.push(Object.assign({ pkg: pkg, torch_free: manifest.packages[pkg].torch_free }, w));
    });
  });
  return out;
}

function matches(w, ignore) {
  if (state.os && ignore !== "os" && w.platform !== state.os) return false;
  if (state.cuda && ignore !== "cuda" && w.cuda !== state.cuda) return false;
  if (state.torch && ignore !== "torch" && w.torch !== state.torch && !w.torch_free) return false;
  // python === null means abi-agnostic (py3-none / abi3): fits every python
  if (state.py && ignore !== "py" && w.python !== null &&
      w.python !== state.py.replace(".", "")) return false;
  return true;
}

function fillSelect(id, values, current, labeler) {
  var sel = el(id);
  sel.innerHTML = "";
  var any = document.createElement("option");
  any.value = ""; any.textContent = "any";
  sel.appendChild(any);
  values.forEach(function(v) {
    var o = document.createElement("option");
    o.value = v; o.textContent = labeler ? labeler(v) : v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  });
}

function pyDotted(v) { return v[0] + "." + v.slice(1); }

function refresh() {
  var ws = allWheels();
  fillSelect("sel-os",
    uniq(ws.filter(function(w){return matches(w,"os");}).map(function(w){return w.platform;})).sort(),
    state.os, function(v){ return OS_LABELS[v] || v; });
  fillSelect("sel-cuda",
    uniq(ws.filter(function(w){return matches(w,"cuda");}).map(function(w){return w.cuda;})).sort(vercmp),
    state.cuda);
  fillSelect("sel-torch",
    uniq(ws.filter(function(w){return matches(w,"torch");}).map(function(w){return w.torch;})).sort(vercmp),
    state.torch);
  fillSelect("sel-py",
    uniq(ws.filter(function(w){return matches(w,"py") && w.python !== null;})
           .map(function(w){return pyDotted(w.python);})).sort(vercmp),
    state.py);

  var chips = el("chips"); chips.innerHTML = "";
  Object.keys(manifest.packages).sort().forEach(function(pkg) {
    var avail = manifest.packages[pkg].wheels.some(function(w) {
      return matches(Object.assign({ torch_free: manifest.packages[pkg].torch_free }, w));
    });
    var c = document.createElement("span");
    c.className = "pkg-chip" + (avail ? "" : " unavailable") + (selected[pkg] && avail ? " selected" : "");
    c.textContent = pkg;
    if (avail) c.onclick = function() { selected[pkg] = !selected[pkg]; refresh(); };
    chips.appendChild(c);
  });

  renderCommand(ws);
  renderUpstream();
}

function siteBase() {
  return location.href.replace(/find\/?(index\.html)?(\?.*)?(#.*)?$/, "");
}

function renderCommand(ws) {
  var pkgs = Object.keys(selected).filter(function(p) { return selected[p]; }).sort();
  var cmdEl = el("cmd"), wheelsEl = el("wheels");
  var hits = ws.filter(function(w) { return matches(w) && selected[w.pkg]; });
  if (!pkgs.length || !state.cuda || !state.torch) {
    cmdEl.className = "placeholder";
    cmdEl.textContent = !pkgs.length ? "select a system and at least one package"
                                     : "pick a CUDA and PyTorch version to get a channel URL";
    wheelsEl.innerHTML = '<span class="placeholder">—</span>';
    return;
  }
  var cuShort = "cu" + state.cuda.replace(".", "");
  // Channel layout mirrors generate_index.py's _COMBO_RE groups:
  // /<cuXXX>/<torchY.Y>/ -- the "torch" prefix is part of the dir name.
  var channel = siteBase() + cuShort + "/torch" + state.torch + "/";
  var torchLine = "pip install torch==" + state.torch + ".* --index-url https://download.pytorch.org/whl/" + cuShort + "\n";
  cmdEl.className = "";
  cmdEl.textContent = torchLine +
    "pip install " + pkgs.map(function(p){return p;}).join(" ") +
    " --extra-index-url " + channel;
  wheelsEl.innerHTML = hits.length
    ? hits.map(function(w) {
        return '<a href="' + w.url + '">' + w.filename + "</a>";
      }).join("<br>")
    : '<span class="no">no wheels match every selection — loosen a filter</span>';
}

function renderUpstream() {
  var tb = el("upstream").querySelector("tbody");
  tb.innerHTML = "";
  var cuShort = state.cuda ? "cu" + state.cuda.replace(".", "") : "";
  var shown = 0;
  UPSTREAM.forEach(function(row) {
    if (cuShort && row.cuda !== cuShort) return;
    if (state.torch && row.torch.indexOf(state.torch) !== 0) return;
    var tr = document.createElement("tr");
    tr.innerHTML = "<td>" + row.cuda + "</td><td>" + row.torch + "</td><td>" +
      row.python.join(" ") + "</td><td>" + row.platforms.join(" ") + "</td>";
    tb.appendChild(tr); shown++;
  });
  var verdict = el("upstream-verdict");
  if (state.cuda && state.torch) {
    var hit = UPSTREAM.some(function(r) {
      return r.cuda === cuShort && r.torch.indexOf(state.torch) === 0 &&
        (!state.os || r.platforms.some(function(p) {
          return state.os === "linux" ? p === "linux_x86_64" : p === state.os ||
                 (state.os === "windows" && p === "windows");
        })) &&
        (!state.py || r.python.indexOf(state.py) !== -1);
    });
    verdict.innerHTML = hit
      ? '<span class="ok">✓ PyTorch upstream ships this combo</span>'
      : '<span class="no">✗ PyTorch upstream does not ship this combo — the farm cannot build it</span>';
  } else { verdict.textContent = ""; }
  if (!shown) tb.innerHTML = '<tr><td colspan="4" class="placeholder">no upstream rows match</td></tr>';
}

["os", "cuda", "torch", "py"].forEach(function(k) {
  el("sel-" + k).addEventListener("change", function(e) {
    state[k] = e.target.value; refresh();
  });
});
el("copy").onclick = function() {
  navigator.clipboard.writeText(el("cmd").textContent).then(function() {
    el("copy").classList.add("copied"); el("copy").textContent = "copied";
    setTimeout(function(){ el("copy").classList.remove("copied"); el("copy").textContent = "copy"; }, 1400);
  });
};

fetch("../packages.json").then(function(r) { return r.json(); }).then(function(m) {
  manifest = m; refresh();
}).catch(function(e) {
  el("chips").innerHTML = '<span class="no">failed to load packages.json: ' + e + "</span>";
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
