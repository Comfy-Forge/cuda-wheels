#!/usr/bin/env bash
# Kept OUT of package.yml on purpose: generate_matrix embeds
# pre_build_script verbatim into a GitHub Actions job OUTPUT, and Actions
# REDACTS any output that looks credential-bearing ('Authorization',
# 'Bearer', ...). When 2e47bbc added the auth header inline, the matrix
# output was dropped, every build/link job saw a null matrix and skipped,
# and the run reported success having built nothing (2026-08-24).
set -euo pipefail

cat > /tmp/pick_cumm.py <<'PY'
import json, os, platform, sys, urllib.request
cu = os.environ["CUW_CUDA_VERSION"].replace(".", "")
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
mach = platform.machine().lower()
plat = "win_amd64" if os.name == "nt" else ("aarch64" if "aarch64" in mach or "arm64" in mach else "x86_64")
req = urllib.request.Request(
    "https://api.github.com/repos/Comfy-Forge/cuda-wheels/releases/tags/cumm-latest")
tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
if tok:
    req.add_header("Authorization", f"Bearer {tok}")
rel = json.load(urllib.request.urlopen(req))
print(next(a["browser_download_url"] for a in rel["assets"]
           if f"+cu{cu}torch" in a["name"] and py in a["name"] and plat in a["name"]))
PY
sed -i 's/^  //' /tmp/pick_cumm.py
CUMM_URL=$(python /tmp/pick_cumm.py)
echo "farm cumm: $CUMM_URL"
python -m pip install "$CUMM_URL"
# Farm wheels exclude libcudart by policy (torch preloads it), but
# spconv's setup imports cumm WITHOUT torch -- put the toolkit libs on
# the loader path for the whole job (harmless no-op on Windows, where
# the CUDA bin dir is already on PATH).
if [ -d "$CUDA_HOME/lib64" ]; then
  echo "LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH" >> $GITHUB_ENV
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi
python -c "import cumm.tensorview, cumm.constants; print('cumm OK:', cumm.constants.TENSORVIEW_INCLUDE_PATH)"
# cumm 0.8.2's dtype headers (tf32.h first, float8.h in cascade)
# specialize std::numeric_limits / use std::float_round_style without
# guaranteeing <limits>: cumm's own TU order supplies it first,
# spconv's generated TUs do not ("float_round_style ... explicit type
# is missing" at tf32.h:254, then float8.h cascade, all 10 shards).
# Graft <limits> into every dtype header that touches those names.
python -c "import importlib.util, pathlib, re; d = pathlib.Path(importlib.util.find_spec('cumm').submodule_search_locations[0]) / 'include/tensorview/gemm/dtypes'; [(p.write_text(re.sub(r'(#include <[^>]+>)', r'#include <limits>\n\1', p.read_text(), count=1)), print('grafted <limits> into', p.name)) for p in sorted(d.glob('*.h')) if re.search(r'float_round_style|float_denorm_style|numeric_limits', p.read_text()) and '#include <limits>' not in p.read_text()]; print('dtype headers <limits> ensured')"
