"""Patch mmcv 1.7.2 to always build with CUDA ops + keep wheel name `mmcv`.

mmcv 1.7.2's setup.py has two conditional behaviors gated on the env
var MMCV_WITH_OPS:

  1. get_extensions() returns [] when MMCV_WITH_OPS != "1" -> pure-Python
     wheel with no CUDA compile.
  2. setup(name=...) flips between `mmcv` (lite) and `mmcv-full` (with ops)
     based on the same env var.

We want behavior #1 = "always build ops" and behavior #2 = "always keep
the name `mmcv`" (matching the PyPI 1.x convention where the published
`mmcv` package IS the full-with-ops version; `mmcv-lite` is the
explicitly-lite sibling).

cuda-wheels' build action doesn't support per-package env vars (only
patch_script + pre_build_script run in a separate shell), so the
cleanest fix is to set `os.environ['MMCV_WITH_OPS'] = '1'` at the top
of setup.py and replace the conditional name expression with a literal
'mmcv'.

Idempotent via the MARKER string.
"""
from pathlib import Path
import sys

setup_file = Path("setup.py")
if not setup_file.exists():
    print("mmcv patch: setup.py not found in cwd, skipping")
    sys.exit(0)

content = setup_file.read_text()

MARKER = "# CUDA-WHEELS PATCH: force MMCV_WITH_OPS=1, name=mmcv"
if MARKER in content:
    print("mmcv patch: already applied (marker present), skipping")
    sys.exit(0)

# Step 1: inject env var override at the top of setup.py, after the
# initial imports. Doing this at module-import time means the env var
# is set before any os.getenv('MMCV_WITH_OPS', ...) check fires.
PROLOGUE = f"""{MARKER}
import os
os.environ['MMCV_WITH_OPS'] = '1'

"""

# Find a clean anchor near the top of setup.py — after the shebang/
# encoding line, after the initial 'import os' (which IS in mmcv's
# setup.py). Just prepend; setup.py's own `import os` later is a no-op.
content = PROLOGUE + content

# Step 2: replace the conditional name expression. mmcv 1.7.2 has
# something like:
#   name='mmcv' if os.getenv('MMCV_WITH_OPS', '0') == '0' else 'mmcv-full',
#
# We force the literal 'mmcv' name regardless of env. Use a tolerant
# replacement that handles slight whitespace variations.
import re
pattern = re.compile(
    r"name\s*=\s*'mmcv'\s+if\s+os\.getenv\(\s*'MMCV_WITH_OPS'\s*,\s*'0'\s*\)\s*==\s*'0'\s+else\s+'mmcv-full'",
)
new_content, n_subs = pattern.subn("name='mmcv'", content)
if n_subs == 0:
    print(
        "mmcv patch: WARNING — could not find the conditional name= "
        "expression in setup.py. The wheel may be published as 'mmcv-full' "
        "instead of 'mmcv'. Check upstream for refactors."
    )
elif n_subs > 1:
    print(f"mmcv patch: WARNING — replaced {n_subs} matches (expected 1)")
content = new_content

setup_file.write_text(content)
print(
    "mmcv patch: applied MMCV_WITH_OPS=1 prologue + forced name='mmcv'; "
    f"name-conditional substitutions: {n_subs}"
)


# ── py3.13+ and setuptools>=81 compatibility (triage 2026-08-24) ────────
# Two independent metadata-time failures that between them killed all 189
# remaining cells before any compiler ran:
#
# 1. PEP 667 (python 3.13): `exec(...)` inside get_version() writes into a
#    SNAPSHOT of locals(), so `return locals()['__version__']` raises
#    KeyError. Killed every py3.13/3.14/3.15 cell (138).
# 2. setuptools 81 removed pkg_resources, which setup.py imports at module
#    scope. Killed the py<=3.12 cells that happened to resolve a new
#    setuptools (51) -- the farm installs it unpinned, hence the scatter.
import re as _re

_sp = Path("setup.py")
_t = _sp.read_text()

_old_ver = """def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""
_new_ver = """def get_version():
    version_file = 'mmcv/version.py'
    _ns = {}
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'), _ns)
    return _ns['__version__']"""
if _old_ver in _t:
    _t = _t.replace(_old_ver, _new_ver)
    print("mmcv patch: get_version() execs into an explicit namespace (PEP 667)")
elif "_ns['__version__']" in _t:
    print("mmcv patch: get_version() already namespaced")
else:
    raise SystemExit(
        "PATCH FAILED: mmcv get_version() does not match the expected "
        "1.7.2 text -- every py3.13+ cell would die at metadata generation "
        "with KeyError: '__version__' (PEP 667)")

_old_pr = ("from pkg_resources import DistributionNotFound, get_distribution, "
           "parse_version")
_new_pr = """try:  # setuptools >= 81 removed pkg_resources
    from pkg_resources import DistributionNotFound, get_distribution, parse_version
except ModuleNotFoundError:  # farm shim
    from importlib.metadata import PackageNotFoundError as DistributionNotFound
    from importlib.metadata import distribution as _distribution
    from packaging.version import parse as parse_version

    class _Dist:
        def __init__(self, d):
            self.version = d.version

    def get_distribution(name):
        return _Dist(_distribution(name))"""
if _old_pr in _t:
    _t = _t.replace(_old_pr, _new_pr)
    print("mmcv patch: pkg_resources -> importlib.metadata shim")
else:
    print("mmcv patch: pkg_resources import not found (already shimmed?)")

_sp.write_text(_t)
