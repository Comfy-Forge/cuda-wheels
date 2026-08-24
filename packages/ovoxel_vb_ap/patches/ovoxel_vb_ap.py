"""Patch o_voxel for wheel building as o_voxel_vb_ap:
1. Rename package from o_voxel to o_voxel_vb_ap
2. Fix MSVC compatibility (size_t narrowing)
3. Fix GCC-only CXX_FLAGS for Windows MSVC builds

Note: postprocess.py and rasterize.py (nvdiffrast deps) are already
removed in the PozzettiAndrea/Trellis.2.drtk source.
"""
# ── Eigen via release tarball, not the gitlab submodule ─────────────────
# gitlab.com rate-limits GitHub runners (HTTP 403 on submodule clone), so
# clone_recursive is OFF for this package and Eigen -- a header-only dep,
# the repo's only submodule -- is fetched as a tarball (dpvo's proven
# approach) into the include path setup.py expects.
import shutil as _sh
import subprocess as _sp
from pathlib import Path as _P

_eigen_dst = _P("o-voxel/third_party/eigen")
if not (_eigen_dst / "Eigen").exists():
    # NOTE: no --retry-all-errors -- AlmaLinux 8 ships curl 7.61, which
    # predates that flag (7.71). -f makes HTTP errors fail loudly.
    _sp.run(["curl", "-sfL", "--retry", "5",
             "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
             "-o", "_eigen.tar.gz"], check=True)
    _sp.run(["tar", "-xzf", "_eigen.tar.gz"], check=True)
    if _eigen_dst.exists():
        _sh.rmtree(_eigen_dst)
    _eigen_dst.parent.mkdir(parents=True, exist_ok=True)
    _sh.move("eigen-3.4.0", str(_eigen_dst))
    _P("_eigen.tar.gz").unlink()
    print("Eigen 3.4.0 headers installed at", _eigen_dst)
# ─────────────────────────────────────────────────────────────────────────


import re
import os
from pathlib import Path

# --- 1. Rename package to o_voxel_vb_ap ---

# pyproject.toml
pyproject = Path("o-voxel/pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "o_voxel"', 'name = "o_voxel_vb_ap"')
pyproject.write_text(content)
print("Renamed package to o_voxel_vb_ap in pyproject.toml")

# setup.py
setup_file = Path("o-voxel/setup.py")
content = setup_file.read_text()
content = content.replace('name="o_voxel"', 'name="o_voxel_vb_ap"')
content = content.replace("name=\"o_voxel._C\"", "name=\"o_voxel_vb_ap._C\"")
content = content.replace("'o_voxel'", "'o_voxel_vb_ap'")
content = content.replace("'o_voxel.convert'", "'o_voxel_vb_ap.convert'")
content = content.replace("'o_voxel.io'", "'o_voxel_vb_ap.io'")
setup_file.write_text(content)
print("Renamed package to o_voxel_vb_ap in setup.py")

# Rename the actual package directory
src_dir = Path("o-voxel/o_voxel")
dst_dir = Path("o-voxel/o_voxel_vb_ap")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed o_voxel/ directory to o_voxel_vb_ap/")

# Update internal imports
for py_file in dst_dir.rglob("*.py"):
    content = py_file.read_text()
    if "from o_voxel" in content or "import o_voxel" in content:
        content = content.replace("from o_voxel", "from o_voxel_vb_ap")
        content = content.replace("import o_voxel", "import o_voxel_vb_ap")
        py_file.write_text(content)
print("Updated internal imports to o_voxel_vb_ap")

# --- 2. Fix size_t narrowing for MSVC ---
for f in ["o-voxel/src/io/filter_neighbor.cpp", "o-voxel/src/io/filter_parent.cpp"]:
    fpath = Path(f)
    if fpath.exists():
        content = fpath.read_text()
        content = re.sub(r'torch::zeros\(\{(\w+),\s*(\w+)\}', r'torch::zeros({(int64_t)\1, (int64_t)\2}', content)
        fpath.write_text(content)
print("Fixed size_t narrowing in filter_*.cpp")

svo_file = Path("o-voxel/src/io/svo.cpp")
if svo_file.exists():
    content = svo_file.read_text()
    content = re.sub(r'\{(\w+)\.size\(\)\}', r'{(int64_t)\1.size()}', content)
    svo_file.write_text(content)
    print("Fixed size_t narrowing in svo.cpp")

# --- 3. Fix GCC-only CXX_FLAGS for MSVC ---
content = setup_file.read_text()
# Let torch pick the C++ standard, and speak MSVC on Windows.
# (Review board 2026-08-24.) The old exact-string rewrite pinned /std:c++17,
# which OVERRIDES torch's own choice: cpp_extension only appends a standard
# when the caller supplied none, so pinning c++17 broke every torch >= 2.13
# cell (its headers need C++20: C7555 designated initializers, C7582
# bit-field NSDMIs). On the vb fork the string never matched at all and the
# patch silently no-opped, leaving GCC flags for cl.exe. Both failure modes
# disappear once we stop having an opinion about the standard.
import os as _os
import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import strip_std_flags, translate_cxx_flags_for_msvc, require

content, _n_std = strip_std_flags(content)
_n_msvc = 0
if _os.name == "nt":
    content, _n_msvc = translate_cxx_flags_for_msvc(content)
    require(_n_msvc > 0,
            "no GCC cxx flags translated for MSVC in setup.py -- the "
            "extra_compile_args block moved; refusing to build with flags "
            "cl.exe would silently ignore")
require(_n_std > 0,
        "no hardcoded C++-standard flag found in setup.py -- upstream "
        "changed; refusing to build against an unverified flag set")
setup_file.write_text(content)
print(f"Patched setup.py: dropped {_n_std} hardcoded std flag(s), "
      f"translated {_n_msvc} flag(s) for MSVC")
