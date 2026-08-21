"""Patch ovoxel for wheel building:
1. Replace git URL dependencies with simple package names in pyproject.toml
2. Add batched BVH queries to avoid GPU timeout (issue #19)
3. Fix MSVC compatibility (double literals, size_t narrowing)
4. Fix GCC-only CXX_FLAGS for Windows MSVC builds
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
from pathlib import Path

# Replace git URL deps with simple package names in pyproject.toml
pyproject = Path("o-voxel/pyproject.toml")
content = pyproject.read_text()
content = re.sub(r'"cumesh\s*@\s*[^"]*"', '"cumesh"', content)
content = re.sub(r'"flex_gemm\s*@\s*[^"]*"', '"flex_gemm"', content)
pyproject.write_text(content)
print("Replaced git URL dependencies with package names in pyproject.toml")

# Patch postprocess.py for batched BVH queries
postprocess = Path("o-voxel/o_voxel/postprocess.py")
content = postprocess.read_text()

batched_func = '''
def _batched_unsigned_distance(bvh, positions, batch_size=500000, return_uvw=False):
    """Batch unsigned_distance queries to avoid GPU kernel timeout.
    See: https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2/issues/19
    """
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)
    import torch
    distances_list, face_id_list, uvw_list = [], [], []
    for i in range(0, N, batch_size):
        d, f, u = bvh.unsigned_distance(positions[i:min(i+batch_size, N)], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)
    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None
    )

'''

content = re.sub(r'(import cumesh\n)', r'\1' + batched_func, content)
content = content.replace(
    '_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)',
    '_, face_id, uvw = _batched_unsigned_distance(bvh, valid_pos, return_uvw=True)'
)
postprocess.write_text(content)
print("Patched postprocess.py for batched BVH queries")

# MSVC compatibility patches
# Fix 1: Remove 'd' suffix from double literals (MSVC doesn't support this)
# Preserve Eigen types like Vector2d by using negative lookbehind
cpp_file = Path("o-voxel/src/convert/flexible_dual_grid.cpp")
content = cpp_file.read_text()
content = re.sub(r'(?<![a-zA-Z_])(\d+\.?\d*(?:[eE][+-]?\d+)?)d\b', r'\1', content)
cpp_file.write_text(content)
print("Fixed double literal suffix in flexible_dual_grid.cpp")

# Fix 2: Cast size_t to int64_t in torch calls (narrowing conversion error on MSVC)
for f in ["o-voxel/src/io/filter_neighbor.cpp", "o-voxel/src/io/filter_parent.cpp"]:
    fpath = Path(f)
    content = fpath.read_text()
    content = re.sub(r'torch::zeros\(\{(\w+),\s*(\w+)\}', r'torch::zeros({(int64_t)\1, (int64_t)\2}', content)
    fpath.write_text(content)
print("Fixed size_t narrowing in filter_*.cpp")

# Fix 3: Cast size_t in svo.cpp
svo_file = Path("o-voxel/src/io/svo.cpp")
content = svo_file.read_text()
content = re.sub(r'\{(\w+)\.size\(\)\}', r'{(int64_t)\1.size()}', content)
svo_file.write_text(content)
print("Fixed size_t narrowing in svo.cpp")

# Fix 4: Replace GCC-only CXX_FLAGS with MSVC equivalents on Windows
setup_file = Path("o-voxel/setup.py")
content = setup_file.read_text()
old_cxx = """            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag,
            }"""
new_cxx = """            extra_compile_args={
                "cxx": ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"] + cc_flag,
            }"""
if old_cxx in content:
    content = content.replace(old_cxx, new_cxx)
    setup_file.write_text(content)
    print("Patched setup.py CXX_FLAGS for MSVC compatibility")
else:
    print("WARNING: Could not find CXX_FLAGS block in setup.py - source may have changed")
