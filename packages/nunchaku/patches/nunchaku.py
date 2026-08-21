"""Strip nunchaku's own +cuXX.YtorchM.m local version tag.

Upstream setup.py unconditionally appends `+cu{cuda}torch{major.minor}`
(dotted CUDA, e.g. +cu12.8torch2.8) to the version. The farm's rename step
then appends ITS tag, producing a double local version
(1.2.1+cu12.8torch2.8+cu128torch2.8) that fails PEP 440 and the verify
gate's filename check. Keep the base version; the farm owns the tag.
"""
from pathlib import Path

setup_py = Path("setup.py")
content = setup_py.read_text()
old = 'version = f"{version}+cu{cuda_version}torch{torch_major_minor_version}"'
if old not in content:
    raise SystemExit("nunchaku patch: version-suffix line not found -- "
                     "upstream setup.py changed; update this patch")
content = content.replace(old, "pass  # farm rename owns the +cuNNNtorchM.m tag")
setup_py.write_text(content)
print("nunchaku patch: upstream local-version suffix stripped")
