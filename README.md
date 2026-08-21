# cuda-wheels

Pre-built CUDA Python wheels for ML/3D packages that are painful to compile
from source. One rolling GitHub Release per package holds the wheels; a
static PEP 503 index on gh-pages makes them pip-installable.

**[Package Index](https://comfy-forge.github.io/cuda-wheels/)** ·
**[Dashboard](https://comfy-forge.github.io/cuda-wheels/dashboard/)** ·
**[Install Helper](https://comfy-forge.github.io/cuda-wheels/dashboard/install.html)** ·
**[PyTorch CUDA Wheel Matrix](https://comfy-forge.github.io/cuda-wheels/matrix/)**

## What is built

One folder per package under [`packages/`](packages/) — the folder list IS
the package list (no table here to rot). Each folder holds `package.yml`,
optional `pcto_override.yml` / `arch_override.yml` (each explained by the
package's `README.md`), and `patches/`.

Farm-wide policy lives in [`defaults/`](defaults/) — see
[`defaults/README.md`](defaults/README.md) for the three-file split.

## Install

Pick the channel index matching your CUDA and torch, then install normally:

```bash
pip install flash-attn --extra-index-url https://comfy-forge.github.io/cuda-wheels/cu128/torch2.8/
```

The flat root index mixes every combo — select from it with a full pin
(`pkg==1.2.3+cu128torch2.8`), or let the install helper choose.

## Docs

Architecture, build process, ADRs:
[comfy-forge docs](https://pozzettiandrea.github.io/comfy-forge-docs/) →
cuda-wheels section.
