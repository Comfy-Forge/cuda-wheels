# cuda-wheels

Prebuilt CUDA Python wheels for ML/3D packages that are painful to compile
from source. Every package has a rolling GitHub Release holding its wheels,
and a static PEP 503 index on gh-pages makes them pip installable.

If you are a person looking for a wheel, use
[Find your wheel](https://comfy-forge.github.io/cuda-wheels/find/).
Pick your OS, CUDA, PyTorch and Python, tick the packages you want, copy
the command. It also tells you whether PyTorch upstream ships that combo
at all.

If you are a tool (pip, uv, comfy-env), resolve against the
[package index](https://comfy-forge.github.io/cuda-wheels/). Per-combo
channels live at `/cu<ver>/torch<M.m>/`, for example:

```bash
pip install flash-attn --extra-index-url https://comfy-forge.github.io/cuda-wheels/cu128/torch2.8/
```

The flat root index mixes every combo, so either use a channel URL or pin
fully (`pkg==1.2.3+cu128torch2.8`).

Reference pages, same site:
[what PyTorch upstream ships](https://comfy-forge.github.io/cuda-wheels/matrix/)
(the farm builds exactly those cells) and
[GPU architectures](https://comfy-forge.github.io/cuda-wheels/archs/)
(which SM targets the wheels carry and why).

## What is built

One folder per package under [`packages/`](packages/). The folder list is
the package list, there is no table here to go stale. Each folder holds
`package.yml`, optional `pcto_override.yml` and `arch_override.yml` (each
explained by that folder's `README.md`), and `patches/`.

Farm-wide policy lives in [`defaults/`](defaults/), see
[`defaults/README.md`](defaults/README.md) for how the three files split.

## Docs

Architecture, build process and ADRs are in the
[comfy-forge docs](https://pozzettiandrea.github.io/comfy-forge-docs/),
cuda-wheels section.
