# packages/

One folder per package. The folder is the complete unit:

```text
packages/<name>/
  package.yml         source repo + tag, build knobs, links_torch (REQUIRED)
  pcto_override.yml   optional: build_matrix / min_pytorch
  arch_override.yml   optional: arch_list / arch_list_by_cuda
  patches/*.py        optional: pre-build source patches (idempotent Python)
  README.md           quirks; REQUIRED if any override file exists
```

`scripts/package_loader.py` is the only reader — it merges the folder into
one flat config dict and hard-errors on an undeclared `links_torch` or an
unexplained override. Packages with no overrides inherit everything from
[`../defaults/`](../defaults/README.md).

Field reference and how-to:
[the build-process docs](https://pozzettiandrea.github.io/comfy-forge-docs/cuda-wheels/build-process/#how-do-i-add-a-package)
(kept there, not here, so there is one copy). Live examples: `flash_attn/`
(plain + arch override), `natten/` (all files), `llama_cpp_python/`
(torch-free), `ovoxel/` (curated `requires_dist` with sibling pins).
