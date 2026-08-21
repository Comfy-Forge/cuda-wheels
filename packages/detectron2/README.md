# detectron2 — build notes
- **Source:** facebookresearch/detectron2 (v0.6)
- **Quirks:** No patches. Standard PyTorch extension build.

## Curated Requires-Dist

Upstream v0.6 accidentally ships `black==21.4b2` (a code formatter,
exact-pinned to a 2021 beta) as a runtime dep, plus the obsolete
`dataclasses` backport. `requires_dist` in package.yml is upstream's
list minus those two, extras dropped.
