# Package Build Notes

Per-package CUDA constraints, patch summaries, and build quirks.

## CUDA Version Support

All 26 packages support CUDA 12.4 through 13.0 (cu124, cu126, cu128, cu129, cu130).
No packages have an upper CUDA version limit that would block cu129.

### Packages with CUDA-version-specific behavior

| Package | Detail |
|---------|--------|
| flash_attn | setup.py explicitly handles CUDA 12.9+ (SM 101/Thor, family-specific 100f/120f gencode flags) |
| sageattention | Needs CUDA >= 12.0; SM 12.0 (Blackwell) needs >= 12.8. No upper bound. |
| sageattn3 | Blackwell-only (SM 10.0, 12.0). Needs CUDA >= 12.8. Patch replaces runtime GPU detection with TORCH_CUDA_ARCH_LIST parsing. |
| cumesh/cumesh_vb | Patch handles `CUDART_VERSION >= 12090` for CUB's DeviceReduce API change |

(Per-package notes now live in packages/<name>/NOTES.md)
