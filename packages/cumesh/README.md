# cumesh — build notes
- **Source:** JeffreyXiang/CuMesh
- **Quirks:** Needs `curand_dev cufft_dev nvtx` CUDA components. Patch fixes CUB API for CUDA 12.9+ on MSVC.

**CUDA-version-specific behavior:** Patch handles `CUDART_VERSION >= 12090` for CUB's DeviceReduce API change.

## Overrides

`pcto_override.yml`: max_cuda 13.0 -- CUDA 13.2's CCCL removed CUB's
classic two-phase DeviceScan API, which cumesh's compress_ids uses.
