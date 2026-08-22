# cumesh — build notes
- **Source:** JeffreyXiang/CuMesh
- **Quirks:** Needs `curand_dev cufft_dev nvtx` CUDA components. Patch fixes CUB API for CUDA 12.9+ on MSVC.

**CUDA-version-specific behavior:** Patch handles `CUDART_VERSION >= 12090` for CUB's DeviceReduce API change.

## CCCL 3.2 (CUDA 13.2)

CCCL 3.2 removed CUB's in-place `ExclusiveSum(temp, bytes, data, N)`
convenience overload; the patch rewrites src/shared.h's two calls to the
explicit five-argument form (in == out), which every CCCL accepts.
