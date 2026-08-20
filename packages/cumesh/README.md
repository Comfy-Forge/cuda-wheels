# cumesh — build notes
- **Source:** JeffreyXiang/CuMesh
- **Quirks:** Needs `curand_dev cufft_dev nvtx` CUDA components. `max_jobs: 2`. Patch fixes CUB API for CUDA 12.9+ on MSVC.

**CUDA-version-specific behavior:** Patch handles `CUDART_VERSION >= 12090` for CUB's DeviceReduce API change.
