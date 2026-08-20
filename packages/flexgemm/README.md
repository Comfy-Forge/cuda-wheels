# flexgemm / flexgemm_ap / flexgemm_vb — build notes
- **Source:** JeffreyXiang/FlexGEMM (main), PozzettiAndrea/FlexGEMM-ap, visualbruno/FlexGEMM
- **Quirks:** Need `cufft_dev nvtx` CUDA components. Triton dependency (platform-specific: triton on Linux, triton-windows on Windows). flexgemm_vb renames to flex_gemm_vb.
