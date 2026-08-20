# diso — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. Differentiable Iso-Surface Extraction (DiffMC / DiffDMC). Surface extraction for Hunyuan3D 2.1 + V2 and TripoSG; the highest-value missing wheel for ComfyUI-3D-Pack-enved. Upstream has no tags, so the pin is a commit. No patch_script needed: * setup.py builds a single extension `diso._C` from src/*.cu -- no hardcoded TORCH_CUDA_ARCH_LIST, so ours is respected. * it falls back to CppExtension (CUDA-less!) when torch.cuda.is_available() is false, which it is on a runner with no GPU -- but the build action already exports FORCE_CUDA=1 on both platforms, which is exactly the
