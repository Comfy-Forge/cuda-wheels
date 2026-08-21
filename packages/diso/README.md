# diso — build notes

Differentiable Iso-Surface Extraction (DiffMC / DiffDMC). Surface extraction
for Hunyuan3D 2.1 + V2 and TripoSG; was the highest-value missing wheel for
ComfyUI-3D-Pack users. Upstream has no tags, so the pin is a commit.

## Why no patch is needed

- setup.py builds a single extension `diso._C` from `src/*.cu` — no
  hardcoded `TORCH_CUDA_ARCH_LIST`, so the farm's list is respected.
- It falls back to CppExtension (CUDA-less!) when
  `torch.cuda.is_available()` is false, which it is on a runner with no
  GPU — but the build action already exports `FORCE_CUDA=1` on both
  platforms, which is exactly the escape hatch its `get_extensions()`
  checks.
- nvcc flags come from `$NVCC_FLAGS`, defaulting to `-O3`.
- The `cxx: ["-O3"]` flag is gcc syntax; MSVC emits warning D9002 and
  ignores it rather than failing, so Windows is left unpatched on purpose.

## Hard arch floor: sm_60 (enforced by arch_override.yml)

diso explicitly instantiates `CUDualMC<double, int>` (src/cudualmc.cu:1144)
and its backward pass calls `atomicAdd` on double accumulators
(cudualmc.cu:738-750, cumc.cu:440-444). CUDA provides
`atomicAdd(double*, double)` only from sm_60 (Pascal) onward, so compiling
for sm_50 fails with "no instance of overloaded function atomicAdd".

The shared policy's cu124/cu126 rows include 5.0 (Maxwell, matching
torch's own wheels), so `arch_override.yml` drops it for this package:
cu124/cu126 build 6.0 and up, cu128+ inherit the policy rows (which
already start at 7.0). The override was briefly deleted while the farm
policy itself excluded Maxwell; it returned when 5.0 did.
