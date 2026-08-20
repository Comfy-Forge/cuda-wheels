# gsplat_maskgaussian — build notes

## Overrides

This package deviates from the farm defaults (`defaults/`). Each
override file carries the detailed rationale in its comments; summary:

- **`arch_override.yml`** — overrides the GPU arch lists: `arch_list_by_cuda`. Inherit the full combinations matrix from _defaults.yml; restrict to Linux only. Override arch_list_by_cuda for cu12.4/12.6 to drop sm_5.0 and sm_6.0 (Maxwell/Pascal) — gsplat HEAD uses cg::labeled_partition which requires sm_70+. Without this override, cu12.4/12.6 builds fail with "namespace 'cooperative_groups' has no member 'labeled_partition'". cu12.8+ default arch_lists already start at 7.0 / 7.5 so no override is needed there.
