# torch_spline_conv — build notes
- **Source:** rusty1s/pytorch_spline_conv
- **Quirks:** No patches.

## Upstream PyTorch Phantom Gaps

These CUDA/torch/python/platform combos are in our build matrix but PyTorch never published a wheel:

| Combo | Reason |
|-------|--------|
| cu124/torch2.5/cp313/windows | PyTorch only published cp313 Linux for torch 2.5+cu124 |
| cu129/torch2.10/*/windows | torch 2.10+cu129 is linux-only upstream |
