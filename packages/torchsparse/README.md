# torchsparse — build notes

Builds with the plain farm defaults. It used to carry a
`pcto_override.yml` with `min_pytorch: "2.4.0"`, but the grid's oldest
torch row is 2.4.1, so a 2.4.0 floor excluded nothing — the override was
a no-op and was deleted. If the farm ever adds torch rows older than 2.4,
re-check whether torchsparse actually builds there before assuming it does.

## OPEN: does not compile against torch 2.11 (2026-08-21)

The aarch64 build at torch 2.11 fails in C++ with `no suitable conversion
function from "const at::DeprecatedTypeProperties..."` -- a torch API
change somewhere after 2.8 (the x86 cell at 2.8 builds). Needs either an
API-compat patch or a documented torch ceiling for this package.
