# torchsparse — build notes

Builds with the plain farm defaults. It used to carry a
`pcto_override.yml` with `min_pytorch: "2.4.0"`, but the grid's oldest
torch row is 2.4.1, so a 2.4.0 floor excluded nothing — the override was
a no-op and was deleted. If the farm ever adds torch rows older than 2.4,
re-check whether torchsparse actually builds there before assuming it does.
