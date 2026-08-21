# nvdiffrast — build notes
- **Source:** NVlabs/nvdiffrast (v0.4.0)
- **Quirks:** No patches.

## Curated Requires-Dist

The opposite fix: upstream under-declares. Our wheel is JIT-only
(compiles plugins at first use) and genuinely needs ninja at runtime;
`requires_dist` adds it alongside numpy.
