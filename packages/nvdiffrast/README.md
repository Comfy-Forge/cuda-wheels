# nvdiffrast — build notes
- **Source:** NVlabs/nvdiffrast (v0.4.0)
- **Quirks:** No patches.

## Curated Requires-Dist

Upstream under-declares, so `requires_dist` restates the real set: just
`numpy`.

**Corrected 2026-08-24.** This entry previously read "our wheel is JIT-only
(compiles plugins at first use) and genuinely needs ninja at runtime", and
`requires_dist` carried `ninja` on that basis. It was wrong. That description
fits nvdiffrast v0.3.x; the pin is **v0.4.0**, which replaced `_get_plugin()`
with a real `CUDAExtension`. Six published wheels (cu124-cu132, all three
platforms) were inspected: each ships a 58-66MB compiled `_nvdiffrast_c`
extension, the wheel holds exactly three `.py` files importing only
`{importlib.metadata, numpy, torch, warnings, _nvdiffrast_c}`, there is no
reference anywhere to `cpp_extension`, `load(`, `ninja` or `_get_plugin`, and
the `.cu` sources a JIT path would compile are not in the wheel at all. The
JIT path is impossible, not merely unused.

`torch` is a genuine runtime dependency but is stripped farm-wide by policy
(the consumer pins the torch family workspace-wide, by version and index,
before a cuda-wheel is selected) — see ADR-0004.

A `verify.allow_pure_python: true` knob was removed in the same correction.
It was inert (C3 consults it only when a wheel has no extensions, and this
one always does) but it would have silently swallowed the exact regression
C3 exists to catch: a CUDA compile quietly degrading to a pure-python wheel.
