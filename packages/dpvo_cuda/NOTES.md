# dpvo_cuda (dpvo-cuda) — build notes
- **Source:** princeton-vl/DPVO
- **Quirks:** Downloads Eigen 3.4.0 headers. Patch fixes PyTorch 2.0+ API (`.type()` -> `.scalar_type()`), MSVC compound literals, DLL exports.
