# cubvh — build notes
- **Source:** ashawkey/cubvh
- **Quirks:** `max_jobs: 2`, recursive clone. Patch (`patches/cubvh.py`): bumps `cpp_standard` 17 -> 20 -- torch 2.13's headers use C++20 features that MSVC/Windows-nvcc hard-error on under c++17.
