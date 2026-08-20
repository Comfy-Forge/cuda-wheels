# defaults/ — what the farm builds, farm-wide

Three files, three different kinds of truth. Every package inherits all of
them unless its own folder overrides a piece (`packages/<name>/
pcto_override.yml`, `arch_override.yml`).

## The files

### `python_cuda_torch_os_policy.yml` — the PCTO axes

Which (Python × CUDA × torch × OS) cells exist. Two kinds of content in
one file, clearly separated:

- **Owned policy** (hand-maintained): `platforms`, `python_min`/`max`,
  `supported_cudas`, and the per-package `defaults` block. Changing these
  is a decision; make it in a reviewed commit.
- **Generated rows** (`combinations`, the final section): derived from
  `scraped_torch_matrix.json` by `scripts/derive_defaults.py`. Never
  hand-edit — edit the policy above or refresh the scrape, then re-run.

### `arch_policy.yml` — the arches

Which GPU architectures each CUDA line compiles for (plus the aarch64
table and per-(cuda, torch) exceptions). **Entirely owned**: arch changes
go through the ADR process (CW-ADR-0012), never through automation.
`scripts/generate_matrix.py` reads this file directly at build time — it
is the only arch source in the repo.

### `scraped_torch_matrix.json` — observed upstream reality

Every (CUDA × torch × Python × platform) combination PyTorch actually
publishes on `download.pytorch.org/whl/`, scraped by
`scripts/fetch_torch_matrix.py` (the `torch-matrix.yml` workflow commits
refreshes). Committed so that grid derivation — and therefore every build
— is a function of the git SHA, never of upstream's availability at build
time.

## Why this split

The three files have **different writers and different change control**:
the scrape is written by a machine observing upstream; the PCTO policy is
written by a human deciding scope; the arch policy is written by a human
through an ADR. Keeping them separate means a diff's meaning is visible
from its filename — a scrape refresh can never smuggle in a policy
change, and a policy change can never be mistaken for upstream drift.

Two design rules follow:

1. **PyTorch's output is an input, never the policy.** The grid widens
   automatically when upstream ships a new torch or CUDA line (via the
   scrape → derivation), but *what we compile for it* — arches, python
   bounds, platforms — only changes when a human says so.
2. **Absence is meaningful.** A CUDA line missing from `supported_cudas`
   or from the aarch64 arch table is deliberately not built (the
   derivation warns loudly rather than guessing).
