"""Fix known bugs in an installed CUDA toolkit's own headers.

Runs once per job from the setup-cuda action, for EVERY package: these are
NVIDIA's bugs, not any package's, so any package whose sources reach the
affected header hits them. Each fix is content-gated and idempotent, so it
is a no-op on toolkit versions that never had the bug or already ship the
upstream fix -- no version lists to maintain here.

Reads CUDA_HOME (or CUDA_PATH). Prints one line per applied fix; exits 0
when nothing matches.
"""
import os
import sys
from pathlib import Path


def fix_clusterlaunchcontrol_llp64(cuda_home: Path) -> str | None:
    """CCCL 2.8 (CUDA 12.9) binds 64-bit "l" asm constraints to long2 members.

    `long` is 32 bits under Windows' LLP64 model, so every Windows TU that
    pulls <cuda/ptx> -- which CUB drags in whenever the arch list includes
    sm_100/sm_120 -- dies with "asm operand type size(4) does not match
    type/size implied by constraint 'l'". Linux/ARM are LP64 and compile it
    fine. Upstream CCCL 3.x fixed this by casting to longlong2; apply the
    same one-token change to the installed copy.
    """
    h = (cuda_home / "include/cuda/__ptx/instructions/generated"
         / "clusterlaunchcontrol.h")
    if not h.exists():
        return None
    text = h.read_text(encoding="utf-8", errors="surrogateescape")
    if "reinterpret_cast<long2*>" not in text:
        return None
    h.write_text(text.replace("reinterpret_cast<long2*>",
                              "reinterpret_cast<longlong2*>"),
                 encoding="utf-8", errors="surrogateescape")
    return (f"clusterlaunchcontrol.h: long2 -> longlong2 "
            f"(CCCL LLP64 asm-constraint bug) [{h}]")


FIXES = [fix_clusterlaunchcontrol_llp64]


def main() -> int:
    raw = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not raw:
        print("patch_cuda_toolkit: no CUDA_HOME/CUDA_PATH -- nothing to do")
        return 0
    cuda_home = Path(raw)
    if not cuda_home.is_dir():
        print(f"patch_cuda_toolkit: {cuda_home} is not a directory -- skipping")
        return 0
    applied = 0
    for fix in FIXES:
        try:
            note = fix(cuda_home)
        except OSError as exc:
            # A read-only or partially installed toolkit must not fail the
            # build here: the compile itself is the real gate.
            print(f"patch_cuda_toolkit: {fix.__name__} could not apply: {exc}")
            continue
        if note:
            print(f"patch_cuda_toolkit: {note}")
            applied += 1
    if not applied:
        print("patch_cuda_toolkit: no known-bad headers in this toolkit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
