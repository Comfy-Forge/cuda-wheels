"""Patch torchsparse v2.0.0 for modern torch (>= 2.8).

Upstream passes `tensor.type()` (a DeprecatedTypeProperties) as the first
argument of AT_DISPATCH_* macros; torch >= 2.8 removed the implicit
conversion to c10::ScalarType, failing with "no suitable conversion
function from 'const at::DeprecatedTypeProperties' to 'c10::ScalarType'".
The modern spelling is `tensor.scalar_type()`.

Only dispatch-argument positions are rewritten (`.type(), "` -- the macro
always follows the arg with the op-name string literal), so other
DeprecatedTypeProperties uses (`.type().is_cuda()` etc.) are untouched.
"""
import re
from pathlib import Path

pattern = re.compile(r"\.type\(\)(\s*,\s*\")")

total = 0
for f in sorted(Path("torchsparse/backend").rglob("*")):
    if f.suffix not in (".cu", ".cpp", ".cc", ".cuh", ".h"):
        continue
    text = f.read_text()
    new_text, n = pattern.subn(r".scalar_type()\1", text)
    if n:
        f.write_text(new_text)
        print(f"  {f}: {n} dispatch arg(s) .type() -> .scalar_type()")
        total += n

if total == 0:
    raise SystemExit(
        "torchsparse patch: no '.type(), \"' dispatch args found -- "
        "upstream changed; update this patch")
print(f"torchsparse patch: rewrote {total} dispatch argument(s)")
