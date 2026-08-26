"""Stop torchao installing its `test/` directory as a top-level package.

Every top-level folder in a wheel lands directly in site-packages, which is one
flat namespace shared by the whole environment. Installing torchao currently claims
the name `test` for the entire interpreter, shadowing anything else that owns it
-- and pip lets whichever distribution installs last overwrite the other's
files, then deletes them on uninstall.

Only the top-level tree is excluded. Exclusion patterns match full dotted
names, so a genuine `torchao.test` subpackage would be untouched.

This package previously had no patch script.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from patch_lib import exclude_top_level_packages  # noqa: E402

exclude_top_level_packages(["test"])
