# SPDX-License-Identifier: Apache-2.0
"""The planning surface must import without a GPU library.

``families/__init__`` documents that a family's ``__init__`` needs no GPU
library and that ``planner()`` returns the family's GPU-free part, and the
daemon runs ``planner(model_type).plan(...)`` in-process. An ``mlx`` import
reached from there therefore takes session start down on a box whose MLX is
missing or broken. The rest of this suite runs where MLX *is* installed, so a
top-level ``import mlx.core`` is invisible to it: this file is the only test
that can see one.

Each probe blocks ``mlx`` with a meta-path finder that raises
``ModuleNotFoundError`` — what CPython raises once every finder has failed —
rather than the ``sys.modules["mlx"] = None`` trick, which models a different
failure. Every probe carries a positive control, so one whose block silently
stopped working fails instead of passing vacuously.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

BLOCKER = '''
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, Block())


def assert_blocked():
    import importlib

    try:
        importlib.import_module("mlx.core")
    except ModuleNotFoundError:
        print("control: mlx blocked")
        return
    print("control: FAILED, mlx imported while blocked")
    raise SystemExit(3)
'''

PLANNER_PROBE = BLOCKER + '''
assert_blocked()

import importlib

families = importlib.import_module("omlx_cuda_helper.families")
print("families: imported")

for model_type in ("qwen3_5", "qwen3_5_moe"):
    module = families.planner(model_type)
    print(f"planner({model_type}): {module.__name__}")
'''

OQH_PROBE = BLOCKER + '''
assert_blocked()

import importlib

try:
    importlib.import_module("omlx_cuda_helper.oqh")
except ModuleNotFoundError as exc:
    print("oqh: blocked by design:", exc)
else:
    print("oqh: FAILED, imported while mlx was blocked")
    raise SystemExit(4)
'''


def _run(tmp_path: Path, name: str, source: str) -> subprocess.CompletedProcess:
    script = tmp_path / f"{name}.py"
    script.write_text(source)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )


def test_the_blocker_blocks_a_module_that_needs_mlx(tmp_path: Path) -> None:
    """The positive control for the probe below: oqh legitimately needs mlx."""
    result = _run(tmp_path, "oqh", OQH_PROBE)

    assert "control: mlx blocked" in result.stdout, result.stdout + result.stderr
    assert "oqh: blocked by design" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_planner_surface_imports_without_mlx(tmp_path: Path) -> None:
    result = _run(tmp_path, "planner", PLANNER_PROBE)

    assert "control: mlx blocked" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "families: imported" in result.stdout
    assert "planner(qwen3_5): omlx_cuda_helper.families.qwen3_5" in result.stdout
    assert "planner(qwen3_5_moe): omlx_cuda_helper.families.qwen3_5_moe" in result.stdout
