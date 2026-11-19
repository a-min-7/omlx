# SPDX-License-Identifier: Apache-2.0
"""Prefill Accelerator: a helper Mac runs the first layers of each prompt chunk.

This Mac keeps the whole model and does all decoding. For each prefill chunk
the helper runs layers ``[0, k)`` and sends back the tensors that cross the cut
plus the chunk's cache state for those layers; this Mac runs ``[k, n)`` and then
applies that state, so every cache ends the chunk at the same offset.
"""

import json
import os
import subprocess
from pathlib import Path

# The shape version of the memory plan the two Macs exchange. The protocol
# version and a family's contract say whether a helper can serve a model at
# all; this says whether both sides read ``plan()``'s answer the same way.
# Bump it when ``shared``/``per_layer`` change meaning, not on every edit.
PLAN_API_VERSION = 1


def _built_commit() -> str | None:
    """A commit the packaging wrote into ``omlx/_version.py``, if any."""
    try:
        from omlx import _version

        return getattr(_version, "SOURCE_COMMIT", None)
    except Exception:
        return None


def _installed_commit() -> str | None:
    """The VCS commit recorded in the wheel's ``direct_url.json``, if any."""
    try:
        from importlib.metadata import distribution

        direct = distribution("omlx").read_text("direct_url.json")
        return (json.loads(direct or "{}").get("vcs_info") or {}).get("commit_id")
    except Exception:
        return None


def _git_commit() -> str | None:
    """HEAD of the checkout this package runs from, if it is one.

    A source checkout *is* the build the process runs, so its HEAD is the
    source — the launchd job sets no ``OMLX_COMMIT``. An installed wheel has
    no ``.git`` and falls through to "unknown".
    """
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        return (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _resolve_commit() -> str:
    """The source this build came from, as far as it can be established.

    Order: the packaging's environment variable, a commit the build wrote
    into ``_version.py``, an installed wheel's ``direct_url.json``, then the
    checkout's HEAD. "unknown" is the last resort, not the default: an
    unconditional "unknown" made ``source_mismatch`` blind to the one skew
    this fleet actually has.
    """
    for value in (
        os.environ.get("OMLX_COMMIT"),
        _built_commit(),
        _installed_commit(),
        _git_commit(),
    ):
        if value:
            return str(value)
    return "unknown"


_COMMIT: str | None = None


def commit() -> str:
    """The cached answer; resolved on first use, not at import."""
    global _COMMIT
    if _COMMIT is None:
        _COMMIT = _resolve_commit()
    return _COMMIT
