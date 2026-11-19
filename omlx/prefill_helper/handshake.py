# SPDX-License-Identifier: Apache-2.0
"""What a helper says about the source it runs and the plan it computes.

``incompatible()`` answers "can this helper serve this family": the data
protocol and the family's contract. It cannot see a source difference — two
builds of the same version answer the same and can still plan memory, or load
a stage, differently. A helper therefore reports the commit it was built from
and the shape version of its memory plan; these two functions judge them.

An absent value is never a mismatch: an older helper does not send them, and
``incompatible()`` still applies to it.
"""

from __future__ import annotations

from . import PLAN_API_VERSION, commit


def plan_mismatch(info: dict, name: str = "the helper") -> str | None:
    theirs = info.get("plan_api_version")
    if theirs is None:
        return None
    if int(theirs) != PLAN_API_VERSION:
        return (
            f"{name} plans memory as version {theirs}; this Mac reads version "
            f"{PLAN_API_VERSION}. Update the older of oMLX and the helper."
        )
    return None


def source_mismatch(
    info: dict, name: str = "the helper", *, ours: str | None = None
) -> str | None:
    # Resolved at call time: an import-time constant read the environment
    # before the packaging could set it and reported "unknown" forever.
    theirs, ours = info.get("commit"), commit() if ours is None else ours
    if not theirs or not ours or "unknown" in (str(theirs), ours):
        return None
    if str(theirs) == ours:
        return None
    return f"{name} was built from {str(theirs)[:12]}, this Mac from {ours[:12]}."
