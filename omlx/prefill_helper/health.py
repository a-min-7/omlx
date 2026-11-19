# SPDX-License-Identifier: Apache-2.0
"""Whether a model's requests go to a helper, and why not.

#79: arming is two keys in ``model_settings.json``. When one is lost — a
restore from a backup that predates the arming, a settings migration — every
request simply runs on this Mac. Nothing fails: no error, no state transition,
no counter. The only symptom is slower prompts, and nothing alerts on slower.

This module names the two states and counts the transitions, so a monitor can
assert "this model is armed" and see the number of requests that did not use
the helper it was armed for.

``half_configured`` is layers without a Mac. A Mac without layers is *not*
half configured: ``prefill_helper_layers: null`` is the supported automatic
split (the helper takes as many layers as fit), so flagging it would cry wolf
on a working configuration.
"""

from __future__ import annotations

import threading
from typing import Any

LOCAL = "local"
HELPER = "helper"
UNARMED = "unarmed"
HALF_CONFIGURED = "half_configured"
UNREACHABLE = "unreachable"
REFUSED = "refused"

# The two arming keys, in the order the settings module declares them.
ARM_KEYS = ("prefill_helper_node_id", "prefill_helper_layers")

_COUNTERS: dict[str, dict[str, int]] = {"path": {}, "reason": {}}
_LOCK = threading.Lock()


def arming(settings: Any) -> dict[str, Any]:
    """The saved arming of one model: where its requests go, and why not."""
    node = getattr(settings, "prefill_helper_node_id", None)
    layers = getattr(settings, "prefill_helper_layers", None)
    if node is None and layers is None:
        return {
            "path": LOCAL,
            "reason": UNARMED,
            "node_id": None,
            "layers": None,
            "automatic_layers": False,
        }
    if node is None:
        # Layers with no Mac: the prefill stays here and nothing says so.
        return {
            "path": LOCAL,
            "reason": HALF_CONFIGURED,
            "node_id": None,
            "layers": layers,
            "automatic_layers": False,
        }
    return {
        "path": HELPER,
        "reason": None,
        "node_id": str(node),
        "layers": layers,
        "automatic_layers": layers is None,
    }


def gate(held: int, info: dict[str, Any], measured_at: float) -> dict[str, Any]:
    """The memory figures, dated and labelled as a capacity, not a target.

    #84: ``layers_fit`` is how many layers would still fit; arming to it
    measured ~40 % slower than a moderate split (k=24 of 39). The numbers are
    one moment's reading (#86 e): co-tenants moved ``budget`` from 98.3 to
    21.7 GiB in an afternoon. ``recommended_layers`` stays ``None`` until a
    stored measurement with provenance exists; no code may invent one.
    """
    return {
        "kind": "capacity",
        "layers_fit": held,
        "max_layers_fit": held,
        "budget": info.get("budget"),
        "measured_at": measured_at,
        "inputs": {
            "available_bytes": info.get("budget"),
            "memory_bytes": info.get("memory"),
            "source": "helper-info",
        },
        "recommended_layers": None,
    }


def note(path: str, reason: str | None = None) -> None:
    """Count one request (or one load) that went this way."""
    with _LOCK:
        paths = _COUNTERS["path"]
        paths[path] = paths.get(path, 0) + 1
        if reason is not None:
            reasons = _COUNTERS["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1


def snapshot() -> dict[str, dict[str, int]]:
    """Process-wide counts since the server started."""
    with _LOCK:
        return {"path": dict(_COUNTERS["path"]), "reason": dict(_COUNTERS["reason"])}


def reset() -> None:
    """The tests' isolation hook."""
    with _LOCK:
        _COUNTERS["path"].clear()
        _COUNTERS["reason"].clear()
