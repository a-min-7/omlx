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
# Armed, but the model has not been loaded, so no manager has tried yet. A
# model nobody asked to run is not an unreachable Mac.
NOT_LOADED = "not_loaded"
# The helper Mac answered; these are its own states, and each names itself.
# "unreachable" is reserved for a connection that actually failed.
NEEDS_FILES = "needs_files"
COPYING = "copying"
STARTING = "starting"
STOPPED = "stopped"
FILES_READY = "files_ready"
FILES_MISSING = "files_missing"

# Every non-serving state where the Mac is not an unanswered one. The last
# two come from the FilePrep surface, which runs before a model loads and has
# no peer record at all — without them it fell through to "unreachable".
STATE_REASONS = {
    "off": STARTING,
    "connecting": STARTING,
    "starting": STARTING,
    "needs_files": NEEDS_FILES,
    "copying": COPYING,
    "stopped": STOPPED,
    "files_ready": FILES_READY,
    "files_missing": FILES_MISSING,
}

# The two arming keys, in the order the settings module declares them.
ARM_KEYS = ("prefill_helper_node_id", "prefill_helper_layers")

# Per model: {"path": {...}, "reason": {...}}. A process-wide total in a
# per-model row would report one model's traffic under another's name.
_COUNTERS: dict[str, dict[str, dict[str, int]]] = {}
_LOCK = threading.Lock()


def _key(model: Any) -> str:
    return "" if model is None else str(model)


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


def gate(
    held: int,
    info: dict[str, Any],
    measured_at: float,
    *,
    budgets: list[int],
    max_layers: int,
) -> dict[str, Any]:
    """The memory figures, dated and labelled as a capacity, not a target.

    #84: ``layers_fit`` is how many layers would still fit; arming to it
    measured ~40 % slower than a moderate split (k=24 of 39), so it is a
    capacity and never a recommendation. The numbers are one moment's reading
    (#86 e): co-tenants moved ``budget`` from 98.3 to 21.7 GiB in an
    afternoon.

    ``max_layers_fit`` is the same fit with the family's split limit removed:
    it exceeds ``layers_fit`` only when that limit binds, and it is not a
    second name for the same number. ``inputs.available_bytes`` is the sum of
    the *clamped* budgets ``fit_layers`` used, so the inputs reproduce the
    output (an unclamped DGX Spark budget does not).

    ``recommended_layers`` stays ``None`` until a stored measurement with
    provenance exists; no code may invent one.
    """
    return {
        "kind": "capacity",
        "layers_fit": held,
        "max_layers_fit": int(max_layers),
        "budget": sum(budgets),
        "measured_at": measured_at,
        "inputs": {
            "available_bytes": sum(budgets),
            "memory_bytes": info.get("memory"),
            "source": "helper-info",
        },
        "recommended_layers": None,
    }


def note(path: str, reason: str | None = None, *, model: Any = None) -> None:
    """Count one request (or one load) that went this way, for one model."""
    with _LOCK:
        counts = _COUNTERS.setdefault(_key(model), {"path": {}, "reason": {}})
        paths = counts["path"]
        paths[path] = paths.get(path, 0) + 1
        if reason is not None:
            reasons = counts["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1


def snapshot(model: Any = None) -> dict[str, dict[str, int]]:
    """One model's counts since the server started (all models if omitted)."""
    with _LOCK:
        counts = _COUNTERS.get(_key(model)) or {"path": {}, "reason": {}}
        return {"path": dict(counts["path"]), "reason": dict(counts["reason"])}


def reset() -> None:
    """The tests' isolation hook."""
    with _LOCK:
        _COUNTERS.clear()
