# SPDX-License-Identifier: Apache-2.0
"""Admin API for the Prefill Accelerator, on both Macs.

This Mac: candidate helpers for a model, the model's helper status and the
copy / retry / stop actions, also while files are copied before the model
loads. The helper Mac: what it is lending and a Stop.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..cluster.registry import get_device_registry
from ..oqh import format as oqh_format
from . import health, launcher, peers
from .adapters import adapter_for_model
from .lending import active_lending, forget_host, recorded_host
from .manager import (
    computer_name,
    file_prep,
    fit_layers,
    helper_budget,
    prepare,
    this_host,
    transport_for,
)

router = APIRouter(prefix="/admin/api/prefill-helper", tags=["prefill-helper"])

_get_engine_pool: Callable[[], Any] = lambda: None  # noqa: E731
# The helper sends up to ~180 MB/s on the tested models: 5 GbE carries that
# with room to spare, 2.5 GbE with little.
_FULL_SPEED_MBPS = 5000


def set_engine_pool_getter(getter: Callable[[], Any]) -> None:
    global _get_engine_pool
    _get_engine_pool = getter


def _entry(model_id: str) -> Any:
    pool = _get_engine_pool()
    entry = pool.get_entry(model_id) if pool is not None else None
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return entry


def _saved_settings(model_id: str) -> Any:
    settings = getattr(_get_engine_pool(), "_settings_manager", None)
    return settings.get_settings(model_id) if settings is not None else None


def _adapter(entry: Any) -> Any:
    return adapter_for_model(
        getattr(entry, "config_model_type", None),
        getattr(entry, "engine_type", None),
        getattr(entry, "model_path", None),
    )


def prepare_unloaded(entry: Any, settings: Any) -> None:
    """Copy the helper's files for a saved choice before the model loads."""
    adapter = _adapter(entry)
    if adapter is None or getattr(entry.engine, "prefill_helper", None) is not None:
        return
    prepare(
        entry.model_path,
        adapter,
        getattr(settings, "prefill_helper_node_id", None),
        getattr(settings, "prefill_helper_layers", None),
    )


def helper_status(entry: Any) -> dict | None:
    """The loaded model's helper, else the copy made before it loads."""
    manager = getattr(entry.engine, "prefill_helper", None)
    if manager is not None:
        return manager.status()
    prep = file_prep(entry.model_path)
    return prep.status() if prep is not None else None


def _candidate(device: dict[str, Any], model_path: str, adapter: Any) -> dict:
    caps = device.get("caps") or {}
    node_id = device["node_id"]
    row = {
        "node_id": node_id,
        "name": device["friendly_name"],
        "chip": caps.get("chip"),
        "ram_gb": caps.get("ram_gb"),
    }
    transport = transport_for(node_id)
    try:
        target = transport.resolve_target(node_id)
        info = transport.remote_info(target)
    except launcher.LaunchError as error:
        return current_peer_row(row, node_id, reason=str(error))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return current_peer_row(row, node_id, reason=str(error))
    # A read-only probe must not write the shared liveness record: this
    # endpoint is polled by the UI, and one failed probe would mark the peer
    # offline for every model whose manager is ready and serving.
    measured_at = time.time()
    shared, per_layer = adapter.memory_plan(model_path)
    limit = adapter.split_limit(len(per_layer))
    budgets = helper_budget(info)
    held = fit_layers(shared, per_layer, budgets, limit)
    lending = info.get("lending")
    if lending:
        lending = {
            "host": lending.get("host"),
            "model": Path(str(lending.get("model") or "")).name,
        }
        if lending == {"host": computer_name(), "model": Path(model_path).name}:
            # The helper already runs for this model.
            lending = None
    max_layers = fit_layers(shared, per_layer, budgets, len(per_layer))
    gate = health.gate(
        held, info, measured_at, budgets=budgets, max_layers=max_layers
    )
    return {
        **row,
        "online": True,
        # What this probe saw; it is not written to the shared peer record.
        "probe": {"at": measured_at, "ok": True, "reason": None},
        "link": target.link,
        "device": info.get("device"),
        "version": info.get("version"),
        "version_ok": transport.incompatible(
            info, adapter, row["name"], oqh_format.checkpoint_version(model_path)
        )
        is None,
        # Distinct from version_ok: the same version can be built from a
        # different commit, and the memory plan has its own shape version.
        "commit": info.get("commit"),
        "plan_api_version": info.get("plan_api_version"),
        "plan_ok": transport.plan_mismatch(info) is None,
        "source_mismatch": transport.source_mismatch(info) is not None,
        "layers_fit": held,
        # The same fit without the family's split limit; not a second name
        # for layers_fit, and not a recommendation either.
        "max_layers_fit": max_layers,
        "budget": sum(budgets),
        "budgets": budgets,
        "measured_at": measured_at,
        "gate": gate,
        "on_battery": bool(info.get("on_battery")),
        "lending": lending,
        "peer": peers.state(node_id).to_dict(),
    }


def current_peer_row(row: dict, node_id: str, *, reason: str) -> dict:
    """A candidate that did not answer, described by the one peer state.

    The failed probe is reported as ``probe``: it dates a sighting, not a
    measurement, so it carries no ``measured_at`` (there is nothing measured)
    and it is never written to the shared record.
    """
    peer = peers.state(node_id)
    return {
        **row,
        "online": False,
        "reason": reason,
        "probe": {"at": time.time(), "ok": False, "reason": reason},
        "peer": peer.to_dict(),
        "backing_off": (
            {
                "attempt": peer.attempt,
                "next_retry_at": peer.next_retry_at,
                "last_error": peer.last_error,
            }
            if peer.attempt and peer.next_retry_at is not None
            else None
        ),
    }


@router.get("/candidates")
async def candidates(model_id: str) -> dict:
    """Paired Macs and whether each can help this model."""
    entry = _entry(model_id)
    adapter = _adapter(entry)
    if adapter is None:
        raise HTTPException(
            status_code=400, detail="This model has no Prefill Accelerator."
        )
    devices = get_device_registry().paired()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max(1, min(len(devices), 4))) as pool:
        rows = await asyncio.gather(
            *[
                loop.run_in_executor(pool, _candidate, d, entry.model_path, adapter)
                for d in devices
            ]
        )
    return {"version": __version__, "devices": rows}


def _check(node_id: str) -> dict:
    """The accelerator checks for one paired Mac, as rows of key, status, detail."""
    registry = get_device_registry()
    device = registry.get(node_id) if registry.is_paired(node_id) else None
    if device is None:
        raise HTTPException(status_code=404, detail="This Mac is not paired.")
    # This Mac is that Mac's host now, not its helper.
    forget_host(node_id)
    rows = []
    transport = transport_for(node_id)
    try:
        target = transport.resolve_target(node_id)
    except (launcher.LaunchError, OSError, subprocess.SubprocessError) as error:
        rows.append({"key": transport.VIA, "status": "fail", "detail": str(error)})
        return {
            "node_id": node_id,
            "name": device["friendly_name"],
            "rows": rows,
            "probe": {"at": time.time(), "ok": False, "reason": str(error)},
        }
    rows.append({"key": transport.VIA, "status": "pass", "detail": target.endpoint})
    fast = (
        target.link in ("thunderbolt", "rdma")
        or (target.speed_mbps or 0) >= _FULL_SPEED_MBPS
    )
    link = {
        "key": "link",
        "status": "pass" if fast else "warn",
        "detail": target.link,
        "speed_mbps": target.speed_mbps,
        "fix": " ".join(target.warnings),
    }
    rows.append(link)
    try:
        info = transport.remote_info(target, host=this_host())
    except (launcher.LaunchError, OSError, subprocess.SubprocessError) as error:
        # The same rule as ``/candidates``: a read-only check reports what it
        # saw and never writes the shared peer record that a serving manager
        # owns.
        rows.append({"key": "version", "status": "fail", "detail": str(error)})
        return {
            "node_id": node_id,
            "name": target.name,
            "rows": rows,
            "probe": {"at": time.time(), "ok": False, "reason": str(error)},
        }
    theirs = info.get("link_mbps")
    if theirs and target.speed_mbps and theirs < target.speed_mbps:
        # A helper daemon reports the speed of its own end; the slower end
        # sets the speed of the link.
        link["speed_mbps"] = theirs
        if theirs < _FULL_SPEED_MBPS:
            link["status"] = "warn"
    problem = transport.incompatible(info, None, target.name)
    version = {
        "key": "version",
        "status": "pass" if problem is None else "fail",
        "detail": info.get("version"),
    }
    if "contracts" in info:
        # A helper daemon has its own version; the model families it can
        # serve decide, not an equal version.
        version.update(families=sorted(info["contracts"]), fix=problem or "")
    else:
        version["expected"] = __version__
    rows.append(version)
    plan_problem = transport.plan_mismatch(info)
    source_problem = transport.source_mismatch(info)
    if info.get("commit") or info.get("plan_api_version") is not None:
        # Its own row: a helper can pass the contract check and still be
        # built from another commit, or plan memory with another shape.
        rows.append(
            {
                "key": "source",
                "status": "warn" if (plan_problem or source_problem) else "pass",
                "detail": info.get("commit") or "unknown",
                "plan_api_version": info.get("plan_api_version"),
                "fix": plan_problem or source_problem or "",
            }
        )
    budgets = helper_budget(info)
    memory = {"key": "memory", "status": "pass", "bytes": sum(budgets)}
    if "budgets" in info:
        memory["gpus"] = len(budgets)
    rows.append(memory)
    rows.append({"key": "disk", "status": "pass", "bytes": info.get("disk_free")})
    if "on_battery" in info:
        rows.append(
            {"key": "power", "status": "warn" if info["on_battery"] else "pass"}
        )
    return {
        "node_id": node_id,
        "name": target.name,
        "rows": rows,
        "probe": {"at": time.time(), "ok": True, "reason": None},
    }


@router.get("/check")
async def check(node_id: str) -> dict:
    """Whether a paired Mac can serve as a prefill helper."""
    return await asyncio.get_running_loop().run_in_executor(None, _check, node_id)


def _copies(node_id: str, delete: str | None = None) -> dict:
    if not get_device_registry().is_paired(node_id):
        raise HTTPException(status_code=404, detail="This Mac is not paired.")
    transport = transport_for(node_id)
    try:
        target = transport.resolve_target(node_id)
        if delete:
            transport.delete_copies(target, [delete])
        return transport.list_copies(target)
    except (launcher.LaunchError, OSError, subprocess.SubprocessError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/copies")
async def copies(node_id: str) -> dict:
    """Model copies this Mac made on a paired Mac, and its free disk."""
    return await asyncio.get_running_loop().run_in_executor(None, _copies, node_id)


@router.post("/copies/delete")
async def delete_copy(node_id: str, name: str) -> dict:
    """Remove one model copy from a paired Mac (never its own model folders)."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _copies, node_id, name
    )


def model_health(settings: Any, status: dict | None) -> dict:
    """Where this model's requests go, and why not (#79).

    Read from the saved arming and the manager's own state, never from a
    stale text: an armed model whose helper is not serving says why
    (``refused`` when the helper answered no, ``unreachable`` otherwise).
    """
    armed = health.arming(settings)
    state = (status or {}).get("state")
    if armed["path"] == health.HELPER and state in ("ready", "working"):
        return {"prefill_path": health.HELPER, "fallback_reason": None, "arming": armed}
    if armed["path"] == health.LOCAL:
        return {
            "prefill_path": health.LOCAL,
            "fallback_reason": armed["reason"],
            "arming": armed,
        }
    if state in (None, "unloaded"):
        # Armed, but this model was never loaded: no manager has tried, so
        # "unreachable" would name a cause nobody observed.
        return {
            "prefill_path": health.LOCAL,
            "fallback_reason": health.NOT_LOADED,
            "arming": armed,
        }
    peer = ((status or {}).get("peer")) or {}
    return {
        "prefill_path": health.LOCAL,
        "fallback_reason": (
            health.REFUSED if peer.get("refused") else health.UNREACHABLE
        ),
        "arming": armed,
    }


@router.get("/models/{model_id}")
async def model_status(model_id: str) -> dict:
    entry = _entry(model_id)
    status = helper_status(entry)
    payload: dict[str, Any] = dict(status) if status else {"state": "unloaded"}
    payload.update(model_health(_saved_settings(model_id), status))
    # This model's counts only: the counters are keyed by model, so another
    # model's traffic cannot appear under this one.
    payload["counters"] = health.snapshot(model_id)
    return payload


@router.post("/models/{model_id}/{action}")
async def model_action(model_id: str, action: str) -> dict:
    if action not in ("check", "copy", "retry", "stop"):
        raise HTTPException(status_code=404, detail=f"Unknown action: {action}")
    entry = _entry(model_id)
    manager = getattr(entry.engine, "prefill_helper", None)
    if manager is not None:
        if action != "check":
            {"copy": manager.copy_files, "retry": manager.retry, "stop": manager.stop}[
                action
            ]()
        return manager.status()
    prep = file_prep(entry.model_path)
    if action == "check":
        # Whether a saved helper already has the files; nothing is copied.
        settings = _saved_settings(model_id)
        node_id = getattr(settings, "prefill_helper_node_id", None)
        if prep is None and node_id and _adapter(entry) is not None:
            prepare(
                entry.model_path,
                _adapter(entry),
                node_id,
                getattr(settings, "prefill_helper_layers", None),
                check_only=True,
            )
        return helper_status(entry) or {"state": "unloaded"}
    if prep is None:
        raise HTTPException(
            status_code=409, detail="Load the model to use the Prefill Accelerator."
        )
    if action == "copy":
        prep.copy_files()
    elif action == "stop":
        prep.cancel()
    else:
        prepare(entry.model_path, prep.adapter, prep.node_id, prep.layers, restart=True)
    return helper_status(entry) or {"state": "unloaded"}


@router.get("/lending")
async def lending() -> dict:
    """On the helper Mac: the running helper and the Mac that set this one up."""
    marker = active_lending()
    host = recorded_host()
    registry = get_device_registry()
    device = (
        registry.get(host["node_id"])
        if host and registry.is_paired(host["node_id"])
        else None
    )
    # A host that is no longer paired here no longer uses this Mac.
    helps = (
        {
            "node_id": host["node_id"],
            "name": device.get("friendly_name") or host["name"],
        }
        if device
        else None
    )
    return {"active": marker is not None, **(marker or {}), "helps": helps}


@router.post("/lending/stop")
async def stop_lending() -> dict:
    marker = active_lending()
    if marker is None:
        return {"active": False}
    # The worker tells its host it was stopped here, so the host stays off.
    os.kill(int(marker["pid"]), signal.SIGTERM)
    return {"active": False, "stopped": True}
