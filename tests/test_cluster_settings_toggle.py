# SPDX-License-Identifier: Apache-2.0
"""Persistence contract for the distributed-inference exposure toggle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from omlx.admin import routes


def test_distributed_opt_in_is_saved_for_the_next_restart():
    settings = MagicMock()
    settings.server = SimpleNamespace(distributed_inference_enabled=False)
    settings.validate.return_value = []
    settings.save.return_value = None
    original = routes._get_global_settings
    routes._get_global_settings = lambda: settings
    try:
        result = asyncio.run(
            routes.update_global_settings(
                request=routes.GlobalSettingsRequest(
                    distributed_inference_enabled=True
                ),
                is_admin=True,
            )
        )
    finally:
        routes._get_global_settings = original

    assert result["success"] is True
    assert "distributed_inference_enabled" not in result["runtime_applied"]
    assert settings.server.distributed_inference_enabled is True
    settings.save.assert_called_once()


def _save(request, settings=None):
    if settings is None:
        settings = MagicMock()
        settings.server = SimpleNamespace(distributed_inference_enabled=False)
    settings.validate.return_value = []
    settings.save.return_value = None
    original = routes._get_global_settings
    routes._get_global_settings = lambda: settings
    try:
        return asyncio.run(
            routes.update_global_settings(request=request, is_admin=True)
        )
    finally:
        routes._get_global_settings = original


def test_a_restart_only_setting_is_reported_as_pending_restart():
    """#87: success stays true, but the save says it is not in effect.

    The toggle is persisted for the next start and never applied at runtime,
    so a bare "saved successfully" let enabling it read as live.
    """
    result = _save(routes.GlobalSettingsRequest(distributed_inference_enabled=True))

    assert result["success"] is True
    assert result["pending_restart"] == ["distributed_inference_enabled"]
    assert "distributed_inference_enabled" in result["message"]


def test_a_runtime_applied_setting_is_not_pending_restart():
    result = _save(routes.GlobalSettingsRequest(mcp_expose_tools=True))

    assert "mcp_expose_tools" in result["runtime_applied"]
    assert result["pending_restart"] == []
    assert result["message"] == "Settings saved successfully."


def test_resending_the_whole_form_does_not_demand_a_restart():
    """The dashboard posts every field; only a changed value is pending."""
    settings = MagicMock()
    settings.server = SimpleNamespace(
        distributed_inference_enabled=False, host="127.0.0.1", port=8000
    )
    settings.cache = SimpleNamespace(
        ssd_cache_dir=None,
        ssd_cache_max_size="10GB",
        hot_cache_only=False,
        hot_cache_write_through=False,
        ane_compile_cache=False,
        gdn_ssd_pending_max_size="2GB",
        gdn_sidecar_state_dtype="fp32",
        initial_cache_blocks=64,
        get_gdn_snapshot_storage=lambda: "auto",
    )
    settings.mcp = SimpleNamespace(config_path=None)
    request = routes.GlobalSettingsRequest(
        host="127.0.0.1",
        port=8000,
        distributed_inference_enabled=True,
        ssd_cache_max_size="10GB",
        hot_cache_only=False,
        hot_cache_write_through=False,
        ane_compile_cache=False,
        gdn_ssd_pending_max_size="2GB",
        gdn_sidecar_precision="fp32",
        initial_cache_blocks=64,
    )

    result = _save(request, settings)

    assert result["pending_restart"] == ["distributed_inference_enabled"]
