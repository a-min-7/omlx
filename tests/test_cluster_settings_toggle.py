# SPDX-License-Identifier: Apache-2.0
"""Persistence contract for the distributed-inference exposure toggle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from omlx.admin import routes
from omlx.settings import GlobalSettings


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


def _run_global_save(settings, payload):
    """Run the real handler against a real GlobalSettings in a temp base_path."""
    original = routes._get_global_settings
    routes._get_global_settings = lambda: settings
    try:
        with patch.object(
            routes,
            "_apply_model_dirs_runtime",
            new=AsyncMock(return_value=(True, "ok")),
        ):
            return asyncio.run(
                routes.update_global_settings(
                    request=routes.GlobalSettingsRequest(**payload),
                    is_admin=True,
                )
            )
    finally:
        routes._get_global_settings = original


def _main_save_payload(settings):
    """The exact body dashboard.js saveGlobalSettings posts, unchanged."""
    return {
        "host": settings.server.host,
        "port": settings.server.port,
        "log_level": settings.server.log_level,
        "sse_keepalive_mode": settings.server.sse_keepalive_mode,
        "burst_decode_mode": settings.server.burst_decode_mode,
        "preserve_mid_system_cache": settings.server.preserve_mid_system_cache,
        "gpu_keep_warm_idle_window": settings.server.gpu_keep_warm_idle_window,
        "distributed_inference_enabled": (
            settings.server.distributed_inference_enabled
        ),
        "max_audio_upload_size": settings.server.max_audio_upload_size,
        "model_dirs": [
            str(d) for d in settings.model.get_model_dirs(settings.base_path)
        ],
        "model_fallback": settings.model.model_fallback,
        "hide_helper_models": settings.model.hide_helper_models,
        "memory_prefill_memory_guard": settings.memory.prefill_memory_guard,
        "memory_guard_tier": settings.memory.memory_guard_tier,
        "memory_guard_custom_ceiling_gb": (
            settings.memory.memory_guard_custom_ceiling_gb
        ),
        "max_concurrent_requests": settings.scheduler.max_concurrent_requests,
        "embedding_batch_size": settings.scheduler.embedding_batch_size,
        "chunked_prefill": settings.scheduler.chunked_prefill,
        "prefill_priority": settings.scheduler.prefill_priority,
        "concurrent_priority": settings.scheduler.concurrent_priority,
        "cache_enabled": settings.cache.enabled,
        # The GET renders the resolved default, and the dashboard sends that.
        "ssd_cache_dir": settings.cache.ssd_cache_dir
        or str(settings.cache.get_ssd_cache_dir(settings.base_path)),
        "ssd_cache_max_size": settings.cache.ssd_cache_max_size,
        "hot_cache_max_size": settings.cache.hot_cache_max_size,
        "initial_cache_blocks": settings.cache.initial_cache_blocks,
        "hot_cache_only": settings.cache.hot_cache_only,
        "hot_cache_write_through": settings.cache.hot_cache_write_through,
        "ane_compile_cache": settings.cache.ane_compile_cache,
        "gdn_snapshot_storage": settings.cache.get_gdn_snapshot_storage(),
        "gdn_ssd_pending_max_size": settings.cache.gdn_ssd_pending_max_size,
        "gdn_sidecar_precision": settings.cache.gdn_sidecar_state_dtype,
        "sampling_max_context_window": settings.sampling.max_context_window,
        "sampling_max_context_window_policy": (
            settings.sampling.max_context_window_policy
        ),
        "sampling_max_tokens": settings.sampling.max_tokens,
        "sampling_temperature": settings.sampling.temperature,
        "sampling_top_p": settings.sampling.top_p,
        "sampling_top_k": settings.sampling.top_k,
        "sampling_repetition_penalty": settings.sampling.repetition_penalty,
        "mcp_config": settings.mcp.config_path,
        "mcp_expose_tools": settings.mcp.expose_tools,
        "usage_history": settings.usage.usage_history,
        "hf_cache_enabled": settings.huggingface.hf_cache_enabled,
        "network_http_proxy": settings.network.http_proxy,
        "network_https_proxy": settings.network.https_proxy,
        "network_no_proxy": settings.network.no_proxy,
        "network_ca_bundle": settings.network.ca_bundle,
        "skip_api_key_verification": settings.auth.skip_api_key_verification,
        "allow_unauthenticated_inference": (
            settings.auth.allow_unauthenticated_inference
        ),
        "idle_timeout_seconds": settings.idle_timeout.idle_timeout_seconds,
    }


def _integration_save_payload(settings):
    """The exact body dashboard.js saveIntegrationSettings posts, unchanged."""
    integrations = settings.integrations
    fields = {
        "integrations_copilot_model": "copilot_model",
        "integrations_codex_model": "codex_model",
        "integrations_opencode_model": "opencode_model",
        "integrations_openclaw_model": "openclaw_model",
        "integrations_hermes_model": "hermes_model",
        "integrations_pi_model": "pi_model",
        "integrations_vibe_model": "vibe_model",
        "integrations_dsh_model": "dsh_model",
        "integrations_openclaw_tools_profile": "openclaw_tools_profile",
        "markitdown_enabled": "markitdown_enabled",
        "markitdown_expose_model": "markitdown_expose_model",
        "markitdown_max_file_size_mb": "markitdown_max_file_size_mb",
        "markitdown_max_files_per_request": "markitdown_max_files_per_request",
        "markitdown_pdf_processing_engine": "markitdown_pdf_processing_engine",
        "web_search_provider": "web_search_provider",
        "web_search_brave_api_key": "web_search_brave_api_key",
        "web_search_searxng_url": "web_search_searxng_url",
        "web_search_ddgs_backends": "web_search_ddgs_backends",
        "web_search_max_results": "web_search_max_results",
        "web_search_content_mode": "web_search_content_mode",
        "web_search_content_truncate": "web_search_content_truncate",
        "web_search_content_max_chars": "web_search_content_max_chars",
        "splash_path": "splash_path",
        "splash_kv_format": "splash_kv_format",
        "splash_activations": "splash_activations",
        "splash_max_memory_gb": "splash_max_memory_gb",
        "splash_ssd_cache_gb": "splash_ssd_cache_gb",
        "splash_ane_prefill": "splash_ane_prefill",
    }
    return {
        request_field: getattr(integrations, attribute)
        for request_field, attribute in fields.items()
    }


def test_the_dashboard_main_save_reports_no_pending_restart(tmp_path):
    """The whole form, unchanged: nothing to restart (F1/F2)."""
    settings = GlobalSettings(base_path=tmp_path)

    result = _run_global_save(settings, _main_save_payload(settings))

    assert result["pending_restart"] == []
    assert result["message"] == "Settings saved successfully."


def test_the_dashboard_integration_save_reports_no_pending_restart(tmp_path):
    """The integration form, unchanged: nothing to restart (F3)."""
    settings = GlobalSettings(base_path=tmp_path)

    result = _run_global_save(settings, _integration_save_payload(settings))

    assert result["pending_restart"] == []


def test_a_changed_restart_only_value_is_still_reported(tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    payload = _main_save_payload(settings)
    payload["distributed_inference_enabled"] = (
        not settings.server.distributed_inference_enabled
    )

    result = _run_global_save(settings, payload)

    assert result["pending_restart"] == ["distributed_inference_enabled"]


def test_a_changed_cache_directory_is_reported(tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    payload = _main_save_payload(settings)
    payload["ssd_cache_dir"] = str(tmp_path / "another-cache")

    result = _run_global_save(settings, payload)

    assert result["pending_restart"] == ["ssd_cache_dir"]


def test_every_request_field_has_a_pre_save_value(tmp_path):
    """The snapshot must know every field a future dashboard can send.

    A field the resolver cannot read falls back to "pending", so a miss is
    silent until an operator sees a false banner; this makes it loud.
    """
    settings = GlobalSettings(base_path=tmp_path)
    missing = [
        field
        for field in routes.GlobalSettingsRequest.model_fields
        if field not in routes._NON_SETTING_FIELDS
        and routes._previous_value(settings, field) is routes._MISSING
    ]
    assert missing == []
