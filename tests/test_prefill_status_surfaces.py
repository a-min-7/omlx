# SPDX-License-Identifier: Apache-2.0
"""The four helper-status surfaces return one shape (#91).

The PR-A fields landed on the model-status route only, so an operator (or an
MCP) reading an action response or the admin model list saw a shape without
``prefill_path``/``fallback_reason`` and concluded the model was fine — which
is the invisibility #79 exists to end. This test enumerates all four and
asserts one serializer serves them.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import omlx.server  # noqa: F401 - import order, as the other admin tests do
from omlx.admin import routes as admin_routes
from omlx.model_settings import ModelSettings
from omlx.prefill_helper import routes as helper_routes

READY = {
    "state": "ready",
    "node_id": "n",
    "held": 24,
    "peer": {"reachable": True, "refused": False, "serving": True},
    "backing_off": None,
    "last_prompt": {
        "tokens": 100,
        "seconds": 1.0,
        "session_prefill_seconds": 1.0,
        "helper_seconds": 0.8,
        "transfer_seconds": 0.1,
        "layers": 24,
        "at": 1.0,
    },
    "splits": {"24": {"warm": True, "requests_since_load": 2, "last_at": 1.0}},
}

# The fields #79/#86 added; every surface must carry them.
REQUIRED = {
    "state",
    "node_id",
    "held",
    "peer",
    "backing_off",
    "last_prompt",
    "splits",
    "prefill_path",
    "fallback_reason",
    "arming",
    "counters",
}

SETTINGS = ModelSettings(prefill_helper_node_id="n", prefill_helper_layers=24)


class _Manager:
    def status(self) -> dict:
        return dict(READY)

    def copy_files(self) -> None:
        pass

    def retry(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def rearm(self) -> None:
        pass


def _pool() -> MagicMock:
    entry = SimpleNamespace(
        model_id="loaded",
        model_path="/models/loaded",
        engine=SimpleNamespace(prefill_helper=_Manager()),
        engine_type="vlm",
        config_model_type="qwen3_5",
    )
    pool = MagicMock()
    pool.get_entry.side_effect = lambda model_id: (
        entry if model_id == "loaded" else None
    )
    pool.get_status.return_value = {
        "models": [
            {
                "id": "loaded",
                "loaded": True,
                "estimated_size": 1000,
                "pinned": False,
                "engine_type": "vlm",
                "model_type": "llm",
            }
        ]
    }
    pool.splash_target.return_value = (None, "")
    pool._settings_manager = SimpleNamespace(get_settings=lambda model_id: SETTINGS)
    return pool


def _helper_client(pool: MagicMock) -> TestClient:
    helper_routes.set_engine_pool_getter(lambda: pool)
    app = FastAPI()
    app.include_router(helper_routes.router)
    return TestClient(app)


def _admin_status(pool: MagicMock) -> dict:
    settings_manager = MagicMock()
    settings_manager.get_all_settings.return_value = {"loaded": SETTINGS}
    settings_manager.get_settings.return_value = SETTINGS
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch.object(
            admin_routes, "_get_settings_manager", return_value=settings_manager
        ),
        patch.object(admin_routes, "_get_server_state", return_value=MagicMock()),
    ):
        listed = asyncio.run(admin_routes.list_models(is_admin=True))
    return listed["models"][0]["prefill_helper"]


def test_the_four_status_surfaces_return_one_shape():
    pool = _pool()
    client = _helper_client(pool)

    status = client.get("/admin/api/prefill-helper/models/loaded").json()
    check = client.post("/admin/api/prefill-helper/models/loaded/check").json()
    action = client.post("/admin/api/prefill-helper/models/loaded/copy").json()
    admin = _admin_status(pool)

    surfaces = {
        "GET /models/{id}": status,
        "POST /models/{id}/check": check,
        "POST /models/{id}/{action}": action,
        "GET /admin/api/models": admin,
    }
    for name, payload in surfaces.items():
        assert payload is not None, name
        assert set(payload) == set(status), name
        assert set(payload) >= REQUIRED, name
        assert payload["prefill_path"] == "helper", name
        assert payload["fallback_reason"] is None, name
        assert payload["arming"]["node_id"] == "n", name
