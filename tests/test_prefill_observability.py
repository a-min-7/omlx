# SPDX-License-Identifier: Apache-2.0
"""The Prefill Accelerator's observability fields and state transitions.

Each test here pins one field or one transition the operators were missing:
where a model's prompts run (#79), one coherent peer state (#86 d), the three
separate spans of a prompt (#86 a), warm against cold per split (#86 b), and
a memory gate that says when it was read and that it is a capacity, not a
target (#86 e, #84).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.model_settings import ModelSettings
from omlx.prefill_helper import handshake, health, peers
from omlx.prefill_helper.manager import PrefillHelperManager
from omlx.prefill_helper.session import PrefillHelperSession


@pytest.fixture(autouse=True)
def _clean_state():
    peers.reset()
    health.reset()
    yield
    peers.reset()
    health.reset()


def _armed(node="helper-mac", layers=24):
    return ModelSettings(prefill_helper_node_id=node, prefill_helper_layers=layers)


# ---------------------------------------------------------------------------
# #79: where the prompts go, and why not


def test_arming_names_the_path_and_the_reason():
    assert health.arming(_armed()) == {
        "path": "helper",
        "reason": None,
        "node_id": "helper-mac",
        "layers": 24,
        "automatic_layers": False,
    }
    assert health.arming(ModelSettings())["reason"] == "unarmed"
    assert health.arming(ModelSettings())["path"] == "local"


def test_layers_without_a_mac_is_the_half_configured_state():
    armed = health.arming(ModelSettings(prefill_helper_layers=24))
    assert (armed["path"], armed["reason"]) == ("local", "half_configured")


def test_a_mac_without_layers_is_the_supported_automatic_split():
    """The reviewer's "exactly one key" rule would flag this working setup.

    ``prefill_helper_layers: null`` is Auto: the helper takes as many layers
    as fit. It is armed, and the status says the layers are automatic.
    """
    armed = health.arming(ModelSettings(prefill_helper_node_id="helper-mac"))
    assert (armed["path"], armed["reason"]) == ("helper", None)
    assert armed["automatic_layers"] is True


def test_counters_count_each_path_and_reason():
    health.note(health.HELPER)
    health.note(health.LOCAL, health.UNREACHABLE)
    health.note(health.LOCAL, health.UNREACHABLE)
    health.note(health.LOCAL, health.UNARMED)
    assert health.snapshot() == {
        "path": {"helper": 1, "local": 3},
        "reason": {"unreachable": 2, "unarmed": 1},
    }


def test_the_load_line_says_which_side_of_the_arming_it_saw(caplog):
    from omlx.engine_pool import EnginePool

    entry = SimpleNamespace(model_id="m1", model_path="/models/m1")
    pool = object.__new__(EnginePool)
    with caplog.at_level("INFO"):
        pool._log_arming(entry, ModelSettings(), None)
        pool._log_arming(entry, _armed(), object())
        pool._log_arming(entry, ModelSettings(prefill_helper_layers=24), object())
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "not armed for m1" in text and "model_settings.json" in text
    assert "armed for m1: node=helper-mac layers=24" in text
    assert "prefill_helper_layers=24 is set but prefill_helper_node_id is not" in text
    # The half-configured case is the one that is an error.
    levels = {record.levelname: record.getMessage() for record in caplog.records}
    assert any(
        record.levelname == "ERROR" and "prefill_helper_layers" in record.getMessage()
        for record in caplog.records
    )
    assert levels.get("ERROR")


# ---------------------------------------------------------------------------
# #86 d: one peer state, and one retry when it comes back


def test_the_offline_online_edge_is_the_only_wakeup():
    calls = []
    peers.wait_for("n", lambda: calls.append(1))
    assert peers.offline("n", "timed out") is False
    assert calls == []
    # The first online observation fires the waiter; the announce storm that
    # follows it must not fire it again.
    assert peers.online("n", source="discovery") is True
    # Already online: not an edge, so no waiter fires even if one waits.
    peers.wait_for("n", lambda: calls.append(2))
    assert peers.online("n", source="discovery") is False
    assert calls == [1]
    assert peers.state("n").attempt == 0


def test_a_first_ever_sighting_does_not_wake_anyone():
    """A starting process sees every paired Mac for the first time."""
    calls = []
    peers.wait_for("n", lambda: calls.append(1))
    peers.online("n", source="candidates")
    assert calls == []


def test_refused_does_not_schedule_the_backoff_a_timeout_does():
    peers.refused("n", "does not support this model family yet.")
    state = peers.state("n")
    assert (state.status, state.next_retry_at) == ("refused", None)
    assert state.to_dict()["refused"] is True
    peers.backing_off("n", 2, 30.0)
    assert peers.state("n").next_retry_at is not None
    peers.clear_backoff("n")
    assert (peers.state("n").attempt, peers.state("n").next_retry_at) == (0, None)


def test_backoff_reads_as_a_schedule_not_as_a_silent_mac(monkeypatch):
    """The status names the retry, and discovery's announce retries once."""
    starts = []
    manager = PrefillHelperManager(
        model_path="/models/m",
        model_type="qwen3_5",
        adapter=None,
        language_model=None,
        attach=lambda session: None,
        chunk_tokens=1,
        launcher=SimpleNamespace(),
    )
    monkeypatch.setattr(manager, "_start", lambda: starts.append(1))
    manager.node_id = "n"
    manager._backing_off(2, "the connection was lost")

    status = manager.status()
    assert status["state"] == "offline"
    assert status["backing_off"]["attempt"] == 2
    assert status["backing_off"]["next_retry_at"] > time.time()
    assert status["backing_off"]["last_error"] == "the connection was lost"
    assert status["peer"]["reachable"] is False

    announce = SimpleNamespace(state="discovered", node_id="n")
    peers.announce_peer(announce)
    # One object serves both surfaces: discovery's sighting flips reachable
    # while the schedule it interrupted is still visible.
    assert manager.status()["peer"]["reachable"] is True
    deadline = time.time() + 5
    while not starts and time.time() < deadline:
        time.sleep(0.01)
    assert starts == [1]
    # Every later announce is not another retry.
    peers.announce_peer(announce)
    peers.announce_peer(announce)
    time.sleep(0.1)
    assert starts == [1]
    manager.close()


def test_a_discovery_change_that_is_not_healthy_does_nothing():
    peers.offline("n", "timed out")
    peers.announce_peer(SimpleNamespace(state="dead", node_id="n"))
    assert peers.state("n").status == "offline"


# ---------------------------------------------------------------------------
# #86 a and b: three spans, and warm per split


def _session(layers=4, work=None):
    client = SimpleNamespace(info={}, alive=True)
    return PrefillHelperSession(
        client,
        adapter=None,
        lm=None,
        layers=layers,
        name="helper",
        work=work or [1.0] * 8,
    )


def test_the_three_spans_are_reported_apart():
    session = _session()
    session.finish(
        "r1",
        tokens=100,
        seconds=7.5,
        layers=4,
        helper_seconds=6.0,
        transfer_seconds=0.9,
    )
    prompt = session.last_prompt
    # A component of the span, never subtracted from another host's clock.
    assert prompt["session_prefill_seconds"] == 7.5
    assert prompt["seconds"] == 7.5  # kept for one release
    assert prompt["helper_seconds"] == 6.0
    assert prompt["transfer_seconds"] == 0.9


def test_helper_seconds_does_not_move_with_the_session_span():
    """The scenario test: same helper work, different wall-clock TTFT.

    Before the split, the single field tracked TTFT across a 7-second range
    and could not attribute any of it; the helper's figure must not.
    """
    session = _session()
    session.finish("a", 100, 7.56, layers=4, helper_seconds=6.0, transfer_seconds=0.8)
    fast = dict(session.last_prompt)
    session.finish("b", 100, 10.74, layers=12, helper_seconds=6.0, transfer_seconds=0.8)
    slow = dict(session.last_prompt)
    assert slow["session_prefill_seconds"] - fast["session_prefill_seconds"] > 3
    assert fast["helper_seconds"] == slow["helper_seconds"] == 6.0


def test_warm_and_requests_are_kept_per_split():
    session = _session()
    session.finish("a", 10, 1.0, layers=24)
    first = session.split_stats()["24"]
    assert (first["requests_since_load"], first["warm"]) == (1, False)
    session.finish("b", 10, 1.0, layers=24)
    assert session.split_stats()["24"] == {
        "requests_since_load": 2,
        "warm": True,
        "last_at": session.split_stats()["24"]["last_at"],
    }
    # A different k is a different split: it starts cold and at zero.
    session.finish("c", 10, 1.0, layers=12)
    assert session.split_stats()["12"]["requests_since_load"] == 1
    assert session.split_stats()["12"]["warm"] is False
    session.finish("d", 10, 1.0, layers=24)
    assert session.split_stats()["24"]["requests_since_load"] == 3


# ---------------------------------------------------------------------------
# #86 e and #84: a dated capacity, never an invented recommendation


def test_the_gate_is_dated_capacity_with_its_inputs():
    gate = health.gate(39, {"budget": 98 << 30, "memory": 128 << 30}, 1700000000.0)
    assert gate["kind"] == "capacity"
    assert (gate["layers_fit"], gate["max_layers_fit"]) == (39, 39)
    assert gate["measured_at"] == 1700000000.0
    assert gate["inputs"] == {
        "available_bytes": 98 << 30,
        "memory_bytes": 128 << 30,
        "source": "helper-info",
    }
    # No stored measurement with provenance exists, so none is claimed.
    assert "recommended_layers" in gate and gate["recommended_layers"] is None


def test_a_candidate_row_dates_its_figures(monkeypatch):
    from omlx.prefill_helper import routes as helper_routes

    info = {
        "version": "0.1.6",
        "data_protocol": 1,
        "contracts": {"qwen3_5": 1},
        "budgets": [40 << 30],
        "budget": 40 << 30,
        "memory": 128 << 30,
        "commit": "abc123",
        "plan_api_version": 1,
    }
    transport = SimpleNamespace(
        resolve_target=lambda node: SimpleNamespace(link="ethernet"),
        remote_info=lambda target: info,
        incompatible=lambda info, adapter, name, oqh=None: None,
        plan_mismatch=handshake.plan_mismatch,
        source_mismatch=handshake.source_mismatch,
    )
    monkeypatch.setattr(helper_routes, "transport_for", lambda node: transport)
    monkeypatch.setattr(helper_routes, "computer_name", lambda: "this-mac")
    adapter = SimpleNamespace(
        memory_plan=lambda path: (1 << 30, [1 << 30] * 60),
        split_limit=lambda count: count,
    )
    device = {"node_id": "n", "friendly_name": "nvidia1", "caps": {}}
    row = helper_routes._candidate(device, "/models/m", adapter)
    assert row["online"] is True
    assert row["version_ok"] is True and row["plan_ok"] is True
    assert row["source_mismatch"] is False
    assert row["commit"] == "abc123" and row["plan_api_version"] == 1
    # Enough layers for one GPU's budget; the old key stays beside the new.
    assert row["layers_fit"] == row["max_layers_fit"] > 0
    assert isinstance(row["measured_at"], float)
    assert row["gate"]["recommended_layers"] is None
    assert row["gate"]["kind"] == "capacity"
    assert row["peer"]["reachable"] is True


def test_a_refusal_is_distinguished_from_an_unreachable_mac(monkeypatch):
    from omlx.prefill_helper import routes as helper_routes

    def unreachable(node):
        raise OSError("No route to host")

    monkeypatch.setattr(
        helper_routes,
        "transport_for",
        lambda node: SimpleNamespace(resolve_target=unreachable),
    )
    device = {"node_id": "n", "friendly_name": "nvidia1", "caps": {}}
    row = helper_routes._candidate(device, "/models/m", None)
    assert row["online"] is False
    assert peers.state("n").status == "offline"
    peers.refused("n", "does not support this model family yet.")
    assert peers.state("n").to_dict()["refused"] is True


# ---------------------------------------------------------------------------
# The handshake that the contract check cannot see


def test_a_plan_or_source_difference_is_flagged_apart_from_version_ok():
    same = {"plan_api_version": 1, "commit": None}
    assert handshake.plan_mismatch(same) is None
    assert handshake.source_mismatch(same) is None
    other_plan = {"plan_api_version": 2, "commit": None}
    assert "plans memory as version 2" in handshake.plan_mismatch(other_plan)
    different = {"plan_api_version": 1, "commit": "0123456789abcdef"}
    message = handshake.source_mismatch(different, ours="fedcba9876543210")
    assert message is not None and "0123456789ab" in message
    # An older helper that does not say is never a mismatch.
    assert handshake.plan_mismatch({}) is None
    assert handshake.source_mismatch({"commit": "unknown"}, ours="x") is None
    assert handshake.source_mismatch({"commit": "abc"}, ours="unknown") is None


# ---------------------------------------------------------------------------
# The status payload is a contract: a monitor reads these keys


def _app(monkeypatch, status, settings):
    from omlx.prefill_helper import routes as helper_routes

    entries = {
        "loaded": SimpleNamespace(engine=SimpleNamespace(prefill_helper=_Fake(status))),
        "idle": SimpleNamespace(engine=None, model_path="/models/idle"),
    }
    helper_routes.set_engine_pool_getter(
        lambda: SimpleNamespace(
            get_entry=entries.get,
            _settings_manager=SimpleNamespace(get_settings=lambda model_id: settings),
        )
    )
    app = FastAPI()
    app.include_router(helper_routes.router)
    return TestClient(app)


class _Fake:
    def __init__(self, status):
        self._status = status

    def status(self):
        return dict(self._status)


READY = {
    "state": "ready",
    "node_id": "n",
    "held": 24,
    "peer": {"reachable": True, "refused": False},
    "backing_off": None,
    "last_prompt": {
        "tokens": 100,
        "seconds": 7.5,
        "session_prefill_seconds": 7.5,
        "helper_seconds": 6.0,
        "transfer_seconds": 0.9,
        "layers": 24,
        "at": 1.0,
    },
    "splits": {"24": {"warm": True, "requests_since_load": 2, "last_at": 1.0}},
}


def test_the_status_payload_keeps_its_keys(monkeypatch):
    client = _app(monkeypatch, READY, _armed())
    payload = client.get("/admin/api/prefill-helper/models/loaded").json()
    assert {
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
    } <= set(payload)
    assert payload["prefill_path"] == "helper" and payload["fallback_reason"] is None
    assert payload["arming"]["node_id"] == "helper-mac"
    assert set(payload["last_prompt"]) == {
        "tokens",
        "seconds",
        "session_prefill_seconds",
        "helper_seconds",
        "transfer_seconds",
        "layers",
        "at",
    }
    assert set(payload["counters"]) == {"path", "reason"}
    # JSON-serialisable as-is: a monitor polls this endpoint.
    json.dumps(payload)


def test_a_ready_model_with_no_helper_says_unreachable(monkeypatch):
    status = {
        **READY,
        "state": "offline",
        "peer": {"reachable": False, "refused": False},
    }
    client = _app(monkeypatch, status, _armed())
    payload = client.get("/admin/api/prefill-helper/models/loaded").json()
    assert payload["prefill_path"] == "local"
    assert payload["fallback_reason"] == "unreachable"


def test_a_refused_helper_says_refused(monkeypatch):
    status = {**READY, "state": "error", "peer": {"reachable": True, "refused": True}}
    client = _app(monkeypatch, status, _armed())
    payload = client.get("/admin/api/prefill-helper/models/loaded").json()
    assert payload["prefill_path"] == "local"
    assert payload["fallback_reason"] == "refused"
