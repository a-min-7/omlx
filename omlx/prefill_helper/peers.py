# SPDX-License-Identifier: Apache-2.0
"""One peer state for every surface that reports a helper Mac.

The status, the candidate rows and the retry schedule each used to read the
helper independently: the manager said ``offline`` from its own failed attempt
while ``/candidates`` said ``online`` from a fresh probe of the same node, and
the status text named a cause ("Grace does not answer") that was not the cause
— the manager was simply sitting in exponential backoff (#86d). A reader could
not tell whether the Mac was down or merely not asked recently.

Everything that observes a peer records it here, so one answer serves all
three, and the reason travels with it:

* ``OFFLINE`` — a probe timed out or a connection broke; a retry is scheduled.
* ``REFUSED`` — the helper answered and said no (version, pairing, memory); a
  retry cannot fix it, so it must not feed the same backoff as a timeout.
* ``ONLINE`` — a probe reached it, or discovery announced it.

A peer that goes from ``OFFLINE``/``REFUSED`` back to ``ONLINE`` wakes exactly
one waiter, once per edge: an mDNS announce every few seconds must not reset
the backoff on each announce, only the first return.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ONLINE = "online"
OFFLINE = "offline"
REFUSED = "refused"


@dataclass
class PeerState:
    """What the process knows about one paired Mac."""

    node_id: str
    status: str | None = None
    last_error: str | None = None
    last_attempt_at: float | None = None
    last_ok_at: float | None = None
    attempt: int = 0
    next_retry_at: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "reachable": None if self.status is None else self.status == ONLINE,
            "refused": self.status == REFUSED,
            "last_error": self.last_error,
            "last_attempt_at": self.last_attempt_at,
            "last_ok_at": self.last_ok_at,
            "source": self.source,
        }


_STATES: dict[str, PeerState] = {}
# A node's callbacks fire once, on the next offline/refused -> online edge.
_WAITERS: dict[str, list[Callable[[], None]]] = {}
_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def state(node_id: str) -> PeerState:
    """A copy, so a caller cannot mutate the record."""
    with _LOCK:
        peer = _STATES.get(node_id)
        if peer is None:
            return PeerState(node_id)
        return PeerState(**vars(peer))


def observe(
    node_id: str,
    status: str,
    *,
    reason: str | None = None,
    source: str = "probe",
    now: float | None = None,
) -> bool:
    """Record what a probe saw; True when this is an offline -> online edge.

    A first observation of a peer that was never seen is not an edge: a
    process starting up should not fire a retry for every paired Mac.
    """
    at = _now() if now is None else now
    with _LOCK:
        peer = _STATES.setdefault(node_id, PeerState(node_id))
        before = peer.status
        peer.status, peer.source, peer.last_attempt_at = status, source, at
        if status == ONLINE:
            peer.last_ok_at, peer.last_error = at, None
            peer.attempt, peer.next_retry_at = 0, None
        else:
            peer.last_error = reason
            if status == REFUSED:
                # No automatic retry is scheduled, so nothing wakes the
                # manager on its own; the peer is as reachable as it gets.
                peer.next_retry_at = None
        edge = status == ONLINE and before in (OFFLINE, REFUSED)
        callbacks: list[Callable[[], None]] = _WAITERS.pop(node_id, []) if edge else []
    # Outside the lock: a waiter may start work that observes this peer again.
    for callback in callbacks:
        callback()
    return edge


def online(node_id: str, *, source: str = "probe") -> bool:
    return observe(node_id, ONLINE, source=source)


def offline(node_id: str, reason: str, *, source: str = "probe") -> bool:
    return observe(node_id, OFFLINE, reason=reason, source=source)


def refused(node_id: str, reason: str, *, source: str = "manager") -> bool:
    """The helper answered and said no; a retry does not help."""
    return observe(node_id, REFUSED, reason=reason, source=source)


def backing_off(
    node_id: str, attempt: int, delay: float, *, now: float | None = None
) -> float:
    """Record the next automatic attempt; returns its wall-clock time."""
    at = _now() if now is None else now
    with _LOCK:
        peer = _STATES.setdefault(node_id, PeerState(node_id))
        peer.attempt, peer.next_retry_at = int(attempt), at + float(delay)
        return peer.next_retry_at


def clear_backoff(node_id: str) -> None:
    with _LOCK:
        peer = _STATES.get(node_id)
        if peer is not None:
            peer.attempt, peer.next_retry_at = 0, None


def wait_for(node_id: str, callback: Callable[[], None]) -> None:
    """Call ``callback`` once when this peer next comes back online."""
    with _LOCK:
        _WAITERS.setdefault(node_id, []).append(callback)


def stop_waiting(node_id: str, callback: Callable[[], None]) -> None:
    with _LOCK:
        waiting = _WAITERS.get(node_id)
        if waiting and callback in waiting:
            waiting.remove(callback)
            if not waiting:
                _WAITERS.pop(node_id, None)


def announce_peer(peer: Any) -> None:
    """A discovery change callback: an announced peer counts as reachable.

    ``peer.state`` is discovery's own vocabulary (``discovered``, ``suspect``,
    ``dead``); only the healthy one feeds this record, and only the first one
    after an outage wakes the manager.
    """
    if getattr(peer, "state", None) != "discovered":
        return
    node_id = getattr(peer, "node_id", None)
    if node_id:
        observe(str(node_id), ONLINE, source="discovery")


def reset() -> None:
    """Forget everything; the tests' isolation hook."""
    with _LOCK:
        _STATES.clear()
        _WAITERS.clear()
