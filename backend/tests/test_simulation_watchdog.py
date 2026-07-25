"""Watchdog / dead-man's-switch policy for the simulation runner.

A running sim's OASIS subprocess outlives the HTTP request and is only stopped by the client's
/stop call. If the client is killed (its session times out) it can never send /stop, so the
subprocess — and its LLM token + CPU burn — would run forever. The runner bounds a sim's lifetime
server-side via ``_watchdog_reason`` (pure policy) fed by ``note_client_poll`` heartbeats. These
tests pin the policy without threads or a live subprocess.
"""
import pytest

from app.services.simulation_runner import SimulationRunner


R = SimulationRunner._watchdog_reason  # (now, started, last_poll, max_dur, heartbeat) -> reason|None


def test_within_bounds_returns_none():
    # polled 10s ago, running 100s, both well under the bounds
    assert R(1000.0, started_monotonic=900.0, last_client_poll=990.0,
             max_duration=5400, heartbeat_timeout=300) is None


def test_stale_heartbeat_trips_dead_mans_switch():
    # last poll was 400s ago (> 300s heartbeat) -> the client is gone
    assert R(1000.0, started_monotonic=900.0, last_client_poll=600.0,
             max_duration=5400, heartbeat_timeout=300) == "no-client-heartbeat"


def test_absolute_max_duration_backstop():
    # still being polled, but the sim has run 6000s (> 5400s ceiling)
    assert R(7000.0, started_monotonic=1000.0, last_client_poll=6999.0,
             max_duration=5400, heartbeat_timeout=300) == "max-duration"


def test_heartbeat_takes_precedence_over_duration():
    # both bounds exceeded; the gone-client reason wins (it's the more specific cause)
    assert R(9000.0, started_monotonic=1000.0, last_client_poll=1000.0,
             max_duration=5400, heartbeat_timeout=300) == "no-client-heartbeat"


def test_zero_bounds_disable_each_check():
    # heartbeat disabled -> stale poll ignored; only the duration ceiling can fire
    assert R(1000.0, started_monotonic=900.0, last_client_poll=100.0,
             max_duration=0, heartbeat_timeout=0) is None
    assert R(7000.0, started_monotonic=1000.0, last_client_poll=6999.0,
             max_duration=5400, heartbeat_timeout=0) == "max-duration"


def test_missing_timestamps_never_trip():
    # a sim not yet armed (no timestamps) is never reaped
    assert R(1000.0, started_monotonic=None, last_client_poll=None,
             max_duration=5400, heartbeat_timeout=300) is None


def test_note_client_poll_only_tracks_running_sims():
    sid_running, sid_unknown = "sim_running_test", "sim_unknown_test"
    SimulationRunner._started_monotonic[sid_running] = 123.0     # simulate an armed sim
    SimulationRunner._last_client_poll.pop(sid_running, None)
    SimulationRunner._last_client_poll.pop(sid_unknown, None)
    try:
        SimulationRunner.note_client_poll(sid_running)
        SimulationRunner.note_client_poll(sid_unknown)           # not armed -> ignored
        assert sid_running in SimulationRunner._last_client_poll
        assert sid_unknown not in SimulationRunner._last_client_poll
    finally:
        SimulationRunner._started_monotonic.pop(sid_running, None)
        SimulationRunner._last_client_poll.pop(sid_running, None)
