"""Pure helpers behind the persona (profile) LLM call guard — app/services/profile_llm.py.

Regression for 2026-09-29: one persona-synthesis request never returned, the worker thread
blocked on it, and the whole prepare stage sat until the consumer's 3600 s watchdog killed the
task. These helpers are what bound a single call (wall clock, not socket idle time), space the
retries, and describe a degradation to the consumer.
"""
import json
import threading
import time

import httpx
import pytest

from app.services import profile_llm as pl


# ---- PROFILE_LLM_TIMEOUT_SECONDS -----------------------------------------------------------

def test_timeout_defaults_to_300_seconds():
    assert pl.resolve_profile_llm_timeout({}) == 300.0


def test_timeout_is_read_from_env():
    assert pl.resolve_profile_llm_timeout({"PROFILE_LLM_TIMEOUT_SECONDS": "120"}) == 120.0


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "-5"])
def test_garbage_or_non_positive_timeout_falls_back_to_default(raw):
    assert pl.resolve_profile_llm_timeout({"PROFILE_LLM_TIMEOUT_SECONDS": raw}) == 300.0


# ---- call_with_deadline --------------------------------------------------------------------

def test_deadline_returns_the_value_of_a_call_that_finishes_in_time():
    assert pl.call_with_deadline(lambda: "persona", timeout=1.0) == "persona"


def test_deadline_reraises_the_error_of_a_call_that_fails_in_time():
    def boom():
        raise ValueError("provider said no")

    with pytest.raises(ValueError, match="provider said no"):
        pl.call_with_deadline(boom, timeout=1.0)


def test_call_that_never_returns_is_bounded_and_raises_a_timeout():
    release = threading.Event()
    started = time.monotonic()
    try:
        with pytest.raises(pl.ProfileLLMTimeout):
            pl.call_with_deadline(lambda: release.wait(30), timeout=0.2)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 2.0, f"a wedged call must cost ~the deadline, took {elapsed:.1f}s"


def test_a_dripping_call_is_bounded_by_wall_clock_not_by_idle_time():
    """A provider that keeps the socket busy defeats an idle (read) timeout; the deadline is
    measured from the start of the call, so activity inside the call does not extend it."""
    stop = threading.Event()

    def drip():
        while not stop.wait(0.05):   # "bytes" every 50 ms, never a response
            pass

    started = time.monotonic()
    try:
        with pytest.raises(pl.ProfileLLMTimeout):
            pl.call_with_deadline(drip, timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        stop.set()

    assert elapsed < 2.0


def test_wedged_call_runs_on_a_daemon_thread_so_it_cannot_block_process_exit():
    release = threading.Event()
    seen = {}

    def wedge():
        seen["daemon"] = threading.current_thread().daemon
        release.wait(30)

    try:
        with pytest.raises(pl.ProfileLLMTimeout):
            pl.call_with_deadline(wedge, timeout=0.2)
    finally:
        release.set()

    assert seen["daemon"] is True


def test_timeout_error_is_a_timeout_error():
    assert issubclass(pl.ProfileLLMTimeout, TimeoutError)


# ---- backoff_delay -------------------------------------------------------------------------

def test_backoff_is_about_one_then_four_seconds():
    assert pl.backoff_delay(0, rng=lambda: 0.5) == pytest.approx(1.0)
    assert pl.backoff_delay(1, rng=lambda: 0.5) == pytest.approx(4.0)


def test_backoff_jitter_stays_within_a_quarter_either_side():
    assert pl.backoff_delay(1, rng=lambda: 0.0) == pytest.approx(3.0)
    assert pl.backoff_delay(1, rng=lambda: 1.0) == pytest.approx(5.0)


# ---- classify_failure ----------------------------------------------------------------------

def _request():
    return httpx.Request("POST", "http://localhost:1/chat/completions")


@pytest.mark.parametrize("exc, expected", [
    (pl.ProfileLLMTimeout("no response within 300s"), "timeout"),
    (TimeoutError("socket"), "timeout"),
    (httpx.ReadTimeout("read", request=_request()), "timeout"),
    (httpx.ConnectError("refused", request=_request()), "connection"),
    (httpx.RemoteProtocolError("peer closed", request=_request()), "connection"),
    (ConnectionResetError("reset"), "connection"),
    (json.JSONDecodeError("Expecting value", "", 0), "bad_json"),
    (ValueError("anything else"), "error"),
])
def test_failures_are_classified_for_the_degradation_reason(exc, expected):
    assert pl.classify_failure(exc) == expected


def test_openai_sdk_timeout_and_connection_errors_are_classified():
    import openai

    assert pl.classify_failure(openai.APITimeoutError(request=_request())) == "timeout"
    assert pl.classify_failure(openai.APIConnectionError(request=_request())) == "connection"


# ---- degradation warnings ------------------------------------------------------------------

GOOGL = {"entity": "GOOGL", "entity_type": "Company", "node_degree": 55, "reason": "timeout"}


def test_no_degradation_means_no_warnings():
    assert pl.degradation_warnings([], total=282) == []


def test_each_degradation_is_one_machine_readable_warning():
    warnings = pl.degradation_warnings([GOOGL], total=282)

    code, payload = warnings[0].split(": ", 1)
    assert code == "PROFILE_DEGRADED"
    assert json.loads(payload) == GOOGL


def test_a_summary_warning_carries_the_degraded_fraction():
    warnings = pl.degradation_warnings([GOOGL], total=282)

    code, payload = warnings[-1].split(": ", 1)
    assert code == "PROFILE_DEGRADED_SUMMARY"
    assert json.loads(payload) == {"degraded": 1, "total": 282, "fraction": round(1 / 282, 4)}


def test_entity_names_survive_the_round_trip_unescaped():
    record = dict(GOOGL, entity="Ireland’s Data Protection Commission: EU")
    warnings = pl.degradation_warnings([record], total=10)

    assert "Ireland’s Data Protection Commission: EU" in warnings[0]
    assert json.loads(warnings[0].split(": ", 1)[1])["entity"] == record["entity"]
