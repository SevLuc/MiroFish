"""Guard rails for the per-entity persona (profile) LLM call of the ``prepare`` stage.

``OasisProfileGenerator`` makes one independent LLM call per graph entity, ``PROFILE_PARALLEL_COUNT``
at a time, and waits for all of them. Until 2026-09-29 nothing bounded one call by wall clock: the
client ran on the SDK defaults (a 600 s *idle* timeout plus two silent SDK retries), so a request
the provider accepted and then never answered blocked its worker thread, the barrier waited on
that thread, and prepare sat until the consumer's 3600 s watchdog killed the whole task.

An idle (socket read) timeout is not enough on its own. It only fires when the socket is silent,
and the same day's log has calls that took 17 to 23 minutes and still succeeded, i.e. calls no
idle timeout ever cut short. So the bound here is a **deadline measured from the start of the
call**, with the socket timeout kept as the second layer that eventually frees a silent thread.

This module is pure (no LLM / Flask / graph imports) so the policy is unit-testable:

* :func:`resolve_profile_llm_timeout` — ``PROFILE_LLM_TIMEOUT_SECONDS`` env (default 300 s): the
  wall-clock cap of ONE attempt, also handed to the HTTP client as its read timeout. 300 s clears
  the healthy tail measured on 2026-09-29 (median 20 s, p95 157 s) with room to spare; with three
  attempts a permanently wedged entity costs about 15 minutes of one worker, not the run.
* :func:`call_with_deadline` — run one call, give up on it after the cap.
* :func:`backoff_delay` — the pause before a retry (about 1 s, then about 4 s, jittered).
* :func:`classify_failure` — ``timeout`` / ``connection`` / ``bad_json`` / ``error``.
* :func:`degradation_warnings` — the ``prepare_warnings`` lines for entities that ended up on the
  rule-based persona. The fork only reports; whether a degradation is material is the consumer's
  call, which is why every line carries the entity, its type and its node degree.
"""
import json
import os
import random
import threading

DEFAULT_PROFILE_LLM_TIMEOUT_SECONDS = 300.0

#: Attempts per entity, the first one included.
PROFILE_LLM_MAX_ATTEMPTS = 3

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_FACTOR = 4.0
BACKOFF_JITTER = 0.25          # +/- 25 %

WARNING_DEGRADED = "PROFILE_DEGRADED"
WARNING_DEGRADED_SUMMARY = "PROFILE_DEGRADED_SUMMARY"

_TIMEOUT_MARKERS = ("timeout",)
_CONNECTION_MARKERS = ("connect", "network", "protocol", "transport")


class ProfileLLMTimeout(TimeoutError):
    """One persona LLM call did not return within the wall-clock cap."""


def resolve_profile_llm_timeout(env=None):
    """Wall-clock cap (seconds) of one persona LLM attempt.
    ``PROFILE_LLM_TIMEOUT_SECONDS`` env, default 300, garbage or <= 0 -> default."""
    env = env if env is not None else os.environ
    raw = (env.get("PROFILE_LLM_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_PROFILE_LLM_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_PROFILE_LLM_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_PROFILE_LLM_TIMEOUT_SECONDS


def call_with_deadline(fn, timeout, *, name="profile-llm"):
    """Return ``fn()``, re-raise what it raised, or raise :class:`ProfileLLMTimeout` once
    ``timeout`` seconds of wall clock have passed since the call started.

    Python cannot kill a thread, so a call that overruns is *abandoned*, not stopped: it keeps
    its thread until the socket timeout (or the provider) ends it, and its late result is
    dropped. The thread is a daemon so an abandoned call can never hold up process exit, which
    a ``ThreadPoolExecutor`` worker would (the pool joins its threads on shutdown)."""
    outcome = {}
    finished = threading.Event()

    def run():
        try:
            outcome["value"] = fn()
        except BaseException as exc:          # handed to the caller, which decides
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=run, name=name, daemon=True).start()
    if not finished.wait(timeout):
        raise ProfileLLMTimeout(f"no response within {timeout:g}s (wall clock)")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def backoff_delay(attempt, *, rng=random.random):
    """Seconds to wait after failed attempt number ``attempt`` (0-based): ~1 s, then ~4 s,
    each jittered by +/- 25 % so 16 workers retrying at once do not re-send in lockstep."""
    base = BACKOFF_BASE_SECONDS * (BACKOFF_FACTOR ** attempt)
    return base * (1.0 + BACKOFF_JITTER * (2.0 * rng() - 1.0))


def classify_failure(exc):
    """``timeout`` | ``connection`` | ``bad_json`` | ``error`` — the degradation reason.

    Matched on the class names of the exception's MRO rather than on imported types, so the
    OpenAI SDK, httpx and requests errors are all covered without this module importing any of
    them (``APITimeoutError`` derives from ``APIConnectionError``, hence timeout is tested first)."""
    if isinstance(exc, json.JSONDecodeError):
        return "bad_json"
    names = [cls.__name__.lower() for cls in type(exc).__mro__]
    if isinstance(exc, TimeoutError) or any(m in n for n in names for m in _TIMEOUT_MARKERS):
        return "timeout"
    if isinstance(exc, ConnectionError) or any(m in n for n in names for m in _CONNECTION_MARKERS):
        return "connection"
    return "error"


def degradation_warnings(degradations, total):
    """``prepare_warnings`` lines for the entities that fell back to the rule-based persona.

    One ``PROFILE_DEGRADED: {json}`` line per entity (``entity``, ``entity_type``, ``node_degree``,
    ``reason``) and one closing ``PROFILE_DEGRADED_SUMMARY: {json}`` line (``degraded``, ``total``,
    ``fraction``). Empty list when nothing degraded, so an empty ``prepare_warnings`` still means
    a clean prepare."""
    degradations = list(degradations or [])
    if not degradations:
        return []
    total = max(int(total or 0), len(degradations))
    lines = [f"{WARNING_DEGRADED}: {json.dumps(record, ensure_ascii=False)}"
             for record in degradations]
    summary = {"degraded": len(degradations), "total": total,
               "fraction": round(len(degradations) / total, 4)}
    lines.append(f"{WARNING_DEGRADED_SUMMARY}: {json.dumps(summary)}")
    return lines
