"""Batch scheduling for the per-agent simulation-config LLM calls (the ``prepare`` stage).

``SimulationConfigGenerator.generate_config`` asks the LLM for activity configs in batches of
``AGENTS_PER_BATCH`` entities. Each batch is one independent, latency-bound call: the prompt only
contains that batch's entities and every ``agent_id`` is fixed from the batch offset before the
call, so batches share no state and their results can be concatenated in batch order whatever
order they complete in. Running them one after another (the original loop) made this stage cost
``num_batches x call latency`` (~18 x ~54 s for a 262-agent cluster) plus every stall in full.

This module is pure (no LLM / Flask / Zep imports) so the scheduling is unit-testable:

* :func:`resolve_agents_per_batch` — ``AGENTS_PER_BATCH`` env (default 15, min 1). Smaller
  batches mean shorter JSON outputs per call (less truncation exposure) and finer retry.
* :func:`resolve_config_parallel_count` — ``CONFIG_PARALLEL_COUNT`` env: ``0``/unset = run every
  batch at once (the natural maximum: more workers than batches does nothing); ``N`` caps the
  in-flight calls, the knob to turn down if the provider starts returning 429s.
* :func:`resolve_config_llm_timeout` — ``CONFIG_LLM_TIMEOUT_SECONDS`` env (default 180 s): the
  wall-clock cap of ONE config-generation LLM attempt (time config, event config, each agent
  batch), measured from the start of the call, and also the HTTP read timeout handed to the OpenAI
  client. Until 2026-10-08 it was only the read timeout, which never fires while the gateway keeps
  the connection fed: calls of 500 to 2 281 s completed through a 60 s "timeout", and one such call
  per cluster held the whole prepare stage until its 3600 s watchdog (all three clusters, 10-08).
* :func:`resolve_config_reasoning_effort` — ``CONFIG_LLM_REASONING_EFFORT`` env (default ``none``):
  the OpenRouter ``reasoning.effort`` sent with every config-generation call. These calls map an
  entity type plus a one-line summary onto a handful of behaviour dials and return ~1 K tokens of
  JSON; the slow answers on 10-08 were 8 K to 47 K reasoning tokens for exactly that, because the
  provider OpenRouter picked turns thinking on by default. ``none`` disables it where the provider
  supports the switch; ``provider`` (or ``default``) sends nothing and leaves it to the provider;
  ``low`` / ``medium`` / ``high`` / ``minimal`` pass through.
* :func:`run_batches` — the executor.
"""
import concurrent.futures
import math
import os

DEFAULT_AGENTS_PER_BATCH = 15
DEFAULT_CONFIG_LLM_TIMEOUT_SECONDS = 180.0
DEFAULT_CONFIG_LLM_REASONING_EFFORT = "none"
#: Values that mean "send no reasoning preference; the provider decides" (the pre-2026-10-08 shape).
_REASONING_PROVIDER_DEFAULT = ("provider", "default", "unset")
_REASONING_OFF = ("none", "off", "0", "false", "disabled")
_REASONING_LEVELS = ("minimal", "low", "medium", "high")


def _env(env):
    return env if env is not None else os.environ


def _int_env(env, name, default, minimum):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def resolve_agents_per_batch(env=None):
    """Entities per LLM call. ``AGENTS_PER_BATCH`` env, default 15, garbage or < 1 -> default."""
    return _int_env(_env(env), "AGENTS_PER_BATCH", DEFAULT_AGENTS_PER_BATCH, minimum=1)


def resolve_config_parallel_count(env=None, num_batches=None):
    """Concurrent batch calls. ``CONFIG_PARALLEL_COUNT`` env; 0/unset/garbage = all batches at
    once. Never more than ``num_batches`` (when given) and never less than 1."""
    env = _env(env)
    raw = (env.get("CONFIG_PARALLEL_COUNT") or "").strip()
    try:
        requested = int(raw) if raw else 0
    except ValueError:
        requested = 0
    if requested < 0:
        requested = 0
    if num_batches is None:
        return requested if requested > 0 else 0
    cap = max(1, int(num_batches))
    return cap if requested == 0 else max(1, min(requested, cap))


def resolve_config_llm_timeout(env=None):
    """Per-request HTTP timeout (seconds) for the config-generation LLM calls.
    ``CONFIG_LLM_TIMEOUT_SECONDS`` env, default 180, garbage or <= 0 -> default."""
    raw = (_env(env).get("CONFIG_LLM_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_CONFIG_LLM_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_CONFIG_LLM_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_CONFIG_LLM_TIMEOUT_SECONDS


def resolve_config_reasoning_effort(env=None):
    """OpenRouter ``reasoning.effort`` for the config-generation calls, or ``None`` to send nothing.

    ``CONFIG_LLM_REASONING_EFFORT`` env: unset/``none``/``off`` -> ``"none"`` (reasoning disabled,
    the default); ``provider``/``default`` -> ``None`` (no preference sent); a known effort level
    passes through; anything else -> the default."""
    raw = (_env(env).get("CONFIG_LLM_REASONING_EFFORT") or "").strip().lower()
    if not raw or raw in _REASONING_OFF:
        return "none"
    if raw in _REASONING_PROVIDER_DEFAULT:
        return None
    if raw in _REASONING_LEVELS:
        return raw
    return DEFAULT_CONFIG_LLM_REASONING_EFFORT


def reasoning_extra_body(effort):
    """The OpenAI-SDK ``extra_body`` carrying an OpenRouter reasoning preference, or ``None``.
    OpenRouter forwards ``reasoning`` only to providers that support it; others ignore it."""
    if not effort:
        return None
    return {"reasoning": {"effort": effort}}


def split_batches(n_items, per_batch):
    """``[(start, end), ...]`` half-open index ranges covering ``0..n_items``."""
    per_batch = max(1, int(per_batch))
    n_items = max(0, int(n_items))
    return [(s, min(s + per_batch, n_items)) for s in range(0, n_items, per_batch)]


def num_batches(n_items, per_batch):
    return math.ceil(max(0, int(n_items)) / max(1, int(per_batch)))


def run_batches(batch_fn, batches, parallel_count, on_done=None):
    """Run ``batch_fn(batch_idx, start, end)`` for every ``(start, end)`` in ``batches`` with up to
    ``parallel_count`` in flight; return the results concatenated in **batch order**.

    ``on_done(completed_count, batch_idx)`` is called from the calling thread each time a batch
    finishes (completion order) so the caller can report progress. An exception in any batch is
    re-raised after the pool drains, the same way the serial loop would have surfaced it.
    """
    batches = list(batches)
    if not batches:
        return []
    width = max(1, min(int(parallel_count) if parallel_count else len(batches), len(batches)))
    results = [None] * len(batches)
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        futures = {pool.submit(batch_fn, i, s, e): i for i, (s, e) in enumerate(batches)}
        completed = 0
        first_error = None
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:          # noqa: BLE001 - re-raised below after the pool drains
                if first_error is None:
                    first_error = exc
                results[i] = []
            completed += 1
            if on_done:
                on_done(completed, i)
    if first_error is not None:
        raise first_error
    out = []
    for r in results:
        out.extend(r or [])
    return out
