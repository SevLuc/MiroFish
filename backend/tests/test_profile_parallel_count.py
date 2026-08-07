"""prepare's persona-generation concurrency is env-tunable, with a sane raised default.
See app/api/simulation.py :: _resolve_profile_parallel_count (B2 — the prepare stage is
LLM-latency-bound and independent per agent, so higher concurrency cuts its wall-clock)."""
from app.api import simulation as sim


def test_explicit_request_value_wins_over_env(monkeypatch):
    monkeypatch.setenv("PROFILE_PARALLEL_COUNT", "20")
    assert sim._resolve_profile_parallel_count(12) == 12


def test_env_used_when_request_absent(monkeypatch):
    monkeypatch.setenv("PROFILE_PARALLEL_COUNT", "16")
    assert sim._resolve_profile_parallel_count(None) == 16


def test_default_is_8_when_unset(monkeypatch):
    monkeypatch.delenv("PROFILE_PARALLEL_COUNT", raising=False)
    assert sim._resolve_profile_parallel_count(None) == 8


def test_garbage_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PROFILE_PARALLEL_COUNT", "not-a-number")
    assert sim._resolve_profile_parallel_count(None) == 8
