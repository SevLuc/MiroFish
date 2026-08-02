"""Reliability of Graphiti structured extraction on OpenRouter (ADR 0009 follow-up).

Root cause pinned by a real 56-ticker AI cold-start crash: the backend drove graphiti-core's
``OpenAIGenericClient`` in ``json_object`` mode (schema only *prompt-injected*, never sent to the
API), so ``gpt-4o-mini`` was free to omit required fields — ``EdgeDuplicate.duplicate_facts`` /
``contradicted_facts`` — and ``resolve_extracted_edge`` raised ``ValidationError``, killing the run
(worker ``maxRetries=0``). graphiti itself does not re-prompt on ``ValidationError``.

These tests pin the two-part fix, both pure (no FalkorDB / LLM / graphiti_core needed):
  * ``_structured_mode`` defaults to ``json_schema`` (schema sent to the API), overridable.
  * ``_generate_validated`` validates each response against the response_model and retries a
    bounded number of times, failing loud only after exhaustion — so one dropped-field response
    can't kill a multi-thousand-edge build.
"""

import asyncio

import pytest
from pydantic import BaseModel

from app.services.graphiti_backend import (
    _generate_validated,
    _structured_mode,
    _validation_retries,
)


class _EdgeDuplicate(BaseModel):
    """Mirror of graphiti's EdgeDuplicate: both fields required (Field(...))."""

    duplicate_facts: list[int]
    contradicted_facts: list[int]


# -- _structured_mode ---------------------------------------------------------

def test_structured_mode_defaults_to_json_schema(monkeypatch):
    monkeypatch.delenv("GRAPHITI_STRUCTURED", raising=False)
    assert _structured_mode() == "json_schema"


def test_structured_mode_env_override_wins(monkeypatch):
    monkeypatch.setenv("GRAPHITI_STRUCTURED", "json_object")
    assert _structured_mode() == "json_object"


# -- _validation_retries ------------------------------------------------------

def test_validation_retries_default_is_at_least_two(monkeypatch):
    monkeypatch.delenv("GRAPHITI_VALIDATION_RETRIES", raising=False)
    assert _validation_retries() >= 2


def test_validation_retries_env_override(monkeypatch):
    monkeypatch.setenv("GRAPHITI_VALIDATION_RETRIES", "5")
    assert _validation_retries() == 5


def test_validation_retries_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("GRAPHITI_VALIDATION_RETRIES", "not-an-int")
    assert _validation_retries() >= 2


# -- _generate_validated ------------------------------------------------------

def test_returns_first_valid_response_without_retry():
    calls = {"n": 0}

    async def gen():
        calls["n"] += 1
        return {"duplicate_facts": [1], "contradicted_facts": []}

    result = asyncio.run(_generate_validated(gen, _EdgeDuplicate, attempts=3))
    assert result == {"duplicate_facts": [1], "contradicted_facts": []}
    assert calls["n"] == 1


def test_retries_on_missing_field_then_succeeds():
    """The exact crash: first response drops the required fields, retry returns them."""
    responses = [
        {"duplicate_facts": [0]},  # contradicted_facts missing -> ValidationError
        {"duplicate_facts": [0], "contradicted_facts": [2]},
    ]
    calls = {"n": 0}

    async def gen():
        r = responses[calls["n"]]
        calls["n"] += 1
        return r

    result = asyncio.run(_generate_validated(gen, _EdgeDuplicate, attempts=3))
    assert result == {"duplicate_facts": [0], "contradicted_facts": [2]}
    assert calls["n"] == 2


def test_raises_validation_error_after_exhausting_attempts():
    calls = {"n": 0}

    async def gen():
        calls["n"] += 1
        return {"duplicate_facts": [0]}  # always missing contradicted_facts

    with pytest.raises(Exception) as exc:
        asyncio.run(_generate_validated(gen, _EdgeDuplicate, attempts=3))
    assert "contradicted_facts" in str(exc.value)
    assert calls["n"] == 3  # tried the full budget, then failed loud


def test_no_response_model_returns_first_result_unvalidated():
    calls = {"n": 0}

    async def gen():
        calls["n"] += 1
        return {"anything": True}

    result = asyncio.run(_generate_validated(gen, None, attempts=3))
    assert result == {"anything": True}
    assert calls["n"] == 1
