import threading

import pytest

from app.services import ontology_generator
from app.services.ontology_generator import OntologyGenerator
from app.utils.llm_client import LLMResponseError


class RecordingLLMClient:
    def __init__(self):
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "entity_types": [],
            "edge_types": [],
            "analysis_summary": "ok",
        }


def _generator_for_test() -> OntologyGenerator:
    generator = OntologyGenerator(llm_client=object())
    generator.MAX_TEXT_LENGTH_FOR_LLM = 2000
    generator.LONG_TEXT_CHUNK_SIZE = 500
    generator.LONG_TEXT_CHUNK_OVERLAP = 0
    generator.MAX_LONG_TEXT_CHUNKS = 3
    generator.MIN_LONG_TEXT_EXCERPT = 120
    return generator


def test_short_ontology_context_keeps_original_text():
    generator = _generator_for_test()

    context = generator._build_document_context(["short document body"])

    assert context == "short document body"
    assert "长文本自动分块摘要" not in context


def test_long_ontology_context_samples_across_document():
    generator = _generator_for_test()
    long_text = "BEGIN" + ("a" * 1050) + "MIDDLE" + ("b" * 1050) + "END"

    context = generator._build_document_context([long_text])

    assert len(context) <= generator.MAX_TEXT_LENGTH_FOR_LLM
    assert "长文本自动分块摘要" in context
    assert "BEGIN" in context
    assert "MIDDLE" in context
    assert "END" in context
    assert "分块 1/" in context
    assert "分块 3/" in context
    assert "分块 5/" in context


def test_very_long_ontology_context_selects_representative_chunks():
    generator = _generator_for_test()
    chunks = ["BEGIN"] + [
        f"CHUNK{i:02d}-" + (str(i) * 490)
        for i in range(12)
    ] + ["FINALEND"]
    long_text = "".join(chunks)

    context = generator._build_document_context([long_text])

    assert len(context) <= generator.MAX_TEXT_LENGTH_FOR_LLM
    assert "BEGIN" in context
    assert "FINALEND" in context
    assert context.count("--- 文档 1 / 分块") == generator.MAX_LONG_TEXT_CHUNKS


def test_ontology_generation_does_not_cap_structured_output_tokens():
    llm = RecordingLLMClient()
    generator = OntologyGenerator(llm_client=llm)

    result = generator.generate(
        document_texts=["A short source document."],
        simulation_requirement="Simulate the public discussion.",
    )

    assert result["analysis_summary"] == "ok"
    assert llm.calls[0]["max_tokens"] is None
    # One request per chat_json call: the retry loop lives in generate(), where it is bounded.
    assert llm.calls[0]["max_attempts"] == 1


_OK = {"entity_types": [], "edge_types": [], "analysis_summary": "ok"}


class ScriptedLLMClient:
    """chat_json plays one scripted outcome per call: an exception to raise, an Event to hang
    on (a request the provider never answers), or the dict to return."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def chat_json(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, threading.Event):
            outcome.wait(5)
            return {}
        return outcome


def _generate(llm):
    return OntologyGenerator(llm_client=llm).generate(
        document_texts=["A short source document."],
        simulation_requirement="Simulate the public discussion.",
    )


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(ontology_generator, "backoff_delay", lambda attempt: 0)


def test_ontology_retries_an_unusable_response(no_backoff):
    # 2026-10-02: a truncated response, then malformed JSON on the single retry, failed the task.
    llm = ScriptedLLMClient(
        LLMResponseError("truncated at the token limit", finish_reason="length"),
        LLMResponseError("invalid JSON (line 136, column 13)"),
        _OK,
    )

    assert _generate(llm)["analysis_summary"] == "ok"
    assert llm.calls == 3


def test_ontology_abandons_a_call_that_overruns_the_wall_clock_cap(monkeypatch, no_backoff):
    monkeypatch.setenv("ONTOLOGY_LLM_TIMEOUT_SECONDS", "0.05")
    hung = threading.Event()
    llm = ScriptedLLMClient(hung, _OK)

    try:
        assert _generate(llm)["analysis_summary"] == "ok"
    finally:
        hung.set()
    assert llm.calls == 2


def test_ontology_fails_loud_once_every_attempt_overran(monkeypatch, no_backoff):
    monkeypatch.setenv("ONTOLOGY_LLM_TIMEOUT_SECONDS", "0.05")
    hung = threading.Event()
    llm = ScriptedLLMClient(hung, hung, hung)

    try:
        with pytest.raises(LLMResponseError):  # the API answers 502 on this, as before
            _generate(llm)
    finally:
        hung.set()
    assert llm.calls == 3


def test_ontology_does_not_retry_a_provider_error(no_backoff):
    llm = ScriptedLLMClient(RuntimeError("HTTP 401"), _OK)

    with pytest.raises(RuntimeError):
        _generate(llm)
    assert llm.calls == 1
