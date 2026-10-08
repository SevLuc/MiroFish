"""The config-generation step of ``prepare`` survives an LLM call that never returns.

Regression for 2026-10-08 (all three clusters, both Cloud Run attempts): persona generation
completed under its fork #15 cap, then the config step (time config -> event config -> agent-config
batches) waited 10 to 50 minutes on single calls OpenRouter had routed to a slow, heavily-reasoning
provider. Its 60 s ``CONFIG_LLM_TIMEOUT_SECONDS`` was only a socket read timeout, which never fires
while the gateway keeps the connection fed (calls of 500 to 2 281 s completed through it), so one
such call per cluster held the whole stage until the consumer's 3600 s prepare watchdog killed the
task. The AI cluster had survived the identical pattern on 10-07 with nine minutes to spare.

Contract under test:
  * one config-generation attempt costs at most CONFIG_LLM_TIMEOUT_SECONDS of wall clock, then
    counts as a failed attempt (3 attempts, lower temperature each time);
  * when the attempts run out, the time config / an agent batch falls back to the rule-based
    config, the step does NOT raise, and the degradation is reported through
    ``SimulationParameters.warnings`` (CONFIG_DEGRADED per call + CONFIG_DEGRADED_SUMMARY);
  * every config call carries the OpenRouter reasoning preference (default: off), overridable
    per env, because the slow answers were 8 K to 47 K reasoning tokens for ~1 K tokens of JSON.
"""
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")

from app.services import config_batches as cb  # noqa: E402
from app.services import simulation_config_generator as scg  # noqa: E402
from app.services.zep_entity_reader import EntityNode  # noqa: E402
from app.utils import openai_chat_compat  # noqa: E402

TIME_CFG = {"total_simulation_hours": 24, "minutes_per_round": 60, "agents_per_hour_min": 1,
            "agents_per_hour_max": 3, "peak_hours": [19, 20], "off_peak_hours": [1, 2],
            "morning_hours": [7], "work_hours": [9, 10], "reasoning": "ok"}
EVENT_CFG = {"hot_topics": ["ai"], "narrative_direction": "up",
             "initial_posts": [{"content": "NVDA beats", "poster_type": "Company"}],
             "reasoning": "ok"}


def _resp(payload, finish_reason="stop"):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                    message=SimpleNamespace(content=content))])


def _entities(n):
    return [EntityNode(uuid=f"u{i}", name=f"Ent{i}", labels=["Entity", "Company"],
                       summary=f"Ent{i} summary", attributes={}) for i in range(n)]


@pytest.fixture
def release():
    event = threading.Event()
    yield event
    event.set()          # free any wedged call threads


@pytest.fixture
def fast(monkeypatch):
    """A tiny wall-clock cap and no retry sleeps, so a hang costs milliseconds in the test."""
    monkeypatch.setenv("CONFIG_LLM_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setattr(scg.time, "sleep", lambda s: None)


# ------------------------------------------------------------------ the knobs (pure module)
def test_reasoning_effort_env():
    assert cb.resolve_config_reasoning_effort({}) == "none"
    assert cb.resolve_config_reasoning_effort({"CONFIG_LLM_REASONING_EFFORT": "off"}) == "none"
    assert cb.resolve_config_reasoning_effort({"CONFIG_LLM_REASONING_EFFORT": "NONE"}) == "none"
    assert cb.resolve_config_reasoning_effort({"CONFIG_LLM_REASONING_EFFORT": "provider"}) is None
    assert cb.resolve_config_reasoning_effort({"CONFIG_LLM_REASONING_EFFORT": "default"}) is None
    assert cb.resolve_config_reasoning_effort({"CONFIG_LLM_REASONING_EFFORT": "low"}) == "low"
    assert cb.resolve_config_reasoning_effort({"CONFIG_LLM_REASONING_EFFORT": "bogus"}) == "none"


def test_reasoning_extra_body():
    assert cb.reasoning_extra_body("none") == {"reasoning": {"effort": "none"}}
    assert cb.reasoning_extra_body("low") == {"reasoning": {"effort": "low"}}
    assert cb.reasoning_extra_body(None) is None
    assert cb.reasoning_extra_body("") is None


# ------------------------------------------------------------------ extra_body passthrough
class _Recorder:
    def __init__(self, result):
        self.result, self.calls = result, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _client(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def test_create_chat_completion_forwards_extra_body():
    rec = _Recorder(_resp({"ok": 1}))
    openai_chat_compat.create_chat_completion(
        _client(rec), model="deepseek/deepseek-v4-flash", messages=[],
        extra_body={"reasoning": {"effort": "none"}})
    assert rec.calls[0]["extra_body"] == {"reasoning": {"effort": "none"}}

    rec = _Recorder(_resp({"ok": 1}))
    openai_chat_compat.create_chat_completion(_client(rec), model="m", messages=[])
    assert "extra_body" not in rec.calls[0]          # legacy request shape untouched


def test_config_calls_carry_reasoning_off_by_default(monkeypatch):
    monkeypatch.delenv("CONFIG_LLM_REASONING_EFFORT", raising=False)
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    seen = []

    def fake_create(client, **kwargs):
        seen.append(kwargs)
        return _resp(TIME_CFG)
    monkeypatch.setattr(scg, "create_chat_completion", fake_create)
    assert gen._call_llm_with_retry("p", "s", stage="time_config") == TIME_CFG
    assert seen[0]["extra_body"] == {"reasoning": {"effort": "none"}}
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_config_calls_can_leave_reasoning_to_the_provider(monkeypatch):
    monkeypatch.setenv("CONFIG_LLM_REASONING_EFFORT", "provider")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    seen = []

    def fake_create(client, **kwargs):
        seen.append(kwargs)
        return _resp(TIME_CFG)
    monkeypatch.setattr(scg, "create_chat_completion", fake_create)
    gen._call_llm_with_retry("p", "s")
    assert seen[0]["extra_body"] is None


# ------------------------------------------------------------------ the client itself
def test_client_has_no_sdk_retries_and_read_timeout_equals_cap(monkeypatch):
    monkeypatch.setenv("CONFIG_LLM_TIMEOUT_SECONDS", "45")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    assert gen.config_llm_timeout == 45.0
    assert gen.client.max_retries == 0
    assert gen.client.timeout.read == 45.0
    assert gen.client.timeout.connect == 30.0


# ------------------------------------------------------------------ the deadline
def test_hung_call_is_abandoned_after_the_cap_and_retried(monkeypatch, fast, release):
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    started = []

    def fake_create(client, **kwargs):
        started.append(time.monotonic())
        release.wait()                       # never answers until the test lets it
        return _resp(TIME_CFG)
    monkeypatch.setattr(scg, "create_chat_completion", fake_create)

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        gen._call_llm_with_retry("p", "s", stage="time_config")
    elapsed = time.monotonic() - t0
    assert len(started) == 3                 # three bounded attempts
    assert elapsed < 2.0                     # ~3 x 0.05 s, not 3 x "forever"


def test_slow_first_attempt_then_good_answer(monkeypatch, fast):
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    calls = []

    def fake_create(client, **kwargs):
        calls.append(kwargs["temperature"])
        if len(calls) == 1:
            threading.Event().wait(0.3)      # over the 0.05 s cap: abandoned (time.sleep is patched)
        return _resp(TIME_CFG)
    monkeypatch.setattr(scg, "create_chat_completion", fake_create)
    assert gen._call_llm_with_retry("p", "s") == TIME_CFG
    assert calls[:2] == [0.7, pytest.approx(0.6)]   # retry at lower temperature


# ------------------------------------------------------------------ fallbacks + reporting
def _scripted(monkeypatch, hang_stage_substrings, release):
    """Fake LLM: good answers for every call except those whose prompt contains one of the
    substrings, which hang (abandoned by the deadline)."""
    def fake_create(client, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if any(s in prompt for s in hang_stage_substrings):
            release.wait()
            return _resp({})
        if "agent_configs" in prompt:
            ids = [e["agent_id"] for e in json.loads(prompt.split("```json")[1].split("```")[0])]
            return _resp({"agent_configs": [{"agent_id": i, "activity_level": 0.9, "posts_per_hour": 2,
                                             "comments_per_hour": 2, "active_hours": [9],
                                             "response_delay_min": 1, "response_delay_max": 2,
                                             "sentiment_bias": 0.8, "stance": "supportive",
                                             "influence_weight": 2.0} for i in ids]})
        if "initial_posts" in prompt:
            return _resp(EVENT_CFG)
        return _resp(TIME_CFG)
    monkeypatch.setattr(scg, "create_chat_completion", fake_create)


def _generate(gen, n):
    return gen.generate_config(simulation_id="sim", project_id="p", graph_id="g",
                               simulation_requirement="req", document_text="doc",
                               entities=_entities(n), enable_twitter=True, enable_reddit=True)


def test_clean_run_has_no_config_warnings(monkeypatch, fast, release):
    monkeypatch.setenv("AGENTS_PER_BATCH", "2")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    _scripted(monkeypatch, [], release)
    params = _generate(gen, 5)
    assert len(params.agent_configs) == 5
    assert all(c.stance == "supportive" for c in params.agent_configs)
    assert [w for w in params.warnings if w.startswith("CONFIG_DEGRADED")] == []


def test_hung_agent_batch_falls_back_to_rules_and_is_reported(monkeypatch, fast, release):
    monkeypatch.setenv("AGENTS_PER_BATCH", "2")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    _scripted(monkeypatch, ['"entity_name": "Ent2"'], release)   # the batch holding Ent2/Ent3 hangs

    t0 = time.monotonic()
    params = _generate(gen, 5)
    assert time.monotonic() - t0 < 3.0                    # bounded, not a 3600 s watchdog
    assert len(params.agent_configs) == 5                  # nobody is skipped
    by_name = {c.entity_name: c for c in params.agent_configs}
    assert by_name["Ent0"].stance == "supportive"           # LLM answer kept where it arrived
    assert by_name["Ent2"].stance == "neutral"              # rule-based fallback for the hung batch
    assert by_name["Ent3"].sentiment_bias == 0.0

    degraded = [w for w in params.warnings if w.startswith("CONFIG_DEGRADED:")]
    assert len(degraded) == 1
    record = json.loads(degraded[0].split(": ", 1)[1])
    assert record["stage"] == "agent_config" and record["reason"] == "timeout"
    assert record["agents"] == 2 and record["entities"] == ["Ent2", "Ent3"]
    summary = json.loads([w for w in params.warnings if w.startswith("CONFIG_DEGRADED_SUMMARY:")][0]
                         .split(": ", 1)[1])
    assert summary == {"degraded_agents": 2, "total_agents": 5, "fraction": 0.4,
                       "time_config_defaulted": False, "calls": 1}


def test_hung_time_config_uses_default_and_is_reported(monkeypatch, fast, release):
    monkeypatch.setenv("AGENTS_PER_BATCH", "5")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    _scripted(monkeypatch, ["total_simulation_hours"], release)   # the time-config prompt hangs
    params = _generate(gen, 3)
    assert params.time_config.minutes_per_round == 60          # the built-in default
    summary = json.loads([w for w in params.warnings if w.startswith("CONFIG_DEGRADED_SUMMARY:")][0]
                         .split(": ", 1)[1])
    assert summary["time_config_defaulted"] is True and summary["degraded_agents"] == 0


def test_degradations_reset_between_runs(monkeypatch, fast, release):
    monkeypatch.setenv("AGENTS_PER_BATCH", "5")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    _scripted(monkeypatch, ["total_simulation_hours"], release)
    assert any(w.startswith("CONFIG_DEGRADED") for w in _generate(gen, 2).warnings)
    release.set()
    _scripted(monkeypatch, [], release)
    assert not any(w.startswith("CONFIG_DEGRADED") for w in _generate(gen, 2).warnings)
