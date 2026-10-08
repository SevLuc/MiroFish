"""The seed-post (event config) guard in SimulationConfigGenerator.

Regression for 2026-09-10: the event-config LLM output hit the token cap, the JSON repair
"succeeded" into a list of bare strings, and _assign_initial_post_agents crashed on .get(), taking
the whole prepare stage (and the trading day) with it. The guard is ordered so that dropping is
the last resort and never silent: validate shape -> retry (lower temperature) -> retry with a
length cap -> keep well-formed posts and flag EVENT_CONFIG_DEGRADED / EVENT_CONFIG_EMPTY.
"""
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")

from app.services import simulation_config_generator as scg  # noqa: E402
from app.services.simulation_manager import SimulationState  # noqa: E402
from app.services.zep_entity_reader import EntityNode  # noqa: E402


def _resp(payload, finish_reason="stop"):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                    message=SimpleNamespace(content=content))])


GOOD = {"hot_topics": ["ai"], "narrative_direction": "up",
        "initial_posts": [{"content": "NVDA beats", "poster_type": "MediaOutlet"},
                          {"content": "INTC guides down", "poster_type": "MediaOutlet"}],
        "reasoning": "ok"}
BAD_STRINGS = {"hot_topics": ["ai"], "narrative_direction": "up",
               "initial_posts": ["NVDA beats", {"content": "INTC guides down", "poster_type": "MediaOutlet"}, 42],
               "reasoning": "truncated"}


def _gen(monkeypatch, responses):
    """Generator whose LLM returns the scripted responses in order (last one repeats)."""
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    calls = {"prompts": []}
    it = iter(responses)
    last = responses[-1]

    def fake_create(client, *, model, messages, temperature=None, max_tokens=None, response_format=None, **kwargs):
        calls["prompts"].append(messages[1]["content"])
        nonlocal last
        try:
            last = next(it)
        except StopIteration:
            pass
        return last
    monkeypatch.setattr(scg, "create_chat_completion", fake_create)
    monkeypatch.setattr(scg.time, "sleep", lambda s: None) if hasattr(scg, "time") else None
    return gen, calls


def _entities():
    return [EntityNode(uuid="u1", name="NVIDIA", labels=["Entity", "MediaOutlet"], summary="", attributes={})]


# ---------------------------------------------------------------- shape validation
def test_well_formed_check():
    assert scg.SimulationConfigGenerator._event_config_is_well_formed(GOOD)
    assert not scg.SimulationConfigGenerator._event_config_is_well_formed(BAD_STRINGS)
    assert not scg.SimulationConfigGenerator._event_config_is_well_formed({"initial_posts": "nope"})
    assert not scg.SimulationConfigGenerator._event_config_is_well_formed({"initial_posts": [{"content": ""}]})
    assert scg.SimulationConfigGenerator._event_config_is_well_formed({"initial_posts": []})
    assert not scg.SimulationConfigGenerator._event_config_is_well_formed("garbage")


def test_salvage_keeps_only_well_formed_posts():
    kept, dropped = scg.SimulationConfigGenerator._salvage_initial_posts(BAD_STRINGS)
    assert kept == [{"content": "INTC guides down", "poster_type": "MediaOutlet"}] and dropped == 2
    assert scg.SimulationConfigGenerator._salvage_initial_posts(None) == ([], 0)


# ---------------------------------------------------------------- the ladder
def test_malformed_first_pass_is_retried_and_good_result_wins(monkeypatch):
    gen, calls = _gen(monkeypatch, [_resp(BAD_STRINGS, finish_reason="length"), _resp(GOOD)])
    gen.event_config_warnings = []
    out = gen._generate_event_config("ctx", "req", _entities())
    assert out["initial_posts"] == GOOD["initial_posts"]
    assert gen.event_config_warnings == []                      # plain retry, nothing to flag
    assert len(calls["prompts"]) == 2 and "输出长度限制" not in calls["prompts"][1]


def test_persistent_malformed_output_triggers_the_short_retry(monkeypatch):
    gen, calls = _gen(monkeypatch, [_resp(BAD_STRINGS)] * 3 + [_resp(GOOD)])
    gen.event_config_warnings = []
    out = gen._generate_event_config("ctx", "req", _entities())
    assert out["initial_posts"] == GOOD["initial_posts"]
    assert len(calls["prompts"]) == 4
    assert "输出长度限制" in calls["prompts"][3]                  # the 4th call asked for fewer, shorter posts
    assert any(w.startswith("EVENT_CONFIG_SHORTENED") for w in gen.event_config_warnings)


def test_all_attempts_malformed_keeps_well_formed_posts_and_flags_degraded(monkeypatch):
    gen, calls = _gen(monkeypatch, [_resp(BAD_STRINGS)])           # every call returns junk
    gen.event_config_warnings = []
    out = gen._generate_event_config("ctx", "req", _entities())
    assert out["initial_posts"] == [{"content": "INTC guides down", "poster_type": "MediaOutlet"}]
    assert out["hot_topics"] == ["ai"]
    assert len(calls["prompts"]) == 6                            # 3 normal + 3 short
    assert any(w.startswith("EVENT_CONFIG_DEGRADED: kept 1") and "dropped 2" in w
               for w in gen.event_config_warnings)


def test_nothing_salvageable_flags_empty_not_crash(monkeypatch):
    gen, _ = _gen(monkeypatch, [_resp({"initial_posts": ["a", "b"]})])
    gen.event_config_warnings = []
    out = gen._generate_event_config("ctx", "req", _entities())
    assert out["initial_posts"] == []
    assert any(w.startswith("EVENT_CONFIG_EMPTY") for w in gen.event_config_warnings)


def test_llm_error_falls_through_the_ladder_to_empty(monkeypatch):
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    gen.event_config_warnings = []

    def boom(*a, **k):
        raise RuntimeError("provider down")
    monkeypatch.setattr(scg, "create_chat_completion", boom)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = gen._generate_event_config("ctx", "req", _entities())
    assert out["initial_posts"] == []
    assert any(w.startswith("EVENT_CONFIG_EMPTY") for w in gen.event_config_warnings)


# ---------------------------------------------------------------- downstream never crashes
def test_assign_initial_post_agents_skips_non_dict_posts():
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    ec = scg.EventConfig(initial_posts=["bare string", {"content": "ok", "poster_type": "MediaOutlet"}, 7])
    agents = [scg.AgentActivityConfig(agent_id=0, entity_uuid="u", entity_name="Reuters", entity_type="MediaOutlet")]
    out = gen._assign_initial_post_agents(ec, agents)               # would have raised AttributeError
    assert len(out.initial_posts) == 1 and out.initial_posts[0]["poster_agent_id"] == 0


# ---------------------------------------------------------------- warnings reach the state
def test_warnings_are_carried_on_params_and_state(monkeypatch):
    gen, _ = _gen(monkeypatch, [_resp(BAD_STRINGS)])
    monkeypatch.setattr(gen, "_generate_time_config",
                        lambda c, n: {"total_simulation_hours": 24, "minutes_per_round": 60,
                                      "agents_per_hour_min": 1, "agents_per_hour_max": 2, "reasoning": ""})
    monkeypatch.setattr(gen, "_generate_agent_configs_batch",
                        lambda context, entities, start_idx, simulation_requirement: [
                            scg.AgentActivityConfig(agent_id=start_idx + i, entity_uuid=e.uuid, entity_name=e.name,
                                                    entity_type="MediaOutlet") for i, e in enumerate(entities)])
    params = gen.generate_config(simulation_id="s", project_id="p", graph_id="g", simulation_requirement="r",
                                 document_text="d", entities=_entities())
    assert any(w.startswith("EVENT_CONFIG_DEGRADED") for w in params.warnings)
    assert "EVENT_CONFIG_DEGRADED" in params.generation_reasoning
    assert json.loads(params.to_json())["warnings"] == params.warnings

    state = SimulationState(simulation_id="s", project_id="p", graph_id="g", prepare_warnings=params.warnings)
    assert state.to_simple_dict()["prepare_warnings"] == params.warnings
    assert state.to_dict()["prepare_warnings"] == params.warnings
