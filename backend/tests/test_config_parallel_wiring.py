"""SimulationConfigGenerator.generate_config runs its agent-config batches through the parallel
runner: env knobs reach the generator and the OpenAI client, batches run concurrently, and the
agent_ids come back contiguous and in order. Every LLM-touching method is stubbed, so this needs
the fork's Python deps but no network."""
import os
import threading
import time

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")

from app.services import simulation_config_generator as scg  # noqa: E402
from app.services.zep_entity_reader import EntityNode  # noqa: E402


def _entities(n):
    return [EntityNode(uuid=f"u{i}", name=f"Entity {i}", labels=["Entity", "Person"],
                       summary="s", attributes={}) for i in range(n)]


def _stub_llm_steps(gen, monkeypatch, batch_latency, tracker):
    monkeypatch.setattr(gen, "_generate_time_config",
                        lambda context, n: {"total_simulation_hours": 24, "minutes_per_round": 60,
                                            "agents_per_hour_min": 2, "agents_per_hour_max": 5,
                                            "reasoning": "stub"})
    monkeypatch.setattr(gen, "_generate_event_config",
                        lambda context, req, entities: {"initial_posts": [], "reasoning": "stub"})
    lock = threading.Lock()

    def fake_batch(context, entities, start_idx, simulation_requirement):
        with lock:
            tracker["in_flight"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["in_flight"])
        time.sleep(batch_latency)
        with lock:
            tracker["in_flight"] -= 1
        return [scg.AgentActivityConfig(agent_id=start_idx + i, entity_uuid=e.uuid, entity_name=e.name,
                                        entity_type="Person") for i, e in enumerate(entities)]
    monkeypatch.setattr(gen, "_generate_agent_configs_batch", fake_batch)


def _generate(gen, n):
    return gen.generate_config(simulation_id="sim_t", project_id="p", graph_id="g",
                               simulation_requirement="req", document_text="doc",
                               entities=_entities(n))


def test_env_knobs_reach_the_generator_and_the_client(monkeypatch):
    monkeypatch.setenv("AGENTS_PER_BATCH", "10")
    monkeypatch.setenv("CONFIG_LLM_TIMEOUT_SECONDS", "90")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    assert gen.agents_per_batch == 10
    assert gen.config_llm_timeout == 90.0
    assert float(gen.client.timeout) == 90.0


def test_defaults_when_env_unset(monkeypatch):
    for k in ("AGENTS_PER_BATCH", "CONFIG_LLM_TIMEOUT_SECONDS", "CONFIG_PARALLEL_COUNT"):
        monkeypatch.delenv(k, raising=False)
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    assert gen.agents_per_batch == 15 and gen.config_llm_timeout == 180.0


def test_batches_run_concurrently_and_ids_stay_contiguous(monkeypatch):
    monkeypatch.setenv("AGENTS_PER_BATCH", "15")
    monkeypatch.delenv("CONFIG_PARALLEL_COUNT", raising=False)       # all batches at once
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    tracker = {"in_flight": 0, "peak": 0}
    _stub_llm_steps(gen, monkeypatch, batch_latency=0.2, tracker=tracker)

    t0 = time.perf_counter()
    params = _generate(gen, 262)                                        # 18 batches
    wall = time.perf_counter() - t0

    print(f"\n262 agents / 18 batches x 0.2 s fake latency: wall {wall:.2f} s, peak in-flight {tracker['peak']}")
    assert [c.agent_id for c in params.agent_configs] == list(range(262))
    assert tracker["peak"] >= 10                                        # really concurrent
    assert wall < 1.0                                                   # serial would be 3.6 s


def test_parallel_count_env_caps_concurrency(monkeypatch):
    monkeypatch.setenv("AGENTS_PER_BATCH", "15")
    monkeypatch.setenv("CONFIG_PARALLEL_COUNT", "3")
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    tracker = {"in_flight": 0, "peak": 0}
    _stub_llm_steps(gen, monkeypatch, batch_latency=0.05, tracker=tracker)
    params = _generate(gen, 90)                                         # 6 batches, 3 wide
    assert tracker["peak"] <= 3
    assert [c.agent_id for c in params.agent_configs] == list(range(90))


def test_progress_callback_reports_each_batch(monkeypatch):
    monkeypatch.setenv("AGENTS_PER_BATCH", "15")
    monkeypatch.delenv("CONFIG_PARALLEL_COUNT", raising=False)
    gen = scg.SimulationConfigGenerator(api_key="k", base_url="http://localhost:1", model_name="m")
    _stub_llm_steps(gen, monkeypatch, batch_latency=0.01, tracker={"in_flight": 0, "peak": 0})
    steps = []
    gen.generate_config(simulation_id="s", project_id="p", graph_id="g", simulation_requirement="r",
                        document_text="d", entities=_entities(45),
                        progress_callback=lambda cur, total, msg: steps.append((cur, total)))
    # time config (1), event config (2), three batches (3..5), platform config (6 = total)
    assert steps[0] == (1, 6) and steps[1] == (2, 6) and steps[-1] == (6, 6)
    assert sorted(s for s, _ in steps[2:5]) == [3, 4, 5]
