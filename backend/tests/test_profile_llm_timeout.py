"""The persona (profile) step survives an LLM call that never returns.

Regression for 2026-09-29 (AI cluster, 282 personas at concurrency 16): 281 personas finished,
the 282nd (GOOGL, the largest-context entity) had its request accepted and then never answered.
Nothing bounded that call by wall clock, so its worker thread blocked, the barrier waited on it,
and the whole prepare stage sat until the consumer's 3600 s watchdog killed the task.

Contract under test:
  * one call costs at most PROFILE_LLM_TIMEOUT_SECONDS of wall clock, then raises (retryable);
  * a failed call is re-sent (3 attempts, backoff, a fresh client after a transport failure);
  * when the attempts run out the entity gets the rule-based persona, the step does NOT raise,
    and the degradation is reported (entity, type, node degree, reason) through prepare_warnings;
  * the barrier therefore completes N/N with N-1 LLM personas and one flagged stub.
"""
import json
import os
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")

from app.services import oasis_profile_generator as opg  # noqa: E402
from app.services import simulation_manager as simulation_manager_module  # noqa: E402
from app.services.simulation_manager import (  # noqa: E402
    SimulationManager,
    SimulationState,
    SimulationStatus,
)
from app.services.zep_entity_reader import EntityNode, FilteredEntities  # noqa: E402

PERSONA = {"bio": "Search and cloud giant.", "persona": "Alphabet speaks through one account.",
           "age": 30, "gender": "other", "mbti": "ISTJ", "country": "US",
           "profession": "Company", "interested_topics": ["AI"]}


def _resp(payload, finish_reason="stop"):
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                    message=SimpleNamespace(content=content))])


def _entity(name, degree=0, entity_type="Company"):
    return EntityNode(uuid=f"u-{name}", name=name, labels=["Entity", entity_type],
                      summary=f"{name} summary", attributes={},
                      related_edges=[{"fact": f"{name} fact {i}"} for i in range(degree)])


class _Hang:
    """Marker: this scripted call never returns (until the test releases it)."""


@pytest.fixture
def release():
    event = threading.Event()
    yield event
    event.set()          # let any wedged call threads finish


def _gen(monkeypatch, release, script, timeout="0.2"):
    """Generator whose LLM follows ``script(entity_name, call_no)`` -> response | Exception | _Hang."""
    monkeypatch.setenv("PROFILE_LLM_TIMEOUT_SECONDS", timeout)
    gen = opg.OasisProfileGenerator(api_key="k", base_url="http://localhost:1", model_name="m",
                                    graph_id=None)
    gen.zep_client = None
    monkeypatch.setattr(gen, "_search_zep_for_entity",
                        lambda entity: {"facts": [], "node_summaries": [], "context": ""})
    calls = []
    lock = threading.Lock()

    def fake_create(client, *, model, messages, temperature=None, max_tokens=None,
                    response_format=None):
        prompt = messages[1]["content"]
        with lock:
            name = next((n for n in script.names if n in prompt), "?")
            call_no = sum(1 for c in calls if c["entity"] == name) + 1
            calls.append({"entity": name, "client": client, "max_tokens": max_tokens})
        outcome = script(name, call_no)
        if outcome is _Hang:
            release.wait(30)
            raise RuntimeError("released by the test")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(opg, "create_chat_completion", fake_create)
    monkeypatch.setattr(opg.time, "sleep", lambda seconds: None)
    return gen, calls


def _script(names, fn):
    fn.names = list(names)
    return fn


def _llm(gen, name="GOOGL", degree=55):
    return gen._generate_profile_with_llm(
        entity_name=name, entity_type="Company", entity_summary=f"{name} summary",
        entity_attributes={}, context="", node_degree=degree)


# ---- the client --------------------------------------------------------------------------

def test_client_has_a_socket_timeout_equal_to_the_cap_and_no_hidden_sdk_retries(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script([], lambda n, c: _resp(PERSONA)), timeout="120")

    assert gen.profile_llm_timeout == 120.0
    assert gen.client.max_retries == 0
    assert isinstance(gen.client.timeout, httpx.Timeout)
    assert gen.client.timeout.read == 120.0
    assert gen.client.timeout.connect == 10.0


def test_default_cap_is_300_seconds(monkeypatch):
    monkeypatch.delenv("PROFILE_LLM_TIMEOUT_SECONDS", raising=False)
    gen = opg.OasisProfileGenerator(api_key="k", base_url="http://localhost:1", model_name="m")

    assert gen.profile_llm_timeout == 300.0
    assert gen.client.timeout.read == 300.0


# ---- (a) a call that never returns is bounded --------------------------------------------

def test_a_call_that_never_returns_costs_about_the_cap_per_attempt(monkeypatch, release):
    gen, calls = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _Hang))

    started = time.monotonic()
    result = _llm(gen)
    elapsed = time.monotonic() - started

    assert len(calls) == 3                       # 3 attempts, each bounded at 0.2 s
    assert elapsed < 3.0, f"3 wedged attempts must cost ~3 x cap, took {elapsed:.1f}s"
    assert result["persona"]                     # and the step still returned a persona


# ---- (b) retry re-sends and succeeds -----------------------------------------------------

def test_a_wedged_first_attempt_is_retried_and_the_second_answer_is_used(monkeypatch, release):
    gen, calls = _gen(monkeypatch, release,
                      _script(["GOOGL"], lambda n, c: _Hang if c == 1 else _resp(PERSONA)))

    result = _llm(gen)

    assert len(calls) == 2
    assert result["bio"] == PERSONA["bio"]
    assert gen.profile_degradations == []


def test_a_connection_error_is_retried(monkeypatch, release):
    request = httpx.Request("POST", "http://localhost:1/chat/completions")
    gen, calls = _gen(monkeypatch, release, _script(
        ["GOOGL"], lambda n, c: httpx.ConnectError("refused", request=request) if c < 3
        else _resp(PERSONA)))

    result = _llm(gen)

    assert len(calls) == 3
    assert result["bio"] == PERSONA["bio"]
    assert gen.profile_degradations == []


def test_the_retry_after_a_transport_failure_uses_a_fresh_client(monkeypatch, release):
    gen, calls = _gen(monkeypatch, release,
                      _script(["GOOGL"], lambda n, c: _Hang if c == 1 else _resp(PERSONA)))
    fresh = object()
    monkeypatch.setattr(gen, "_build_client", lambda: fresh)

    _llm(gen)

    assert calls[0]["client"] is gen.client
    assert calls[1]["client"] is fresh


def test_retries_back_off_about_one_then_four_seconds(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _Hang))
    slept = []
    monkeypatch.setattr(opg.time, "sleep", slept.append)

    _llm(gen)

    assert len(slept) == 2                       # between attempts only, not after the last
    assert 0.75 <= slept[0] <= 1.25
    assert 3.0 <= slept[1] <= 5.0


def test_output_length_is_left_uncapped(monkeypatch, release):
    gen, calls = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _resp(PERSONA)))

    _llm(gen)

    assert calls[0]["max_tokens"] is None


# ---- (c) attempts exhausted -> flagged stub, no raise ------------------------------------

def test_exhausted_attempts_return_the_rule_based_persona_and_record_the_degradation(
        monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _Hang))

    result = _llm(gen, degree=55)

    assert result["persona"] == "GOOGL summary"          # the rule-based default persona
    assert gen.profile_degradations == [
        {"entity": "GOOGL", "entity_type": "Company", "node_degree": 55, "reason": "timeout"}]


def test_the_degradation_is_surfaced_as_prepare_warnings(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _Hang))
    _llm(gen, degree=55)

    warnings = gen.degradation_warnings(total=282)

    assert [w.split(": ", 1)[0] for w in warnings] == ["PROFILE_DEGRADED", "PROFILE_DEGRADED_SUMMARY"]
    assert json.loads(warnings[0].split(": ", 1)[1]) == {
        "entity": "GOOGL", "entity_type": "Company", "node_degree": 55, "reason": "timeout"}


def test_unparseable_output_on_every_attempt_is_a_flagged_degradation_too(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _resp("")))

    result = _llm(gen)

    assert result["persona"]
    assert [d["reason"] for d in gen.profile_degradations] == ["bad_json"]


def test_a_healthy_call_records_no_degradation(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _resp(PERSONA)))

    _llm(gen)

    assert gen.profile_degradations == []
    assert gen.degradation_warnings(total=282) == []


# ---- (d) the barrier completes N-1/N -----------------------------------------------------

def test_barrier_completes_when_one_entity_never_answers(monkeypatch, release):
    names = ["NVDA", "MSFT", "GOOGL", "META", "AMD"]
    gen, _ = _gen(monkeypatch, release, _script(
        names, lambda n, c: _Hang if n == "GOOGL" else _resp(dict(PERSONA, bio=f"{n} llm bio"))))
    entities = [_entity(n, degree=55 if n == "GOOGL" else 3) for n in names]

    started = time.monotonic()
    profiles = gen.generate_profiles_from_entities(entities, use_llm=True, parallel_count=3)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"the barrier must not wait on the wedged entity, took {elapsed:.1f}s"
    assert [p.name for p in profiles] == names                      # N/N, order preserved
    by_name = {p.name: p for p in profiles}
    assert all(by_name[n].bio == f"{n} llm bio" for n in names if n != "GOOGL")   # N-1 from the LLM
    assert by_name["GOOGL"].persona == "GOOGL summary"              # 1 flagged stub
    assert gen.profile_degradations == [
        {"entity": "GOOGL", "entity_type": "Company", "node_degree": 55, "reason": "timeout"}]
    assert json.loads(gen.degradation_warnings(total=len(profiles))[-1].split(": ", 1)[1]) == {
        "degraded": 1, "total": 5, "fraction": 0.2}


def test_a_crash_outside_the_llm_call_is_flagged_as_well(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["NVDA", "GOOGL"], lambda n, c: _resp(PERSONA)))

    def boom(entity):
        if entity.name == "GOOGL":
            raise RuntimeError("context build failed")
        return ""
    monkeypatch.setattr(gen, "_build_entity_context", boom)

    profiles = gen.generate_profiles_from_entities(
        [_entity("NVDA", 2), _entity("GOOGL", 55)], use_llm=True, parallel_count=2)

    assert [p.name for p in profiles] == ["NVDA", "GOOGL"]
    assert gen.profile_degradations == [
        {"entity": "GOOGL", "entity_type": "Company", "node_degree": 55,
         "reason": "error:RuntimeError"}]


def test_degradations_are_reset_between_runs(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(
        ["GOOGL", "NVDA"], lambda n, c: _Hang if n == "GOOGL" else _resp(PERSONA)))

    gen.generate_profiles_from_entities([_entity("GOOGL", 55)], use_llm=True, parallel_count=1)
    gen.generate_profiles_from_entities([_entity("NVDA", 3)], use_llm=True, parallel_count=1)

    assert gen.profile_degradations == []


# ---- diagnosability: a stalled call must be visible in the logs --------------------------

class _Log:
    def __init__(self):
        self.lines = []

    def _add(self, msg, *args, **kwargs):
        self.lines.append(msg % args if args else msg)

    debug = info = warning = error = _add


def test_each_attempt_logs_a_start_and_an_end_with_timing(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release,
                  _script(["GOOGL"], lambda n, c: _Hang if c == 1 else _resp(PERSONA)))
    log = _Log()
    monkeypatch.setattr(opg, "logger", log)

    _llm(gen)

    profile_llm = [line for line in log.lines if line.startswith("PROFILE_LLM ")]
    assert [line.split()[1] for line in profile_llm] == ["start", "end", "start", "end"]
    assert 'entity="GOOGL"' in profile_llm[0] and "attempt=1/3" in profile_llm[0]
    assert "outcome=timeout" in profile_llm[1] and "elapsed=" in profile_llm[1]
    assert "attempt=2/3" in profile_llm[2]
    assert "outcome=ok" in profile_llm[3]


def test_a_degradation_is_logged_loudly(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script(["GOOGL"], lambda n, c: _Hang))
    log = _Log()
    monkeypatch.setattr(opg, "logger", log)

    _llm(gen, degree=55)

    degraded = [line for line in log.lines if line.startswith("PROFILE_DEGRADED ")]
    assert degraded == ['PROFILE_DEGRADED entity="GOOGL" entity_type=Company node_degree=55 '
                        'reason=timeout']


def test_response_headers_are_logged_as_first_byte(monkeypatch, release):
    """The event hook on the HTTP client: headers arriving = the provider answered, so a later
    silence is a slow/dripping body rather than a request that was never picked up."""
    gen, _ = _gen(monkeypatch, release, _script([], lambda n, c: _resp(PERSONA)))
    log = _Log()
    monkeypatch.setattr(opg, "logger", log)
    response = httpx.Response(200, request=httpx.Request("POST", "http://localhost:1/x"))

    with opg.profile_llm_call_context(entity="GOOGL", attempt=2):
        gen._log_first_byte(response)

    assert len(log.lines) == 1
    assert log.lines[0].startswith('PROFILE_LLM first_byte entity="GOOGL" attempt=2 ')
    assert "status=200" in log.lines[0] and "elapsed=" in log.lines[0]


def test_first_byte_hook_is_silent_outside_a_persona_call(monkeypatch, release):
    gen, _ = _gen(monkeypatch, release, _script([], lambda n, c: _resp(PERSONA)))
    log = _Log()
    monkeypatch.setattr(opg, "logger", log)

    gen._log_first_byte(httpx.Response(200, request=httpx.Request("POST", "http://localhost:1/x")))

    assert log.lines == []


# ---- prepare: profile warnings reach prepare_warnings, next to the config step's ---------

def test_prepare_merges_profile_degradations_with_config_warnings(tmp_path, monkeypatch):
    profile_warning = ('PROFILE_DEGRADED: {"entity": "GOOGL", "entity_type": "Company", '
                       '"node_degree": 55, "reason": "timeout"}')
    config_warning = "EVENT_CONFIG_DEGRADED: kept 1 well-formed initial posts, dropped 2"

    class Reader:
        def filter_defined_entities(self, **kwargs):
            return FilteredEntities(entities=[_entity("GOOGL", 55)], entity_types={"Company"},
                                    total_count=1, filtered_count=1)

    class ProfileGenerator:
        def __init__(self, graph_id=None):
            pass

        def generate_profiles_from_entities(self, entities, **kwargs):
            return [SimpleNamespace(name=e.name) for e in entities]

        def degradation_warnings(self, total=None):
            assert total == 1
            return [profile_warning]

        def save_profiles(self, **kwargs):
            pass

    class ConfigGenerator:
        def generate_config(self, **kwargs):
            return SimpleNamespace(to_json=lambda: "{}", generation_reasoning="ok",
                                   warnings=[config_warning])

    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_manager_module, "ZepEntityReader", Reader)
    monkeypatch.setattr(simulation_manager_module, "OasisProfileGenerator", ProfileGenerator)
    monkeypatch.setattr(simulation_manager_module, "SimulationConfigGenerator", ConfigGenerator)

    manager = SimulationManager()
    state = SimulationState(simulation_id="sim_degraded", project_id="project", graph_id="graph",
                            status=SimulationStatus.CREATED)
    manager._save_simulation_state(state)

    result = manager.prepare_simulation(simulation_id=state.simulation_id,
                                        simulation_requirement="requirement",
                                        document_text="document")

    assert result.status == SimulationStatus.READY
    assert result.prepare_warnings == [profile_warning, config_warning]
    persisted = json.loads((tmp_path / "sim_degraded" / "state.json").read_text(encoding="utf-8"))
    assert persisted["prepare_warnings"] == [profile_warning, config_warning]
