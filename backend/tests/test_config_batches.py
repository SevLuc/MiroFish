"""Batch scheduling for the prepare-stage agent-config LLM calls (app/services/config_batches.py).

Pure-module tests (no LLM, no Flask): the env knobs, the batch split, ordering under out-of-order
completion, the concurrency cap, error propagation, and a wall-clock test that shows the speedup
the parallel runner buys over the original serial loop. Run with ``-s`` to see the timings.
"""
import importlib.util
import pathlib
import random
import threading
import time

_SERVICES = pathlib.Path(__file__).resolve().parents[1] / "app" / "services"
_spec = importlib.util.spec_from_file_location("config_batches", _SERVICES / "config_batches.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


# --------------------------------------------------------------------------- knobs
def test_agents_per_batch_env():
    assert cb.resolve_agents_per_batch({}) == 15
    assert cb.resolve_agents_per_batch({"AGENTS_PER_BATCH": "5"}) == 5
    assert cb.resolve_agents_per_batch({"AGENTS_PER_BATCH": "0"}) == 15
    assert cb.resolve_agents_per_batch({"AGENTS_PER_BATCH": "lots"}) == 15


def test_parallel_count_defaults_to_all_batches():
    assert cb.resolve_config_parallel_count({}, num_batches=18) == 18
    assert cb.resolve_config_parallel_count({"CONFIG_PARALLEL_COUNT": "0"}, num_batches=18) == 18
    assert cb.resolve_config_parallel_count({"CONFIG_PARALLEL_COUNT": "4"}, num_batches=18) == 4
    assert cb.resolve_config_parallel_count({"CONFIG_PARALLEL_COUNT": "50"}, num_batches=18) == 18
    assert cb.resolve_config_parallel_count({"CONFIG_PARALLEL_COUNT": "-3"}, num_batches=18) == 18
    assert cb.resolve_config_parallel_count({"CONFIG_PARALLEL_COUNT": "x"}, num_batches=18) == 18
    assert cb.resolve_config_parallel_count({}, num_batches=0) == 1


def test_llm_timeout_env():
    assert cb.resolve_config_llm_timeout({}) == 180.0
    assert cb.resolve_config_llm_timeout({"CONFIG_LLM_TIMEOUT_SECONDS": "90"}) == 90.0
    assert cb.resolve_config_llm_timeout({"CONFIG_LLM_TIMEOUT_SECONDS": "0"}) == 180.0
    assert cb.resolve_config_llm_timeout({"CONFIG_LLM_TIMEOUT_SECONDS": "soon"}) == 180.0


def test_split_batches_covers_every_index_once():
    assert cb.split_batches(262, 15) == [(s, min(s + 15, 262)) for s in range(0, 262, 15)]
    assert len(cb.split_batches(262, 15)) == 18 == cb.num_batches(262, 15)
    assert cb.split_batches(0, 15) == []
    assert cb.split_batches(7, 15) == [(0, 7)]


# --------------------------------------------------------------------------- runner
def _fake_batch(latency, jitter=0.0, tracker=None):
    """A stand-in for _generate_agent_configs_batch: sleeps, returns the agent ids it was given."""
    lock = threading.Lock()

    def fn(batch_idx, start, end):
        if tracker is not None:
            with lock:
                tracker["in_flight"] += 1
                tracker["peak"] = max(tracker["peak"], tracker["in_flight"])
        time.sleep(latency + random.uniform(0, jitter))
        if tracker is not None:
            with lock:
                tracker["in_flight"] -= 1
        return list(range(start, end))
    return fn


def test_results_keep_batch_order_when_batches_finish_out_of_order():
    random.seed(7)
    batches = cb.split_batches(97, 10)
    out = cb.run_batches(_fake_batch(0.01, jitter=0.05), batches, parallel_count=0)
    assert out == list(range(97))          # agent_id 0..96, contiguous, in order


def test_parallel_count_caps_in_flight_calls():
    tracker = {"in_flight": 0, "peak": 0}
    batches = cb.split_batches(90, 15)     # 6 batches
    t0 = time.perf_counter()
    cb.run_batches(_fake_batch(0.15, tracker=tracker), batches, parallel_count=2)
    wall = time.perf_counter() - t0
    assert tracker["peak"] <= 2
    assert wall >= 0.40                    # 6 batches / 2 wide = 3 waves x 0.15 s


def test_progress_callback_counts_every_batch_once():
    seen = []
    batches = cb.split_batches(45, 15)
    cb.run_batches(_fake_batch(0.01), batches, parallel_count=0,
                   on_done=lambda completed, idx: seen.append((completed, idx)))
    assert [c for c, _ in seen] == [1, 2, 3]
    assert sorted(i for _, i in seen) == [0, 1, 2]


def test_a_failing_batch_is_raised_after_the_pool_drains():
    def fn(batch_idx, start, end):
        if batch_idx == 1:
            raise RuntimeError("boom")
        return list(range(start, end))
    try:
        cb.run_batches(fn, cb.split_batches(45, 15), parallel_count=0)
    except RuntimeError as e:
        assert str(e) == "boom"
    else:
        raise AssertionError("expected the batch error to propagate")


def test_empty_input_returns_empty():
    assert cb.run_batches(_fake_batch(0.0), [], parallel_count=0) == []


# --------------------------------------------------------------------------- speed
def test_parallel_runner_beats_the_serial_loop():
    """The point of the change. 18 batches (a 262-agent AI cluster) at a fake 0.15 s latency:
    serial ~2.7 s, all-at-once ~0.15 s. Production latency is ~54 s per batch, so the same ratio
    is ~16 min vs ~1 min."""
    batches = cb.split_batches(262, 15)
    assert len(batches) == 18

    t0 = time.perf_counter()
    serial = cb.run_batches(_fake_batch(0.15), batches, parallel_count=1)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = cb.run_batches(_fake_batch(0.15), batches, parallel_count=0)
    t_parallel = time.perf_counter() - t0

    print(f"\n18 batches x 0.15 s: serial {t_serial:.2f} s, parallel(all) {t_parallel:.2f} s, "
          f"speedup x{t_serial / t_parallel:.1f}")
    assert serial == parallel == list(range(262))
    assert t_serial >= 2.5
    assert t_parallel < t_serial / 4


def test_speed_scales_with_parallel_count():
    """Wall follows ceil(batches / width): 18 batches at width 1/3/6/18 -> 18/6/3/1 waves."""
    batches = cb.split_batches(262, 15)
    walls = {}
    for width in (1, 3, 6, 18):
        t0 = time.perf_counter()
        cb.run_batches(_fake_batch(0.05), batches, parallel_count=width)
        walls[width] = time.perf_counter() - t0
    print("\n" + ", ".join(f"width {w}: {t:.2f} s" for w, t in walls.items()))
    assert walls[1] > walls[3] > walls[6] > walls[18]
    assert walls[18] < walls[1] / 8
