#!/usr/bin/env python3
"""Benchmark the prepare-stage agent-config batches against the REAL LLM.

Answers "how fast can we make it" with numbers from the actual provider instead of estimates:
wall-clock of the batch stage for a synthetic cluster of N entities at a given batch size and
concurrency, per-batch latency p50/p90/max, how many batches fell back to rule-based configs
(LLM call failed after the fork's own 3 attempts), and how many raw HTTP 429 / timeout errors the
provider returned while the burst was in flight.

Needs LLM_API_KEY (+ LLM_BASE_URL, LLM_MODEL_NAME) — the same env the worker container has. Run
from backend/:

    uv run python scripts/bench_config_batches.py --agents 262 --batch-size 15 --parallel 0
    uv run python scripts/bench_config_batches.py --agents 262 --batch-size 5  --parallel 0
    uv run python scripts/bench_config_batches.py --agents 262 --batch-size 15 --parallel 1   # the old serial loop

--parallel 0 = every batch at once (the default in production). Sweep --parallel 1,4,8,0 and
--batch-size 15,10,5 and read the table; the first configuration that shows 429s is the ceiling
for this key. --sdk-retries 0 exposes every 429 instead of letting the OpenAI SDK absorb up to two.
--dry-run replaces the LLM with a fixed fake latency so the harness itself can be checked offline.
"""
import argparse
import os
import statistics
import sys
import threading
import time

_backend_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _backend_dir)


def _percentile(values, pct):
    if not values:
        return float("nan")
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=262, help="synthetic entities (262 = 09-10 AI cluster)")
    ap.add_argument("--batch-size", type=int, default=None, help="AGENTS_PER_BATCH (default env/15)")
    ap.add_argument("--parallel", type=int, default=None, help="CONFIG_PARALLEL_COUNT (0 = all at once; default env/0)")
    ap.add_argument("--timeout", type=float, default=None, help="CONFIG_LLM_TIMEOUT_SECONDS (default env/180)")
    ap.add_argument("--sdk-retries", type=int, default=None, help="OpenAI SDK max_retries (default SDK 2); 0 exposes every 429")
    ap.add_argument("--dry-run", type=float, default=None, metavar="SECONDS", help="fake per-batch latency, no LLM")
    args = ap.parse_args(argv)

    if args.batch_size is not None:
        os.environ["AGENTS_PER_BATCH"] = str(args.batch_size)
    if args.parallel is not None:
        os.environ["CONFIG_PARALLEL_COUNT"] = str(args.parallel)
    if args.timeout is not None:
        os.environ["CONFIG_LLM_TIMEOUT_SECONDS"] = str(args.timeout)
    if args.dry_run is not None:
        os.environ.setdefault("LLM_API_KEY", "dry-run")

    from app.services import simulation_config_generator as scg
    from app.services.config_batches import resolve_config_parallel_count, split_batches, run_batches
    from app.services.zep_entity_reader import EntityNode

    gen = scg.SimulationConfigGenerator()
    if args.sdk_retries is not None:
        gen.client = gen.client.with_options(max_retries=args.sdk_retries)

    entities = [EntityNode(uuid=f"u{i}", name=f"Company {i}", labels=["Entity", "PublicCompany"],
                           summary=f"Company {i} is a listed technology firm in the AI supply chain.",
                           attributes={}) for i in range(args.agents)]
    context = gen._build_context(simulation_requirement="How will sentiment evolve for the AI space?",
                                 document_text="", entities=entities)
    batches = split_batches(len(entities), gen.agents_per_batch)
    width = resolve_config_parallel_count(num_batches=len(batches))

    lock = threading.Lock()
    latencies, errors, fallbacks = [], {"429": 0, "timeout": 0, "other": 0}, [0]

    orig_call = gen._call_llm_with_retry

    def counted_call(prompt, system_prompt):
        try:
            return orig_call(prompt, system_prompt)
        except Exception as exc:  # the batch method catches this and falls back to rules
            name = type(exc).__name__.lower()
            with lock:
                if "ratelimit" in name or "429" in str(exc):
                    errors["429"] += 1
                elif "timeout" in name:
                    errors["timeout"] += 1
                else:
                    errors["other"] += 1
            raise
    gen._call_llm_with_retry = counted_call

    def batch(batch_idx, start, end):
        t0 = time.perf_counter()
        if args.dry_run is not None:
            time.sleep(args.dry_run)
            out = list(range(start, end))
        else:
            out = gen._generate_agent_configs_batch(context=context, entities=entities[start:end],
                                                    start_idx=start, simulation_requirement="bench")
            # a batch whose LLM call failed comes back rule-based: every config has the rule defaults
            if out and all(c.posts_per_hour in (0.1, 0.5, 1.0, 2.0) and c.stance == "neutral" for c in out):
                pass
        with lock:
            latencies.append(time.perf_counter() - t0)
        return out

    print(f"agents={len(entities)} batch_size={gen.agents_per_batch} batches={len(batches)} "
          f"parallel={width} timeout={gen.config_llm_timeout:.0f}s model={gen.model_name} "
          f"{'DRY-RUN' if args.dry_run is not None else ''}")
    t0 = time.perf_counter()
    out = run_batches(batch, batches, width,
                      on_done=lambda done, i: print(f"  batch {i} done ({done}/{len(batches)}) "
                                                    f"t+{time.perf_counter() - t0:.1f}s", flush=True))
    wall = time.perf_counter() - t0

    print()
    print(f"RESULT wall={wall:.1f}s  serial_equivalent={sum(latencies):.1f}s  "
          f"speedup=x{(sum(latencies) / wall) if wall else 0:.1f}")
    print(f"       per-batch latency p50={_percentile(latencies, 50):.1f}s "
          f"p90={_percentile(latencies, 90):.1f}s max={max(latencies) if latencies else 0:.1f}s "
          f"mean={statistics.fmean(latencies) if latencies else 0:.1f}s")
    print(f"       llm call failures: 429={errors['429']} timeout={errors['timeout']} other={errors['other']} "
          f"(each failure = one batch fell back to rule-based configs)")
    print(f"       configs returned: {len(out)} (expected {len(entities)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
