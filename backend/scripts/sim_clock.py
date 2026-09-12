"""Simulated-clock helpers shared by the OASIS runner scripts.

Pure functions with no OASIS / camel imports, so the round-to-clock mapping and the per-round
agent eligibility rule are unit-testable (``backend/tests/test_sim_clock.py``).

Background. The runners map round ``N`` to simulated clock time ``N * minutes_per_round`` counted
from 00:00, and only wake agents whose LLM-generated ``active_hours`` contain that hour. A short
capped run (``--max-rounds 7`` at the usual 60 minutes per round) therefore spends every round
between 00:00 and 06:00, when almost no agent is active, and produces no actions beyond the
initial posts. Two knobs fix that without touching the LLM-generated config:

* ``start_hour`` shifts the simulated clock so round 0 starts at that hour (e.g. 9 -> the seven
  capped rounds cover 09:00 to 15:00);
* ``ignore_active_hours`` drops the clock gate entirely: every agent is eligible every round,
  still subject to its ``activity_level`` and the per-round target count. This is the right
  setting when the simulated population is global (no shared day/night rhythm).
"""
import random

DEFAULT_ACTIVE_HOURS = list(range(8, 23))
DEFAULT_PEAK_HOURS = [9, 10, 11, 14, 15, 20, 21, 22]
DEFAULT_OFF_PEAK_HOURS = [0, 1, 2, 3, 4, 5]


def simulated_clock(round_num, minutes_per_round, start_hour=0):
    """Return ``(simulated_hour, simulated_day)`` for ``round_num`` (0-based).

    ``start_hour`` is where the clock stands at round 0; hours wrap at 24 and roll the day.
    """
    minutes = int(start_hour) * 60 + round_num * minutes_per_round
    hour = (minutes // 60) % 24
    day = minutes // (60 * 24) + 1
    return hour, day


def activity_multiplier(time_config, current_hour, ignore_active_hours=False):
    """Peak / off-peak scaling of the per-round activation target (1.0 when the clock is ignored)."""
    if ignore_active_hours:
        return 1.0
    peak_hours = time_config.get("peak_hours", DEFAULT_PEAK_HOURS)
    off_peak_hours = time_config.get("off_peak_hours", DEFAULT_OFF_PEAK_HOURS)
    if current_hour in peak_hours:
        return time_config.get("peak_activity_multiplier", 1.5)
    if current_hour in off_peak_hours:
        return time_config.get("off_peak_activity_multiplier", 0.3)
    return 1.0


def eligible_agent_ids(agent_configs, current_hour, ignore_active_hours=False, rng=random):
    """Agent ids that may act this round: awake at ``current_hour`` (unless ignored) and passing
    their own ``activity_level`` draw. ``rng`` is injectable for tests."""
    ids = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", DEFAULT_ACTIVE_HOURS)
        activity_level = cfg.get("activity_level", 0.5)
        if not ignore_active_hours and current_hour not in active_hours:
            continue
        if rng.random() < activity_level:
            ids.append(agent_id)
    return ids
