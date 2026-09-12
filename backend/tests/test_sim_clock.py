"""The simulated clock and per-round agent eligibility (backend/scripts/sim_clock.py).

Regression for the empty-rounds bug: a capped run (max_rounds=7 at 60 min/round) spends every
round at simulated 00:00-06:00, outside every agent's active_hours, so no agent ever acts and the
run ends with total_actions == the initial posts. ``start_hour`` and ``ignore_active_hours`` are
the two fixes; both must leave the default (start_hour=0, gate on) byte-for-byte as before.
"""
import importlib.util
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("sim_clock", _SCRIPTS / "sim_clock.py")
sim_clock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and sim_clock)


class _Rng:
    """Deterministic stand-in for ``random``: every draw returns ``value``."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


AGENTS = [
    {"agent_id": 1, "active_hours": list(range(8, 23)), "activity_level": 0.5},
    {"agent_id": 2, "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23], "activity_level": 0.5},
    {"agent_id": 3},   # defaults: active 8..22, activity 0.5
]


def test_default_clock_starts_at_midnight():
    assert [sim_clock.simulated_clock(r, 60) for r in range(7)] == [(h, 1) for h in range(7)]


def test_start_hour_shifts_the_clock():
    assert [sim_clock.simulated_clock(r, 60, start_hour=9)[0] for r in range(7)] == list(range(9, 16))


def test_clock_wraps_past_midnight_into_the_next_day():
    assert sim_clock.simulated_clock(3, 60, start_hour=22) == (1, 2)
    assert sim_clock.simulated_clock(1, 30, start_hour=23) == (23, 1)


def test_seven_capped_rounds_at_midnight_wake_nobody():
    """The bug: rounds 0..6 = hours 0..6, in no agent's active_hours, even with a generous draw."""
    for hour in range(7):
        assert sim_clock.eligible_agent_ids(AGENTS, hour, rng=_Rng(0.0)) == []


def test_start_hour_nine_wakes_agents_in_working_hours():
    assert sim_clock.eligible_agent_ids(AGENTS, 9, rng=_Rng(0.0)) == [1, 2, 3]
    assert sim_clock.eligible_agent_ids(AGENTS, 15, rng=_Rng(0.0)) == [1, 3]   # agent 2 is out at 15


def test_ignore_active_hours_makes_every_agent_eligible_at_any_hour():
    for hour in (0, 3, 6):
        assert sim_clock.eligible_agent_ids(AGENTS, hour, ignore_active_hours=True, rng=_Rng(0.0)) == [1, 2, 3]


def test_activity_level_still_filters_when_the_clock_is_ignored():
    # draw 0.9 >= activity_level 0.5 for every agent -> nobody, gate or no gate
    assert sim_clock.eligible_agent_ids(AGENTS, 10, ignore_active_hours=True, rng=_Rng(0.9)) == []


def test_multiplier_follows_peak_and_off_peak_unless_ignored():
    tc = {"peak_hours": [20], "off_peak_hours": [3], "peak_activity_multiplier": 1.5,
          "off_peak_activity_multiplier": 0.05}
    assert sim_clock.activity_multiplier(tc, 20) == 1.5
    assert sim_clock.activity_multiplier(tc, 3) == 0.05
    assert sim_clock.activity_multiplier(tc, 12) == 1.0
    assert sim_clock.activity_multiplier(tc, 20, ignore_active_hours=True) == 1.0
    assert sim_clock.activity_multiplier(tc, 3, ignore_active_hours=True) == 1.0


def test_multiplier_defaults_match_the_runner_defaults():
    assert sim_clock.activity_multiplier({}, 9) == 1.5      # default peak hours include 9
    assert sim_clock.activity_multiplier({}, 2) == 0.3      # default off-peak 0..5
    assert sim_clock.activity_multiplier({}, 13) == 1.0
