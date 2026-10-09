"""`/api/simulation/prepare/status` must answer from the task when the caller passes its task_id.

Until 2026-10-09 the route returned the "already prepared" shape (status=ready, no `result`) as
soon as the profile/config files existed on disk, before it looked the task up — so a poller
that sent both ids never received the task result, and `prepare_warnings` (PROFILE_DEGRADED*,
EVENT_CONFIG_*, CONFIG_DEGRADED*) never left the backend: 0 of 134 worker result.json carried one.
"""
import json

import pytest

from app import create_app
from app.config import Config
from app.models.task import TaskManager, TaskStatus

WARNINGS = [
    'PROFILE_DEGRADED: {"entity": "NVDA", "entity_type": "Techcompany", "node_degree": 7, "reason": "timeout"}',
    'PROFILE_DEGRADED_SUMMARY: {"degraded": 1, "total": 52, "fraction": 0.0192}',
]


@pytest.fixture
def prepared_sim(tmp_path, monkeypatch):
    """A simulation whose prepared files already exist on disk (the shape that used to win)."""
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    sim_dir = tmp_path / "sim_prepared"
    sim_dir.mkdir()
    (sim_dir / "state.json").write_text(
        json.dumps({"status": "ready", "config_generated": True}), encoding="utf-8"
    )
    (sim_dir / "reddit_profiles.json").write_text("[]", encoding="utf-8")
    (sim_dir / "twitter_profiles.csv").write_text("user_id\n", encoding="utf-8")
    (sim_dir / "simulation_config.json").write_text("{}", encoding="utf-8")
    return "sim_prepared"


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _status(client, **body):
    response = client.post("/api/simulation/prepare/status", json=body)
    assert response.status_code == 200, response.data
    return response.json["data"]


def test_completed_task_result_wins_over_the_files_on_disk(prepared_sim, client):
    manager = TaskManager()
    task_id = manager.create_task("prepare")
    manager.complete_task(task_id, result={"simulation_id": prepared_sim, "prepare_warnings": list(WARNINGS)})

    data = _status(client, simulation_id=prepared_sim, task_id=task_id)

    assert data["status"] == TaskStatus.COMPLETED.value
    assert data["result"]["prepare_warnings"] == WARNINGS
    assert data["already_prepared"] is False


def test_a_running_task_is_reported_as_running_even_once_the_files_exist(prepared_sim, client):
    manager = TaskManager()
    task_id = manager.create_task("prepare")
    manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=90)

    data = _status(client, simulation_id=prepared_sim, task_id=task_id)

    assert data["status"] == TaskStatus.PROCESSING.value
    assert "result" not in data or not data["result"]


def test_unknown_task_id_still_falls_back_to_the_files(prepared_sim, client):
    data = _status(client, simulation_id=prepared_sim, task_id="task_from_a_previous_process")

    assert data["status"] == "ready"
    assert data["already_prepared"] is True
    assert data["task_id"] == "task_from_a_previous_process"


def test_no_task_id_keeps_the_file_based_answer(prepared_sim, client):
    data = _status(client, simulation_id=prepared_sim)

    assert data["status"] == "ready"
    assert data["already_prepared"] is True
