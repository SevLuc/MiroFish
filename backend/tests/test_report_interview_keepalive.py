"""The report gate must accept a rounds-complete sim that is still ALIVE in its
wait-for-commands loop (RunnerStatus.RUNNING at 100%), so the report agent can
interview the live OASIS env before it is closed. A *mid-run* RUNNING sim must
still be rejected. See app/api/report.py :: _rounds_complete_alive and the
trade-gpt microfish_pipeline reorder (report BEFORE close-env)."""
from types import SimpleNamespace

from flask import Flask

from app.api import report as report_api
from app.services.simulation_runner import RunnerStatus


def _json_result(result):
    if isinstance(result, tuple):
        response, status = result
    else:
        response, status = result, result.status_code
    return response.get_json(), status


def _wire(monkeypatch, *, runner_status, all_platforms_completed, project):
    """Wire report_api so generate_report reaches (and clears, or not) the run-status gate."""
    simulation = SimpleNamespace(project_id="proj-1", graph_id="graph-1")
    monkeypatch.setattr(report_api, "SimulationManager",
                        lambda: SimpleNamespace(get_simulation=lambda _s: simulation))
    monkeypatch.setattr(report_api.ReportManager, "get_report_by_simulation",
                        classmethod(lambda _cls, _s: None))
    monkeypatch.setattr(report_api.SimulationRunner, "get_run_state",
                        classmethod(lambda _cls, _s: SimpleNamespace(
                            runner_status=runner_status, project_id="proj-1",
                            graph_id="graph-1")))
    monkeypatch.setattr(report_api.SimulationRunner, "_check_all_platforms_completed",
                        classmethod(lambda _cls, _state: all_platforms_completed))
    monkeypatch.setattr(report_api.ZepGraphMemoryManager, "get_updater",
                        classmethod(lambda _cls, _s: None))
    monkeypatch.setattr(report_api.ProjectManager, "get_project",
                        classmethod(lambda _cls, _pid: project))


def _call():
    app = Flask(__name__)
    with app.test_request_context("/api/report/generate", method="POST",
                                  json={"simulation_id": "sim-1"}):
        return _json_result(report_api.generate_report())


def test_rounds_complete_but_alive_running_sim_passes_the_run_status_gate(monkeypatch):
    # RUNNING + all platforms finished => the interview window. Project is None so the
    # request stops at the NEXT check (projectNotFound, 404) — which proves the run-status
    # gate ACCEPTED it (a rejected run-status gate returns 409 before ever reaching project).
    _wire(monkeypatch, runner_status=RunnerStatus.RUNNING,
          all_platforms_completed=True, project=None)
    body, status = _call()
    assert status == 404
    assert "completed or stopped simulation is required" not in (body.get("error") or "")


def test_midrun_running_sim_is_still_rejected(monkeypatch):
    # RUNNING but NOT all platforms finished => a genuinely mid-run sim, still rejected.
    _wire(monkeypatch, runner_status=RunnerStatus.RUNNING,
          all_platforms_completed=False, project=SimpleNamespace())
    body, status = _call()
    assert status == 409
    assert "completed or stopped simulation is required" in body["error"]
