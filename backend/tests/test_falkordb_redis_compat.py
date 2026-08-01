"""Regression guard for the falkordb-client <-> redis-py version pairing (ADR 0009).

falkordb 1.6.2's ``Is_Cluster()`` copies async-connection kwargs into a **sync** ``redis.Redis(**kwargs)``.
redis-py 8.x carries a kwarg (``himport_registry``) the sync client rejects, so ``FalkorDriver()`` init
raises ``TypeError`` before the graph is ever touched — which is exactly what broke the first AI parity
run. The backend pins ``redis>=5,<8``; these tests fail loudly if that pin ever drifts.
"""

import shutil
import subprocess
import time

import pytest


def test_redis_py_is_below_8():
    """The falkordb client targets redis-py 7.x; 8.x breaks FalkorDriver init (himport_registry)."""
    redis = pytest.importorskip("redis")
    major = int(redis.__version__.split(".")[0])
    assert major < 8, (
        f"redis-py {redis.__version__} is incompatible with falkordb 1.6.2's Is_Cluster(); "
        "pin redis<8 (see pyproject).")


@pytest.mark.skipif(shutil.which("redis-server") is None or shutil.which("redis-cli") is None,
                    reason="redis-server/redis-cli not installed")
def test_falkordriver_init_against_plain_redis():
    """Reproduce the exact failing path: FalkorDriver()'s Is_Cluster does a sync .info() call.

    A plain ``redis-server`` answers ``INFO`` (no FalkorDB module needed), so this exercises the
    client/redis-py pairing that raised the himport_registry TypeError, without the module.
    """
    pytest.importorskip("graphiti_core")
    pytest.importorskip("falkordb")
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    port = 6403
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(20):
            out = subprocess.run(["redis-cli", "-p", str(port), "ping"],
                                 capture_output=True, text=True)
            if out.stdout.strip() == "PONG":
                break
            time.sleep(0.2)
        # Must not raise TypeError('himport_registry') — that is the regression.
        FalkorDriver(host="localhost", port=port, database="default_db")
    finally:
        subprocess.run(["redis-cli", "-p", str(port), "shutdown", "nosave"],
                       capture_output=True)
        proc.wait(timeout=10)
