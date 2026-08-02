"""``GRAPH_BACKEND`` seam — pick the graph engine at runtime (ADR 0009).

MiroFish's cluster graphs originally lived in **Zep Cloud**. ADR 0009 (in the trade-gpt repo)
migrates them to a **self-hosted Graphiti** engine driving a local **FalkorDB**, with Zep kept
dormant and *selectable* so the project stays a complete picture and the two backends can be A/B'd.

This module is the single switch. The legacy Zep service classes (``GraphBuilderService``,
``ZepEntityReader``, ``ZepToolsService``) call :func:`use_graphiti` at their raw graph touchpoints
and, when it is true, delegate to the shared :class:`GraphitiBackend` returned by
:func:`get_graphiti_backend`. When ``GRAPH_BACKEND=zep`` every legacy Zep path runs exactly as before.

Default is ``graphiti`` (ADR 0009): the self-hosted engine is the going-forward backend; ``zep`` is
opt-in. One ``GraphitiBackend`` serves every cluster (clusters are isolated by Graphiti ``group_id``,
which is MiroFish's ``graph_id``), so it is a process-wide singleton over the single local FalkorDB.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_BACKEND = "graphiti"

_backend = None  # cached GraphitiBackend singleton (one FalkorDB per worker process)


def graph_backend_name() -> str:
    """Selected backend: ``graphiti`` (default) or ``zep``. Unknown/empty → the default."""
    name = (os.environ.get("GRAPH_BACKEND") or "").strip().lower()
    return name if name in ("graphiti", "zep") else DEFAULT_BACKEND


def use_graphiti() -> bool:
    """True when graph reads/writes/search should route to the self-hosted Graphiti backend."""
    return graph_backend_name() == "graphiti"


def use_zep() -> bool:
    """True when the legacy Zep Cloud backend is selected (dormant-by-default per ADR 0009)."""
    return graph_backend_name() == "zep"


def get_graphiti_backend():
    """Return the process-wide :class:`GraphitiBackend` (one FalkorDB serves every cluster).

    Imported lazily so a ``GRAPH_BACKEND=zep`` process never imports ``graphiti_core``/``fastembed``.
    """
    global _backend
    if _backend is None:
        from .graphiti_backend import GraphitiBackend
        _backend = GraphitiBackend()
    return _backend


def reset_graphiti_backend(new=None) -> None:
    """Drop (or replace) the cached backend — for tests and for a clean worker teardown."""
    global _backend
    if _backend is not None and new is None:
        try:
            _backend.close()
        except Exception:  # best-effort; the process is tearing down anyway
            pass
    _backend = new
