"""Graphiti graph backend (self-hosted replacement for Zep Cloud).

This is the ``graphiti`` half of the ``GRAPH_BACKEND`` seam (ADR 0009 in the trade-gpt repo). It
drives Zep's own open-source engine, `graphiti-core <https://github.com/getzep/graphiti>`_, against a
**FalkorDB** instance on ``localhost`` — the same engine and the same bi-temporal data model Zep runs,
just self-hosted, so nothing downstream (persona generation, simulation, report) changes.

Why this is a drop-in for the Zep reader/tools:
    MiroFish already reads Graphiti's *own* field names through the Zep API — node ``summary`` /
    ``labels`` / ``attributes`` and edge ``fact`` / ``valid_at`` / ``invalid_at`` / ``expired_at``.
    Those are exactly the fields ``graphiti_core.nodes.EntityNode`` and
    ``graphiti_core.edges.EntityEdge`` expose, so :meth:`get_all_nodes` / :meth:`get_all_edges` /
    :meth:`search_graph` return the **same dict shapes** as ``zep_entity_reader`` / ``zep_tools``.

Deliberately self-contained (only ``graphiti_core`` + ``pydantic`` + stdlib) so it can be unit-tested
and imported without the rest of the MiroFish backend. Configuration is read from the environment:

    ================================  ==========================================================
    ``OPENAI_API_KEY``                extraction + embedding key (replaces ``ZEP_API_KEY``)
    ``GRAPHITI_MODEL``                extraction model (default ``gpt-4o-mini`` — ADR 0009)
    ``GRAPHITI_EMBED_MODEL``          embedding model (default ``text-embedding-3-small``)
    ``FALKORDB_HOST`` / ``_PORT``     FalkorDB address (default ``localhost`` / ``6379``)
    ================================  ==========================================================

The FalkorDB *process* itself and its RDB-file persistence to GCS are managed separately by
``falkordb_lifecycle`` — this module only talks to an already-running FalkorDB.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger("mirofish.graphiti_backend")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
#: Cap on nodes/edges pulled per read-back page (Graphiti paginates via ``uuid_cursor``).
READ_PAGE_SIZE = 500
#: Reserved attribute keys Graphiti manages itself — never surface them as ontology attributes.
_RESERVED_ATTRS = {"uuid", "name", "group_id", "labels", "summary", "fact", "created_at",
                   "valid_at", "invalid_at", "expired_at", "attributes", "episodes"}


def _pascal(name: str) -> str:
    return "".join(w.capitalize() for w in str(name).replace("-", "_").split("_") if w) or "Entity"


def _model_from_attributes(type_name: str, attributes: list[dict]) -> type[BaseModel]:
    """Build a Pydantic model whose fields are an ontology type's attributes.

    Graphiti's ``entity_types`` / ``edge_types`` are ``{type_name: PydanticModel}`` where the model's
    fields tell the extractor which attributes to pull. MiroFish's ontology gives each type a list of
    ``{name, type, description}`` attributes; we map every one to an optional ``str`` field (the
    extractor fills what it can). A type with no attributes becomes a bare marker model.
    """
    fields: dict[str, Any] = {}
    for attr in attributes or []:
        key = str((attr or {}).get("name", "")).strip()
        if not key or key in _RESERVED_ATTRS:
            continue
        desc = str((attr or {}).get("description", "") or "")
        fields[key] = (Optional[str], Field(default=None, description=desc[:200]))
    return create_model(_pascal(type_name), __base__=BaseModel, **fields)


def ontology_to_graphiti_types(ontology: dict) -> tuple[dict[str, type[BaseModel]],
                                                         dict[str, type[BaseModel]],
                                                         dict[tuple[str, str], list[str]]]:
    """Convert MiroFish's generated ontology JSON into Graphiti's typed-extraction inputs.

    Returns ``(entity_types, edge_types, edge_type_map)`` for :meth:`Graphiti.add_episode`:
      * ``entity_types`` — ``{PascalName: model}`` from ``ontology['entity_types']``.
      * ``edge_types``   — ``{UPPER_SNAKE: model}`` from ``ontology['edge_types']``.
      * ``edge_type_map``— ``{(SourceType, TargetType): [edge_names]}`` from each edge's
        ``source_targets``, so a relation is only considered between the entity types it connects.
    """
    entity_types: dict[str, type[BaseModel]] = {}
    for ent in (ontology or {}).get("entity_types", []) or []:
        raw = str((ent or {}).get("name", "")).strip()
        if not raw:
            continue
        name = _pascal(raw)
        entity_types[name] = _model_from_attributes(name, (ent or {}).get("attributes", []))

    edge_types: dict[str, type[BaseModel]] = {}
    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for edge in (ontology or {}).get("edge_types", []) or []:
        raw = str((edge or {}).get("name", "")).strip()
        if not raw:
            continue
        ename = raw.upper().replace(" ", "_").replace("-", "_")
        edge_types[ename] = _model_from_attributes(ename, (edge or {}).get("attributes", []))
        for st in (edge or {}).get("source_targets", []) or []:
            src, tgt = _pascal((st or {}).get("source", "")), _pascal((st or {}).get("target", ""))
            edge_type_map.setdefault((src, tgt), []).append(ename)
    return entity_types, edge_types, edge_type_map


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else None


class GraphitiBackend:
    """Talks to a running FalkorDB via graphiti-core, returning MiroFish's dict shapes.

    One backend serves every cluster; clusters are isolated by Graphiti ``group_id`` (so ``AI`` /
    ``GENE`` / ``FUSION`` share one FalkorDB database — matching how a single Zep account held
    multiple graphs). All public methods are synchronous wrappers over graphiti-core's async API,
    run on a dedicated event loop so a Flask request thread can call them directly.
    """

    def __init__(self, *, host: Optional[str] = None, port: Optional[int] = None,
                 model: Optional[str] = None, embed_model: Optional[str] = None,
                 api_key: Optional[str] = None, database: str = "default_db"):
        self.host = host or os.environ.get("FALKORDB_HOST", "localhost")
        self.port = int(port or os.environ.get("FALKORDB_PORT", "6379"))
        self.model = model or os.environ.get("GRAPHITI_MODEL", DEFAULT_MODEL)
        self.embed_model = embed_model or os.environ.get("GRAPHITI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.database = database
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set (extraction + embeddings)")
        self._graphiti = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -- lifecycle -------------------------------------------------------- #
    def _loop_get(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        return self._loop_get().run_until_complete(coro)

    def _client(self):
        """Lazily build the Graphiti client bound to FalkorDB + OpenAI (extraction + embeddings)."""
        if self._graphiti is None:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from graphiti_core.llm_client import LLMConfig, OpenAIClient
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

            driver = FalkorDriver(host=self.host, port=self.port, database=self.database)
            llm = OpenAIClient(config=LLMConfig(api_key=self.api_key, model=self.model))
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.embed_model))
            self._graphiti = Graphiti(graph_driver=driver, llm_client=llm, embedder=embedder)
        return self._graphiti

    def init_indices(self) -> None:
        """One-time index/constraint setup (BM25 + vector). Idempotent; run on a fresh DB file."""
        self._run(self._client().build_indices_and_constraints())

    def close(self) -> None:
        if self._graphiti is not None:
            try:
                self._run(self._client().close())
            except Exception:  # best-effort — the process is about to exit anyway
                logger.debug("graphiti close() failed", exc_info=True)
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    # -- build / append --------------------------------------------------- #
    def add_documents(self, cluster: str, chunks: list[str], *, ontology: Optional[dict] = None,
                      reference_time: Optional[datetime] = None,
                      progress=None) -> int:
        """Ingest ``chunks`` into the ``cluster`` graph (create on first sight, append thereafter).

        Each chunk is one Graphiti *episode* (``source=text``), tagged with ``group_id=cluster`` so
        it accretes into that cluster's persistent bi-temporal graph. Superseded facts are
        invalidated (``invalid_at`` set), not duplicated — the same behaviour Zep provided. Returns
        the number of chunks ingested.
        """
        from graphiti_core.nodes import EpisodeType

        entity_types, edge_types, edge_type_map = (
            ontology_to_graphiti_types(ontology) if ontology else ({}, {}, {}))
        ref = reference_time or datetime.now(timezone.utc)
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            self._run(self._client().add_episode(
                name=f"{cluster}-chunk-{i}",
                episode_body=chunk,
                source_description="MiroFish news chunk",
                reference_time=ref,
                source=EpisodeType.text,
                group_id=cluster,
                entity_types=entity_types or None,
                edge_types=edge_types or None,
                edge_type_map=edge_type_map or None,
            ))
            if progress:
                progress(f"ingested {i + 1}/{total}", (i + 1) / max(total, 1))
        return total

    # -- read-back (persona generation) ----------------------------------- #
    def _all(self, cls, cluster: str) -> list:
        driver, out, cursor = self._client().driver, [], None
        while True:
            page = self._run(cls.get_by_group_ids(
                driver, [cluster], limit=READ_PAGE_SIZE, uuid_cursor=cursor))
            if not page:
                break
            out.extend(page)
            if len(page) < READ_PAGE_SIZE:
                break
            cursor = page[-1].uuid
        return out

    def get_all_nodes(self, cluster: str) -> list[dict]:
        """Every node in the cluster graph, in the exact shape ``zep_entity_reader`` returned."""
        return [{
            "uuid": n.uuid,
            "name": n.name or "",
            "labels": list(n.labels or []),
            "summary": n.summary or "",
            "attributes": dict(n.attributes or {}),
        } for n in self._all(_entity_node(), cluster)]

    def get_all_edges(self, cluster: str) -> list[dict]:
        """Every edge (fact) in the cluster graph, with the bi-temporal fields personas read."""
        return [{
            "uuid": e.uuid,
            "name": e.name or "",
            "fact": e.fact or "",
            "source_node_uuid": e.source_node_uuid,
            "target_node_uuid": e.target_node_uuid,
            "attributes": dict(e.attributes or {}),
            "valid_at": _iso(e.valid_at),
            "invalid_at": _iso(e.invalid_at),
            "expired_at": _iso(e.expired_at),
        } for e in self._all(_entity_edge(), cluster)]

    # -- search (simulation-time) ----------------------------------------- #
    def search_graph(self, cluster: str, query: str, *, limit: int = 10) -> dict:
        """Hybrid search over the cluster graph; returns facts + temporal validity for the agents.

        Mirrors ``zep_tools.search_graph``: a list of facts (each with ``valid_at``/``invalid_at``/
        ``expired_at``) plus the raw edges, so the simulation's time-aware reasoning is unchanged.
        """
        edges = self._run(self._client().search(query, group_ids=[cluster], num_results=limit))
        facts = [{
            "fact": e.fact or "",
            "name": e.name or "",
            "valid_at": _iso(e.valid_at),
            "invalid_at": _iso(e.invalid_at),
            "expired_at": _iso(e.expired_at),
        } for e in edges]
        return {"facts": facts, "edges": edges}

    # -- lifecycle -------------------------------------------------------- #
    def delete_cluster(self, cluster: str) -> None:
        """Drop a cluster's whole graph (rarely used; MiroFish appends, it does not rebuild)."""
        self._run(_entity_node().delete_by_group_id(self._client().driver, cluster))


def _entity_node():
    from graphiti_core.nodes import EntityNode
    return EntityNode


def _entity_edge():
    from graphiti_core.edges import EntityEdge
    return EntityEdge
