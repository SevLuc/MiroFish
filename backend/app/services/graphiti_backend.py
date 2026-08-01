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

Imported without ``graphiti_core`` present, the pure-logic helpers (e.g. ``ontology_to_graphiti_types``)
still work — the engine is imported lazily inside the methods that need it. Configuration is read
from the environment, and is designed to run on **MiroFish's existing OpenRouter credential** with
**no new key** (ADR 0009):

    ================================  ==========================================================
    ``LLM_API_KEY`` / ``LLM_BASE_URL`` reused for extraction — OpenRouter proxies ``gpt-4o-mini``
                                      (falls back to ``OPENAI_API_KEY`` / OpenAI default base_url)
    ``GRAPHITI_MODEL``                extraction model (default ``gpt-4o-mini``; on OpenRouter set
                                      ``openai/gpt-4o-mini``)
    ``GRAPHITI_EMBEDDER``             ``local`` (default — free fastembed, no key) or ``openai``
    ``GRAPHITI_EMBED_MODEL``          embedding model when ``GRAPHITI_EMBEDDER=openai``
    ``FASTEMBED_CACHE_DIR``           where the local ONNX model lives (bake it at image build so a
                                      run never downloads it)
    ``GRAPHITI_RERANKER``             ``passthrough`` (default — free, keeps hybrid-rank order) or
                                      ``openai`` (LLM reranker via the same OpenRouter creds)
    ``FALKORDB_HOST`` / ``_PORT``     FalkorDB address (default ``localhost`` / ``6379``)
    ================================  ==========================================================

OpenRouter has no embeddings endpoint, so embeddings default to a **local** fastembed model — that is
what lets the whole backend run on the existing OpenRouter key alone. The FalkorDB *process* and its
RDB-file persistence to GCS are managed separately by the trade-gpt worker (``falkordb_session``);
this module only talks to an already-running FalkorDB.
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
#: Free local embedding model (fastembed / ONNX — no torch, no GPU, no API key). Small & fast; ample
#: for news-entity similarity on a few-MB graph.
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
#: Cap on nodes/edges pulled per read-back page (Graphiti paginates via ``uuid_cursor``).
READ_PAGE_SIZE = 500
#: Reserved attribute keys Graphiti manages itself — never surface them as ontology attributes.
_RESERVED_ATTRS = {"uuid", "name", "group_id", "labels", "summary", "fact", "created_at",
                   "valid_at", "invalid_at", "expired_at", "attributes", "episodes"}


def _pascal(name: str) -> str:
    # Split on spaces/underscores/hyphens so multi-word ontology names become a single valid
    # PascalCase label ("ai chip" -> "AiChip"); a space would be fragile as a Graphiti/Cypher label.
    cleaned = str(name).replace("-", "_").replace(" ", "_")
    return "".join(w.capitalize() for w in cleaned.split("_") if w) or "Entity"


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
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 embedder: Optional[str] = None, database: str = "default_db"):
        self.host = host or os.environ.get("FALKORDB_HOST", "localhost")
        self.port = int(port or os.environ.get("FALKORDB_PORT", "6379"))
        # Extraction: reuse MiroFish's existing OpenRouter creds so NO new key is needed — OpenRouter
        # proxies gpt-4o-mini. Fall back to OPENAI_API_KEY (+ OpenAI default base_url) if that is set.
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or None
        self.model = model or os.environ.get("GRAPHITI_MODEL", DEFAULT_MODEL)
        # Embeddings: OpenRouter has none, so default to a free LOCAL embedder (zero new credentials).
        self.embedder_kind = (embedder or os.environ.get("GRAPHITI_EMBEDDER", "local")).lower()
        self.embed_model = embed_model or os.environ.get("GRAPHITI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.database = database
        if not self.api_key:
            raise ValueError("no extraction key: set LLM_API_KEY (OpenRouter) or OPENAI_API_KEY")
        self._graphiti = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._indices_built = False

    # -- lifecycle -------------------------------------------------------- #
    def _loop_get(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro):
        return self._loop_get().run_until_complete(coro)

    def _client(self):
        """Lazily build the Graphiti client bound to FalkorDB, the extraction LLM, and an embedder."""
        if self._graphiti is None:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from graphiti_core.llm_client import LLMConfig, OpenAIClient

            driver = FalkorDriver(host=self.host, port=self.port, database=self.database)
            # OpenAI-compatible client; base_url points it at OpenRouter (or OpenAI when unset).
            llm = OpenAIClient(config=LLMConfig(
                api_key=self.api_key, model=self.model, base_url=self.base_url))
            # cross_encoder MUST be passed: Graphiti's default is an OpenAIRerankerClient that reads
            # OPENAI_API_KEY at construction, which we don't set (we run on OpenRouter). Default to a
            # free passthrough reranker (no key/model/network); GRAPHITI_RERANKER=openai opts into the
            # LLM reranker on our OpenRouter creds.
            self._graphiti = Graphiti(
                graph_driver=driver, llm_client=llm, embedder=self._build_embedder(),
                cross_encoder=self._build_cross_encoder())
        return self._graphiti

    def _build_cross_encoder(self):
        """Passthrough reranker by default (free, no key); OpenRouter LLM reranker when selected.

        Graphiti already hybrid-ranks (vector + BM25) before the cross-encoder, so preserving that
        order is a sound €0 default. ``GRAPHITI_RERANKER=openai`` uses the LLM reranker via the same
        OpenRouter creds (only if that model/endpoint supports the reranking call)."""
        if (os.environ.get("GRAPHITI_RERANKER") or "passthrough").lower() == "openai":
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from graphiti_core.llm_client import LLMConfig
            return OpenAIRerankerClient(config=LLMConfig(
                api_key=self.api_key, model=self.model, base_url=self.base_url))
        return _make_passthrough_reranker()

    def _build_embedder(self):
        """Local fastembed by default (free, no key); OpenAI embeddings when explicitly selected."""
        if self.embedder_kind == "openai":
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("GRAPHITI_EMBEDDER=openai requires OPENAI_API_KEY "
                                 "(OpenRouter has no embeddings endpoint)")
            return OpenAIEmbedder(config=OpenAIEmbedderConfig(
                api_key=key, embedding_model=self.embed_model))
        return _make_local_embedder()

    def init_indices(self) -> None:
        """One-time index/constraint setup (BM25 + vector). Idempotent; run on a fresh DB file."""
        self._run(self._client().build_indices_and_constraints())
        self._indices_built = True

    def _ensure_indices(self) -> None:
        """Build indices once per process before the first write (cheap no-op if already present)."""
        if not self._indices_built:
            self.init_indices()

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

        self._ensure_indices()
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


def _make_passthrough_reranker():
    """A free, no-key cross-encoder that keeps Graphiti's existing hybrid-ranking order.

    Graphiti requires a ``cross_encoder``; its default (``OpenAIRerankerClient``) needs an
    ``OPENAI_API_KEY`` we don't set. Search results already arrive vector+BM25-ranked, so returning
    them in order (descending scores) is a sound €0 default with no model, key, or network. Defined
    lazily so this module imports without ``graphiti_core`` present."""
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class _PassthroughReranker(CrossEncoderClient):
        async def rank(self, query: str, passages: list) -> list:
            n = len(passages)
            return [(p, (n - i) / n) for i, p in enumerate(passages)]

    return _PassthroughReranker()


def _make_local_embedder(model_name: Optional[str] = None):
    """A free, no-key Graphiti embedder backed by fastembed (ONNX; no torch/GPU/API key).

    Defined lazily so this module imports without ``graphiti_core``/``fastembed`` present. Lets the
    whole backend run on MiroFish's existing OpenRouter key (extraction) + local embeddings, needing
    zero new credentials — OpenRouter itself has no embeddings endpoint (ADR 0009).
    """
    from graphiti_core.embedder.client import EmbedderClient
    from fastembed import TextEmbedding

    # A fixed cache dir (env-set) lets the worker image bake the ONNX model at BUILD time so the
    # weekly run never depends on a HuggingFace download — key for a hands-off multi-year job.
    cache_dir = os.environ.get("FASTEMBED_CACHE_DIR") or None
    model = TextEmbedding(model_name=model_name or DEFAULT_LOCAL_EMBED_MODEL, cache_dir=cache_dir)

    class _LocalEmbedder(EmbedderClient):
        async def create(self, input_data):
            text = input_data if isinstance(input_data, str) else " ".join(map(str, input_data))
            return next(iter(model.embed([text]))).tolist()

        async def create_batch(self, input_data_list):
            return [vec.tolist() for vec in model.embed(list(input_data_list))]

    return _LocalEmbedder()
