"""GRAPH_BACKEND seam tests (ADR 0009).

Verify that under ``GRAPH_BACKEND=graphiti`` the three legacy Zep service classes
(``ZepEntityReader`` / ``ZepToolsService`` / ``GraphBuilderService``):

  * construct with **no** ``ZEP_API_KEY`` and never build a Zep client, and
  * route their graph reads/writes/search to the injected ``GraphitiBackend`` (returning the same
    dict / ``NodeInfo`` / ``EdgeInfo`` / ``SearchResult`` shapes MiroFish already consumes).

The backend is faked via ``graph_backend.reset_graphiti_backend``, so no FalkorDB, graphiti-core or
LLM is needed. Importing the service modules still pulls their normal deps (flask, zep_cloud, …);
where those are absent (a bare checkout) the whole module is skipped rather than failed.
"""

import os

import pytest

# The service modules import flask / zep_cloud at module load; skip cleanly if unavailable.
pytest.importorskip("flask")
pytest.importorskip("zep_cloud")

os.environ["GRAPH_BACKEND"] = "graphiti"
os.environ.setdefault("LLM_API_KEY", "test-openrouter-key")
os.environ.pop("ZEP_API_KEY", None)

# Importing the service package runs app/services/__init__.py, which pulls the full pipeline
# (camel-oasis, openai, …). On a bare checkout those are absent — skip rather than error.
try:
    from app.services import graph_backend  # noqa: E402
    from app.services.zep_entity_reader import ZepEntityReader  # noqa: E402
    from app.services.zep_tools import ZepToolsService, NodeInfo, EdgeInfo, SearchResult  # noqa: E402
    from app.services.graph_builder import GraphBuilderService, BatchSubmission  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"service deps unavailable: {exc}", allow_module_level=True)


class _FakeBackend:
    """Stand-in for GraphitiBackend recording writes and returning canned reads."""

    def __init__(self):
        self.added = []
        self.deleted = []
        self.nodes = [{"uuid": "n1", "name": "NVDA", "labels": ["Company"],
                       "summary": "GPU maker", "attributes": {"ticker": "NVDA"}}]
        self.edges = [{"uuid": "e1", "name": "SUPPLIES", "fact": "MU supplies HBM to NVDA",
                       "source_node_uuid": "n2", "target_node_uuid": "n1", "attributes": {},
                       "valid_at": "2026-01-01T00:00:00+00:00", "invalid_at": None,
                       "expired_at": None}]

    def get_all_nodes(self, cluster):
        return list(self.nodes)

    def get_all_edges(self, cluster):
        return list(self.edges)

    def search_graph(self, cluster, query, *, limit=10):
        class _E:  # graphiti edge-like object (attribute access)
            uuid, name, fact = "e1", "SUPPLIES", "MU supplies HBM to NVDA"
            source_node_uuid, target_node_uuid = "n2", "n1"
        return {"facts": [{"fact": _E.fact}], "edges": [_E()]}

    def add_documents(self, cluster, chunks, *, ontology=None, reference_time=None, progress=None):
        self.added.append((cluster, list(chunks), ontology))
        if progress:
            progress("done", 1.0)
        return len(chunks)

    def delete_cluster(self, cluster):
        self.deleted.append(cluster)

    def close(self):
        pass


@pytest.fixture
def fake_backend():
    fb = _FakeBackend()
    graph_backend.reset_graphiti_backend(fb)
    yield fb
    graph_backend.reset_graphiti_backend()


def test_services_construct_without_zep_key_under_graphiti(fake_backend):
    # No ZEP_API_KEY in env, yet all three construct and hold no Zep client.
    assert ZepEntityReader().client is None
    assert ZepToolsService().client is None
    assert GraphBuilderService().client is None


def test_reader_reads_route_to_graphiti(fake_backend):
    reader = ZepEntityReader()
    nodes = reader.get_all_nodes("AI")
    edges = reader.get_all_edges("AI")
    assert nodes[0]["name"] == "NVDA" and nodes[0]["labels"] == ["Company"]
    assert edges[0]["fact"] == "MU supplies HBM to NVDA"

    # filter_defined_entities builds on those primitives -> typed node survives, enriched by the edge
    filtered = reader.filter_defined_entities(graph_id="AI", enrich_with_edges=True)
    assert filtered.filtered_count == 1
    ent = filtered.entities[0]
    assert ent.name == "NVDA"
    assert any(e["fact"] == "MU supplies HBM to NVDA" for e in ent.related_edges)

    # get_entity_with_context resolves the node from the group set (no single-node Zep call)
    ctx = reader.get_entity_with_context("AI", "n1")
    assert ctx is not None and ctx.name == "NVDA"
    assert reader.get_entity_with_context("AI", "missing") is None


def test_tools_reads_and_search_route_to_graphiti(fake_backend):
    tools = ZepToolsService()
    nodes = tools.get_all_nodes("AI")
    edges = tools.get_all_edges("AI")
    assert isinstance(nodes[0], NodeInfo) and nodes[0].name == "NVDA"
    assert isinstance(edges[0], EdgeInfo) and edges[0].valid_at == "2026-01-01T00:00:00+00:00"

    res = tools.search_graph("AI", "who supplies NVDA", limit=5)
    assert isinstance(res, SearchResult)
    assert "MU supplies HBM to NVDA" in res.facts
    assert res.edges[0]["source_node_uuid"] == "n2"


def test_builder_build_sequence_routes_to_graphiti(fake_backend):
    builder = GraphBuilderService()
    onto = {"entity_types": [{"name": "Company", "attributes": []}], "edge_types": []}

    seen = {}
    gid = builder.create_graph(name="AI", graph_id_callback=lambda g: seen.setdefault("id", g))
    assert gid and seen["id"] == gid            # id generated + callback fired, no Zep create
    builder.set_ontology(gid, onto)             # stashed for the ingest
    sub = builder.add_text_batches(gid, ["chunk-a", "chunk-b"])
    assert isinstance(sub, BatchSubmission) and sub.item_count == 2
    # ontology stashed by set_ontology flowed into the ingest
    assert fake_backend.added[0][0] == gid and fake_backend.added[0][2] == onto

    # _wait_for_batch is a no-op for graphiti (extraction already ran)
    assert builder._wait_for_batch(sub) == []
    # get_batch_summary reports "not resumable" so the API never enters batch-resume
    assert builder.get_batch_summary("graphiti:x") is None

    data = builder.get_graph_data(gid)
    assert data["node_count"] == 1 and data["edge_count"] == 1
    assert data["edges"][0]["target_node_name"] == "NVDA"   # UUID resolved to a name

    builder.delete_graph(gid)
    assert fake_backend.deleted == [gid]


def test_append_passes_ontology_through(fake_backend):
    # Append path: a fresh builder with no set_ontology must still apply the ontology it is handed.
    builder = GraphBuilderService()
    onto = {"entity_types": [{"name": "Company", "attributes": []}], "edge_types": []}
    builder.add_text_batches("AI", ["news"], ontology=onto)
    assert fake_backend.added[-1][2] == onto
