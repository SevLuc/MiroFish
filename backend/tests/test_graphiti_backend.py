"""Unit tests for the Graphiti backend's pure logic (no FalkorDB / LLM needed).

Covers ``ontology_to_graphiti_types`` — the conversion of MiroFish's generated ontology JSON into
Graphiti's typed-extraction inputs — which is the one piece with real branching worth pinning.
The FalkorDB/LLM-dependent paths are exercised by the end-to-end parity test, not here.
"""

from app.services.graphiti_backend import ontology_to_graphiti_types


def _onto():
    return {
        "entity_types": [
            {"name": "semiconductor_company",
             "attributes": [{"name": "ticker", "type": "text", "description": "stock symbol"},
                            {"name": "summary", "type": "text", "description": "reserved — dropped"}]},
            {"name": "Person", "attributes": []},
        ],
        "edge_types": [
            {"name": "supplies to", "attributes": [],
             "source_targets": [{"source": "semiconductor_company", "target": "semiconductor_company"}]},
        ],
    }


def test_entity_types_are_pascal_cased_and_have_no_attributes_by_default():
    # Default: type NAMES (labels) are produced, but models carry NO attribute fields — FalkorDB
    # can't store the nested attribute values graphiti would spread onto nodes (see
    # _extract_attributes_enabled). Typed labels are what MiroFish's persona filter needs.
    entity_types, _, _ = ontology_to_graphiti_types(_onto())
    assert set(entity_types) == {"SemiconductorCompany", "Person"}
    assert list(entity_types["SemiconductorCompany"].model_fields) == []
    assert list(entity_types["Person"].model_fields) == []


def test_attributes_become_fields_when_explicitly_enabled(monkeypatch):
    # Opt-in path (GRAPHITI_EXTRACT_ATTRIBUTES=true): attributes map to model fields, with
    # graphiti-reserved names (summary/name/uuid/...) filtered out.
    monkeypatch.setenv("GRAPHITI_EXTRACT_ATTRIBUTES", "true")
    entity_types, _, _ = ontology_to_graphiti_types(_onto())
    assert "ticker" in entity_types["SemiconductorCompany"].model_fields
    assert "summary" not in entity_types["SemiconductorCompany"].model_fields
    assert list(entity_types["Person"].model_fields) == []


def test_edge_types_are_upper_snake_and_map_to_their_source_target_pairs():
    _, edge_types, edge_type_map = ontology_to_graphiti_types(_onto())
    assert set(edge_types) == {"SUPPLIES_TO"}
    assert edge_type_map == {("SemiconductorCompany", "SemiconductorCompany"): ["SUPPLIES_TO"]}


def test_empty_or_missing_ontology_is_safe():
    assert ontology_to_graphiti_types({}) == ({}, {}, {})
    assert ontology_to_graphiti_types({"entity_types": None, "edge_types": None}) == ({}, {}, {})


def test_unnamed_types_are_skipped():
    entity_types, edge_types, _ = ontology_to_graphiti_types(
        {"entity_types": [{"attributes": []}], "edge_types": [{"name": "", "source_targets": []}]})
    assert entity_types == {} and edge_types == {}
