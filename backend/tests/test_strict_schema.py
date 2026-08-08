"""make_strict_schema turns a raw model_json_schema() into an OpenAI-strict schema:
additionalProperties:false + every property required + optionals kept nullable, recursively.
See app/services/strict_schema.py and the GRAPHITI_STRUCTURED=json_schema_strict path."""
import copy

from app.services.strict_schema import make_strict_schema


def test_object_gets_additionalprops_false_and_all_required_without_mutating_input():
    s = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    frozen = copy.deepcopy(s)
    o = make_strict_schema(s)
    assert o["additionalProperties"] is False
    assert sorted(o["required"]) == ["a", "b"]
    assert s == frozen                                   # input untouched


def test_optional_field_becomes_required_but_stays_nullable():
    s = {"type": "object",
         "properties": {"a": {"type": "string"},
                        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
         "required": ["a"]}
    o = make_strict_schema(s)
    assert sorted(o["required"]) == ["a", "note"]
    assert {"type": "null"} in o["properties"]["note"]["anyOf"]


def test_non_nullable_optional_is_made_nullable():
    s = {"type": "object", "properties": {"a": {"type": "string"}}}
    o = make_strict_schema(s)
    assert "null" in o["properties"]["a"]["type"]


def test_nested_object_and_defs_and_array_items_are_recursed():
    s = {"type": "object",
         "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/E"}}},
         "$defs": {"E": {"type": "object", "properties": {"n": {"type": "string"}}}}}
    o = make_strict_schema(s)
    e = o["$defs"]["E"]
    assert e["additionalProperties"] is False and e["required"] == ["n"]
