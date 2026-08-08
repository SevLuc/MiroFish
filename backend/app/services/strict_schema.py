"""Turn a Pydantic/JSON schema into an OpenAI **strict** structured-output schema.

OpenAI strict structured outputs (gpt-5 family, gpt-4o-2024-08-06+) require, on every object:
``additionalProperties: false`` and **every** property listed in ``required`` — otherwise the API
returns ``400 invalid_json_schema``. graphiti-core's generic client sends the raw
``model_json_schema()`` (neither), which older models tolerated but gpt-5 rejects.

This transforms a schema to meet those rules while preserving "optional" semantics: a property that
wasn't required is added to ``required`` but made **nullable** (so the model may still emit ``null``).
Recurses through ``properties``, ``$defs``/``definitions``, array ``items``/``prefixItems``, and the
``anyOf``/``allOf``/``oneOf`` combinators. Pure + stdlib-only (never mutates the input).
"""
import copy

_COMBINATORS = ("anyOf", "allOf", "oneOf", "prefixItems")


def make_strict_schema(schema):
    """Return a deep-copied, OpenAI-strict-compliant version of ``schema``."""
    return _strict(copy.deepcopy(schema))


def _strict(node):
    if isinstance(node, list):
        return [_strict(n) for n in node]
    if not isinstance(node, dict):
        return node

    for defs_key in ("$defs", "definitions"):
        if isinstance(node.get(defs_key), dict):
            node[defs_key] = {k: _strict(v) for k, v in node[defs_key].items()}

    props = node.get("properties")
    if isinstance(props, dict):
        for k in list(props):
            props[k] = _strict(props[k])
        required = set(node.get("required") or [])
        for k in props:
            if k not in required:                 # optional -> keep nullable, but strict needs it required
                props[k] = _make_nullable(props[k])
        node["required"] = list(props.keys())
        node["additionalProperties"] = False

    if "items" in node:
        node["items"] = _strict(node["items"])

    for comb in _COMBINATORS:
        if isinstance(node.get(comb), list):
            node[comb] = [_strict(n) for n in node[comb]]

    return node


def _make_nullable(p):
    """Allow ``null`` as a value of property schema ``p`` (idempotent)."""
    if not isinstance(p, dict):
        return p
    t = p.get("type")
    if isinstance(t, str):
        if t != "null":
            p["type"] = [t, "null"]
    elif isinstance(t, list):
        if "null" not in t:
            p["type"] = t + ["null"]
    elif isinstance(p.get("anyOf"), list):
        if not any(isinstance(o, dict) and o.get("type") == "null" for o in p["anyOf"]):
            p["anyOf"] = p["anyOf"] + [{"type": "null"}]
    elif "$ref" in p:
        p["anyOf"] = [{"$ref": p.pop("$ref")}, {"type": "null"}]
    else:
        p["type"] = "null"
    return p
