from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from rag_app.pipeline.structured_extraction_protocol import (
    PROMPT_VERSION,
    SchemaPreflightLimitError,
    SchemaPreflightLimits,
    SchemaTokenCounterError,
    SchemaTokenLimitError,
    StructuredExtractionProtocolError,
    TableAlignmentConflictError,
    TablePredictionError,
    UnsupportedSchemaKeywordError,
    UnsupportedTableSchemaError,
    build_request_hash,
    build_varex_prompt,
    canonical_json_bytes,
    canonical_json_sha256,
    chunk_nested_schema,
    is_schema_echo,
    merge_table_array_predictions,
    normalize_extraction_schema,
    normalize_request_schema,
    resolve_local_refs,
    split_table_schema,
)


def _word_tokens(value: str) -> int:
    return len(value.split())


def test_canonical_json_and_request_hash_are_order_independent() -> None:
    first = {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "integer"}}}
    second = {"properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "type": "object"}
    source_hash = "a" * 64

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert build_request_hash(
        first,
        source_sha256=source_hash,
        model="ibm-granite/granite-vision-4.1-4b",
        model_revision="abc123",
    ) == build_request_hash(
        second,
        source_sha256=source_hash,
        model="ibm-granite/granite-vision-4.1-4b",
        model_revision="abc123",
    )


def test_request_hash_changes_with_protocol_inputs() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    source_hash = "b" * 64
    model = "ibm-granite/granite-vision-4.1-4b"
    base = build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
    )

    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-2",
    )
    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
        prompt_version=PROMPT_VERSION + "-next",
    )
    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
        flat_nested_max_tokens=2048,
    )
    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
        max_leaves=12,
    )
    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
        max_schema_tokens=1024,
    )
    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
        tokenizer_policy="granite-tokenizer@rev-2",
    )
    assert base != build_request_hash(
        schema,
        source_sha256=source_hash,
        model=model,
        model_revision="rev-1",
        limits=SchemaPreflightLimits(max_chunks=8),
    )


def test_resolves_local_defs_without_mutating_input_and_makes_leaves_nullable() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "address": {
                "type": "object",
                "properties": {
                    "zip": {"type": "string"},
                    "city": {"type": "string"},
                },
            }
        },
        "properties": {
            "name": {"type": "string"},
            "address": {"$ref": "#/$defs/address", "description": "Mailing address"},
        },
    }
    original = copy.deepcopy(schema)

    normalized = normalize_extraction_schema(schema)

    assert schema == original
    assert "$defs" not in normalized
    assert list(normalized["properties"]) == ["address", "name"]
    assert normalized["required"] == ["address", "name"]
    address = normalized["properties"]["address"]
    assert address["description"] == "Mailing address"
    assert address["type"] == ["object", "null"]
    assert address["properties"]["city"]["type"] == ["string", "null"]


def test_rejects_external_unresolved_and_cyclic_refs() -> None:
    with pytest.raises(StructuredExtractionProtocolError, match="external"):
        resolve_local_refs({"type": "object", "properties": {"x": {"$ref": "other.json#/x"}}})
    with pytest.raises(StructuredExtractionProtocolError, match="unresolved"):
        resolve_local_refs({"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}})

    cyclic = {
        "type": "object",
        "$defs": {
            "a": {"$ref": "#/$defs/b"},
            "b": {"$ref": "#/$defs/a"},
        },
        "properties": {"x": {"$ref": "#/$defs/a"}},
    }
    with pytest.raises(StructuredExtractionProtocolError, match="cyclic"):
        resolve_local_refs(cyclic)


@pytest.mark.parametrize(
    "keyword",
    [
        "allOf",
        "anyOf",
        "oneOf",
        "if",
        "dependentRequired",
        "patternProperties",
        "const",
    ],
)
def test_schema_validation_is_fail_closed_for_unsupported_semantics(keyword: str) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        keyword: [],
    }

    with pytest.raises(UnsupportedSchemaKeywordError, match=keyword):
        normalize_extraction_schema(schema)


def test_schema_validation_rejects_unknown_keywords_and_schema_additional_properties() -> None:
    unknown = {
        "type": "object",
        "properties": {"value": {"type": "string", "x-vendor": True}},
    }
    dynamic = {
        "type": "object",
        "properties": {},
        "additionalProperties": {"type": "string"},
    }

    with pytest.raises(UnsupportedSchemaKeywordError, match="x-vendor"):
        normalize_extraction_schema(unknown)
    with pytest.raises(UnsupportedSchemaKeywordError, match="additionalProperties"):
        normalize_extraction_schema(dynamic)

    union = {
        "type": "object",
        "properties": {"value": {"type": ["string", "number"]}},
    }
    with pytest.raises(UnsupportedSchemaKeywordError, match="multi-type unions"):
        normalize_extraction_schema(union)

    mismatched = {
        "type": "object",
        "properties": {"value": {"type": "string", "properties": {}}},
    }
    with pytest.raises(StructuredExtractionProtocolError, match="object type"):
        normalize_extraction_schema(mismatched)


def test_table_schema_is_explicitly_unsupported() -> None:
    table = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
            }
        },
    }

    with pytest.raises(UnsupportedTableSchemaError, match="table splitting"):
        normalize_extraction_schema(table)
    with pytest.raises(UnsupportedTableSchemaError):
        chunk_nested_schema(table, token_counter=_word_tokens)


def test_nested_chunking_is_stable_and_bounded_by_leaf_count() -> None:
    schema = {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": {
                    f"field_{index:02d}": {"type": "string"}
                    for index in reversed(range(25))
                },
            }
        },
    }

    chunks = chunk_nested_schema(
        schema,
        token_counter=_word_tokens,
        max_schema_tokens=100_000,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["c000", "c001"]
    assert [chunk.leaf_count for chunk in chunks] == [24, 1]
    assert all(chunk.leaf_count <= 24 for chunk in chunks)
    assert chunks[0].paths[0] == "/section/field_00"
    assert chunks[1].paths == ("/section/field_24",)
    assert list(chunks[0].schema["properties"]["section"]["properties"])[0] == "field_00"


def test_injected_token_counter_splits_and_rejects_oversized_single_leaf() -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "description": "short"},
            "b": {"type": "string", "description": "short"},
        },
    }
    baseline = chunk_nested_schema(schema, token_counter=len, max_schema_tokens=10_000)
    one_leaf_size = len(json.dumps(baseline[0].schema, ensure_ascii=False, indent=2)) - 20

    chunks = chunk_nested_schema(schema, token_counter=len, max_schema_tokens=one_leaf_size)
    assert [chunk.leaf_count for chunk in chunks] == [1, 1]

    huge = {
        "type": "object",
        "properties": {"a": {"type": "string", "description": "x" * 500}},
    }
    with pytest.raises(SchemaTokenLimitError, match="/a"):
        chunk_nested_schema(huge, token_counter=len, max_schema_tokens=100)


def test_preflight_caps_input_bytes_depth_refs_and_total_leaves() -> None:
    oversized = {
        "type": "object",
        "properties": {"value": {"type": "string", "description": "x" * 100}},
    }
    with pytest.raises(SchemaPreflightLimitError, match="maximum size"):
        normalize_extraction_schema(
            oversized,
            limits=SchemaPreflightLimits(max_input_bytes=80),
        )

    nested = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }
        },
    }
    with pytest.raises(SchemaPreflightLimitError, match="maximum depth"):
        normalize_extraction_schema(
            nested,
            limits=SchemaPreflightLimits(max_depth=4),
        )

    repeated_ref = {
        "type": "object",
        "$defs": {"value": {"type": "string"}},
        "properties": {
            "a": {"$ref": "#/$defs/value"},
            "b": {"$ref": "#/$defs/value"},
        },
    }
    with pytest.raises(SchemaPreflightLimitError, match=r"\$ref expansions"):
        normalize_extraction_schema(
            repeated_ref,
            limits=SchemaPreflightLimits(max_ref_expansions=1),
        )

    three_leaves = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in ("a", "b", "c")},
    }
    with pytest.raises(SchemaPreflightLimitError, match="total leaves"):
        normalize_extraction_schema(
            three_leaves,
            limits=SchemaPreflightLimits(max_total_leaves=2),
        )


def test_preflight_caps_chunks_and_total_schema_tokens() -> None:
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in ("a", "b", "c")},
    }

    with pytest.raises(SchemaPreflightLimitError, match="maximum chunks"):
        chunk_nested_schema(
            schema,
            token_counter=len,
            max_leaves=1,
            max_schema_tokens=100_000,
            limits=SchemaPreflightLimits(max_chunks=2),
        )
    with pytest.raises(SchemaPreflightLimitError, match="total schema tokens"):
        chunk_nested_schema(
            schema,
            token_counter=len,
            max_schema_tokens=100_000,
            limits=SchemaPreflightLimits(max_total_schema_tokens=10),
        )


def test_default_preflight_caps_logical_chunks_at_sixteen() -> None:
    assert SchemaPreflightLimits().max_chunks == 16


@pytest.mark.parametrize(
    "token_counter",
    [lambda _value: -1, lambda _value: "10", lambda _value: 1 / 0],
)
def test_token_counter_failures_are_controlled(token_counter: object) -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    with pytest.raises(SchemaTokenCounterError):
        chunk_nested_schema(schema, token_counter=token_counter)  # type: ignore[arg-type]


def test_detects_direct_and_wrapped_schema_echo_only() -> None:
    schema = {
        "type": "object",
        "properties": {"invoice_number": {"type": ["string", "null"]}},
        "required": ["invoice_number"],
    }

    assert is_schema_echo(schema, expected_schema=schema)
    assert is_schema_echo({"result": schema}, expected_schema=schema)
    assert is_schema_echo({"result": [schema]}, expected_schema=schema)
    assert is_schema_echo([schema], expected_schema=schema)
    assert not is_schema_echo({"invoice_number": "INV-42"}, expected_schema=schema)
    assert not is_schema_echo({"invoice_number": None}, expected_schema=schema)


def test_detects_echo_inside_an_expected_result_field() -> None:
    expected = {
        "type": "object",
        "properties": {"result": {"type": ["string", "null"]}},
        "required": ["result"],
    }

    assert is_schema_echo({"result": expected}, expected_schema=expected)


def test_schema_echo_ignores_expected_fields_named_like_schema_and_their_lists() -> None:
    expected = {
        "type": "object",
        "properties": {
            "type": {"type": ["string", "null"]},
            "properties": {"type": ["string", "null"]},
            "line_items": {"type": ["array", "null"], "items": {"type": "string"}},
        },
        "required": ["line_items", "properties", "type"],
    }
    instance = {
        "type": "object",
        "properties": "customer supplied value",
        "line_items": [{"type": "object", "properties": {"code": {"type": "string"}}}],
        "producer_metadata": {"type": "object", "properties": {"source": {"type": "string"}}},
    }

    assert not is_schema_echo(instance, expected_schema=expected)
    assert is_schema_echo({"wrapper": [expected]}, expected_schema=expected)


def test_varex_prompt_requires_instance_and_nulls() -> None:
    schema = normalize_extraction_schema(
        {"type": "object", "properties": {"invoice_number": {"type": "string"}}}
    )
    prompt = build_varex_prompt(schema)

    assert "Return null for fields you cannot find." in prompt
    assert "Return ONLY valid JSON." in prompt
    assert "Return an instance of the JSON with extracted values" in prompt
    assert '"invoice_number"' in prompt


def _table_schema(*, columns: int = 20, second_table: bool = False) -> dict:
    row_properties = {
        "id": {"type": "string"},
        "name": {"type": "string"},
        **{f"value_{index:02d}": {"type": "string"} for index in range(columns)},
    }
    properties: dict = {
        "document_number": {"type": "string"},
        "rows": {
            "type": "array",
            "items": {"type": "object", "properties": row_properties},
        },
    }
    if second_table:
        properties["approvals"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        }
    return {"type": "object", "properties": properties}


def _chunk_prediction(chunk, rows: list[dict]) -> dict:
    array_name = chunk.array_path.removeprefix("/")
    fields = (*chunk.anchor_fields, *chunk.value_fields)
    return {array_name: [{field: row[field] for field in fields} for row in rows]}


def test_table_split_repeats_two_anchors_and_bounds_value_columns() -> None:
    schema = _table_schema(columns=20, second_table=True)
    plan = split_table_schema(
        schema,
        token_counter=_word_tokens,
        max_schema_tokens=100_000,
    )

    assert len(plan.scalar_chunks) == 1
    assert [table.array_path for table in plan.tables] == ["/approvals", "/rows"]
    rows = plan.tables[1]
    assert rows.anchor_fields == ("id", "name")
    assert [len(chunk.value_fields) for chunk in rows.chunks] == [8, 8, 4]
    assert all(chunk.anchor_fields == ("id", "name") for chunk in rows.chunks)
    assert all(len(chunk.value_fields) <= 8 for chunk in rows.chunks)
    assert all(
        set(chunk.schema["properties"]) == {"rows"}
        for chunk in rows.chunks
    )


def test_request_hash_supports_tables_and_fingerprints_table_policy() -> None:
    schema = _table_schema(columns=2)
    normalized = normalize_request_schema(schema)
    base = build_request_hash(
        schema,
        source_sha256="d" * 64,
        model="granite",
        model_revision="revision",
    )

    assert normalized["properties"]["rows"]["type"] == ["array", "null"]
    assert base != build_request_hash(
        schema,
        source_sha256="d" * 64,
        model="granite",
        model_revision="revision",
        table_max_columns=4,
    )
    assert base != build_request_hash(
        schema,
        source_sha256="d" * 64,
        model="granite",
        model_revision="revision",
        table_max_anchors=1,
    )
    assert base != build_request_hash(
        schema,
        source_sha256="d" * 64,
        model="granite",
        model_revision="revision",
        table_max_rows=25,
    )
    assert base != build_request_hash(
        schema,
        source_sha256="d" * 64,
        model="granite",
        model_revision="revision",
        table_max_cell_bytes=2048,
    )


def test_table_merge_uses_unique_composite_anchors_across_reordered_chunks() -> None:
    table = split_table_schema(
        _table_schema(columns=9),
        token_counter=_word_tokens,
        max_columns=5,
        max_schema_tokens=100_000,
    ).tables[0]
    source_rows = [
        {
            "id": "A",
            "name": "Alpha",
            **{f"value_{index:02d}": f"a{index}" for index in range(9)},
        },
        {
            "id": "B",
            "name": "Beta",
            **{f"value_{index:02d}": f"b{index}" for index in range(9)},
        },
    ]
    predictions = {
        table.chunks[0].chunk_id: _chunk_prediction(table.chunks[0], source_rows),
        table.chunks[1].chunk_id: _chunk_prediction(
            table.chunks[1], list(reversed(source_rows))
        ),
    }

    merged = merge_table_array_predictions(table, predictions)["rows"]

    assert merged == source_rows


def test_table_merge_rejects_empty_or_duplicate_planned_anchors() -> None:
    table = split_table_schema(
        _table_schema(columns=9),
        token_counter=_word_tokens,
        max_columns=5,
        max_schema_tokens=100_000,
    ).tables[0]
    source_rows = [
        {
            "id": "",
            "name": "",
            **{f"value_{index:02d}": f"a{index}" for index in range(9)},
        },
        {
            "id": "",
            "name": "",
            **{f"value_{index:02d}": f"b{index}" for index in range(9)},
        },
    ]
    predictions = {
        chunk.chunk_id: _chunk_prediction(chunk, source_rows)
        for chunk in table.chunks
    }
    with pytest.raises(TableAlignmentConflictError, match="non-empty unique anchors"):
        merge_table_array_predictions(table, predictions)

    duplicate_rows = [{**row, "id": "same", "name": "same"} for row in source_rows]
    predictions = {
        chunk.chunk_id: _chunk_prediction(
            chunk,
            duplicate_rows if index == 0 else list(reversed(duplicate_rows)),
        )
        for index, chunk in enumerate(table.chunks)
    }
    with pytest.raises(TableAlignmentConflictError, match="non-empty unique anchors"):
        merge_table_array_predictions(table, predictions)


def test_table_merge_rejects_anchor_set_conflicts_and_missing_chunks() -> None:
    table = split_table_schema(
        _table_schema(columns=9),
        token_counter=_word_tokens,
        max_columns=5,
        max_schema_tokens=100_000,
    ).tables[0]
    first_rows = [
        {
            "id": "A",
            "name": "Alpha",
            **{f"value_{index:02d}": f"a{index}" for index in range(9)},
        }
    ]
    second_rows = [{**first_rows[0], "id": "B"}]
    predictions = {
        table.chunks[0].chunk_id: _chunk_prediction(table.chunks[0], first_rows),
        table.chunks[1].chunk_id: _chunk_prediction(table.chunks[1], second_rows),
    }
    with pytest.raises(TableAlignmentConflictError, match="different anchor sets"):
        merge_table_array_predictions(table, predictions)

    predictions.pop(table.chunks[1].chunk_id)
    with pytest.raises(TablePredictionError, match="every planned chunk"):
        merge_table_array_predictions(table, predictions)


def test_table_split_and_predictions_fail_closed_on_unsupported_shapes() -> None:
    nested_table = {
        "type": "object",
        "properties": {
            "wrapper": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                        },
                    }
                },
            }
        },
    }
    with pytest.raises(UnsupportedTableSchemaError, match="nested table"):
        split_table_schema(
            nested_table,
            token_counter=_word_tokens,
            max_schema_tokens=100_000,
        )

    plan = split_table_schema(
        _table_schema(columns=1),
        token_counter=_word_tokens,
        max_schema_tokens=100_000,
    ).tables[0]
    chunk = plan.chunks[0]
    with pytest.raises(TablePredictionError, match="every requested field"):
        merge_table_array_predictions(
            plan,
            {chunk.chunk_id: {"rows": [{"id": "A"}]}},
        )


def test_table_split_allows_primitive_array_values_but_never_uses_them_as_anchors() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "code": {"type": "string"},
                    },
                },
            }
        },
    }

    table = split_table_schema(
        schema,
        token_counter=_word_tokens,
        max_schema_tokens=100_000,
    ).tables[0]

    assert table.anchor_fields == ("code",)
    assert table.chunks[0].value_fields == ("tags",)

    array_only = copy.deepcopy(schema)
    del array_only["properties"]["rows"]["items"]["properties"]["code"]
    ordinal_table = split_table_schema(
        array_only,
        token_counter=_word_tokens,
        max_schema_tokens=100_000,
    ).tables[0]
    assert ordinal_table.anchor_fields == ()
    chunk = ordinal_table.chunks[0]
    valid = {chunk.chunk_id: {"rows": [{"tags": ["a", "b"]}]}}
    assert merge_table_array_predictions(ordinal_table, valid) == {
        "rows": [{"tags": ["a", "b"]}]
    }
    with pytest.raises(TablePredictionError, match="expected array"):
        merge_table_array_predictions(
            ordinal_table,
            {chunk.chunk_id: {"rows": [{"tags": "not-an-array"}]}},
        )

    two_arrays = copy.deepcopy(array_only)
    two_arrays["properties"]["rows"]["items"]["properties"]["labels"] = {
        "type": "array",
        "items": {"type": "string"},
    }
    with pytest.raises(UnsupportedTableSchemaError, match="one safe chunk"):
        split_table_schema(
            two_arrays,
            token_counter=lambda value: value.count('"type"'),
            max_schema_tokens=6,
        )


def test_table_prediction_validates_enum_limits_and_row_cap() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "pattern": "^[A-Z]+$"},
                        "status": {"type": "string", "enum": ["open", "closed"]},
                        "values": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                            "maxItems": 2,
                        },
                    },
                },
            }
        },
    }
    table = split_table_schema(
        schema,
        token_counter=_word_tokens,
        max_schema_tokens=100_000,
    ).tables[0]
    chunk = table.chunks[0]
    with pytest.raises(TablePredictionError, match="outside enum"):
        merge_table_array_predictions(
            table,
            {
                chunk.chunk_id: {
                    "rows": [{"code": "ABC", "status": "unknown", "values": [1]}]
                }
            },
        )
    with pytest.raises(TablePredictionError, match="maximum rows"):
        merge_table_array_predictions(
            table,
            {
                chunk.chunk_id: {
                    "rows": [
                        {"code": "ABC", "status": "open", "values": [1]}
                        for _ in range(51)
                    ]
                }
            },
        )
    duplicate = {"code": "ABC", "status": "open", "values": [1]}
    with pytest.raises(TablePredictionError, match="uniqueItems"):
        merge_table_array_predictions(
            table,
            {chunk.chunk_id: {"rows": [duplicate, duplicate]}},
        )
    with pytest.raises(TablePredictionError, match="exceeds byte limit"):
        merge_table_array_predictions(
            table,
            {
                chunk.chunk_id: {
                    "rows": [{"code": "A" * 4097, "status": "open", "values": [1]}]
                }
            },
        )


def test_table_protocol_rejects_limits_above_hard_caps() -> None:
    schema = _table_schema(columns=9)
    with pytest.raises(ValueError, match="max_columns"):
        split_table_schema(
            schema,
            token_counter=_word_tokens,
            max_columns=9,
        )
    with pytest.raises(ValueError, match="max_anchors"):
        split_table_schema(
            schema,
            token_counter=_word_tokens,
            max_anchors=3,
        )
    with pytest.raises(ValueError, match="max_rows"):
        split_table_schema(
            schema,
            token_counter=_word_tokens,
            max_rows=51,
        )
    with pytest.raises(ValueError, match="max_cell_bytes"):
        split_table_schema(
            schema,
            token_counter=_word_tokens,
            max_cell_bytes=4097,
        )
    with pytest.raises(ValueError, match="table_max_columns"):
        build_request_hash(
            schema,
            source_sha256="e" * 64,
            model="granite",
            model_revision="revision",
            table_max_columns=9,
        )
    with pytest.raises(ValueError, match="table_max_rows"):
        build_request_hash(
            schema,
            source_sha256="e" * 64,
            model="granite",
            model_revision="revision",
            table_max_rows=51,
        )
    with pytest.raises(ValueError, match="table_max_cell_bytes"):
        build_request_hash(
            schema,
            source_sha256="e" * 64,
            model="granite",
            model_revision="revision",
            table_max_cell_bytes=4097,
        )
