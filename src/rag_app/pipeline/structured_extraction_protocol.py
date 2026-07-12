"""Deterministic protocol-v3 primitives for schema-guided extraction."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

PROTOCOL_VERSION = 3
PROMPT_VERSION = "granite-varex-split-v1"
DEFAULT_MAX_LEAVES = 24
DEFAULT_MAX_SCHEMA_TOKENS = 2048
DEFAULT_TOKENIZER_POLICY = "model-tokenizer-v1"
DEFAULT_TABLE_MAX_COLUMNS = 8
DEFAULT_TABLE_MAX_ANCHORS = 2
DEFAULT_TABLE_MAX_ROWS = 50
DEFAULT_TABLE_MAX_CELL_BYTES = 4096

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANNOTATION_KEYS = {
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
_ALLOWED_SCHEMA_KEYS = _ANNOTATION_KEYS | {
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalProperties",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "multipleOf",
    "pattern",
    "properties",
    "required",
    "type",
    "uniqueItems",
}
_UNSUPPORTED_SEMANTIC_KEYS = {
    "allOf",
    "anyOf",
    "contains",
    "const",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "else",
    "if",
    "maxContains",
    "maxProperties",
    "minContains",
    "minProperties",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_JSON_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_SCHEMA_MARKER_KEYS = {
    "$defs",
    "$ref",
    "$schema",
    "additionalProperties",
    "enum",
    "items",
    "properties",
    "required",
    "type",
}


class StructuredExtractionProtocolError(ValueError):
    """The schema cannot be handled without weakening the protocol."""


class UnsupportedTableSchemaError(StructuredExtractionProtocolError):
    """A table schema cannot be represented by the bounded table protocol."""


class TablePredictionError(StructuredExtractionProtocolError):
    """A table chunk prediction does not match its requested schema."""


class TableAlignmentConflictError(TablePredictionError):
    """Table chunks cannot be joined without guessing row identity or values."""


class SchemaTokenLimitError(StructuredExtractionProtocolError):
    """A single schema leaf cannot fit the configured schema token budget."""


class UnsupportedSchemaKeywordError(StructuredExtractionProtocolError):
    """The schema uses semantics that this deterministic protocol cannot preserve."""


class SchemaPreflightLimitError(StructuredExtractionProtocolError):
    """The schema exceeds a bounded protocol preflight limit."""


class SchemaTokenCounterError(StructuredExtractionProtocolError):
    """The injected tokenizer returned an unusable result."""


@dataclass(frozen=True)
class SchemaPreflightLimits:
    max_input_bytes: int = 256 * 1024
    max_depth: int = 32
    max_ref_expansions: int = 256
    max_total_leaves: int = 1000
    max_chunks: int = 16
    max_total_schema_tokens: int = 32_768

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_input_bytes,
                self.max_depth,
                self.max_ref_expansions,
                self.max_total_leaves,
                self.max_chunks,
                self.max_total_schema_tokens,
            )
        ):
            raise ValueError("all schema preflight limits must be positive")


DEFAULT_PREFLIGHT_LIMITS = SchemaPreflightLimits()


@dataclass(frozen=True)
class SchemaChunk:
    chunk_id: str
    paths: tuple[str, ...]
    schema: dict[str, Any]
    leaf_count: int
    schema_tokens: int


@dataclass(frozen=True)
class TableSchemaChunk:
    chunk_id: str
    array_path: str
    anchor_fields: tuple[str, ...]
    value_fields: tuple[str, ...]
    schema: dict[str, Any]
    schema_tokens: int


@dataclass(frozen=True)
class TableArrayPlan:
    array_path: str
    fields: tuple[str, ...]
    anchor_fields: tuple[str, ...]
    max_rows: int
    max_cell_bytes: int
    chunks: tuple[TableSchemaChunk, ...]


@dataclass(frozen=True)
class TableExtractionPlan:
    scalar_chunks: tuple[SchemaChunk, ...]
    tables: tuple[TableArrayPlan, ...]
    total_schema_tokens: int


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and immutable artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _check_json_structure(
    value: Any,
    *,
    limits: SchemaPreflightLimits,
    label: str,
) -> None:
    active_containers: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        if depth > limits.max_depth:
            raise SchemaPreflightLimitError(
                f"{label} exceeds maximum depth {limits.max_depth}"
            )
        if isinstance(item, Mapping):
            container_id = id(item)
            if container_id in active_containers:
                raise StructuredExtractionProtocolError(f"{label} contains a container cycle")
            active_containers.add(container_id)
            try:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise StructuredExtractionProtocolError(
                            f"{label} object keys must be strings"
                        )
                    visit(child, depth + 1)
            finally:
                active_containers.remove(container_id)
            return
        if isinstance(item, list):
            container_id = id(item)
            if container_id in active_containers:
                raise StructuredExtractionProtocolError(f"{label} contains a container cycle")
            active_containers.add(container_id)
            try:
                for child in item:
                    visit(child, depth + 1)
            finally:
                active_containers.remove(container_id)
            return
        if item is None or isinstance(item, (bool, int, str)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        raise StructuredExtractionProtocolError(f"{label} contains a non-JSON value")

    visit(value, 1)


def _validate_schema_node(node: Mapping[str, Any], *, path: str = "#") -> None:
    keys = set(node)
    unsupported = sorted(keys & _UNSUPPORTED_SEMANTIC_KEYS)
    if unsupported:
        raise UnsupportedSchemaKeywordError(
            f"unsupported schema semantics at {path}: {', '.join(unsupported)}"
        )
    unknown = sorted(keys - _ALLOWED_SCHEMA_KEYS)
    if unknown:
        raise UnsupportedSchemaKeywordError(
            f"unsupported schema keyword at {path}: {', '.join(unknown)}"
        )

    ref = node.get("$ref")
    if ref is not None:
        if not isinstance(ref, str):
            raise StructuredExtractionProtocolError(f"$ref must be a string at {path}")
        structural_siblings = keys - {"$ref"} - _ANNOTATION_KEYS
        if structural_siblings:
            raise UnsupportedSchemaKeywordError(
                f"structural siblings beside $ref are unsupported at {path}"
            )
        return

    schema_type = node.get("type")
    declared_types: set[str] = set()
    if schema_type is not None:
        type_values = [schema_type] if isinstance(schema_type, str) else schema_type
        if (
            not isinstance(type_values, list)
            or not type_values
            or not all(isinstance(item, str) and item in _JSON_SCHEMA_TYPES for item in type_values)
        ):
            raise StructuredExtractionProtocolError(f"invalid schema type at {path}")
        if len(type_values) != len(set(type_values)):
            raise StructuredExtractionProtocolError(f"schema type contains duplicates at {path}")
        declared_types = set(type_values)
        non_null_types = declared_types - {"null"}
        if len(non_null_types) > 1:
            raise UnsupportedSchemaKeywordError(
                f"multi-type unions are unsupported at {path}"
            )

    for keyword in ("$id", "$schema", "format", "pattern"):
        if keyword in node and not isinstance(node[keyword], str):
            raise StructuredExtractionProtocolError(f"{keyword} must be a string at {path}")
    for keyword in ("maxItems", "maxLength", "minItems", "minLength"):
        value = node.get(keyword)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise StructuredExtractionProtocolError(
                f"{keyword} must be a non-negative integer at {path}"
            )
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise StructuredExtractionProtocolError(f"uniqueItems must be boolean at {path}")
    for keyword in (
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maximum",
        "minimum",
        "multipleOf",
    ):
        value = node.get(keyword)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise StructuredExtractionProtocolError(f"{keyword} must be numeric at {path}")
    if "multipleOf" in node and node["multipleOf"] <= 0:
        raise StructuredExtractionProtocolError(f"multipleOf must be positive at {path}")

    definitions = node.get("$defs")
    if definitions is not None:
        if not isinstance(definitions, Mapping):
            raise StructuredExtractionProtocolError(f"$defs must be an object at {path}")
        for name, child in definitions.items():
            if not isinstance(child, Mapping):
                raise StructuredExtractionProtocolError(
                    f"definition must be a schema object at {path}/$defs/{_encode_pointer_token(name)}"
                )
            _validate_schema_node(
                child,
                path=f"{path}/$defs/{_encode_pointer_token(name)}",
            )

    properties = node.get("properties")
    if properties is not None:
        if declared_types and "object" not in declared_types:
            raise StructuredExtractionProtocolError(
                f"properties require object type at {path}"
            )
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError(f"properties must be an object at {path}")
        for name, child in properties.items():
            if not isinstance(child, Mapping):
                raise StructuredExtractionProtocolError(
                    f"property must be a schema object at {path}/properties/{_encode_pointer_token(name)}"
                )
            _validate_schema_node(
                child,
                path=f"{path}/properties/{_encode_pointer_token(name)}",
            )

    items = node.get("items")
    if items is not None:
        if properties is not None or (declared_types and "array" not in declared_types):
            raise StructuredExtractionProtocolError(f"items require array type at {path}")
        if not isinstance(items, Mapping):
            raise UnsupportedSchemaKeywordError(
                f"tuple or boolean items are unsupported at {path}/items"
            )
        _validate_schema_node(items, path=f"{path}/items")

    required = node.get("required")
    if required is not None and (
        not isinstance(required, list)
        or not all(isinstance(item, str) for item in required)
        or len(required) != len(set(required))
    ):
        raise StructuredExtractionProtocolError(f"required must contain unique strings at {path}")
    if required is not None:
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError(f"required needs properties at {path}")
        unknown_required = sorted(set(required) - set(properties))
        if unknown_required:
            raise StructuredExtractionProtocolError(
                f"required names unknown properties at {path}: {', '.join(unknown_required)}"
            )
    if "additionalProperties" in node and not isinstance(node["additionalProperties"], bool):
        raise UnsupportedSchemaKeywordError(
            f"schema-valued additionalProperties is unsupported at {path}"
        )
    if "additionalProperties" in node and declared_types and "object" not in declared_types:
        raise StructuredExtractionProtocolError(
            f"additionalProperties requires object type at {path}"
        )
    if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
        raise StructuredExtractionProtocolError(f"enum must be a non-empty array at {path}")


def _preflight_input(schema: Mapping[str, Any], *, limits: SchemaPreflightLimits) -> None:
    _check_json_structure(schema, limits=limits, label="input schema")
    try:
        input_size = len(canonical_json_bytes(dict(schema)))
    except (TypeError, ValueError, RecursionError) as exc:
        raise StructuredExtractionProtocolError("input schema is not canonical JSON") from exc
    if input_size > limits.max_input_bytes:
        raise SchemaPreflightLimitError(
            f"input schema exceeds maximum size {limits.max_input_bytes} bytes"
        )
    _validate_schema_node(schema)


def _decode_pointer_token(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _encode_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _lookup_local_ref(root: Mapping[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise StructuredExtractionProtocolError(f"external or unsupported $ref: {ref}")
    current: Any = root
    for token in ref[2:].split("/"):
        if not isinstance(current, Mapping):
            raise StructuredExtractionProtocolError(f"invalid local $ref: {ref}")
        key = _decode_pointer_token(token)
        if key not in current:
            raise StructuredExtractionProtocolError(f"unresolved local $ref: {ref}")
        current = current[key]
    return current


def resolve_local_refs(
    schema: Mapping[str, Any],
    *,
    limits: SchemaPreflightLimits = DEFAULT_PREFLIGHT_LIMITS,
) -> dict[str, Any]:
    """Inline local JSON Pointer refs without mutating the caller's schema."""

    _preflight_input(schema, limits=limits)
    root = copy.deepcopy(dict(schema))
    ref_expansions = 0

    def resolve(node: Mapping[str, Any], stack: tuple[str, ...], depth: int) -> dict[str, Any]:
        nonlocal ref_expansions
        if depth > limits.max_depth:
            raise SchemaPreflightLimitError(
                f"resolved schema exceeds maximum depth {limits.max_depth}"
            )
        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str):  # Guarded by preflight; retained for type narrowing.
                raise StructuredExtractionProtocolError("$ref must be a string")
            if ref in stack:
                chain = " -> ".join((*stack, ref))
                raise StructuredExtractionProtocolError(f"cyclic local $ref: {chain}")
            ref_expansions += 1
            if ref_expansions > limits.max_ref_expansions:
                raise SchemaPreflightLimitError(
                    f"schema exceeds maximum $ref expansions {limits.max_ref_expansions}"
                )
            raw_target = _lookup_local_ref(root, ref)
            if not isinstance(raw_target, Mapping):
                raise StructuredExtractionProtocolError(f"$ref target must be an object: {ref}")
            target = resolve(raw_target, (*stack, ref), depth + 1)
            if not isinstance(target, Mapping):
                raise StructuredExtractionProtocolError(f"$ref target must be an object: {ref}")
            merged = dict(target)
            for key in _ANNOTATION_KEYS:
                if key in node:
                    merged[key] = copy.deepcopy(node[key])
            return merged

        resolved: dict[str, Any] = {}
        for key, item in node.items():
            if key == "$defs":
                continue
            if key == "properties":
                if not isinstance(item, Mapping):  # Guarded by preflight.
                    raise StructuredExtractionProtocolError("properties must be an object")
                resolved[key] = {
                    name: resolve(child, stack, depth + 1)
                    for name, child in item.items()
                    if isinstance(child, Mapping)
                }
            elif key == "items":
                if not isinstance(item, Mapping):  # Guarded by preflight.
                    raise StructuredExtractionProtocolError("items must be a schema object")
                resolved[key] = resolve(item, stack, depth + 1)
            else:
                resolved[key] = copy.deepcopy(item)
        return resolved

    resolved = resolve(root, (), 1)
    _check_json_structure(resolved, limits=limits, label="resolved schema")
    return resolved


def _normalized_type(value: Any, *, nullable: bool) -> str | list[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise StructuredExtractionProtocolError("schema type must be a string or string array")
    unique = sorted(set(values) - {"null"})
    if nullable:
        unique.append("null")
    return unique[0] if len(unique) == 1 else unique


def _is_object_schema(schema: Mapping[str, Any]) -> bool:
    value = schema.get("type")
    return value == "object" or (isinstance(value, list) and "object" in value) or "properties" in schema


def _is_array_schema(schema: Mapping[str, Any]) -> bool:
    value = schema.get("type")
    return value == "array" or (isinstance(value, list) and "array" in value)


def _stable_nullable_schema(
    value: Any,
    *,
    is_root: bool = False,
    allow_table: bool = False,
) -> Any:
    if isinstance(value, list):
        return [_stable_nullable_schema(item, allow_table=allow_table) for item in value]
    if not isinstance(value, Mapping):
        return value

    node = dict(value)
    if _is_object_schema(node):
        properties = node.get("properties", {})
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError("object properties must be an object")
        stable_properties = {
            key: _stable_nullable_schema(properties[key], allow_table=allow_table)
            for key in sorted(properties)
        }
        out = {
            key: _stable_nullable_schema(item, allow_table=allow_table)
            for key, item in sorted(node.items())
            if key not in {"properties", "required", "type"}
        }
        out["type"] = "object" if is_root else ["object", "null"]
        out["properties"] = stable_properties
        out["required"] = sorted(stable_properties)
        return out

    if _is_array_schema(node):
        items = node.get("items", {})
        if isinstance(items, Mapping) and _is_object_schema(items) and not allow_table:
            raise UnsupportedTableSchemaError(
                "array-of-object schemas require table splitting and merge support"
            )
        out = {
            key: _stable_nullable_schema(item, allow_table=allow_table)
            for key, item in sorted(node.items())
            if key not in {"items", "type"}
        }
        out["type"] = _normalized_type(node.get("type", "array"), nullable=True)
        if isinstance(items, Mapping):
            out["items"] = _stable_nullable_schema(
                items,
                is_root=_is_object_schema(items),
                allow_table=allow_table,
            )
        return out

    out = {
        key: _stable_nullable_schema(item, allow_table=allow_table)
        for key, item in sorted(node.items())
        if key != "type"
    }
    normalized_type = _normalized_type(node.get("type"), nullable=True)
    if normalized_type is not None:
        out["type"] = normalized_type
    if "enum" in out and isinstance(out["enum"], list) and None not in out["enum"]:
        out["enum"] = [*out["enum"], None]
    return out


def normalize_extraction_schema(
    schema: Mapping[str, Any],
    *,
    limits: SchemaPreflightLimits = DEFAULT_PREFLIGHT_LIMITS,
) -> dict[str, Any]:
    """Resolve refs, reject table schemas, make requested leaves nullable, and sort properties."""

    resolved = normalize_request_schema(schema, limits=limits)
    if not _is_object_schema(resolved):
        raise StructuredExtractionProtocolError("root extraction schema must have type object")
    normalized = _stable_nullable_schema(resolved, is_root=True)
    if not isinstance(normalized, dict):
        raise StructuredExtractionProtocolError("normalized schema must be an object")
    _check_json_structure(normalized, limits=limits, label="normalized schema")
    total_leaves = len(_leaf_paths(normalized))
    if total_leaves > limits.max_total_leaves:
        raise SchemaPreflightLimitError(
            f"schema exceeds maximum total leaves {limits.max_total_leaves}"
        )
    return normalized


def _schema_leaf_count(schema: Mapping[str, Any]) -> int:
    if _is_object_schema(schema):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError("object properties must be an object")
        return sum(
            _schema_leaf_count(child)
            for child in properties.values()
            if isinstance(child, Mapping)
        )
    if _is_array_schema(schema):
        items = schema.get("items")
        if isinstance(items, Mapping) and _is_object_schema(items):
            return _schema_leaf_count(items)
    return 1


def normalize_request_schema(
    schema: Mapping[str, Any],
    *,
    limits: SchemaPreflightLimits = DEFAULT_PREFLIGHT_LIMITS,
) -> dict[str, Any]:
    """Canonical nullable schema used by request hashes, including table arrays."""

    resolved = resolve_local_refs(schema, limits=limits)
    if not _is_object_schema(resolved):
        raise StructuredExtractionProtocolError("root extraction schema must have type object")
    normalized = _stable_nullable_schema(resolved, is_root=True, allow_table=True)
    if not isinstance(normalized, dict):
        raise StructuredExtractionProtocolError("normalized request schema must be an object")
    _check_json_structure(normalized, limits=limits, label="normalized request schema")
    total_leaves = _schema_leaf_count(normalized)
    if total_leaves > limits.max_total_leaves:
        raise SchemaPreflightLimitError(
            f"schema exceeds maximum total leaves {limits.max_total_leaves}"
        )
    return normalized


def build_varex_prompt(schema: Mapping[str, Any]) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False)
    return (
        "Extract structured data from this document.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{schema_text}\n\n"
        "Return null for fields you cannot find.\n"
        "Return ONLY valid JSON.\n"
        "Return an instance of the JSON with extracted values, not the schema itself."
    )


def _leaf_paths(schema: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if _is_object_schema(schema):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError("object properties must be an object")
        paths: list[tuple[str, ...]] = []
        for key in sorted(properties):
            child = properties[key]
            if not isinstance(child, Mapping):
                raise StructuredExtractionProtocolError("property schema must be an object")
            paths.extend(_leaf_paths(child, (*prefix, key)))
        return paths
    return [prefix]


def _select_paths(schema: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> dict[str, Any]:
    if not paths:
        raise ValueError("paths must not be empty")

    def select(node: Mapping[str, Any], relative_paths: Sequence[tuple[str, ...]]) -> dict[str, Any]:
        if not _is_object_schema(node):
            return copy.deepcopy(dict(node))
        properties = node.get("properties", {})
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError("object properties must be an object")
        selected_properties: dict[str, Any] = {}
        for key in sorted({path[0] for path in relative_paths if path}):
            child = properties[key]
            if not isinstance(child, Mapping):
                raise StructuredExtractionProtocolError("property schema must be an object")
            tails = [path[1:] for path in relative_paths if path and path[0] == key]
            selected_properties[key] = select(child, tails)
        result = {
            key: copy.deepcopy(value)
            for key, value in sorted(node.items())
            if key not in {"properties", "required"}
        }
        result["properties"] = selected_properties
        result["required"] = sorted(selected_properties)
        return result

    return select(schema, paths)


def _pointer(path: Sequence[str]) -> str:
    return "/" + "/".join(_encode_pointer_token(part) for part in path)


def _count_tokens(token_counter: Callable[[str], int], value: str) -> int:
    try:
        count = token_counter(value)
    except Exception as exc:
        raise SchemaTokenCounterError("token_counter failed") from exc
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SchemaTokenCounterError("token_counter must return a non-negative integer")
    return count


def chunk_nested_schema(
    schema: Mapping[str, Any],
    *,
    token_counter: Callable[[str], int],
    max_leaves: int = DEFAULT_MAX_LEAVES,
    max_schema_tokens: int = DEFAULT_MAX_SCHEMA_TOKENS,
    limits: SchemaPreflightLimits = DEFAULT_PREFLIGHT_LIMITS,
) -> tuple[SchemaChunk, ...]:
    """Greedily split Flat/Nested schemas; array-of-object tables fail explicitly."""

    if max_leaves <= 0:
        raise ValueError("max_leaves must be positive")
    if max_schema_tokens <= 0:
        raise ValueError("max_schema_tokens must be positive")
    normalized = normalize_extraction_schema(schema, limits=limits)
    leaf_paths = _leaf_paths(normalized)
    if not leaf_paths:
        return ()

    groups: list[list[tuple[str, ...]]] = []
    current: list[tuple[str, ...]] = []
    for path in leaf_paths:
        candidate = [*current, path]
        candidate_schema = _select_paths(normalized, candidate)
        candidate_text = json.dumps(candidate_schema, ensure_ascii=False, indent=2, allow_nan=False)
        candidate_tokens = _count_tokens(token_counter, candidate_text)
        if current and (len(candidate) > max_leaves or candidate_tokens > max_schema_tokens):
            groups.append(current)
            if len(groups) >= limits.max_chunks:
                raise SchemaPreflightLimitError(
                    f"schema exceeds maximum chunks {limits.max_chunks}"
                )
            current = [path]
            single_schema = _select_paths(normalized, current)
            single_text = json.dumps(single_schema, ensure_ascii=False, indent=2, allow_nan=False)
            if _count_tokens(token_counter, single_text) > max_schema_tokens:
                raise SchemaTokenLimitError(f"single schema leaf exceeds token limit: {_pointer(path)}")
        else:
            if candidate_tokens > max_schema_tokens:
                raise SchemaTokenLimitError(f"single schema leaf exceeds token limit: {_pointer(path)}")
            current = candidate
    if current:
        groups.append(current)
    if len(groups) > limits.max_chunks:
        raise SchemaPreflightLimitError(f"schema exceeds maximum chunks {limits.max_chunks}")

    chunks: list[SchemaChunk] = []
    total_schema_tokens = 0
    for index, group in enumerate(groups):
        chunk_schema = _select_paths(normalized, group)
        schema_text = json.dumps(chunk_schema, ensure_ascii=False, indent=2, allow_nan=False)
        schema_tokens = _count_tokens(token_counter, schema_text)
        total_schema_tokens += schema_tokens
        if total_schema_tokens > limits.max_total_schema_tokens:
            raise SchemaPreflightLimitError(
                "schema chunks exceed maximum total schema tokens "
                f"{limits.max_total_schema_tokens}"
            )
        chunks.append(
            SchemaChunk(
                chunk_id=f"c{index:03d}",
                paths=tuple(_pointer(path) for path in group),
                schema=chunk_schema,
                leaf_count=len(group),
                schema_tokens=schema_tokens,
            )
        )
    return tuple(chunks)


def _contains_table_array(schema: Mapping[str, Any]) -> bool:
    if _is_array_schema(schema):
        items = schema.get("items")
        return isinstance(items, Mapping) and _is_object_schema(items)
    if not _is_object_schema(schema):
        return False
    properties = schema.get("properties", {})
    return isinstance(properties, Mapping) and any(
        isinstance(child, Mapping) and _contains_table_array(child)
        for child in properties.values()
    )


def _root_subset_schema(
    root: Mapping[str, Any], properties: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result = {
        key: copy.deepcopy(value)
        for key, value in sorted(root.items())
        if key not in {"$defs", "properties", "required", "type"}
    }
    result["type"] = "object"
    result["properties"] = {
        key: copy.deepcopy(properties[key]) for key in sorted(properties)
    }
    result["required"] = sorted(properties)
    return result


def _table_anchor_rank(name: str) -> tuple[int, int, str] | None:
    priorities = ("id", "number", "no", "code", "name", "date")
    lowered = name.lower()
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", lowered) if token)
    for index, priority in enumerate(priorities):
        if lowered == priority:
            return (index, 0, lowered)
        if priority in tokens:
            return (index, 1, lowered)
    return None


def _table_chunk_schema(
    root: Mapping[str, Any],
    *,
    array_name: str,
    array_schema: Mapping[str, Any],
    item_schema: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    item_properties = item_schema.get("properties", {})
    if not isinstance(item_properties, Mapping):
        raise StructuredExtractionProtocolError("table item properties must be an object")
    normalized_properties = {
        field: _stable_nullable_schema(item_properties[field]) for field in fields
    }
    item_result = {
        key: copy.deepcopy(value)
        for key, value in sorted(item_schema.items())
        if key not in {"properties", "required", "type"}
    }
    item_result["type"] = "object"
    item_result["properties"] = normalized_properties
    item_result["required"] = list(fields)

    array_result = {
        key: copy.deepcopy(value)
        for key, value in sorted(array_schema.items())
        if key not in {"items", "type"}
    }
    array_result["type"] = ["array", "null"]
    array_result["items"] = item_result
    return _root_subset_schema(root, {array_name: array_result})


def split_table_schema(
    schema: Mapping[str, Any],
    *,
    token_counter: Callable[[str], int],
    max_columns: int = DEFAULT_TABLE_MAX_COLUMNS,
    max_anchors: int = DEFAULT_TABLE_MAX_ANCHORS,
    max_rows: int = DEFAULT_TABLE_MAX_ROWS,
    max_cell_bytes: int = DEFAULT_TABLE_MAX_CELL_BYTES,
    max_leaves: int = DEFAULT_MAX_LEAVES,
    max_schema_tokens: int = DEFAULT_MAX_SCHEMA_TOKENS,
    limits: SchemaPreflightLimits = DEFAULT_PREFLIGHT_LIMITS,
) -> TableExtractionPlan:
    """Split root table arrays independently and keep non-table fields in Nested chunks."""

    if not 1 <= max_columns <= DEFAULT_TABLE_MAX_COLUMNS:
        raise ValueError(
            f"table max_columns must be between 1 and {DEFAULT_TABLE_MAX_COLUMNS}"
        )
    if not 1 <= max_anchors <= DEFAULT_TABLE_MAX_ANCHORS:
        raise ValueError(
            f"table max_anchors must be between 1 and {DEFAULT_TABLE_MAX_ANCHORS}"
        )
    if not 1 <= max_rows <= DEFAULT_TABLE_MAX_ROWS:
        raise ValueError(f"table max_rows must be between 1 and {DEFAULT_TABLE_MAX_ROWS}")
    if not 1 <= max_cell_bytes <= DEFAULT_TABLE_MAX_CELL_BYTES:
        raise ValueError(
            f"table max_cell_bytes must be between 1 and {DEFAULT_TABLE_MAX_CELL_BYTES}"
        )
    resolved = normalize_request_schema(schema, limits=limits)
    if not _is_object_schema(resolved):
        raise StructuredExtractionProtocolError("root table schema must have type object")
    properties = resolved.get("properties", {})
    if not isinstance(properties, Mapping):
        raise StructuredExtractionProtocolError("root table properties must be an object")

    scalar_properties: dict[str, Mapping[str, Any]] = {}
    raw_tables: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for name in sorted(properties):
        child = properties[name]
        if not isinstance(child, Mapping):
            raise StructuredExtractionProtocolError("property schema must be an object")
        items = child.get("items") if _is_array_schema(child) else None
        if isinstance(items, Mapping) and _is_object_schema(items):
            raw_tables.append((name, child, items))
        elif _contains_table_array(child):
            raise UnsupportedTableSchemaError(
                f"nested table arrays are unsupported: {_pointer((name,))}"
            )
        else:
            scalar_properties[name] = child
    if not raw_tables:
        raise UnsupportedTableSchemaError("schema does not contain a root array-of-object table")

    scalar_chunks: tuple[SchemaChunk, ...] = ()
    if scalar_properties:
        scalar_schema = _root_subset_schema(resolved, scalar_properties)
        scalar_chunks = chunk_nested_schema(
            scalar_schema,
            token_counter=token_counter,
            max_leaves=max_leaves,
            max_schema_tokens=max_schema_tokens,
            limits=limits,
        )

    table_plans: list[TableArrayPlan] = []
    chunk_count = len(scalar_chunks)
    total_schema_tokens = sum(chunk.schema_tokens for chunk in scalar_chunks)
    for table_index, (array_name, array_schema, item_schema) in enumerate(raw_tables):
        item_properties = item_schema.get("properties", {})
        if not isinstance(item_properties, Mapping) or not item_properties:
            raise UnsupportedTableSchemaError(
                f"table item requires non-empty properties: {_pointer((array_name,))}"
            )
        fields = tuple(sorted(item_properties))
        table_fields: list[str] = []
        anchor_candidates: list[str] = []
        for field in fields:
            field_schema = item_properties[field]
            if not isinstance(field_schema, Mapping):
                raise StructuredExtractionProtocolError("table field schema must be an object")
            is_array = _is_array_schema(field_schema)
            array_items = field_schema.get("items") if is_array else None
            if _is_object_schema(field_schema) or (
                is_array and isinstance(array_items, Mapping) and _is_object_schema(array_items)
            ):
                raise UnsupportedTableSchemaError(
                    f"nested table column is unsupported: {_pointer((array_name, field))}"
                )
            table_fields.append(field)
            if not is_array:
                anchor_candidates.append(field)
        ranked_anchors = sorted(
            (
                (rank, field)
                for field in anchor_candidates
                if (rank := _table_anchor_rank(field)) is not None
            ),
            key=lambda item: item[0],
        )
        anchor_fields = tuple(field for _, field in ranked_anchors[:max_anchors])
        if not anchor_fields and anchor_candidates:
            anchor_fields = (anchor_candidates[0],)
        value_fields = [field for field in table_fields if field not in anchor_fields]
        if not anchor_fields and len(value_fields) > max_columns:
            raise UnsupportedTableSchemaError(
                f"anchorless table cannot be split safely: {_pointer((array_name,))}"
            )

        field_groups: list[tuple[str, ...]] = []
        if not value_fields:
            field_groups.append(())
        else:
            current: list[str] = []
            for field in value_fields:
                candidate = [*current, field]
                selected = (*anchor_fields, *candidate)
                candidate_schema = _table_chunk_schema(
                    resolved,
                    array_name=array_name,
                    array_schema=array_schema,
                    item_schema=item_schema,
                    fields=selected,
                )
                candidate_tokens = _count_tokens(
                    token_counter,
                    json.dumps(candidate_schema, ensure_ascii=False, indent=2, allow_nan=False),
                )
                if current and (
                    len(candidate) > max_columns or candidate_tokens > max_schema_tokens
                ):
                    field_groups.append(tuple(current))
                    current = [field]
                else:
                    if candidate_tokens > max_schema_tokens:
                        raise SchemaTokenLimitError(
                            "single table column exceeds token limit: "
                            f"{_pointer((array_name, field))}"
                        )
                    current = candidate
            if current:
                field_groups.append(tuple(current))
        if not anchor_fields and len(field_groups) > 1:
            raise UnsupportedTableSchemaError(
                f"anchorless table exceeds one safe chunk: {_pointer((array_name,))}"
            )

        chunks: list[TableSchemaChunk] = []
        for group_index, group in enumerate(field_groups):
            selected = (*anchor_fields, *group)
            chunk_schema = _table_chunk_schema(
                resolved,
                array_name=array_name,
                array_schema=array_schema,
                item_schema=item_schema,
                fields=selected,
            )
            schema_tokens = _count_tokens(
                token_counter,
                json.dumps(chunk_schema, ensure_ascii=False, indent=2, allow_nan=False),
            )
            if schema_tokens > max_schema_tokens:
                raise SchemaTokenLimitError(
                    f"table anchor schema exceeds token limit: {_pointer((array_name,))}"
                )
            chunk_count += 1
            total_schema_tokens += schema_tokens
            if chunk_count > limits.max_chunks:
                raise SchemaPreflightLimitError(
                    f"schema exceeds maximum chunks {limits.max_chunks}"
                )
            if total_schema_tokens > limits.max_total_schema_tokens:
                raise SchemaPreflightLimitError(
                    "schema chunks exceed maximum total schema tokens "
                    f"{limits.max_total_schema_tokens}"
                )
            chunks.append(
                TableSchemaChunk(
                    chunk_id=f"t{table_index:03d}-c{group_index:03d}",
                    array_path=_pointer((array_name,)),
                    anchor_fields=anchor_fields,
                    value_fields=group,
                    schema=chunk_schema,
                    schema_tokens=schema_tokens,
                )
            )
        table_plans.append(
            TableArrayPlan(
                array_path=_pointer((array_name,)),
                fields=fields,
                anchor_fields=anchor_fields,
                max_rows=max_rows,
                max_cell_bytes=max_cell_bytes,
                chunks=tuple(chunks),
            )
        )
    return TableExtractionPlan(
        scalar_chunks=scalar_chunks,
        tables=tuple(table_plans),
        total_schema_tokens=total_schema_tokens,
    )


def _table_array_name(array_path: str) -> str:
    if not array_path.startswith("/") or "/" in array_path[1:]:
        raise ValueError("table array path must address one root property")
    return _decode_pointer_token(array_path[1:])


def _json_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _validate_table_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    max_cell_bytes: int,
    depth: int = 0,
) -> None:
    if depth > 8:
        raise TablePredictionError(f"table value nesting is too deep at {path}")
    try:
        payload_size = len(canonical_json_bytes(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise TablePredictionError(f"table value is not canonical JSON at {path}") from exc
    if payload_size > max_cell_bytes:
        raise TablePredictionError(f"table value exceeds byte limit at {path}")

    raw_type = schema.get("type")
    allowed_types = (
        {raw_type}
        if isinstance(raw_type, str)
        else set(raw_type)
        if isinstance(raw_type, list)
        else set()
    )
    if value is None:
        if allowed_types and "null" not in allowed_types:
            raise TablePredictionError(f"null is not allowed at {path}")
        return
    non_null_types = allowed_types - {"null"}
    expected_type = next(iter(non_null_types), None)
    if expected_type == "string" and not isinstance(value, str):
        raise TablePredictionError(f"expected string at {path}")
    if expected_type == "boolean" and not isinstance(value, bool):
        raise TablePredictionError(f"expected boolean at {path}")
    if expected_type == "integer" and not (
        _json_number(value) and (isinstance(value, int) or float(value).is_integer())
    ):
        raise TablePredictionError(f"expected integer at {path}")
    if expected_type == "number" and not _json_number(value):
        raise TablePredictionError(f"expected number at {path}")
    if expected_type == "array" and not isinstance(value, list):
        raise TablePredictionError(f"expected array at {path}")
    if expected_type == "object" or isinstance(value, Mapping):
        raise TablePredictionError(f"nested object table values are unsupported at {path}")

    enum = schema.get("enum")
    if isinstance(enum, list):
        encoded = canonical_json_bytes(value)
        if not any(canonical_json_bytes(candidate) == encoded for candidate in enum):
            raise TablePredictionError(f"value is outside enum at {path}")
    if isinstance(value, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise TablePredictionError(f"string is shorter than minLength at {path}")
        if isinstance(max_length, int) and len(value) > max_length:
            raise TablePredictionError(f"string exceeds maxLength at {path}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matches = re.search(pattern, value) is not None
            except re.error as exc:
                raise StructuredExtractionProtocolError(
                    f"invalid schema pattern at {path}"
                ) from exc
            if not matches:
                raise TablePredictionError(f"string does not match pattern at {path}")
    if _json_number(value):
        numeric = Decimal(str(value))
        for keyword, predicate in (
            ("minimum", lambda bound: numeric >= bound),
            ("maximum", lambda bound: numeric <= bound),
            ("exclusiveMinimum", lambda bound: numeric > bound),
            ("exclusiveMaximum", lambda bound: numeric < bound),
        ):
            bound = schema.get(keyword)
            if bound is not None and not predicate(Decimal(str(bound))):
                raise TablePredictionError(f"number violates {keyword} at {path}")
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            try:
                if numeric % Decimal(str(multiple_of)) != 0:
                    raise TablePredictionError(f"number violates multipleOf at {path}")
            except InvalidOperation as exc:
                raise StructuredExtractionProtocolError(
                    f"invalid multipleOf at {path}"
                ) from exc
    if isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise TablePredictionError(f"array is shorter than minItems at {path}")
        if isinstance(max_items, int) and len(value) > max_items:
            raise TablePredictionError(f"array exceeds maxItems at {path}")
        if schema.get("uniqueItems") is True:
            encoded_items = [canonical_json_bytes(item) for item in value]
            if len(encoded_items) != len(set(encoded_items)):
                raise TablePredictionError(f"array violates uniqueItems at {path}")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_table_value(
                    item,
                    items,
                    path=f"{path}/{index}",
                    max_cell_bytes=max_cell_bytes,
                    depth=depth + 1,
                )


def validate_nested_prediction(
    value: Any,
    schema: Mapping[str, Any],
    *,
    max_value_bytes: int = 4 * 1024 * 1024,
) -> None:
    """Проверить точный экземпляр поддерживаемой Flat/Nested-схемы."""

    if max_value_bytes <= 0:
        raise ValueError("max_value_bytes must be positive")

    def visit(node: Any, node_schema: Mapping[str, Any], path: str, depth: int) -> None:
        if depth > 32:
            raise TablePredictionError(f"prediction nesting is too deep at {path}")
        raw_type = node_schema.get("type")
        allowed_types = (
            {raw_type}
            if isinstance(raw_type, str)
            else set(raw_type)
            if isinstance(raw_type, list)
            else set()
        )
        if node is None:
            if "null" not in allowed_types:
                raise TablePredictionError(f"null is not allowed at {path}")
            return
        non_null_types = allowed_types - {"null"}
        expected_type = next(iter(non_null_types), None)
        if expected_type != "object":
            _validate_table_value(
                node,
                node_schema,
                path=path,
                max_cell_bytes=max_value_bytes,
                depth=depth,
            )
            return
        if not isinstance(node, Mapping):
            raise TablePredictionError(f"expected object at {path}")
        properties = node_schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError(f"object properties are invalid at {path}")
        expected_keys = set(properties)
        actual_keys = set(node)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise TablePredictionError(
                f"object keys mismatch at {path}: missing={missing} extra={extra}"
            )
        for key in sorted(expected_keys):
            child_schema = properties[key]
            if not isinstance(child_schema, Mapping):
                raise StructuredExtractionProtocolError(f"property schema is invalid at {path}/{key}")
            visit(node[key], child_schema, f"{path}/{_encode_pointer_token(key)}", depth + 1)

    try:
        payload_size = len(canonical_json_bytes(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise TablePredictionError("prediction is not canonical JSON") from exc
    if payload_size > max_value_bytes:
        raise TablePredictionError("prediction exceeds byte limit")
    visit(value, schema, "", 0)


def _table_prediction_rows(
    chunk: TableSchemaChunk,
    prediction: Any,
    *,
    max_rows: int,
    max_cell_bytes: int,
) -> list[dict[str, Any]] | None:
    array_name = _table_array_name(chunk.array_path)
    if not isinstance(prediction, Mapping) or set(prediction) != {array_name}:
        raise TablePredictionError(
            f"chunk {chunk.chunk_id} must contain only {chunk.array_path}"
        )
    value = prediction[array_name]
    if value is None:
        return None
    if not isinstance(value, list):
        raise TablePredictionError(f"chunk {chunk.chunk_id} table value must be an array or null")
    if len(value) > max_rows:
        raise TablePredictionError(
            f"chunk {chunk.chunk_id} exceeds maximum rows {max_rows}"
        )
    requested = (*chunk.anchor_fields, *chunk.value_fields)
    expected = set(requested)
    array_schema = chunk.schema["properties"][array_name]
    min_items = array_schema.get("minItems")
    max_items = array_schema.get("maxItems")
    if isinstance(min_items, int) and len(value) < min_items:
        raise TablePredictionError(f"chunk {chunk.chunk_id} is shorter than minItems")
    if isinstance(max_items, int) and len(value) > max_items:
        raise TablePredictionError(f"chunk {chunk.chunk_id} exceeds maxItems")
    item_schema = array_schema["items"]
    field_schemas = item_schema["properties"]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise TablePredictionError(
                f"chunk {chunk.chunk_id} row {index} must contain every requested field only"
            )
        normalized_row: dict[str, Any] = {}
        for field in requested:
            _validate_table_value(
                row[field],
                field_schemas[field],
                path=f"{chunk.array_path}/{index}/{_encode_pointer_token(field)}",
                max_cell_bytes=max_cell_bytes,
            )
            normalized_row[field] = copy.deepcopy(row[field])
        rows.append(normalized_row)
    if array_schema.get("uniqueItems") is True:
        encoded_rows = [canonical_json_bytes(row) for row in rows]
        if len(encoded_rows) != len(set(encoded_rows)):
            raise TablePredictionError(f"chunk {chunk.chunk_id} violates uniqueItems")
    return rows


def _empty_table_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _table_anchor_key(
    row: Mapping[str, Any], anchor_fields: Sequence[str]
) -> tuple[bytes, ...] | None:
    if not anchor_fields:
        return None
    values = [row[field] for field in anchor_fields]
    if any(_empty_table_value(value) for value in values):
        return None
    try:
        return tuple(canonical_json_bytes(value) for value in values)
    except (TypeError, ValueError, RecursionError) as exc:
        raise TablePredictionError("table anchor is not canonical JSON") from exc


def _merge_table_value(current: Any, incoming: Any, *, field: str) -> Any:
    if _empty_table_value(current):
        return copy.deepcopy(incoming)
    if _empty_table_value(incoming) or current == incoming:
        return current
    raise TableAlignmentConflictError(f"conflicting non-empty table value for {field}")


def merge_table_array_predictions(
    plan: TableArrayPlan,
    predictions: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]] | None]:
    """Merge one table's chunks only when repeated anchors prove row identity."""

    expected_chunk_ids = {chunk.chunk_id for chunk in plan.chunks}
    if set(predictions) != expected_chunk_ids:
        raise TablePredictionError("table predictions must contain every planned chunk exactly once")
    row_sets = [
        _table_prediction_rows(
            chunk,
            predictions[chunk.chunk_id],
            max_rows=plan.max_rows,
            max_cell_bytes=plan.max_cell_bytes,
        )
        for chunk in plan.chunks
    ]
    array_name = _table_array_name(plan.array_path)
    if all(rows is None for rows in row_sets):
        return {array_name: None}
    if any(rows is None for rows in row_sets):
        raise TableAlignmentConflictError("table chunks disagree between null and row data")
    concrete_rows = [rows for rows in row_sets if rows is not None]
    if len(concrete_rows) == 1:
        return {
            array_name: [
                {field: copy.deepcopy(row[field]) for field in plan.fields}
                for row in concrete_rows[0]
            ]
        }
    if not plan.anchor_fields:
        raise TableAlignmentConflictError("multi-chunk table requires stable anchor fields")

    keyed_rows: list[dict[tuple[bytes, ...], dict[str, Any]]] = []
    keyed_orders: list[list[tuple[bytes, ...]]] = []
    for rows in concrete_rows:
        mapping: dict[tuple[bytes, ...], dict[str, Any]] = {}
        order: list[tuple[bytes, ...]] = []
        for row in rows:
            key = _table_anchor_key(row, plan.anchor_fields)
            if key is None or key in mapping:
                raise TableAlignmentConflictError(
                    "multi-chunk table requires non-empty unique anchors"
                )
            mapping[key] = row
            order.append(key)
        keyed_rows.append(mapping)
        keyed_orders.append(order)

    reference_keys = set(keyed_rows[0])
    if any(set(mapping) != reference_keys for mapping in keyed_rows[1:]):
        raise TableAlignmentConflictError("table chunks contain different anchor sets")
    aligned = [
        [mapping[key] for mapping in keyed_rows]
        for key in keyed_orders[0]
    ]

    merged_rows: list[dict[str, Any]] = []
    for row_group in aligned:
        merged: dict[str, Any] = {}
        for row in row_group:
            for field, value in row.items():
                if field in merged:
                    merged[field] = _merge_table_value(merged[field], value, field=field)
                else:
                    merged[field] = copy.deepcopy(value)
        merged_rows.append({field: merged[field] for field in plan.fields})
    return {array_name: merged_rows}


def is_schema_echo(value: Any, *, expected_schema: Mapping[str, Any] | None = None) -> bool:
    """Detect direct or wrapped JSON Schema output without rejecting normal instances."""

    if expected_schema is not None:
        properties = expected_schema.get("properties", {})
        expected_keys = set(properties) if isinstance(properties, Mapping) else set()
        try:
            expected_bytes = canonical_json_bytes(expected_schema)
        except (TypeError, ValueError, RecursionError):
            expected_bytes = None
    else:
        expected_keys = set()
        expected_bytes = None

    def looks_like_schema(node: Any) -> bool:
        if not isinstance(node, Mapping):
            return False
        keys = set(node)
        if keys & {"$defs", "$ref", "$schema"}:
            return True
        schema_type = node.get("type")
        type_values = [schema_type] if isinstance(schema_type, str) else schema_type
        valid_type = (
            isinstance(type_values, list)
            and bool(type_values)
            and all(isinstance(item, str) and item in _JSON_SCHEMA_TYPES for item in type_values)
        )
        properties = node.get("properties")
        if valid_type and isinstance(properties, Mapping):
            return all(isinstance(child, Mapping) for child in properties.values())
        return bool(valid_type and (keys & (_SCHEMA_MARKER_KEYS - {"type"})))

    stack: list[tuple[Any, bool, int, bool]] = [(value, True, 1, False)]
    visited = 0
    while stack:
        node, is_root, depth, suppress_shape = stack.pop()
        visited += 1
        if visited > 10_000 or depth > 64:
            return False
        if isinstance(node, Mapping):
            if expected_bytes is not None:
                try:
                    if canonical_json_bytes(node) == expected_bytes:
                        return True
                except (TypeError, ValueError, RecursionError):
                    return False
            node_keys = set(node)
            is_expected_instance = bool(
                is_root and expected_keys and node_keys & expected_keys
            )
            if not suppress_shape and not is_expected_instance and looks_like_schema(node):
                return True
            child_suppresses_shape = suppress_shape or is_expected_instance
            for child in node.values():
                stack.append((child, False, depth + 1, child_suppresses_shape))
        elif isinstance(node, list):
            stack.extend((child, False, depth + 1, suppress_shape) for child in node)
    return False


def build_request_hash(
    schema: Mapping[str, Any],
    *,
    source_sha256: str,
    model: str,
    model_revision: str,
    prompt_version: str = PROMPT_VERSION,
    split_policy_version: str = "nested-v1",
    tokenizer_policy: str = DEFAULT_TOKENIZER_POLICY,
    max_leaves: int = DEFAULT_MAX_LEAVES,
    max_schema_tokens: int = DEFAULT_MAX_SCHEMA_TOKENS,
    table_max_columns: int = DEFAULT_TABLE_MAX_COLUMNS,
    table_max_anchors: int = DEFAULT_TABLE_MAX_ANCHORS,
    table_max_rows: int = DEFAULT_TABLE_MAX_ROWS,
    table_max_cell_bytes: int = DEFAULT_TABLE_MAX_CELL_BYTES,
    flat_nested_max_tokens: int = 4096,
    table_max_tokens: int = 8192,
    limits: SchemaPreflightLimits = DEFAULT_PREFLIGHT_LIMITS,
) -> str:
    """Hash logical request inputs; retry timestamps and endpoint are intentionally excluded."""

    if not _SHA256_RE.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be lowercase hexadecimal")
    if not all((model, model_revision, prompt_version, split_policy_version, tokenizer_policy)):
        raise ValueError(
            "model, revision, prompt version, split policy, and tokenizer policy must be non-empty"
        )
    if not 1 <= table_max_columns <= DEFAULT_TABLE_MAX_COLUMNS:
        raise ValueError(
            f"table_max_columns must be between 1 and {DEFAULT_TABLE_MAX_COLUMNS}"
        )
    if not 1 <= table_max_anchors <= DEFAULT_TABLE_MAX_ANCHORS:
        raise ValueError(
            f"table_max_anchors must be between 1 and {DEFAULT_TABLE_MAX_ANCHORS}"
        )
    if not 1 <= table_max_rows <= DEFAULT_TABLE_MAX_ROWS:
        raise ValueError(f"table_max_rows must be between 1 and {DEFAULT_TABLE_MAX_ROWS}")
    if not 1 <= table_max_cell_bytes <= DEFAULT_TABLE_MAX_CELL_BYTES:
        raise ValueError(
            "table_max_cell_bytes must be between 1 and "
            f"{DEFAULT_TABLE_MAX_CELL_BYTES}"
        )
    if any(
        value <= 0
        for value in (
            max_leaves,
            max_schema_tokens,
            table_max_columns,
            table_max_anchors,
            table_max_rows,
            table_max_cell_bytes,
            flat_nested_max_tokens,
            table_max_tokens,
        )
    ):
        raise ValueError("token limits must be positive")
    normalized = normalize_request_schema(schema, limits=limits)
    fingerprint = {
        "flat_nested_max_tokens": flat_nested_max_tokens,
        "max_leaves": max_leaves,
        "max_schema_tokens": max_schema_tokens,
        "model": model,
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "preflight_limits": {
            "max_chunks": limits.max_chunks,
            "max_depth": limits.max_depth,
            "max_input_bytes": limits.max_input_bytes,
            "max_ref_expansions": limits.max_ref_expansions,
            "max_total_leaves": limits.max_total_leaves,
            "max_total_schema_tokens": limits.max_total_schema_tokens,
        },
        "protocol_version": PROTOCOL_VERSION,
        "response_format": "json_object",
        "schema_sha256": canonical_json_sha256(normalized),
        "source_sha256": source_sha256,
        "split_policy_version": split_policy_version,
        "table_max_tokens": table_max_tokens,
        "table_max_columns": table_max_columns,
        "table_max_anchors": table_max_anchors,
        "table_max_rows": table_max_rows,
        "table_max_cell_bytes": table_max_cell_bytes,
        "temperature": 0,
        "tokenizer_policy": tokenizer_policy,
    }
    return canonical_json_sha256(fingerprint)
