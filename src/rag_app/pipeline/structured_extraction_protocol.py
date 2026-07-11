"""Deterministic protocol-v3 primitives for schema-guided extraction."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 3
PROMPT_VERSION = "granite-varex-split-v1"
DEFAULT_MAX_LEAVES = 24
DEFAULT_MAX_SCHEMA_TOKENS = 2048
DEFAULT_TOKENIZER_POLICY = "model-tokenizer-v1"

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
    """Array-of-object schemas require the separate table split/merge protocol."""


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


def _stable_nullable_schema(value: Any, *, is_root: bool = False) -> Any:
    if isinstance(value, list):
        return [_stable_nullable_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    node = dict(value)
    if _is_object_schema(node):
        properties = node.get("properties", {})
        if not isinstance(properties, Mapping):
            raise StructuredExtractionProtocolError("object properties must be an object")
        stable_properties = {
            key: _stable_nullable_schema(properties[key])
            for key in sorted(properties)
        }
        out = {
            key: _stable_nullable_schema(item)
            for key, item in sorted(node.items())
            if key not in {"properties", "required", "type"}
        }
        out["type"] = "object" if is_root else ["object", "null"]
        out["properties"] = stable_properties
        out["required"] = sorted(stable_properties)
        return out

    if _is_array_schema(node):
        items = node.get("items", {})
        if isinstance(items, Mapping) and _is_object_schema(items):
            raise UnsupportedTableSchemaError(
                "array-of-object schemas require table splitting and merge support"
            )
        out = {
            key: _stable_nullable_schema(item)
            for key, item in sorted(node.items())
            if key != "type"
        }
        out["type"] = _normalized_type(node.get("type", "array"), nullable=True)
        return out

    out = {key: _stable_nullable_schema(item) for key, item in sorted(node.items()) if key != "type"}
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

    resolved = resolve_local_refs(schema, limits=limits)
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
    if any(
        value <= 0
        for value in (
            max_leaves,
            max_schema_tokens,
            flat_nested_max_tokens,
            table_max_tokens,
        )
    ):
        raise ValueError("token limits must be positive")
    normalized = normalize_extraction_schema(schema, limits=limits)
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
        "temperature": 0,
        "tokenizer_policy": tokenizer_policy,
    }
    return canonical_json_sha256(fingerprint)
