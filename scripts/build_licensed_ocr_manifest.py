"""Build a deterministic manifest for the license-clear OCR corpus.

The builder is deliberately offline: it reads a lightweight JSONL candidate
catalog and never downloads document images, PDFs, or dataset archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_DEFAULT_SEED = "docragenslate-ocr-v1"
_HASH_ALGORITHM = "sha256-nul-v1"
_CANONICAL_JSON_ALGORITHM = "sha256-json-sort-keys-ascii-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_PUBTABLES_IMAGES_PATH = "PubTables-v2_Full-Documents_test_images.tar.gz"
_PUBTABLES_IMAGES_SHA256 = "0d42821fb1dce5713a86c327bec5fabbe214bb5ebbd0cfc75cd2ef89b7c7230e"
_PUBTABLES_TABLES_PATH = "PubTables-v2_Full-Documents_test_tables.tar.gz"
_PUBTABLES_TABLES_SHA256 = "dfd10e0dc4cb3e92d0f521e8a135e6e96094d8e90e130fb3fe25c9fa31b3a3de"
_VAREX_SHARDS = {
    "data/benchmark-00000-of-00004.parquet": (
        "f0328edd6242318f97eb85fdd63466ec6a9db1482b7edd3ddb92de2bc535e147"
    ),
    "data/benchmark-00001-of-00004.parquet": (
        "6eee390d60212571269d1002fe335d635f149e5850a0ed05dd1b3dc77a8a5d07"
    ),
    "data/benchmark-00002-of-00004.parquet": (
        "03c5411bb91eaaacedd77ee280e4e92359fa18cb26f5bf9a0cfab53481c4c194"
    ),
    "data/benchmark-00003-of-00004.parquet": (
        "b2304eb66183063d18da6d93bf6ce73283fa5dfa5b39611082f1b85a5f2874c7"
    ),
}
_VAREX_RECORD_FIELDS = ["doc_id", "ground_truth", "schema", "split"]
_AI2D_IMAGE_ARCHIVE_URL = (
    "https://ai2-public-datasets.s3.us-west-2.amazonaws.com/diagrams/ai2d-all.zip"
)
_AI2D_IMAGE_ARCHIVE_SHA256 = (
    "1a6b77eebb8b7dbdf76a0ba6ca76c2f97ce8f81d8ee33b06593aa722e54c4786"
)
_AI2D_ANNOTATION_ARCHIVE_URL = (
    "https://www.kielipankki.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip"
)
_AI2D_ANNOTATION_ARCHIVE_SHA256 = (
    "eb11d67507e08eb9bfd0f5944da7ca32cfcffa13e119b04ac5054effa65a759a"
)
_MWS_LEGACY_REVISION = "e204166bde25f7dcaaffb9313b855de67b516e5d"
_MWS_METADATA_URL = (
    "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/"
    f"{_MWS_LEGACY_REVISION}/metadata.jsonl"
)
_MWS_METADATA_SHA256 = "c234a569583858bfab13399169ec9951da12edf6d88a5cd4c4efae8a1fd4197d"
_MWS_RECORD_FIELDS = ["answers", "dataset_name", "file_name", "id", "question", "type"]
_MWS_SOURCE_TYPES = {
    "document_parsing_ru": "document parsing ru",
    "full_page_ocr_ru": "full-page OCR ru",
    "key_information_extraction_ru": "key information extraction ru",
    "reasoning_vqa_ru": "reasoning VQA ru",
    "text_grounding_ru": "text grounding ru",
}


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    source_url: str
    revision: str
    license: str
    quotas: Mapping[str, int]
    expected_inputs: Mapping[str, int]
    license_components: Mapping[str, str] | None = None
    source_components: Mapping[str, Mapping[str, str]] | None = None


DATASETS: dict[str, DatasetSpec] = {
    "pubtables_v2": DatasetSpec(
        source="kensho/PubTables-v2",
        source_url="https://huggingface.co/datasets/kensho/PubTables-v2",
        revision="aa575e798cb00a296925e2086addb3e3fd9a1903",
        license="CDLA-Permissive-2.0",
        quotas={"pages_2": 8, "pages_3": 4, "pages_4": 2},
        expected_inputs={"pages_2": 2, "pages_3": 3, "pages_4": 4},
    ),
    "varex": DatasetSpec(
        source="ibm-research/VAREX",
        source_url="https://huggingface.co/datasets/ibm-research/VAREX",
        revision="2dfc3386a4567c7d56bf1abf4d12ff42afed27b6",
        license="CDLA-Permissive-2.0",
        quotas={"Flat": 10, "Nested": 10, "Table": 10},
        expected_inputs={"Flat": 2, "Nested": 2, "Table": 2},
    ),
    "ai2d_rst": DatasetSpec(
        source="AllenAI/AI2D + AI2D-RST v1.1",
        source_url=_AI2D_IMAGE_ARCHIVE_URL,
        revision="content-addressed-source-components",
        license="CC-BY-4.0 AND CC-BY-SA-4.0",
        quotas={"diagram": 12},
        expected_inputs={"diagram": 1},
        license_components={
            "annotations": "CC-BY-4.0",
            "source_images": "CC-BY-SA-4.0",
        },
        source_components={
            "annotations": {
                "source": "Kielipankki/AI2D-RST-v1.1",
                "source_url": _AI2D_ANNOTATION_ARCHIVE_URL,
                "revision": _AI2D_ANNOTATION_ARCHIVE_SHA256,
            },
            "source_images": {
                "source": "AllenAI/AI2D",
                "source_url": _AI2D_IMAGE_ARCHIVE_URL,
                "revision": _AI2D_IMAGE_ARCHIVE_SHA256,
            },
        },
    ),
    "mws_vision_bench": DatasetSpec(
        source="MTSAIR/MWS-Vision-Bench",
        source_url="https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench",
        revision="b8d473734b79343cac2b74f692a29ab191c7d11d",
        license="MIT AND CC-BY-4.0",
        license_components={"benchmark_code": "MIT", "source_assets": "CC-BY-4.0"},
        quotas={
            "document_parsing_ru": 6,
            "full_page_ocr_ru": 6,
            "key_information_extraction_ru": 6,
            "reasoning_vqa_ru": 6,
            "text_grounding_ru": 6,
        },
        expected_inputs={
            "document_parsing_ru": 1,
            "full_page_ocr_ru": 1,
            "key_information_extraction_ru": 1,
            "reasoning_vqa_ru": 1,
            "text_grounding_ru": 1,
        },
    ),
}


def _selection_hash(seed: str, dataset: str, stratum: str, canonical_id: str) -> str:
    value = "\0".join((seed, dataset, stratum, canonical_id)).encode()
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _catalog_hash(candidates: Sequence[dict[str, Any]]) -> str:
    rows = sorted(_canonical_json(candidate) for candidate in candidates)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
        digest.update(b"\n")
    return digest.hexdigest()


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(
                raw_line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {constant}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"candidate on line {line_number} must be a JSON object")
        candidates.append(value)
    return candidates


def _require_string(metadata: Mapping[str, Any], key: str, context: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: metadata.{key} must be a non-empty string")
    return value


def _safe_pinned_path(uri: str, expected_prefix: str, context: str) -> None:
    parsed = urlparse(uri)
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError(f"{context}.uri must not contain query, fragment or params")
    raw_path_lower = parsed.path.lower()
    if "%2f" in raw_path_lower or "%2e" in raw_path_lower:
        raise ValueError(f"{context}.uri must not contain encoded path separators")
    decoded_path = unquote(parsed.path)
    if not decoded_path.startswith(expected_prefix) or decoded_path == expected_prefix:
        raise ValueError(f"{context}.uri must use the exact pinned source revision path")
    if any(segment in {"", ".", ".."} for segment in decoded_path.split("/")[1:]):
        raise ValueError(f"{context}.uri contains an unsafe path segment")


def _validate_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _validate_member(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty relative POSIX path")
    decoded = unquote(value)
    if decoded != value or "\\" in value or value.startswith("/"):
        raise ValueError(f"{context} must not contain encoded, absolute or backslash paths")
    if any(segment in {"", ".", ".."} for segment in value.split("/")):
        raise ValueError(f"{context} contains an unsafe path segment")
    return value


def _validate_container(value: Any, *, formats: set[str], context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if set(value) != {"uri", "sha256", "format"}:
        raise ValueError(f"{context} fields must be exactly uri, sha256 and format")
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"{context}.uri must be an absolute URI")
    parsed = urlparse(uri)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment or parsed.params:
        raise ValueError(f"{context}.uri must be an immutable absolute HTTPS URI")
    sha256 = _validate_sha256(value.get("sha256"), f"{context}.sha256")
    format_ = value.get("format")
    if format_ not in formats:
        raise ValueError(f"{context}.format must be one of {sorted(formats)}")
    return {"uri": uri, "sha256": sha256, "format": format_}


def _validate_reference(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    kind = value.get("kind")
    if kind == "direct":
        if set(value) != {"kind", "uri", "sha256"}:
            raise ValueError(f"{context} direct fields must be exactly kind, uri and sha256")
        uri = value.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"{context}.uri must be an absolute URI")
        return {
            "kind": kind,
            "uri": uri,
            "sha256": _validate_sha256(value.get("sha256"), f"{context}.sha256"),
        }
    if kind == "archive_member":
        if set(value) != {"kind", "container", "member", "sha256"}:
            raise ValueError(
                f"{context} archive_member fields must be exactly kind, container, member and sha256"
            )
        return {
            "kind": kind,
            "container": _validate_container(
                value.get("container"), formats={"tar.gz", "zip"}, context=f"{context}.container"
            ),
            "member": _validate_member(value.get("member"), f"{context}.member"),
            "sha256": _validate_sha256(value.get("sha256"), f"{context}.sha256"),
        }
    if kind == "parquet_field":
        if set(value) != {"kind", "container", "row_index", "field", "sha256"}:
            raise ValueError(
                f"{context} parquet_field fields must be exactly kind, container, row_index, field and sha256"
            )
        row_index = value.get("row_index")
        field = value.get("field")
        if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
            raise ValueError(f"{context}.row_index must be a non-negative integer")
        if not isinstance(field, str) or not field:
            raise ValueError(f"{context}.field must be a non-empty string")
        return {
            "kind": kind,
            "container": _validate_container(
                value.get("container"), formats={"parquet"}, context=f"{context}.container"
            ),
            "row_index": row_index,
            "field": field,
            "sha256": _validate_sha256(value.get("sha256"), f"{context}.sha256"),
        }
    raise ValueError(f"{context}.kind must be direct, archive_member or parquet_field")


def _validate_record_reference(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    kind = value.get("kind")
    if kind not in {"parquet_row", "jsonl_row"}:
        raise ValueError(f"{context}.kind must be parquet_row or jsonl_row")
    if set(value) != {"kind", "container", "row_index", "fields", "sha256"}:
        raise ValueError(
            f"{context} fields must be exactly kind, container, row_index, fields and sha256"
        )
    row_index = value.get("row_index")
    if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
        raise ValueError(f"{context}.row_index must be a non-negative integer")
    fields = value.get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
        or fields != sorted(set(fields))
    ):
        raise ValueError(f"{context}.fields must be a non-empty sorted list of unique strings")
    format_ = "parquet" if kind == "parquet_row" else "jsonl"
    return {
        "kind": kind,
        "container": _validate_container(
            value.get("container"), formats={format_}, context=f"{context}.container"
        ),
        "row_index": row_index,
        "fields": fields,
        "sha256": _validate_sha256(value.get("sha256"), f"{context}.sha256"),
    }


def _expected_roles(dataset: str, stratum: str, expected_inputs: int) -> set[str]:
    if dataset == "pubtables_v2":
        return {f"page_{index}" for index in range(1, expected_inputs + 1)}
    if dataset == "varex":
        return {"image_200dpi", "image_50dpi"}
    if dataset in {"ai2d_rst", "mws_vision_bench"}:
        return {"image"}
    raise ValueError(f"no input-role schema for dataset: {dataset}")


def _hf_uri(source: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{source}/resolve/{revision}/{path}"


def _require_container(
    reference: Mapping[str, Any],
    *,
    kind: str,
    uri: str,
    sha256: str | None,
    format_: str,
    context: str,
) -> None:
    if reference.get("kind") != kind:
        raise ValueError(f"{context}.kind must be {kind}")
    container = reference.get("container")
    if not isinstance(container, Mapping):
        raise ValueError(f"{context}.container is required")
    if container.get("uri") != uri or container.get("format") != format_:
        raise ValueError(f"{context}.container must use the exact pinned source container")
    if sha256 is not None and container.get("sha256") != sha256:
        raise ValueError(f"{context}.container.sha256 must match the pinned container")


def _asset_identity(reference: Mapping[str, Any]) -> str:
    kind = reference["kind"]
    if kind == "direct":
        return str(reference["uri"])
    container_uri = reference["container"]["uri"]
    if kind == "archive_member":
        return f"{container_uri}#member={reference['member']}"
    if kind == "parquet_field":
        return f"{container_uri}#row={reference['row_index']}&field={reference['field']}"
    return f"{container_uri}#row={reference['row_index']}"


def _validate_source_inputs(
    dataset: str,
    inputs: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    spec: DatasetSpec,
    context: str,
) -> None:
    by_role = {item["role"]: item for item in inputs}
    if dataset == "pubtables_v2":
        uri = _hf_uri(spec.source, spec.revision, _PUBTABLES_IMAGES_PATH)
        document_id = metadata["document_id"]
        for ordinal, page_index in enumerate(metadata["page_indices"], start=1):
            item = by_role[f"page_{ordinal}"]
            _require_container(
                item,
                kind="archive_member",
                uri=uri,
                sha256=_PUBTABLES_IMAGES_SHA256,
                format_="tar.gz",
                context=f"{context}.inputs[{ordinal - 1}]",
            )
            expected_member = f"Full Documents/test/images/{document_id}_page_{page_index}.jpg"
            if item.get("member") != expected_member:
                raise ValueError(f"{context}: PubTables input member does not match document/page metadata")
        return
    if dataset == "varex":
        bindings: set[tuple[str, str, int]] = set()
        for role, field in (("image_200dpi", "image"), ("image_50dpi", "image_50dpi")):
            item = by_role[role]
            if item.get("kind") != "parquet_field" or item.get("field") != field:
                raise ValueError(f"{context}: {role} must reference VAREX parquet field {field!r}")
            container = item["container"]
            matched = [
                path
                for path, sha256 in _VAREX_SHARDS.items()
                if container == {
                    "uri": _hf_uri(spec.source, spec.revision, path),
                    "sha256": sha256,
                    "format": "parquet",
                }
            ]
            if not matched:
                raise ValueError(f"{context}: VAREX input must use a pinned benchmark parquet shard")
            bindings.add((container["uri"], container["sha256"], item["row_index"]))
        if len(bindings) != 1:
            raise ValueError(f"{context}: VAREX inputs must use the same parquet row")
        return
    if dataset == "ai2d_rst":
        item = by_role["image"]
        _require_container(
            item,
            kind="archive_member",
            uri=_AI2D_IMAGE_ARCHIVE_URL,
            sha256=_AI2D_IMAGE_ARCHIVE_SHA256,
            format_="zip",
            context=f"{context}.inputs[0]",
        )
        if item.get("member") != f"ai2d/images/{metadata['diagram_id']}.png":
            raise ValueError(f"{context}: AI2D image member must match metadata.diagram_id")
        return
    item = by_role["image"]
    if item.get("kind") != "direct":
        raise ValueError(f"{context}: MWS image must be a direct reference")
    uri = item["uri"]
    parsed = urlparse(uri)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ValueError(f"{context}: MWS image must use trusted HTTPS Hugging Face storage")
    prefix = f"/datasets/{spec.source}/resolve/{spec.revision}/"
    _safe_pinned_path(uri, prefix, f"{context}.inputs[0]")
    expected_uri = _hf_uri(spec.source, spec.revision, metadata["image_path"])
    if uri != expected_uri or not metadata["image_path"].startswith("images/"):
        raise ValueError(f"{context}: MWS image URI must match metadata.image_path")


def _schema_property_reaches_type(schema: Mapping[str, Any], target_type: str) -> bool:
    properties = schema.get("properties")
    definitions = schema.get("$defs")
    if not isinstance(properties, Mapping):
        return False
    stack: list[Any] = list(properties.values())
    visited_references: set[str] = set()
    visited_nodes = 0
    while stack:
        value = stack.pop()
        visited_nodes += 1
        if visited_nodes > 10_000:
            raise ValueError("schema exceeds structural validation limit")
        if isinstance(value, list):
            stack.extend(value)
            continue
        if not isinstance(value, Mapping):
            continue
        if value.get("type") == target_type:
            return True
        reference = value.get("$ref")
        prefix = "#/$defs/"
        if (
            isinstance(reference, str)
            and reference.startswith(prefix)
            and "/" not in reference.removeprefix(prefix)
            and isinstance(definitions, Mapping)
            and reference not in visited_references
        ):
            visited_references.add(reference)
            target = definitions.get(reference.removeprefix(prefix))
            if isinstance(target, Mapping):
                stack.append(target)
        nested_properties = value.get("properties")
        if isinstance(nested_properties, Mapping):
            stack.extend(nested_properties.values())
        for keyword in ("items", "allOf", "anyOf", "oneOf"):
            nested = value.get(keyword)
            if isinstance(nested, (Mapping, list)):
                stack.append(nested)
    return False


def _contains_list(value: Any) -> bool:
    stack = [value]
    visited_nodes = 0
    while stack:
        nested = stack.pop()
        visited_nodes += 1
        if visited_nodes > 10_000:
            raise ValueError("ground_truth exceeds structural validation limit")
        if isinstance(nested, list):
            return True
        if isinstance(nested, Mapping):
            stack.extend(nested.values())
    return False


def _validate_metadata(
    dataset: str,
    stratum: str,
    metadata: Any,
    *,
    canonical_id: str,
    group_id: str,
    spec: DatasetSpec,
    expected_inputs: int,
    inputs: Sequence[Mapping[str, Any]],
    context: str,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"{context}: metadata must be an object")
    if dataset == "pubtables_v2":
        document_id = _require_string(metadata, "document_id", context)
        _require_string(metadata, "table_id", context)
        page_indices = metadata.get("page_indices")
        if (
            not isinstance(page_indices, list)
            or len(page_indices) != expected_inputs
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in page_indices
            )
            or page_indices != sorted(set(page_indices))
        ):
            raise ValueError(
                f"{context}: metadata.page_indices must contain {expected_inputs} unique sorted integers"
            )
        annotation = _validate_reference(
            metadata.get("annotation"), context=f"{context}.metadata.annotation"
        )
        _require_container(
            annotation,
            kind="archive_member",
            uri=_hf_uri(spec.source, spec.revision, _PUBTABLES_TABLES_PATH),
            sha256=_PUBTABLES_TABLES_SHA256,
            format_="tar.gz",
            context=f"{context}.metadata.annotation",
        )
        if annotation.get("member") != f"Full Documents/test/tables/{document_id}_tables.json":
            raise ValueError(f"{context}: PubTables annotation member must match document_id")
        expected_group_id = document_id
    elif dataset == "varex":
        expected_group_id = _require_string(metadata, "doc_id", context)
        _validate_source_inputs(dataset, inputs, metadata, spec=spec, context=context)
        source_record = _validate_record_reference(
            metadata.get("source_record"), context=f"{context}.metadata.source_record"
        )
        schema = metadata.get("schema")
        ground_truth = metadata.get("ground_truth")
        if not isinstance(schema, dict) or not schema:
            raise ValueError(f"{context}: metadata.schema must be a non-empty object")
        if not isinstance(ground_truth, dict) or not ground_truth:
            raise ValueError(f"{context}: metadata.ground_truth must be a non-empty object")
        properties = schema.get("properties")
        if schema.get("type") != "object" or not isinstance(properties, dict) or not properties:
            raise ValueError(f"{context}: metadata.schema requires object type and non-empty properties")
        if stratum == "Nested" and not _schema_property_reaches_type(schema, "object"):
            raise ValueError(f"{context}: Nested schema requires a nested object property")
        if stratum == "Nested" and not any(isinstance(value, dict) for value in ground_truth.values()):
            raise ValueError(f"{context}: Nested ground_truth requires a nested object value")
        if stratum == "Table" and not _schema_property_reaches_type(schema, "array"):
            raise ValueError(f"{context}: Table schema requires an array property")
        if stratum == "Table" and not _contains_list(ground_truth):
            raise ValueError(f"{context}: Table ground_truth requires an array value")
        input_binding = inputs[0]
        if (
            source_record["kind"] != "parquet_row"
            or source_record["container"] != input_binding["container"]
            or source_record["row_index"] != input_binding["row_index"]
            or source_record["fields"] != _VAREX_RECORD_FIELDS
        ):
            raise ValueError(f"{context}: VAREX source_record must bind the selected parquet row")
        expected_record_hash = _canonical_json_hash(
            {
                "doc_id": expected_group_id,
                "ground_truth": ground_truth,
                "schema": schema,
                "split": stratum,
            }
        )
        if source_record["sha256"] != expected_record_hash:
            raise ValueError(f"{context}: VAREX source_record sha256 does not match canonical JSON")
    elif dataset == "ai2d_rst":
        expected_group_id = _require_string(metadata, "diagram_id", context)
        annotation = _validate_reference(
            metadata.get("annotation"), context=f"{context}.metadata.annotation"
        )
        _require_container(
            annotation,
            kind="archive_member",
            uri=_AI2D_ANNOTATION_ARCHIVE_URL,
            sha256=_AI2D_ANNOTATION_ARCHIVE_SHA256,
            format_="zip",
            context=f"{context}.metadata.annotation",
        )
        expected_member = f"ai2d-rst-v1-1/json/ai2d-rst/{expected_group_id}.png.json"
        if annotation.get("member") != expected_member:
            raise ValueError(f"{context}: AI2D annotation member must match metadata.diagram_id")
    elif dataset == "mws_vision_bench":
        expected_group_id = _require_string(metadata, "image_path", context)
        question = _require_string(metadata, "question", context)
        dataset_name = _require_string(metadata, "dataset_name", context)
        if metadata.get("task_type") != stratum:
            raise ValueError(f"{context}: metadata.task_type must match stratum {stratum!r}")
        answers = metadata.get("answers")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(answer, str) or not answer for answer in answers)
        ):
            raise ValueError(f"{context}: metadata.answers must be a non-empty list of strings")
        source_record = _validate_record_reference(
            metadata.get("source_record"), context=f"{context}.metadata.source_record"
        )
        if (
            source_record["kind"] != "jsonl_row"
            or source_record["container"]
            != {"uri": _MWS_METADATA_URL, "sha256": _MWS_METADATA_SHA256, "format": "jsonl"}
            or source_record["fields"] != _MWS_RECORD_FIELDS
        ):
            raise ValueError(f"{context}: MWS source_record must use the pinned legacy metadata row")
        expected_record_hash = _canonical_json_hash(
            {
                "answers": answers,
                "dataset_name": dataset_name,
                "file_name": expected_group_id,
                "id": canonical_id,
                "question": question,
                "type": _MWS_SOURCE_TYPES[stratum],
            }
        )
        if source_record["sha256"] != expected_record_hash:
            raise ValueError(f"{context}: MWS source_record sha256 does not match canonical JSON")
    else:
        raise ValueError(f"no metadata schema for dataset: {dataset}")
    if group_id != expected_group_id:
        raise ValueError(
            f"{context}: group_id must equal the physical content id {expected_group_id!r}"
        )
    normalized = dict(metadata)
    _validate_source_inputs(dataset, inputs, normalized, spec=spec, context=context)
    if dataset == "varex":
        schema_hash = _canonical_json_hash(normalized["schema"])
        ground_truth_hash = _canonical_json_hash(normalized["ground_truth"])
        for key, value in (
            ("schema_canonical_sha256", schema_hash),
            ("ground_truth_canonical_sha256", ground_truth_hash),
        ):
            if key in normalized and normalized[key] != value:
                raise ValueError(f"{context}: metadata.{key} does not match canonical content")
            normalized[key] = value
    if dataset == "mws_vision_bench":
        record_hash = expected_record_hash
        key = "source_record_canonical_sha256"
        if key in normalized and normalized[key] != record_hash:
            raise ValueError(f"{context}: metadata.{key} does not match the bound record")
        normalized[key] = record_hash
    return normalized


def _validate_candidate(
    candidate: dict[str, Any],
    *,
    specs: Mapping[str, DatasetSpec],
) -> dict[str, Any]:
    allowed_fields = {"dataset", "stratum", "canonical_id", "group_id", "inputs", "metadata"}
    unknown_fields = set(candidate) - allowed_fields
    if unknown_fields:
        raise ValueError(f"candidate contains unsupported fields: {sorted(unknown_fields)}")
    dataset = candidate.get("dataset")
    if not isinstance(dataset, str) or dataset not in specs:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    stratum = candidate.get("stratum")
    if not isinstance(stratum, str) or stratum not in specs[dataset].quotas:
        raise ValueError(f"unsupported stratum for {dataset}: {stratum!r}")
    canonical_id = candidate.get("canonical_id")
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        raise ValueError("canonical_id must be a non-empty string")
    group_id = candidate.get("group_id")
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError(f"{dataset}/{canonical_id}: group_id must be a non-empty string")
    context = f"{dataset}/{canonical_id}"
    inputs = candidate.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{context}: inputs must be a non-empty list")
    expected = specs[dataset].expected_inputs[stratum]
    if len(inputs) != expected:
        raise ValueError(
            f"{context}: stratum {stratum!r} requires {expected} inputs, "
            f"got {len(inputs)}"
        )
    normalized_inputs: list[dict[str, Any]] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"{context}: input {index} must be an object")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{context}: input {index} requires a non-empty role")
        reference_value = {key: value for key, value in item.items() if key != "role"}
        reference = _validate_reference(
            reference_value,
            context=f"{context}.inputs[{index}]",
        )
        normalized_inputs.append({"role": role, **reference})
    roles = [item["role"] for item in normalized_inputs]
    expected_roles = _expected_roles(dataset, stratum, expected)
    if set(roles) != expected_roles or len(roles) != len(set(roles)):
        raise ValueError(f"{context}: input roles must be exactly {sorted(expected_roles)}")
    normalized_inputs.sort(key=lambda item: item["role"])
    identities = [_asset_identity(item) for item in normalized_inputs]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{context}: input asset references must be unique")
    input_hashes = [item["sha256"] for item in normalized_inputs]
    if len(input_hashes) != len(set(input_hashes)):
        raise ValueError(f"{context}: input sha256 values must be unique")
    metadata = _validate_metadata(
        dataset,
        stratum,
        candidate.get("metadata"),
        canonical_id=canonical_id,
        group_id=group_id,
        spec=specs[dataset],
        expected_inputs=expected,
        inputs=normalized_inputs,
        context=context,
    )
    return {
        "dataset": dataset,
        "stratum": stratum,
        "canonical_id": canonical_id,
        "group_id": group_id,
        "inputs": normalized_inputs,
        "metadata": metadata,
    }


def _select_dataset(
    dataset: str,
    candidates: Sequence[dict[str, Any]],
    *,
    seed: str,
    spec: DatasetSpec,
) -> list[dict[str, Any]]:
    ranked: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in spec.quotas}
    for candidate in candidates:
        selection_hash = _selection_hash(
            seed,
            dataset,
            candidate["stratum"],
            candidate["canonical_id"],
        )
        ranked[candidate["stratum"]].append({**candidate, "selection_hash": selection_hash})
    for stratum, values in ranked.items():
        values.sort(key=lambda item: (item["selection_hash"], item["canonical_id"]))
        best_by_group: dict[str, dict[str, Any]] = {}
        for candidate in values:
            best_by_group.setdefault(candidate["group_id"], candidate)
        ranked[stratum] = list(best_by_group.values())

    # Bipartite b-matching: quota slots on the left, physical groups on the right.
    # Augmenting paths preserve deterministic hash order but can move an earlier
    # group to another stratum, avoiding the false shortages of greedy selection.
    strata = sorted(
        spec.quotas,
        key=lambda stratum: (len(ranked[stratum]) / spec.quotas[stratum], stratum),
    )
    slots = [
        (stratum, slot_index)
        for stratum in strata
        for slot_index in range(spec.quotas[stratum])
    ]
    group_to_slot: dict[str, tuple[str, int]] = {}
    slot_to_candidate: dict[tuple[str, int], dict[str, Any]] = {}

    def augment(slot: tuple[str, int], seen_groups: set[str]) -> bool:
        stratum, _ = slot
        for candidate in ranked[stratum]:
            group_id = candidate["group_id"]
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            previous_slot = group_to_slot.get(group_id)
            if previous_slot is None or augment(previous_slot, seen_groups):
                group_to_slot[group_id] = slot
                slot_to_candidate[slot] = candidate
                return True
        return False

    unmatched: list[tuple[str, int]] = []
    for slot in slots:
        if not augment(slot, set()):
            unmatched.append(slot)
    if unmatched:
        matched_counts = {
            stratum: sum(slot[0] == stratum for slot in slot_to_candidate)
            for stratum in spec.quotas
        }
        shortages = {
            stratum: spec.quotas[stratum] - matched_counts[stratum]
            for stratum in sorted(spec.quotas)
            if matched_counts[stratum] < spec.quotas[stratum]
        }
        raise ValueError(f"not enough unique groups for {dataset}: {shortages}")

    selected = list(slot_to_candidate.values())
    return sorted(selected, key=lambda item: (item["stratum"], item["selection_hash"]))


def _validate_selected_assets(selected: Sequence[dict[str, Any]]) -> None:
    seen_references: dict[str, tuple[str, str]] = {}
    seen_hashes: dict[str, tuple[str, str]] = {}
    seen_metadata_references: dict[str, tuple[str, str]] = {}
    for candidate in selected:
        owner = (candidate["dataset"], candidate["group_id"])
        for item in candidate["inputs"]:
            for key, value, seen in (
                ("reference", _asset_identity(item), seen_references),
                ("sha256", item["sha256"], seen_hashes),
            ):
                previous = seen.get(value)
                if previous is not None and previous != owner:
                    raise ValueError(
                        f"duplicate selected input {key} across groups: {previous} and {owner}"
                    )
                seen[value] = owner
        reference_key = {
            "pubtables_v2": "annotation",
            "varex": "source_record",
            "ai2d_rst": "annotation",
            "mws_vision_bench": "source_record",
        }[candidate["dataset"]]
        reference_identity = _asset_identity(candidate["metadata"][reference_key])
        previous = seen_metadata_references.get(reference_identity)
        if previous is not None and previous != owner:
            raise ValueError(
                f"duplicate selected metadata reference across groups: {previous} and {owner}"
            )
        seen_metadata_references[reference_identity] = owner


def _validate_specs(specs: Mapping[str, DatasetSpec]) -> None:
    if not specs:
        raise ValueError("at least one dataset spec is required")
    for dataset, spec in specs.items():
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset spec: {dataset}")
        pinned = DATASETS[dataset]
        pinned_metadata = (
            "source",
            "source_url",
            "revision",
            "license",
            "license_components",
            "source_components",
        )
        if any(getattr(spec, field) != getattr(pinned, field) for field in pinned_metadata):
            raise ValueError(f"{dataset}: source and license metadata must match the pinned spec")
        if not spec.source or not spec.source_url or not spec.revision or not spec.license:
            raise ValueError(f"{dataset}: source, source_url, revision and license are required")
        if set(spec.quotas) != set(spec.expected_inputs):
            raise ValueError(f"{dataset}: quotas and expected_inputs strata must match")
        if not set(spec.quotas) <= set(pinned.quotas):
            raise ValueError(f"{dataset}: unsupported quota stratum")
        if any(spec.expected_inputs[stratum] != pinned.expected_inputs[stratum] for stratum in spec.quotas):
            raise ValueError(f"{dataset}: expected_inputs must match the pinned spec")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (*spec.quotas.values(), *spec.expected_inputs.values())
        ):
            raise ValueError(f"{dataset}: quotas and expected_inputs must be positive")
        if any(spec.quotas[stratum] > pinned.quotas[stratum] for stratum in spec.quotas):
            raise ValueError(f"{dataset}: quota exceeds the pinned corpus design")


def build_manifest(
    candidates: Iterable[dict[str, Any]],
    *,
    seed: str = _DEFAULT_SEED,
    specs: Mapping[str, DatasetSpec] = DATASETS,
) -> dict[str, Any]:
    if not seed:
        raise ValueError("seed must be non-empty")
    _validate_specs(specs)
    validated = [_validate_candidate(candidate, specs=specs) for candidate in candidates]
    identities: set[tuple[str, str]] = set()
    for candidate in validated:
        identity = (candidate["dataset"], candidate["canonical_id"])
        if identity in identities:
            raise ValueError(f"duplicate candidate: {'/'.join(identity)}")
        identities.add(identity)

    by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in specs}
    for candidate in validated:
        by_dataset[candidate["dataset"]].append(candidate)

    selected: list[dict[str, Any]] = []
    for dataset in sorted(specs):
        selected.extend(
            _select_dataset(dataset, by_dataset[dataset], seed=seed, spec=specs[dataset])
        )
    _validate_selected_assets(selected)

    dataset_metadata: dict[str, Any] = {}
    summary_by_dataset: dict[str, Any] = {}
    for dataset in sorted(specs):
        spec = specs[dataset]
        metadata: dict[str, Any] = {
            "source": spec.source,
            "source_url": spec.source_url,
            "revision": spec.revision,
            "license": spec.license,
            "quotas": dict(spec.quotas),
        }
        if spec.license_components is not None:
            metadata["license_components"] = dict(spec.license_components)
        if spec.source_components is not None:
            metadata["source_components"] = {
                name: dict(component) for name, component in spec.source_components.items()
            }
        dataset_metadata[dataset] = metadata
        items = [item for item in selected if item["dataset"] == dataset]
        summary_by_dataset[dataset] = {
            "selected_units": len(items),
            "input_count": sum(len(item["inputs"]) for item in items),
            "by_stratum": {
                stratum: sum(item["stratum"] == stratum for item in items)
                for stratum in sorted(spec.quotas)
            },
        }

    return {
        "manifest_version": 1,
        "selection": {
            "algorithm": _HASH_ALGORITHM,
            "seed": seed,
            "catalog_sha256": _catalog_hash(validated),
            "canonical_json_algorithm": _CANONICAL_JSON_ALGORITHM,
            "group_policy": "claimed_identity_only; one selected unit per dataset/group_id",
        },
        "verification_state": "metadata_only_unverified",
        "materialization_requirements": {
            "verify_every_referenced_object_bytes_against_sha256": True,
            "verify_group_id_against_trusted_physical_identity": True,
            "reject_manifest_on_any_mismatch": True,
        },
        "datasets": dataset_metadata,
        "summary": {
            "candidate_count": len(validated),
            "selected_units": len(selected),
            "input_count": sum(len(item["inputs"]) for item in selected),
            "by_dataset": summary_by_dataset,
        },
        "selected": selected,
    }


def _spec_document(specs: Mapping[str, DatasetSpec] = DATASETS) -> dict[str, Any]:
    return {
        dataset: {
            "source": spec.source,
            "source_url": spec.source_url,
            "revision": spec.revision,
            "license": spec.license,
            "quotas": dict(spec.quotas),
            "expected_inputs": dict(spec.expected_inputs),
            **(
                {"license_components": dict(spec.license_components)}
                if spec.license_components is not None
                else {}
            ),
            **(
                {
                    "source_components": {
                        name: dict(component) for name, component in spec.source_components.items()
                    }
                }
                if spec.source_components is not None
                else {}
            ),
        }
        for dataset, spec in sorted(specs.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an offline, hash-ranked OCR acceptance manifest from JSONL metadata."
    )
    parser.add_argument("candidates", type=Path, nargs="?", help="lightweight candidate catalog (JSONL)")
    parser.add_argument("output", type=Path, nargs="?", help="output manifest JSON")
    parser.add_argument("--seed", default=_DEFAULT_SEED)
    parser.add_argument(
        "--print-specs",
        action="store_true",
        help="print pinned dataset metadata and catalog quotas without reading candidates",
    )
    args = parser.parse_args()
    if args.print_specs:
        print(json.dumps(_spec_document(), ensure_ascii=False, indent=2))
        return
    if args.candidates is None or args.output is None:
        parser.error("candidates and output are required unless --print-specs is used")

    manifest = build_manifest(_read_candidates(args.candidates), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected {manifest['summary']['selected_units']} units / "
        f"{manifest['summary']['input_count']} inputs -> {args.output}"
    )


if __name__ == "__main__":
    main()
