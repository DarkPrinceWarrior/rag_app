"""Build the offline candidate catalog consumed by build_licensed_ocr_manifest.py."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_CHUNK_SIZE = 1024 * 1024
_PUB_IMAGES_PATH = "PubTables-v2_Full-Documents_test_images.tar.gz"
_PUB_IMAGES_SHA256 = "0d42821fb1dce5713a86c327bec5fabbe214bb5ebbd0cfc75cd2ef89b7c7230e"
_PUB_TABLES_PATH = "PubTables-v2_Full-Documents_test_tables.tar.gz"
_PUB_TABLES_SHA256 = "dfd10e0dc4cb3e92d0f521e8a135e6e96094d8e90e130fb3fe25c9fa31b3a3de"
_PUB_REVISION = "aa575e798cb00a296925e2086addb3e3fd9a1903"
_PUB_IMAGE_RE = re.compile(r"Full Documents/test/images/(.+)_page_([0-9]+)\.jpg")
_PUB_TABLE_RE = re.compile(r"Full Documents/test/tables/(.+)_tables\.json")

_VAREX_REVISION = "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6"
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
_VAREX_COLUMNS = (
    "doc_id",
    "split",
    "image",
    "image_50dpi",
    "schema",
    "ground_truth",
)

_AI2D_IMAGES_PATH = "ai2d-all.zip"
_AI2D_IMAGES_URL = (
    "https://ai2-public-datasets.s3.us-west-2.amazonaws.com/diagrams/ai2d-all.zip"
)
_AI2D_IMAGES_SHA256 = "1a6b77eebb8b7dbdf76a0ba6ca76c2f97ce8f81d8ee33b06593aa722e54c4786"
_AI2D_ANNOTATIONS_PATH = "ai2d-rst-v1-1.zip"
_AI2D_ANNOTATIONS_URL = (
    "https://www.kielipankki.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip"
)
_AI2D_ANNOTATIONS_SHA256 = (
    "eb11d67507e08eb9bfd0f5944da7ca32cfcffa13e119b04ac5054effa65a759a"
)
_AI2D_IMAGE_RE = re.compile(r"ai2d/images/([^/]+)\.png")
_AI2D_ANNOTATION_RE = re.compile(r"ai2d-rst-v1-1/json/ai2d-rst/([^/]+)\.png\.json")

_MWS_METADATA_PATH = "metadata.jsonl"
_MWS_METADATA_URL = (
    "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/"
    "e204166bde25f7dcaaffb9313b855de67b516e5d/metadata.jsonl"
)
_MWS_METADATA_SHA256 = "c234a569583858bfab13399169ec9951da12edf6d88a5cd4c4efae8a1fd4197d"
_MWS_IMAGE_REVISION = "b8d473734b79343cac2b74f692a29ab191c7d11d"
_MWS_TYPE_TO_STRATUM = {
    "document parsing ru": "document_parsing_ru",
    "full-page OCR ru": "full_page_ocr_ru",
    "key information extraction ru": "key_information_extraction_ru",
    "reasoning VQA ru": "reasoning_vqa_ru",
    "text grounding ru": "text_grounding_ru",
}
_MWS_FIELDS = ["answers", "dataset_name", "file_name", "id", "question", "type"]

_QUOTAS = {
    "pubtables_v2": {"pages_2": 8, "pages_3": 4, "pages_4": 2},
    "varex": {"Flat": 10, "Nested": 10, "Table": 10},
    "ai2d_rst": {"diagram": 12},
    "mws_vision_bench": {
        "document_parsing_ru": 6,
        "full_page_ocr_ru": 6,
        "key_information_extraction_ru": 6,
        "reasoning_vqa_ru": 6,
        "text_grounding_ru": 6,
    },
}


@dataclass(frozen=True)
class SourcePins:
    pub_images_sha256: str = _PUB_IMAGES_SHA256
    pub_tables_sha256: str = _PUB_TABLES_SHA256
    varex_shards: Mapping[str, str] | None = None
    ai2d_images_sha256: str = _AI2D_IMAGES_SHA256
    ai2d_annotations_sha256: str = _AI2D_ANNOTATIONS_SHA256
    mws_metadata_sha256: str = _MWS_METADATA_SHA256

    def resolved_varex_shards(self) -> Mapping[str, str]:
        return _VAREX_SHARDS if self.varex_shards is None else self.varex_shards


@dataclass(frozen=True)
class SafetyLimits:
    max_members: int = 100_000
    max_member_bytes: int = 64 * 1024 * 1024
    max_total_member_bytes: int = 32 * 1024 * 1024 * 1024
    max_jsonl_line_bytes: int = 4 * 1024 * 1024
    max_image_index_bytes: int = 16 * 1024 * 1024
    max_image_index_entries: int = 100_000


class ParquetAdapter(Protocol):
    def iter_rows(self, path: Path, columns: Sequence[str]) -> Iterable[Mapping[str, Any]]: ...


class _PyArrowAdapter:
    def iter_rows(self, path: Path, columns: Sequence[str]) -> Iterable[Mapping[str, Any]]:
        try:
            parquet = importlib.import_module("pyarrow.parquet")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "VAREX generation requires optional pyarrow or an injected parquet adapter"
            ) from exc
        parquet_file = parquet.ParquetFile(path)
        return (
            row
            for batch in parquet_file.iter_batches(batch_size=32, columns=list(columns))
            for row in batch.to_pylist()
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _strict_json_loads(value: str | bytes) -> Any:
    return json.loads(
        value,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {constant}")
        ),
    )


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _file_sha256(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    digest = hashlib.sha256()
    try:
        file_fd = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
        )
    except OSError as exc:
        raise ValueError(f"cannot open verified regular file safely: {path}") from exc
    with os.fdopen(file_fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"verified source must be a regular file: {path}")
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    before_snapshot = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_snapshot = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_snapshot != after_snapshot:
        raise ValueError(f"verified source changed while hashing: {path}")
    return digest.hexdigest(), after_snapshot


def _verify_file(path: Path, expected_sha256: str, context: str) -> tuple[int, int, int, int]:
    actual, snapshot = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{context}: container SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return snapshot


def _verify_unchanged(
    path: Path,
    expected_sha256: str,
    snapshot: tuple[int, int, int, int],
    context: str,
) -> None:
    if _verify_file(path, expected_sha256, context) != snapshot:
        raise ValueError(f"{context}: container changed during candidate generation")


def _safe_member(name: str, context: str) -> str:
    if not name or "\\" in name or name.startswith("/"):
        raise ValueError(f"{context}: unsafe archive member path {name!r}")
    parts = name.rstrip("/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{context}: unsafe archive member path {name!r}")
    return name


def _checked_member_size(
    size: int, *, total: int, count: int, limits: SafetyLimits, context: str
) -> int:
    if count > limits.max_members:
        raise ValueError(f"{context}: archive exceeds member count limit")
    if size < 0 or size > limits.max_member_bytes:
        raise ValueError(f"{context}: archive member exceeds size limit")
    total += size
    if total > limits.max_total_member_bytes:
        raise ValueError(f"{context}: archive exceeds total uncompressed size limit")
    return total


def _tar_files(path: Path, limits: SafetyLimits) -> Iterator[tuple[str, bytes]]:
    seen: set[str] = set()
    total = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for count, member in enumerate(archive, start=1):
            if count > limits.max_members:
                raise ValueError(f"{path}: archive exceeds member count limit")
            name = _safe_member(member.name, str(path))
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"{path}: links and special archive members are forbidden")
            if name in seen:
                raise ValueError(f"{path}: duplicate archive member {name!r}")
            seen.add(name)
            total = _checked_member_size(
                member.size, total=total, count=count, limits=limits, context=str(path)
            )
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"{path}: cannot read archive member {name!r}")
            payload = stream.read(limits.max_member_bytes + 1)
            if len(payload) != member.size:
                raise ValueError(f"{path}: truncated archive member {name!r}")
            yield name, payload


def _zip_index(path: Path, limits: SafetyLimits) -> dict[str, zipfile.ZipInfo]:
    seen: set[str] = set()
    indexed: dict[str, zipfile.ZipInfo] = {}
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > limits.max_members:
            raise ValueError(f"{path}: archive exceeds member count limit")
        for count, info in enumerate(infos, start=1):
            name = _safe_member(info.filename, str(path))
            if info.is_dir():
                continue
            if name in seen:
                raise ValueError(f"{path}: duplicate archive member {name!r}")
            seen.add(name)
            mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if info.flag_bits & 0x1 or (
                file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}
            ):
                raise ValueError(f"{path}: links, special, and encrypted ZIP members are forbidden")
            total = _checked_member_size(
                info.file_size, total=total, count=count, limits=limits, context=str(path)
            )
            indexed[name] = info
    return indexed


def _zip_member_hashes(
    path: Path, members: Iterable[str], limits: SafetyLimits
) -> dict[str, str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for member in sorted(members):
            info = archive.getinfo(member)
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as stream:
                while chunk := stream.read(_CHUNK_SIZE):
                    size += len(chunk)
                    if size > limits.max_member_bytes:
                        raise ValueError(f"{path}: archive member exceeds size limit")
                    digest.update(chunk)
            if size != info.file_size:
                raise ValueError(f"{path}: truncated archive member {member!r}")
            result[member] = digest.hexdigest()
    return result


def _container(uri: str, sha256: str, format_: str) -> dict[str, str]:
    return {"uri": uri, "sha256": sha256, "format": format_}


def _archive_reference(
    container: Mapping[str, str], member: str, sha256: str
) -> dict[str, Any]:
    return {
        "kind": "archive_member",
        "container": dict(container),
        "member": member,
        "sha256": sha256,
    }


def _hf_uri(source: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{source}/resolve/{revision}/{path}"


def _first_identifier(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("table_id", "structure_id", "id"):
            identifier = value.get(key)
            if isinstance(identifier, (str, int)) and str(identifier):
                return str(identifier)
        for nested in value.values():
            identifier = _first_identifier(nested)
            if identifier is not None:
                return identifier
    elif isinstance(value, list):
        for nested in value:
            identifier = _first_identifier(nested)
            if identifier is not None:
                return identifier
    return None


def _pubtables_candidates(
    root: Path, pins: SourcePins, limits: SafetyLimits
) -> list[dict[str, Any]]:
    images_path = root / _PUB_IMAGES_PATH
    tables_path = root / _PUB_TABLES_PATH
    images_snapshot = _verify_file(images_path, pins.pub_images_sha256, "PubTables images")
    tables_snapshot = _verify_file(tables_path, pins.pub_tables_sha256, "PubTables annotations")
    pages: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    for member, payload in _tar_files(images_path, limits):
        match = _PUB_IMAGE_RE.fullmatch(member)
        if match is None:
            continue
        document_id, page_text = match.groups()
        page_index = int(page_text)
        if page_index in pages[document_id]:
            raise ValueError(f"PubTables: duplicate document page {document_id}/{page_index}")
        pages[document_id][page_index] = (member, _sha256_bytes(payload))
    annotations: dict[str, tuple[str, str, str]] = {}
    for member, payload in _tar_files(tables_path, limits):
        match = _PUB_TABLE_RE.fullmatch(member)
        if match is None:
            continue
        document_id = match.group(1)
        if document_id in annotations:
            raise ValueError(f"PubTables: duplicate annotation document {document_id!r}")
        try:
            value = _strict_json_loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"PubTables: invalid annotation JSON {member!r}") from exc
        table_id = _first_identifier(value) or _sha256_bytes(payload)[:16]
        annotations[document_id] = (member, _sha256_bytes(payload), table_id)
    image_container = _container(
        _hf_uri("kensho/PubTables-v2", _PUB_REVISION, _PUB_IMAGES_PATH),
        pins.pub_images_sha256,
        "tar.gz",
    )
    table_container = _container(
        _hf_uri("kensho/PubTables-v2", _PUB_REVISION, _PUB_TABLES_PATH),
        pins.pub_tables_sha256,
        "tar.gz",
    )
    candidates: list[dict[str, Any]] = []
    for document_id in sorted(set(pages) & set(annotations)):
        available = sorted(pages[document_id])
        annotation_member, annotation_sha, table_id = annotations[document_id]
        for count in (2, 3, 4):
            if len(available) < count:
                continue
            selected_pages = available[:count]
            candidates.append(
                {
                    "dataset": "pubtables_v2",
                    "stratum": f"pages_{count}",
                    "canonical_id": f"{document_id}:pages_{count}",
                    "group_id": document_id,
                    "inputs": [
                        {
                            "role": f"page_{ordinal}",
                            **_archive_reference(
                                image_container,
                                pages[document_id][page_index][0],
                                pages[document_id][page_index][1],
                            ),
                        }
                        for ordinal, page_index in enumerate(selected_pages, start=1)
                    ],
                    "metadata": {
                        "document_id": document_id,
                        "table_id": table_id,
                        "page_indices": selected_pages,
                        "annotation": _archive_reference(
                            table_container, annotation_member, annotation_sha
                        ),
                    },
                }
            )
    _verify_unchanged(
        images_path, pins.pub_images_sha256, images_snapshot, "PubTables images"
    )
    _verify_unchanged(
        tables_path, pins.pub_tables_sha256, tables_snapshot, "PubTables annotations"
    )
    return candidates


def _json_object(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = _strict_json_loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: invalid JSON") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{context}: expected a non-empty JSON object")
    return value


def _image_bytes(value: Any, context: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping):
        return _image_bytes(value.get("bytes"), context)
    raise ValueError(f"{context}: image field does not contain raw bytes")


def _varex_candidates(
    root: Path, pins: SourcePins, adapter: ParquetAdapter, limits: SafetyLimits
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for relative_path, container_sha in sorted(pins.resolved_varex_shards().items()):
        _safe_member(relative_path, "VAREX shard path")
        path = root / relative_path
        snapshot = _verify_file(path, container_sha, f"VAREX {relative_path}")
        container = _container(
            _hf_uri("ibm-research/VAREX", _VAREX_REVISION, relative_path),
            container_sha,
            "parquet",
        )
        for row_index, row in enumerate(adapter.iter_rows(path, _VAREX_COLUMNS)):
            if not isinstance(row, Mapping):
                raise ValueError(f"VAREX {relative_path}/{row_index}: row must be an object")
            doc_id = row.get("doc_id")
            stratum = row.get("split")
            if not isinstance(doc_id, str) or not doc_id or stratum not in {"Flat", "Nested", "Table"}:
                raise ValueError(f"VAREX {relative_path}/{row_index}: invalid doc_id or split")
            if doc_id in seen_doc_ids:
                raise ValueError(f"VAREX: duplicate doc_id {doc_id!r}")
            seen_doc_ids.add(doc_id)
            schema = _json_object(row.get("schema"), f"VAREX {doc_id}.schema")
            ground_truth = _json_object(
                row.get("ground_truth"), f"VAREX {doc_id}.ground_truth"
            )
            image = _image_bytes(row.get("image"), f"VAREX {doc_id}.image")
            image_50 = _image_bytes(row.get("image_50dpi"), f"VAREX {doc_id}.image_50dpi")
            if len(image) > limits.max_member_bytes or len(image_50) > limits.max_member_bytes:
                raise ValueError(f"VAREX {doc_id}: image field exceeds size limit")
            record = {
                "doc_id": doc_id,
                "ground_truth": ground_truth,
                "schema": schema,
                "split": stratum,
            }
            if len(_canonical_json(record)) > limits.max_member_bytes:
                raise ValueError(f"VAREX {doc_id}: canonical row exceeds size limit")
            candidates.append(
                {
                    "dataset": "varex",
                    "stratum": stratum,
                    "canonical_id": doc_id,
                    "group_id": doc_id,
                    "inputs": [
                        {
                            "role": "image_200dpi",
                            "kind": "parquet_field",
                            "container": dict(container),
                            "row_index": row_index,
                            "field": "image",
                            "sha256": _sha256_bytes(image),
                        },
                        {
                            "role": "image_50dpi",
                            "kind": "parquet_field",
                            "container": dict(container),
                            "row_index": row_index,
                            "field": "image_50dpi",
                            "sha256": _sha256_bytes(image_50),
                        },
                    ],
                    "metadata": {
                        "doc_id": doc_id,
                        "schema": schema,
                        "ground_truth": ground_truth,
                        "source_record": {
                            "kind": "parquet_row",
                            "container": dict(container),
                            "row_index": row_index,
                            "fields": ["doc_id", "ground_truth", "schema", "split"],
                            "sha256": _canonical_hash(record),
                        },
                    },
                }
            )
        _verify_unchanged(path, container_sha, snapshot, f"VAREX {relative_path}")
    return candidates


def _ai2d_candidates(
    root: Path, pins: SourcePins, limits: SafetyLimits
) -> list[dict[str, Any]]:
    images_path = root / _AI2D_IMAGES_PATH
    annotations_path = root / _AI2D_ANNOTATIONS_PATH
    images_snapshot = _verify_file(images_path, pins.ai2d_images_sha256, "AI2D images")
    annotations_snapshot = _verify_file(
        annotations_path, pins.ai2d_annotations_sha256, "AI2D annotations"
    )
    image_index = _zip_index(images_path, limits)
    annotation_index = _zip_index(annotations_path, limits)
    images = {
        match.group(1): member
        for member in image_index
        if (match := _AI2D_IMAGE_RE.fullmatch(member)) is not None
    }
    annotations = {
        match.group(1): member
        for member in annotation_index
        if (match := _AI2D_ANNOTATION_RE.fullmatch(member)) is not None
    }
    ids = sorted(set(images) & set(annotations))
    image_hashes = _zip_member_hashes(images_path, (images[id_] for id_ in ids), limits)
    annotation_hashes = _zip_member_hashes(
        annotations_path, (annotations[id_] for id_ in ids), limits
    )
    image_container = _container(_AI2D_IMAGES_URL, pins.ai2d_images_sha256, "zip")
    annotation_container = _container(
        _AI2D_ANNOTATIONS_URL, pins.ai2d_annotations_sha256, "zip"
    )
    candidates = [
        {
            "dataset": "ai2d_rst",
            "stratum": "diagram",
            "canonical_id": id_,
            "group_id": id_,
            "inputs": [
                {
                    "role": "image",
                    **_archive_reference(
                        image_container, images[id_], image_hashes[images[id_]]
                    ),
                }
            ],
            "metadata": {
                "diagram_id": id_,
                "annotation": _archive_reference(
                    annotation_container,
                    annotations[id_],
                    annotation_hashes[annotations[id_]],
                ),
            },
        }
        for id_ in ids
    ]
    _verify_unchanged(images_path, pins.ai2d_images_sha256, images_snapshot, "AI2D images")
    _verify_unchanged(
        annotations_path,
        pins.ai2d_annotations_sha256,
        annotations_snapshot,
        "AI2D annotations",
    )
    return candidates


def _load_image_index(path: Path, limits: SafetyLimits) -> dict[str, str]:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limits.max_image_index_bytes + 1)
        if len(payload) > limits.max_image_index_bytes:
            raise ValueError("MWS image index exceeds file size limit")
        value = _strict_json_loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MWS image index: {path}") from exc
    envelope = isinstance(value, dict)
    if envelope:
        if (
            set(value) != {"dataset", "revision", "images"}
            or value.get("dataset") != "MTSAIR/MWS-Vision-Bench"
            or value.get("revision") != _MWS_IMAGE_REVISION
            or not isinstance(value.get("images"), list)
        ):
            raise ValueError("MWS image index envelope does not match the pinned dataset revision")
        value = value["images"]
    if not isinstance(value, list):
        raise ValueError("MWS image index must be a JSON array or a pinned envelope")
    if len(value) > limits.max_image_index_entries:
        raise ValueError("MWS image index exceeds entry count limit")
    result: dict[str, str] = {}
    seen_hashes: set[str] = set()
    for index, item in enumerate(value):
        expected_fields = {"path", "sha256", "size_bytes"} if envelope else {"path", "sha256"}
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError(
                f"MWS image index item {index} must contain exactly {sorted(expected_fields)}"
            )
        image_path = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(image_path, str) or not image_path.startswith("images/"):
            raise ValueError(f"MWS image index item {index} has an invalid path")
        _safe_member(image_path, f"MWS image index item {index}")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"MWS image index item {index} has an invalid sha256")
        if envelope and (
            not isinstance(item["size_bytes"], int)
            or isinstance(item["size_bytes"], bool)
            or item["size_bytes"] < 1
        ):
            raise ValueError(f"MWS image index item {index} has an invalid size_bytes")
        if image_path in result or sha256 in seen_hashes:
            raise ValueError("MWS image index contains duplicate paths or hashes")
        result[image_path] = sha256
        seen_hashes.add(sha256)
    return result


def _mws_candidates(
    root: Path, image_index_path: Path, pins: SourcePins, limits: SafetyLimits
) -> list[dict[str, Any]]:
    metadata_path = root / _MWS_METADATA_PATH
    metadata_snapshot = _verify_file(metadata_path, pins.mws_metadata_sha256, "MWS metadata")
    images = _load_image_index(image_index_path, limits)
    container = _container(_MWS_METADATA_URL, pins.mws_metadata_sha256, "jsonl")
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with metadata_path.open("rb") as stream:
        row_index = 0
        while line := stream.readline(limits.max_jsonl_line_bytes + 1):
            if len(line) > limits.max_jsonl_line_bytes:
                raise ValueError(f"MWS metadata row {row_index} exceeds line size limit")
            if not line.strip():
                raise ValueError(f"MWS metadata row {row_index} is blank")
            try:
                row = _strict_json_loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"MWS metadata row {row_index} is invalid JSON") from exc
            if not isinstance(row, dict) or any(field not in row for field in _MWS_FIELDS):
                raise ValueError(f"MWS metadata row {row_index} lacks required fields")
            canonical_id = str(row["id"])
            image_path = row["file_name"]
            stratum = _MWS_TYPE_TO_STRATUM.get(row["type"])
            answers = row["answers"]
            if (
                not canonical_id
                or canonical_id in seen_ids
                or not isinstance(image_path, str)
                or image_path not in images
                or stratum is None
                or not isinstance(row["dataset_name"], str)
                or not row["dataset_name"]
                or not isinstance(row["question"], str)
                or not row["question"]
                or not isinstance(answers, list)
                or not answers
                or any(not isinstance(answer, str) or not answer for answer in answers)
            ):
                raise ValueError(f"MWS metadata row {row_index} is invalid or unindexed")
            seen_ids.add(canonical_id)
            record = {field: row[field] for field in _MWS_FIELDS}
            candidates.append(
                {
                    "dataset": "mws_vision_bench",
                    "stratum": stratum,
                    "canonical_id": canonical_id,
                    "group_id": image_path,
                    "inputs": [
                        {
                            "role": "image",
                            "kind": "direct",
                            "uri": _hf_uri(
                                "MTSAIR/MWS-Vision-Bench", _MWS_IMAGE_REVISION, image_path
                            ),
                            "sha256": images[image_path],
                        }
                    ],
                    "metadata": {
                        "image_path": image_path,
                        "task_type": stratum,
                        "dataset_name": row["dataset_name"],
                        "question": row["question"],
                        "answers": answers,
                        "source_record": {
                            "kind": "jsonl_row",
                            "container": dict(container),
                            "row_index": row_index,
                            "fields": list(_MWS_FIELDS),
                            "sha256": _canonical_hash(record),
                        },
                    },
                }
            )
            row_index += 1
    _verify_unchanged(
        metadata_path, pins.mws_metadata_sha256, metadata_snapshot, "MWS metadata"
    )
    return candidates


def _has_unique_group_capacity(
    items: Sequence[Mapping[str, Any]], quotas: Mapping[str, int]
) -> bool:
    slots = [stratum for stratum, quota in sorted(quotas.items()) for _ in range(quota)]
    group_to_slot: dict[str, int] = {}

    def augment(slot_index: int, seen: set[str]) -> bool:
        stratum = slots[slot_index]
        for item in items:
            if item["stratum"] != stratum:
                continue
            group = str(item["group_id"])
            if group in seen:
                continue
            seen.add(group)
            prior = group_to_slot.get(group)
            if prior is None or augment(prior, seen):
                group_to_slot[group] = slot_index
                return True
        return False

    return all(augment(index, set()) for index in range(len(slots)))


def _require_capacity(candidates: Sequence[Mapping[str, Any]]) -> None:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_dataset[str(candidate["dataset"])].append(candidate)
    for dataset, quotas in _QUOTAS.items():
        available = Counter(str(item["stratum"]) for item in by_dataset[dataset])
        shortages = {
            stratum: quota - available[stratum]
            for stratum, quota in quotas.items()
            if available[stratum] < quota
        }
        if shortages:
            raise ValueError(f"candidate catalog lacks quota capacity for {dataset}: {shortages}")
        if not _has_unique_group_capacity(by_dataset[dataset], quotas):
            raise ValueError(f"candidate catalog lacks unique-group quota capacity for {dataset}")


def build_candidates(
    source_root: Path,
    mws_image_index: Path,
    *,
    pins: SourcePins | None = None,
    limits: SafetyLimits | None = None,
    parquet_adapter: ParquetAdapter | None = None,
    require_quotas: bool = True,
) -> list[dict[str, Any]]:
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {root}")
    pins = SourcePins() if pins is None else pins
    limits = SafetyLimits() if limits is None else limits
    adapter = _PyArrowAdapter() if parquet_adapter is None else parquet_adapter
    candidates = [
        *_pubtables_candidates(root, pins, limits),
        *_varex_candidates(root, pins, adapter, limits),
        *_ai2d_candidates(root, pins, limits),
        *_mws_candidates(root, mws_image_index, pins, limits),
    ]
    candidates.sort(
        key=lambda item: (item["dataset"], item["stratum"], item["canonical_id"])
    )
    identities: set[tuple[str, str]] = set()
    for candidate in candidates:
        identity = (candidate["dataset"], candidate["canonical_id"])
        if identity in identities:
            raise ValueError(f"duplicate generated candidate: {'/'.join(identity)}")
        identities.add(identity)
    if require_quotas:
        _require_capacity(candidates)
    return candidates


def write_candidates(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            for candidate in candidates:
                stream.write(_canonical_json(candidate))
                stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mws-image-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("candidates.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = build_candidates(args.source_root, args.mws_image_index)
    write_candidates(args.output, candidates)
    print(f"wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
