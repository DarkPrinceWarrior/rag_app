"""Materialize and byte-verify a license-clear OCR acceptance manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path, PurePosixPath
from typing import IO, Any, Protocol, TypeGuard
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

_CHUNK_SIZE = 1024 * 1024
_SHA256_HEX_LENGTH = 64
_VERIFIED_MANIFEST = "manifest.verified.json"
_HASH_ALGORITHM = "sha256-nul-v1"
_CANONICAL_JSON_ALGORITHM = "sha256-json-sort-keys-ascii-v1"
_DEFAULT_MAX_CONTAINER_BYTES = 4 * 1024 * 1024 * 1024
_DEFAULT_MAX_MEMBER_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
_DEFAULT_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
_DEFAULT_MAX_ARCHIVE_MEMBERS = 100_000
_HF_CDN_HOSTS = {
    "cas-bridge.xethub.hf.co",
    "cdn-lfs-eu-1.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs.hf.co",
    "us.aws.cdn.hf.co",
}
_AI2D_IMAGE_ARCHIVE = (
    "https://ai2-public-datasets.s3.us-west-2.amazonaws.com/diagrams/ai2d-all.zip"
)
_AI2D_ANNOTATION_ARCHIVE = (
    "https://www.kielipankki.fi/download/AI2D-RST/v1.1/ai2d-rst-v1-1.zip"
)
_MWS_LEGACY_REVISION = "e204166bde25f7dcaaffb9313b855de67b516e5d"
_MWS_METADATA_URI = (
    "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/"
    f"{_MWS_LEGACY_REVISION}/metadata.jsonl"
)
_MWS_RECORD_FIELDS = ["answers", "dataset_name", "file_name", "id", "question", "type"]
_MWS_SOURCE_TYPES = {
    "document_parsing_ru": "document parsing ru",
    "full_page_ocr_ru": "full-page OCR ru",
    "key_information_extraction_ru": "key information extraction ru",
    "reasoning_vqa_ru": "reasoning VQA ru",
    "text_grounding_ru": "text grounding ru",
}
_VAREX_RECORD_FIELDS = ["doc_id", "ground_truth", "schema", "split"]
_PUBTABLES_IMAGES_URI = (
    "https://huggingface.co/datasets/kensho/PubTables-v2/resolve/"
    "aa575e798cb00a296925e2086addb3e3fd9a1903/"
    "PubTables-v2_Full-Documents_test_images.tar.gz"
)
_PUBTABLES_TABLES_URI = (
    "https://huggingface.co/datasets/kensho/PubTables-v2/resolve/"
    "aa575e798cb00a296925e2086addb3e3fd9a1903/"
    "PubTables-v2_Full-Documents_test_tables.tar.gz"
)
_PINNED_CONTAINER_SHA256: dict[str, str] = {
    _PUBTABLES_IMAGES_URI: "0d42821fb1dce5713a86c327bec5fabbe214bb5ebbd0cfc75cd2ef89b7c7230e",
    _PUBTABLES_TABLES_URI: "dfd10e0dc4cb3e92d0f521e8a135e6e96094d8e90e130fb3fe25c9fa31b3a3de",
    (
        "https://huggingface.co/datasets/ibm-research/VAREX/resolve/"
        "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/data/benchmark-00000-of-00004.parquet"
    ): "f0328edd6242318f97eb85fdd63466ec6a9db1482b7edd3ddb92de2bc535e147",
    (
        "https://huggingface.co/datasets/ibm-research/VAREX/resolve/"
        "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/data/benchmark-00001-of-00004.parquet"
    ): "6eee390d60212571269d1002fe335d635f149e5850a0ed05dd1b3dc77a8a5d07",
    (
        "https://huggingface.co/datasets/ibm-research/VAREX/resolve/"
        "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/data/benchmark-00002-of-00004.parquet"
    ): "03c5411bb91eaaacedd77ee280e4e92359fa18cb26f5bf9a0cfab53481c4c194",
    (
        "https://huggingface.co/datasets/ibm-research/VAREX/resolve/"
        "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6/data/benchmark-00003-of-00004.parquet"
    ): "b2304eb66183063d18da6d93bf6ce73283fa5dfa5b39611082f1b85a5f2874c7",
    _AI2D_ANNOTATION_ARCHIVE: (
        "eb11d67507e08eb9bfd0f5944da7ca32cfcffa13e119b04ac5054effa65a759a"
    ),
    _AI2D_IMAGE_ARCHIVE: "1a6b77eebb8b7dbdf76a0ba6ca76c2f97ce8f81d8ee33b06593aa722e54c4786",
    _MWS_METADATA_URI: "c234a569583858bfab13399169ec9951da12edf6d88a5cd4c4efae8a1fd4197d",
}
_PINNED_DATASETS: dict[str, dict[str, Any]] = {
    "pubtables_v2": {
        "source": "kensho/PubTables-v2",
        "source_url": "https://huggingface.co/datasets/kensho/PubTables-v2",
        "revision": "aa575e798cb00a296925e2086addb3e3fd9a1903",
        "license": "CDLA-Permissive-2.0",
    },
    "varex": {
        "source": "ibm-research/VAREX",
        "source_url": "https://huggingface.co/datasets/ibm-research/VAREX",
        "revision": "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6",
        "license": "CDLA-Permissive-2.0",
    },
    "ai2d_rst": {
        "source": "AllenAI/AI2D + AI2D-RST v1.1",
        "source_url": _AI2D_IMAGE_ARCHIVE,
        "revision": "content-addressed-source-components",
        "license": "CC-BY-4.0 AND CC-BY-SA-4.0",
        "license_components": {
            "annotations": "CC-BY-4.0",
            "source_images": "CC-BY-SA-4.0",
        },
        "source_components": {
            "annotations": {
                "source": "Kielipankki/AI2D-RST-v1.1",
                "source_url": _AI2D_ANNOTATION_ARCHIVE,
                "revision": "eb11d67507e08eb9bfd0f5944da7ca32cfcffa13e119b04ac5054effa65a759a",
            },
            "source_images": {
                "source": "AllenAI/AI2D",
                "source_url": _AI2D_IMAGE_ARCHIVE,
                "revision": "1a6b77eebb8b7dbdf76a0ba6ca76c2f97ce8f81d8ee33b06593aa722e54c4786",
            },
        },
    },
    "mws_vision_bench": {
        "source": "MTSAIR/MWS-Vision-Bench",
        "source_url": "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench",
        "revision": "b8d473734b79343cac2b74f692a29ab191c7d11d",
        "license": "MIT AND CC-BY-4.0",
        "license_components": {"benchmark_code": "MIT", "source_assets": "CC-BY-4.0"},
    },
}


class HttpClient(Protocol):
    def open(
        self,
        uri: str,
        *,
        timeout_s: float,
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[IO[bytes]]: ...


class ParquetAdapter(Protocol):
    def read_field(self, container: Path, *, row_index: int, field: str) -> bytes: ...

    def read_row(
        self,
        container: Path,
        *,
        row_index: int,
        fields: Sequence[str],
    ) -> Mapping[str, Any]: ...


class _CheckedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], None]) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        target = urljoin(req.full_url, newurl)
        self._validator(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


class UrllibHttpClient:
    @contextmanager
    def open(
        self,
        uri: str,
        *,
        timeout_s: float,
        redirect_validator: Callable[[str], None],
    ) -> Iterator[IO[bytes]]:
        opener = build_opener(_CheckedRedirectHandler(redirect_validator))
        request = Request(uri, headers={"User-Agent": "DocRAGenslate-materializer/2.0"})
        response = opener.open(request, timeout=timeout_s)
        try:
            final_url = getattr(response, "geturl", lambda: uri)()
            redirect_validator(final_url)
            yield response
        finally:
            response.close()


class PyArrowParquetAdapter:
    @staticmethod
    def _parquet_module() -> Any:
        try:
            return importlib.import_module("pyarrow.parquet")
        except ModuleNotFoundError as exc:
            if exc.name == "pyarrow" or str(exc.name).startswith("pyarrow."):
                raise RuntimeError(
                    "parquet reference requires optional dependency 'pyarrow'; "
                    "install pyarrow or inject a ParquetAdapter"
                ) from exc
            raise

    def _read_values(
        self,
        container: Path,
        *,
        row_index: int,
        fields: Sequence[str],
    ) -> dict[str, Any]:
        parquet = self._parquet_module()
        parquet_file = parquet.ParquetFile(container)
        offset = 0
        for batch in parquet_file.iter_batches(batch_size=1024, columns=list(fields)):
            if row_index < offset + batch.num_rows:
                local_index = row_index - offset
                return {
                    field: batch.column(position)[local_index].as_py()
                    for position, field in enumerate(fields)
                }
            offset += batch.num_rows
        raise IndexError(f"parquet row_index out of range: {row_index}")

    def read_field(self, container: Path, *, row_index: int, field: str) -> bytes:
        value = self._read_values(container, row_index=row_index, fields=(field,))[field]
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, Mapping) and isinstance(value.get("bytes"), bytes):
            return value["bytes"]
        return _canonical_json_bytes(value)

    def read_row(
        self,
        container: Path,
        *,
        row_index: int,
        fields: Sequence[str],
    ) -> Mapping[str, Any]:
        return self._read_values(container, row_index=row_index, fields=fields)


@dataclass(frozen=True)
class MaterializedObject:
    relative_path: str
    absolute_path: Path
    size_bytes: int
    sha256: str


@dataclass
class MaterializationState:
    stage_dir: Path
    container_source_dir: Path | None
    http_client: HttpClient
    parquet_adapter: ParquetAdapter
    max_container_bytes: int
    max_member_bytes: int
    max_total_bytes: int
    max_archive_uncompressed_bytes: int
    max_archive_members: int
    timeout_s: float
    total_bytes: int = 0

    def add_bytes(self, size: int) -> None:
        self.total_bytes += size
        if self.total_bytes > self.max_total_bytes:
            raise ValueError("materialization exceeds max_total_bytes")

    def remaining_limit(self, per_object_limit: int) -> int:
        remaining = self.max_total_bytes - self.total_bytes
        if remaining < 1:
            raise ValueError("materialization exceeds max_total_bytes")
        return min(per_object_limit, remaining)


def _is_sha256(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _safe_pinned_path(uri: str, prefix: str, context: str) -> None:
    parsed = urlparse(uri)
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError(f"{context}: query, fragment and params are forbidden")
    raw_path = parsed.path.lower()
    if "%2f" in raw_path or "%2e" in raw_path:
        raise ValueError(f"{context}: encoded path separators are forbidden")
    path = unquote(parsed.path)
    if not path.startswith(prefix) or path == prefix:
        raise ValueError(f"{context}: URI does not contain the exact pinned revision path")
    if any(segment in {"", ".", ".."} for segment in path.split("/")[1:]):
        raise ValueError(f"{context}: unsafe path segment")


def _validate_source_uri(dataset: str, uri: str) -> None:
    spec = _PINNED_DATASETS[dataset]
    parsed = urlparse(uri)
    context = f"{dataset} source URI"
    if dataset == "ai2d_rst":
        if uri not in {_AI2D_IMAGE_ARCHIVE, _AI2D_ANNOTATION_ARCHIVE}:
            raise ValueError(f"{context}: URI is not an exact pinned AI2D container")
        return
    if dataset == "mws_vision_bench" and uri == _MWS_METADATA_URI:
        return
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ValueError(f"{context}: only trusted HTTPS Hugging Face storage is allowed")
    prefix = f"/datasets/{spec['source']}/resolve/{spec['revision']}/"
    _safe_pinned_path(uri, prefix, context)


def _redirect_validator(source_uri: str, dataset: str) -> Callable[[str], None]:
    source = urlparse(source_uri)

    def validate(target_uri: str) -> None:
        target = urlparse(target_uri)
        if target.scheme != "https":
            raise ValueError("HTTP redirect must keep HTTPS")
        if target.netloc == source.netloc:
            if source_uri == _MWS_METADATA_URI:
                expected_path = (
                    "/api/resolve-cache/datasets/MTSAIR/MWS-Vision-Bench/"
                    f"{_MWS_LEGACY_REVISION}/metadata.jsonl"
                )
                if (
                    target.path == expected_path
                    and target.username is None
                    and target.port is None
                    and not target.params
                    and not target.fragment
                ):
                    return
            _validate_source_uri(dataset, target_uri)
            return
        if source.netloc == "huggingface.co" and target.netloc in _HF_CDN_HOSTS:
            if not target.path or target.username is not None or target.port is not None:
                raise ValueError("Hugging Face CDN redirect target is malformed")
            return
        raise ValueError(f"redirect target is not allowlisted: {target.netloc}")

    return validate


def _validate_dataset_metadata(datasets: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("manifest.datasets must be a non-empty object")
    validated: dict[str, dict[str, Any]] = {}
    for dataset, metadata in datasets.items():
        if dataset not in _PINNED_DATASETS or not isinstance(metadata, dict):
            raise ValueError(f"unsupported dataset metadata: {dataset!r}")
        pinned = _PINNED_DATASETS[dataset]
        for field in ("source", "source_url", "revision", "license"):
            if metadata.get(field) != pinned[field]:
                raise ValueError(f"{dataset}: {field} does not match the pinned source")
        for field in ("license_components", "source_components"):
            if metadata.get(field) != pinned.get(field):
                raise ValueError(f"{dataset}: {field} does not match the pinned source")
        validated[dataset] = metadata
    return validated


def _expected_roles(dataset: str, stratum: Any) -> set[str]:
    if dataset == "pubtables_v2" and stratum in {"pages_2", "pages_3", "pages_4"}:
        count = int(stratum.removeprefix("pages_"))
        return {f"page_{index}" for index in range(1, count + 1)}
    if dataset == "varex" and stratum in {"Flat", "Nested", "Table"}:
        return {"image_200dpi", "image_50dpi"}
    if dataset == "ai2d_rst" and stratum == "diagram":
        return {"image"}
    if dataset == "mws_vision_bench" and stratum in {
        "document_parsing_ru",
        "full_page_ocr_ru",
        "key_information_extraction_ru",
        "reasoning_vqa_ru",
        "text_grounding_ru",
    }:
        return {"image"}
    raise ValueError(f"{dataset}: unsupported stratum {stratum!r}")


def _safe_member_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError(f"{context}: archive member must be a safe POSIX path")
    member = PurePosixPath(value)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"{context}: archive member path traversal is forbidden")
    return member.as_posix()


def _reference_sha(reference: Mapping[str, Any], context: str) -> str:
    sha256 = reference.get("sha256")
    if not _is_sha256(sha256):
        raise ValueError(f"{context}.sha256 must be lowercase hexadecimal SHA-256")
    return sha256


def _validate_container(container: Any, dataset: str, expected_format: str, context: str) -> None:
    if not isinstance(container, Mapping):
        raise ValueError(f"{context}.container must be an object")
    if set(container) != {"uri", "sha256", "format"}:
        raise ValueError(f"{context}.container requires uri, sha256 and format only")
    uri = container.get("uri")
    if not isinstance(uri, str):
        raise ValueError(f"{context}.container.uri must be a string")
    _reference_sha(container, f"{context}.container")
    if container.get("format") != expected_format:
        raise ValueError(f"{context}.container.format must be {expected_format!r}")
    _validate_source_uri(dataset, uri)
    pinned_sha256 = _PINNED_CONTAINER_SHA256.get(uri)
    if dataset in {"pubtables_v2", "varex", "mws_vision_bench"} and pinned_sha256 is None:
        raise ValueError(f"{context}.container.uri is not an exact pinned container")
    if pinned_sha256 is not None and container.get("sha256") != pinned_sha256:
        raise ValueError(f"{context}.container.sha256 does not match the pinned container")


def _validate_reference(reference: Any, dataset: str, context: str) -> None:
    if not isinstance(reference, Mapping):
        raise ValueError(f"{context} must be an object")
    kind = reference.get("kind")
    _reference_sha(reference, context)
    if kind == "direct":
        allowed = {"kind", "uri", "sha256", "role"}
        if set(reference) - allowed or not isinstance(reference.get("uri"), str):
            raise ValueError(f"{context}: invalid direct reference")
        if dataset != "mws_vision_bench":
            raise ValueError(f"{context}: direct references are allowed only for MWS images")
        _validate_source_uri(dataset, reference["uri"])
        return
    if kind == "archive_member":
        allowed = {"kind", "container", "member", "sha256", "role"}
        if set(reference) - allowed:
            raise ValueError(f"{context}: invalid archive_member reference")
        container = reference.get("container")
        archive_format = container.get("format") if isinstance(container, Mapping) else None
        if archive_format not in {"tar.gz", "zip"}:
            raise ValueError(f"{context}.container.format must be 'tar.gz' or 'zip'")
        _validate_container(container, dataset, archive_format, context)
        if dataset not in {"pubtables_v2", "ai2d_rst"}:
            raise ValueError(f"{context}: archive members are unsupported for {dataset}")
        _safe_member_name(reference.get("member"), context)
        return
    if kind == "parquet_field":
        allowed = {"kind", "container", "row_index", "field", "sha256", "role"}
        if set(reference) - allowed:
            raise ValueError(f"{context}: invalid parquet_field reference")
        _validate_container(reference.get("container"), dataset, "parquet", context)
        if dataset != "varex":
            raise ValueError(f"{context}: parquet fields are allowed only for VAREX")
        _validate_row_index(reference.get("row_index"), context)
        if not isinstance(reference.get("field"), str) or not reference["field"]:
            raise ValueError(f"{context}.field must be a non-empty string")
        return
    if kind in {"parquet_row", "jsonl_row"}:
        allowed = {"kind", "container", "row_index", "fields", "sha256", "role"}
        if set(reference) - allowed:
            raise ValueError(f"{context}: invalid {kind} reference")
        expected_format = "parquet" if kind == "parquet_row" else "jsonl"
        _validate_container(reference.get("container"), dataset, expected_format, context)
        if kind == "parquet_row" and dataset != "varex":
            raise ValueError(f"{context}: parquet rows are allowed only for VAREX")
        if kind == "jsonl_row" and dataset != "mws_vision_bench":
            raise ValueError(f"{context}: JSONL rows are allowed only for MWS")
        _validate_row_index(reference.get("row_index"), context)
        fields = reference.get("fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or fields != sorted(set(fields))
        ):
            raise ValueError(f"{context}.fields must be sorted unique non-empty strings")
        return
    raise ValueError(f"{context}.kind is unsupported: {kind!r}")


def _validate_row_index(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context}.row_index must be a non-negative integer")
    return value


def _validate_source_bindings(candidate: Mapping[str, Any], index: int) -> None:
    dataset = candidate["dataset"]
    metadata = candidate["metadata"]
    inputs = {item["role"]: item for item in candidate["inputs"]}
    context = f"selected[{index}]"
    if dataset == "pubtables_v2":
        document_id = metadata["document_id"]
        page_indices = metadata.get("page_indices")
        expected_count = len(_expected_roles(dataset, candidate.get("stratum")))
        if (
            not isinstance(page_indices, list)
            or len(page_indices) != expected_count
            or any(
                not isinstance(page_index, int)
                or isinstance(page_index, bool)
                or page_index < 0
                for page_index in page_indices
            )
            or page_indices != sorted(set(page_indices))
        ):
            raise ValueError(
                f"{context}: PubTables page_indices must match the stratum page count"
            )
        for ordinal, page_index in enumerate(page_indices, start=1):
            item = inputs.get(f"page_{ordinal}")
            expected_member = f"Full Documents/test/images/{document_id}_page_{page_index}.jpg"
            if (
                not isinstance(item, Mapping)
                or item.get("kind") != "archive_member"
                or item.get("container", {}).get("uri") != _PUBTABLES_IMAGES_URI
                or item.get("member") != expected_member
            ):
                raise ValueError(f"{context}: PubTables page reference does not match metadata")
        annotation = metadata["annotation"]
        expected_member = f"Full Documents/test/tables/{document_id}_tables.json"
        if (
            annotation.get("kind") != "archive_member"
            or annotation.get("container", {}).get("uri") != _PUBTABLES_TABLES_URI
            or annotation.get("member") != expected_member
        ):
            raise ValueError(f"{context}: PubTables annotation reference is not pinned")
        return
    if dataset == "varex":
        first = inputs["image_200dpi"]
        second = inputs["image_50dpi"]
        if first.get("kind") != "parquet_field" or first.get("field") != "image":
            raise ValueError(f"{context}: VAREX 200dpi input must bind parquet field 'image'")
        if second.get("kind") != "parquet_field" or second.get("field") != "image_50dpi":
            raise ValueError(f"{context}: VAREX 50dpi input must bind parquet field 'image_50dpi'")
        if (
            first.get("container") != second.get("container")
            or first.get("row_index") != second.get("row_index")
        ):
            raise ValueError(f"{context}: VAREX inputs must bind the same parquet row")
        record = metadata["source_record"]
        if (
            record.get("kind") != "parquet_row"
            or record.get("container") != first.get("container")
            or record.get("row_index") != first.get("row_index")
            or record.get("fields") != _VAREX_RECORD_FIELDS
        ):
            raise ValueError(f"{context}: VAREX source_record must bind the selected parquet row")
        expected_record = {
            "doc_id": metadata.get("doc_id"),
            "ground_truth": metadata.get("ground_truth"),
            "schema": metadata.get("schema"),
            "split": candidate.get("stratum"),
        }
        if record.get("sha256") != _canonical_json_hash(expected_record):
            raise ValueError(f"{context}: VAREX source_record canonical hash mismatch")
        return
    if dataset == "ai2d_rst":
        diagram_id = metadata["diagram_id"]
        image = inputs["image"]
        annotation = metadata["annotation"]
        if (
            image.get("kind") != "archive_member"
            or image.get("container", {}).get("uri") != _AI2D_IMAGE_ARCHIVE
            or image.get("member") != f"ai2d/images/{diagram_id}.png"
        ):
            raise ValueError(f"{context}: AI2D image reference is not pinned")
        if (
            annotation.get("kind") != "archive_member"
            or annotation.get("container", {}).get("uri") != _AI2D_ANNOTATION_ARCHIVE
            or annotation.get("member")
            != f"ai2d-rst-v1-1/json/ai2d-rst/{diagram_id}.png.json"
        ):
            raise ValueError(f"{context}: AI2D annotation reference is not pinned")
        return
    image = inputs["image"]
    expected_image_uri = (
        "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/"
        f"{_PINNED_DATASETS[dataset]['revision']}/{metadata['image_path']}"
    )
    if image.get("kind") != "direct" or image.get("uri") != expected_image_uri:
        raise ValueError(f"{context}: MWS image URI does not match metadata.image_path")
    record = metadata["source_record"]
    if (
        record.get("kind") != "jsonl_row"
        or record.get("container", {}).get("uri") != _MWS_METADATA_URI
        or record.get("fields") != _MWS_RECORD_FIELDS
    ):
        raise ValueError(f"{context}: MWS source_record is not pinned")
    expected_record = {
        "answers": metadata.get("answers"),
        "dataset_name": metadata.get("dataset_name"),
        "file_name": metadata.get("image_path"),
        "id": candidate.get("canonical_id"),
        "question": metadata.get("question"),
        "type": _MWS_SOURCE_TYPES[candidate["stratum"]],
    }
    if record.get("sha256") != _canonical_json_hash(expected_record):
        raise ValueError(f"{context}: MWS source_record canonical hash mismatch")


def _validate_source_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be a JSON object")
    if manifest.get("manifest_version") != 1:
        raise ValueError("source manifest_version must be 1")
    if manifest.get("verification_state") != "metadata_only_unverified":
        raise ValueError("source manifest verification_state must be metadata_only_unverified")
    selection = manifest.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("algorithm") != _HASH_ALGORITHM
        or not isinstance(selection.get("seed"), str)
        or not selection["seed"]
        or not _is_sha256(selection.get("catalog_sha256"))
        or selection.get("canonical_json_algorithm") != _CANONICAL_JSON_ALGORITHM
        or not str(selection.get("group_policy", "")).startswith("claimed_identity_only")
    ):
        raise ValueError("source manifest selection provenance policy is invalid")
    requirements = manifest.get("materialization_requirements")
    required_flags = {
        "verify_every_referenced_object_bytes_against_sha256",
        "verify_group_id_against_trusted_physical_identity",
        "reject_manifest_on_any_mismatch",
    }
    if not isinstance(requirements, dict) or any(
        requirements.get(flag) is not True for flag in required_flags
    ):
        raise ValueError("source manifest lacks mandatory materialization requirements")
    datasets = _validate_dataset_metadata(manifest.get("datasets"))
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("source manifest selected must be a non-empty list")
    claimed_groups: set[tuple[str, str]] = set()
    for index, candidate in enumerate(selected):
        if not isinstance(candidate, dict) or candidate.get("dataset") not in datasets:
            raise ValueError(f"selected[{index}] has invalid dataset metadata")
        dataset = candidate["dataset"]
        canonical_id = candidate.get("canonical_id")
        stratum = candidate.get("stratum")
        if not isinstance(canonical_id, str) or not canonical_id:
            raise ValueError(f"selected[{index}].canonical_id is required")
        if not isinstance(stratum, str):
            raise ValueError(f"selected[{index}].stratum is required")
        expected_selection_hash = hashlib.sha256(
            "\0".join((selection["seed"], dataset, stratum, canonical_id)).encode()
        ).hexdigest()
        if candidate.get("selection_hash") != expected_selection_hash:
            raise ValueError(f"selected[{index}].selection_hash does not match provenance")
        group_id = candidate.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"selected[{index}].group_id is required")
        claimed_group = (dataset, group_id)
        if claimed_group in claimed_groups:
            raise ValueError(f"selected[{index}] repeats claimed group_id")
        claimed_groups.add(claimed_group)
        inputs = candidate.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"selected[{index}].inputs must be a non-empty list")
        roles = [item.get("role") for item in inputs if isinstance(item, Mapping)]
        expected_roles = _expected_roles(dataset, candidate.get("stratum"))
        if len(roles) != len(inputs) or set(roles) != expected_roles or len(roles) != len(set(roles)):
            raise ValueError(f"selected[{index}].inputs roles must be exactly {sorted(expected_roles)}")
        for input_index, reference in enumerate(inputs):
            _validate_reference(reference, dataset, f"selected[{index}].inputs[{input_index}]")
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"selected[{index}].metadata must be an object")
        physical_id_key = {
            "pubtables_v2": "document_id",
            "varex": "doc_id",
            "ai2d_rst": "diagram_id",
            "mws_vision_bench": "image_path",
        }[dataset]
        if metadata.get(physical_id_key) != group_id:
            raise ValueError(f"selected[{index}].group_id does not match metadata.{physical_id_key}")
        reference_key = {
            "pubtables_v2": "annotation",
            "varex": "source_record",
            "ai2d_rst": "annotation",
            "mws_vision_bench": "source_record",
        }[dataset]
        unexpected_key = "source_record" if reference_key == "annotation" else "annotation"
        if unexpected_key in metadata:
            raise ValueError(f"selected[{index}].metadata contains unexpected {unexpected_key}")
        _validate_reference(
            metadata.get(reference_key),
            dataset,
            f"selected[{index}].metadata.{reference_key}",
        )
        if dataset == "varex":
            for value_key, hash_key in (
                ("schema", "schema_canonical_sha256"),
                ("ground_truth", "ground_truth_canonical_sha256"),
            ):
                if metadata.get(hash_key) != _canonical_json_hash(metadata.get(value_key)):
                    raise ValueError(f"selected[{index}].metadata.{hash_key} does not match content")
        if dataset == "mws_vision_bench":
            record = {
                "id": candidate.get("canonical_id"),
                "file_name": metadata.get("image_path"),
                "dataset_name": metadata.get("dataset_name"),
                "question": metadata.get("question"),
                "answers": metadata.get("answers"),
                "type": _MWS_SOURCE_TYPES[candidate["stratum"]],
            }
            if metadata.get("source_record_canonical_sha256") != _canonical_json_hash(record):
                raise ValueError(
                    f"selected[{index}].metadata.source_record_canonical_sha256 does not match content"
                )
        _validate_source_bindings(candidate, index)
    actual_summary_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset, metadata in datasets.items():
        quotas = metadata.get("quotas")
        if (
            not isinstance(quotas, dict)
            or not quotas
            or any(
                not isinstance(stratum, str)
                or not isinstance(quota, int)
                or isinstance(quota, bool)
                or quota < 1
                for stratum, quota in quotas.items()
            )
        ):
            raise ValueError(f"{dataset}: quotas must be positive integer counts")
        items = [candidate for candidate in selected if candidate["dataset"] == dataset]
        actual_counts = {
            stratum: sum(candidate["stratum"] == stratum for candidate in items)
            for stratum in sorted(quotas)
        }
        if actual_counts != dict(sorted(quotas.items())) or any(
            candidate["stratum"] not in quotas for candidate in items
        ):
            raise ValueError(f"{dataset}: selected strata do not satisfy declared quotas")
        actual_summary_by_dataset[dataset] = {
            "selected_units": len(items),
            "input_count": sum(len(candidate["inputs"]) for candidate in items),
            "by_stratum": actual_counts,
        }
    summary = manifest.get("summary")
    actual_summary = {
        "selected_units": len(selected),
        "input_count": sum(len(candidate["inputs"]) for candidate in selected),
        "by_dataset": actual_summary_by_dataset,
    }
    if (
        not isinstance(summary, dict)
        or not isinstance(summary.get("candidate_count"), int)
        or isinstance(summary.get("candidate_count"), bool)
        or summary["candidate_count"] < len(selected)
        or any(summary.get(key) != value for key, value in actual_summary.items())
    ):
        raise ValueError("source manifest summary does not match selected entries")
    return manifest


def _stream_to_atomic(
    stream: IO[bytes],
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
    deadline: float,
) -> MaterializedObject:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with part.open("xb") as output:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("object materialization deadline exceeded")
                chunk = stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"object exceeds max bytes: {max_bytes}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
        if destination.exists():
            part.unlink()
        else:
            os.replace(part, destination)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return MaterializedObject("", destination, size, expected_sha256)


def _object_destination(stage_dir: Path, dataset: str, sha256: str) -> tuple[Path, str]:
    relative = Path("objects") / dataset / sha256[:2] / sha256
    return stage_dir / relative, relative.as_posix()


def _container_destination(stage_dir: Path, sha256: str) -> tuple[Path, str]:
    relative = Path("containers") / sha256[:2] / sha256
    return stage_dir / relative, relative.as_posix()


def _download_http(
    uri: str,
    destination: Path,
    *,
    dataset: str,
    expected_sha256: str,
    max_bytes: int,
    state: MaterializationState,
) -> MaterializedObject:
    validator = _redirect_validator(uri, dataset)
    with state.http_client.open(
        uri,
        timeout_s=state.timeout_s,
        redirect_validator=validator,
    ) as stream:
        return _stream_to_atomic(
            stream,
            destination,
            expected_sha256=expected_sha256,
            max_bytes=max_bytes,
            deadline=time.monotonic() + state.timeout_s,
        )


def _ensure_container(
    container: Mapping[str, Any],
    *,
    dataset: str,
    state: MaterializationState,
    cache: dict[str, MaterializedObject],
    digest_datasets: dict[str, str],
) -> MaterializedObject:
    sha256 = _reference_sha(container, "container")
    uri = container["uri"]
    _validate_source_uri(dataset, uri)
    previous_dataset = digest_datasets.setdefault(sha256, dataset)
    if previous_dataset != dataset:
        raise ValueError(f"verified digest occurs across datasets: {previous_dataset} and {dataset}")
    cached = cache.get(sha256)
    if cached is not None:
        return cached
    destination, relative = _container_destination(state.stage_dir, sha256)
    local_stream: IO[bytes] | None = None
    if state.container_source_dir is not None:
        directory_fd = os.open(
            state.container_source_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            try:
                source_fd = os.open(
                    sha256,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                source_fd = None
            except OSError as exc:
                raise ValueError(
                    f"local container cache entry cannot be opened safely: {sha256}"
                ) from exc
        finally:
            os.close(directory_fd)
        if source_fd is not None:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                os.close(source_fd)
                raise ValueError(f"local container cache entry is not a regular file: {sha256}")
            local_stream = os.fdopen(source_fd, "rb")
    if local_stream is not None:
        with local_stream as stream:
            materialized = _stream_to_atomic(
                stream,
                destination,
                expected_sha256=sha256,
                max_bytes=state.remaining_limit(state.max_container_bytes),
                deadline=time.monotonic() + state.timeout_s,
            )
    else:
        materialized = _download_http(
            uri,
            destination,
            dataset=dataset,
            expected_sha256=sha256,
            max_bytes=state.remaining_limit(state.max_container_bytes),
            state=state,
        )
    materialized = MaterializedObject(relative, destination, materialized.size_bytes, sha256)
    cache[sha256] = materialized
    state.add_bytes(materialized.size_bytes)
    return materialized


def _register_derived_digest(
    sha256: str,
    dataset: str,
    digest_datasets: dict[str, str],
) -> None:
    previous_dataset = digest_datasets.setdefault(sha256, dataset)
    if previous_dataset != dataset:
        raise ValueError(f"verified digest occurs across datasets: {previous_dataset} and {dataset}")


def _materialize_bytes(
    payload: bytes,
    *,
    dataset: str,
    expected_sha256: str,
    state: MaterializationState,
) -> MaterializedObject:
    if len(payload) > state.remaining_limit(state.max_member_bytes):
        raise ValueError(f"derived object exceeds max member bytes: {state.max_member_bytes}")
    destination, relative = _object_destination(state.stage_dir, dataset, expected_sha256)
    result = _stream_to_atomic(
        io.BytesIO(payload),
        destination,
        expected_sha256=expected_sha256,
        max_bytes=state.remaining_limit(state.max_member_bytes),
        deadline=time.monotonic() + state.timeout_s,
    )
    state.add_bytes(result.size_bytes)
    return MaterializedObject(relative, destination, result.size_bytes, expected_sha256)


def _extract_tar_member(
    container: Path,
    member_name: str,
    *,
    dataset: str,
    expected_sha256: str,
    state: MaterializationState,
) -> MaterializedObject:
    destination, relative = _object_destination(state.stage_dir, dataset, expected_sha256)
    found = 0
    declared_total = 0
    member_count = 0
    result: MaterializedObject | None = None
    with container.open("rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as archive:
        for info in archive:
            member_count += 1
            if member_count > state.max_archive_members:
                raise ValueError("tar archive exceeds member count cap")
            safe_name = _safe_member_name(info.name.rstrip("/"), "tar member")
            if info.issym() or info.islnk() or not (info.isfile() or info.isdir()):
                raise ValueError(f"tar archive contains forbidden member type: {safe_name}")
            if info.isfile():
                if info.size < 0:
                    raise ValueError("tar member has negative size")
                declared_total += info.size
                if declared_total > state.max_archive_uncompressed_bytes:
                    raise ValueError("tar archive exceeds total uncompressed bytes cap")
            if safe_name != member_name:
                continue
            if not info.isfile():
                raise ValueError("requested tar member is not a regular file")
            found += 1
            if found > 1:
                raise ValueError("tar archive contains duplicate requested member")
            if info.size > state.remaining_limit(state.max_member_bytes):
                raise ValueError("tar member exceeds max member bytes")
            stream = archive.extractfile(info)
            if stream is None:
                raise ValueError("tar member cannot be read")
            with stream:
                extracted = _stream_to_atomic(
                    stream,
                    destination,
                    expected_sha256=expected_sha256,
                    max_bytes=state.remaining_limit(state.max_member_bytes),
                    deadline=time.monotonic() + state.timeout_s,
                )
            result = MaterializedObject(relative, destination, extracted.size_bytes, expected_sha256)
    if found != 1 or result is None:
        raise ValueError(f"tar archive member not found: {member_name}")
    state.add_bytes(result.size_bytes)
    return result


def _zip_member_is_safe(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    if info.flag_bits & 0x1:
        return False
    file_type = stat.S_IFMT(mode)
    if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
        return False
    return True


def _extract_zip_member(
    container: Path,
    member_name: str,
    *,
    dataset: str,
    expected_sha256: str,
    state: MaterializationState,
) -> MaterializedObject:
    destination, relative = _object_destination(state.stage_dir, dataset, expected_sha256)
    with zipfile.ZipFile(container) as archive:
        infos = archive.infolist()
        if len(infos) > state.max_archive_members:
            raise ValueError("zip archive exceeds member count cap")
        total = 0
        names: set[str] = set()
        target: zipfile.ZipInfo | None = None
        for info in infos:
            safe_name = _safe_member_name(info.filename.rstrip("/"), "zip member")
            if safe_name in names:
                raise ValueError(f"zip archive contains duplicate member: {safe_name}")
            names.add(safe_name)
            if not _zip_member_is_safe(info):
                raise ValueError(f"zip archive contains link, special, or encrypted member: {safe_name}")
            total += info.file_size
            if total > state.max_archive_uncompressed_bytes:
                raise ValueError("zip archive exceeds total uncompressed bytes cap")
            if safe_name == member_name:
                target = info
        if target is None or target.is_dir():
            raise ValueError(f"zip archive member not found or not a file: {member_name}")
        if target.file_size > state.remaining_limit(state.max_member_bytes):
            raise ValueError("zip member exceeds max member bytes")
        with archive.open(target) as stream:
            extracted = _stream_to_atomic(
                stream,
                destination,
                expected_sha256=expected_sha256,
                max_bytes=state.remaining_limit(state.max_member_bytes),
                deadline=time.monotonic() + state.timeout_s,
            )
    result = MaterializedObject(relative, destination, extracted.size_bytes, expected_sha256)
    state.add_bytes(result.size_bytes)
    return result


def _read_jsonl_row(
    container: Path,
    *,
    row_index: int,
    fields: Sequence[str],
    max_row_bytes: int,
) -> bytes:
    with container.open("rb") as stream:
        for _index in range(row_index + 1):
            line = stream.readline(max_row_bytes + 1)
            if not line:
                raise IndexError(f"jsonl row_index out of range: {row_index}")
            if len(line) > max_row_bytes:
                raise ValueError("jsonl row exceeds max member bytes")
    try:
        row = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"jsonl row {row_index} is not valid UTF-8 JSON") from exc
    if not isinstance(row, Mapping):
        raise ValueError(f"jsonl row {row_index} must be an object")
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"jsonl row {row_index} misses fields: {missing}")
    return _canonical_json_bytes({field: row[field] for field in fields})


def _normalize_parquet_row(
    dataset: str,
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    normalized = {field: row[field] for field in fields}
    if dataset != "varex":
        return normalized
    for field in ("schema", "ground_truth"):
        value = normalized.get(field)
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"VAREX parquet {field} is not UTF-8 JSON") from exc
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"VAREX parquet {field} is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"VAREX parquet {field} must decode to an object")
        normalized[field] = dict(value)
    return normalized


def _materialize_reference(
    reference: dict[str, Any],
    *,
    dataset: str,
    state: MaterializationState,
    container_cache: dict[str, MaterializedObject],
    digest_datasets: dict[str, str],
) -> tuple[MaterializedObject, dict[str, Any]]:
    kind = reference["kind"]
    expected_sha256 = _reference_sha(reference, f"{dataset}/{kind}")
    _register_derived_digest(expected_sha256, dataset, digest_datasets)
    if kind == "direct":
        destination, relative = _object_destination(state.stage_dir, dataset, expected_sha256)
        downloaded = _download_http(
            reference["uri"],
            destination,
            dataset=dataset,
            expected_sha256=expected_sha256,
            max_bytes=state.remaining_limit(state.max_member_bytes),
            state=state,
        )
        state.add_bytes(downloaded.size_bytes)
        return (
            MaterializedObject(relative, destination, downloaded.size_bytes, expected_sha256),
            {"kind": "direct"},
        )

    container_ref = reference["container"]
    container = _ensure_container(
        container_ref,
        dataset=dataset,
        state=state,
        cache=container_cache,
        digest_datasets=digest_datasets,
    )
    derivation: dict[str, Any] = {
        "kind": kind,
        "container_sha256": container.sha256,
        "container_materialized_path": container.relative_path,
    }
    if kind == "archive_member":
        member = _safe_member_name(reference["member"], f"{dataset}/archive_member")
        derivation.update({"archive_format": container_ref["format"], "member": member})
        if container_ref["format"] == "tar.gz":
            result = _extract_tar_member(
                container.absolute_path,
                member,
                dataset=dataset,
                expected_sha256=expected_sha256,
                state=state,
            )
        else:
            result = _extract_zip_member(
                container.absolute_path,
                member,
                dataset=dataset,
                expected_sha256=expected_sha256,
                state=state,
            )
        return result, derivation
    row_index = _validate_row_index(reference["row_index"], f"{dataset}/{kind}")
    derivation["row_index"] = row_index
    if kind == "parquet_field":
        field = reference["field"]
        payload = state.parquet_adapter.read_field(
            container.absolute_path,
            row_index=row_index,
            field=field,
        )
        if not isinstance(payload, bytes):
            raise TypeError("ParquetAdapter.read_field must return bytes")
        derivation["field"] = field
    else:
        fields = reference["fields"]
        if kind == "parquet_row":
            row = state.parquet_adapter.read_row(
                container.absolute_path,
                row_index=row_index,
                fields=fields,
            )
            if not isinstance(row, Mapping) or set(row) != set(fields):
                raise ValueError("ParquetAdapter.read_row must return every requested field only")
            normalized_row = _normalize_parquet_row(dataset, row, fields)
            payload = _canonical_json_bytes(normalized_row)
            if dataset == "varex":
                derivation["row_normalization"] = "varex-json-fields-v1"
        else:
            payload = _read_jsonl_row(
                container.absolute_path,
                row_index=row_index,
                fields=fields,
                max_row_bytes=state.max_member_bytes,
            )
        derivation["fields"] = list(fields)
        derivation["canonical_json_algorithm"] = _CANONICAL_JSON_ALGORITHM
    return (
        _materialize_bytes(
            payload,
            dataset=dataset,
            expected_sha256=expected_sha256,
            state=state,
        ),
        derivation,
    )


def _trusted_identity(inputs: list[dict[str, Any]]) -> dict[str, str]:
    identity = [
        {"role": item["role"], "sha256": item["sha256"]}
        for item in sorted(inputs, key=lambda value: value["role"])
    ]
    return {
        "algorithm": "sha256-canonical-role-digests-v1",
        "digest": hashlib.sha256(_canonical_json_bytes(identity)).hexdigest(),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    part = path.with_name(path.name + ".part")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    with part.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(part, path)


def materialize_manifest(
    source_manifest: Path,
    output_dir: Path,
    *,
    http_client: HttpClient | None = None,
    parquet_adapter: ParquetAdapter | None = None,
    container_source_dir: Path | None = None,
    max_manifest_bytes: int = 32 * 1024 * 1024,
    max_container_bytes: int = _DEFAULT_MAX_CONTAINER_BYTES,
    max_member_bytes: int = _DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
    max_archive_uncompressed_bytes: int = _DEFAULT_MAX_ARCHIVE_BYTES,
    max_archive_members: int = _DEFAULT_MAX_ARCHIVE_MEMBERS,
    timeout_s: float = 300.0,
) -> Path:
    for name, value in (
        ("max_manifest_bytes", max_manifest_bytes),
        ("max_container_bytes", max_container_bytes),
        ("max_member_bytes", max_member_bytes),
        ("max_total_bytes", max_total_bytes),
        ("max_archive_uncompressed_bytes", max_archive_uncompressed_bytes),
        ("max_archive_members", max_archive_members),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if source_manifest.stat().st_size > max_manifest_bytes:
        raise ValueError("source manifest exceeds max_manifest_bytes")
    source_bytes = source_manifest.read_bytes()
    if len(source_bytes) > max_manifest_bytes:
        raise ValueError("source manifest exceeds max_manifest_bytes")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    manifest = _validate_source_manifest(json.loads(source_bytes))
    verified = copy.deepcopy(manifest)
    if container_source_dir is not None:
        container_source_dir = container_source_dir.resolve(strict=True)
        if not container_source_dir.is_dir():
            raise ValueError("container_source_dir must be a directory")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    state = MaterializationState(
        stage_dir=stage_dir,
        container_source_dir=container_source_dir,
        http_client=http_client or UrllibHttpClient(),
        parquet_adapter=parquet_adapter or PyArrowParquetAdapter(),
        max_container_bytes=max_container_bytes,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
        max_archive_uncompressed_bytes=max_archive_uncompressed_bytes,
        max_archive_members=max_archive_members,
        timeout_s=timeout_s,
    )
    container_cache: dict[str, MaterializedObject] = {}
    digest_datasets: dict[str, str] = {}
    verified_reference_count = 0
    try:
        for candidate in verified["selected"]:
            dataset = candidate["dataset"]
            for item in candidate["inputs"]:
                materialized, derivation = _materialize_reference(
                    item,
                    dataset=dataset,
                    state=state,
                    container_cache=container_cache,
                    digest_datasets=digest_datasets,
                )
                item["materialized_path"] = materialized.relative_path
                item["size_bytes"] = materialized.size_bytes
                item["derivation"] = derivation
                verified_reference_count += 1
            metadata_key = {
                "pubtables_v2": "annotation",
                "varex": "source_record",
                "ai2d_rst": "annotation",
                "mws_vision_bench": "source_record",
            }[dataset]
            reference = candidate["metadata"][metadata_key]
            materialized, derivation = _materialize_reference(
                reference,
                dataset=dataset,
                state=state,
                container_cache=container_cache,
                digest_datasets=digest_datasets,
            )
            reference["materialized_path"] = materialized.relative_path
            reference["size_bytes"] = materialized.size_bytes
            reference["derivation"] = derivation
            verified_reference_count += 1
            candidate["trusted_physical_identity"] = _trusted_identity(candidate["inputs"])

        verified["verification_state"] = "bytes_verified"
        verified["source_manifest_sha256"] = source_hash
        verified["byte_verification"] = {
            "algorithm": "sha256",
            "verified_references": verified_reference_count,
            "verified_unique_containers": len(container_cache),
            "total_materialized_bytes": state.total_bytes,
            "cross_dataset_digest_duplicates": 0,
        }
        verified.pop("materialization_requirements", None)
        manifest_path = stage_dir / _VERIFIED_MANIFEST
        _write_json_atomic(manifest_path, verified)
        os.replace(stage_dir, output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return output_dir / _VERIFIED_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize and byte-verify an OCR manifest.")
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--container-source-dir",
        type=Path,
        help="optional local cache whose regular-file names are pinned container SHA-256 values",
    )
    parser.add_argument("--max-manifest-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-container-bytes", type=int, default=_DEFAULT_MAX_CONTAINER_BYTES)
    parser.add_argument("--max-member-bytes", type=int, default=_DEFAULT_MAX_MEMBER_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=_DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument(
        "--max-archive-uncompressed-bytes",
        type=int,
        default=_DEFAULT_MAX_ARCHIVE_BYTES,
    )
    parser.add_argument("--max-archive-members", type=int, default=_DEFAULT_MAX_ARCHIVE_MEMBERS)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    verified_path = materialize_manifest(
        args.source_manifest,
        args.output_dir,
        container_source_dir=args.container_source_dir,
        max_manifest_bytes=args.max_manifest_bytes,
        max_container_bytes=args.max_container_bytes,
        max_member_bytes=args.max_member_bytes,
        max_total_bytes=args.max_total_bytes,
        max_archive_uncompressed_bytes=args.max_archive_uncompressed_bytes,
        max_archive_members=args.max_archive_members,
        timeout_s=args.timeout,
    )
    print(verified_path)


if __name__ == "__main__":
    main()
