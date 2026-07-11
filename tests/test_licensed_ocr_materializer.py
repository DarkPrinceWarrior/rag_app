from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import runpy
import stat
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "materialize_licensed_ocr_manifest.py")
)
_AI2D_ANNOTATION_ARCHIVE = _SCRIPT["_AI2D_ANNOTATION_ARCHIVE"]
_AI2D_IMAGE_ARCHIVE = _SCRIPT["_AI2D_IMAGE_ARCHIVE"]
_DEFAULT_MAX_CONTAINER_BYTES = _SCRIPT["_DEFAULT_MAX_CONTAINER_BYTES"]
_MWS_METADATA_URI = _SCRIPT["_MWS_METADATA_URI"]
_MWS_RECORD_FIELDS = _SCRIPT["_MWS_RECORD_FIELDS"]
_PINNED_CONTAINER_SHA256 = _SCRIPT["_PINNED_CONTAINER_SHA256"]
_PINNED_DATASETS = _SCRIPT["_PINNED_DATASETS"]
_PUBTABLES_IMAGES_URI = _SCRIPT["_PUBTABLES_IMAGES_URI"]
_PUBTABLES_TABLES_URI = _SCRIPT["_PUBTABLES_TABLES_URI"]
_VAREX_RECORD_FIELDS = _SCRIPT["_VAREX_RECORD_FIELDS"]
materialize_manifest = _SCRIPT["materialize_manifest"]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_hash(value: Any) -> str:
    return _sha(_canonical_bytes(value))


def _dataset_metadata(dataset: str) -> dict[str, Any]:
    return dict(_PINNED_DATASETS[dataset])


def _container(uri: str, payload: bytes, format_: str) -> dict[str, str]:
    return {"uri": uri, "sha256": _sha(payload), "format": format_}


def _direct(uri: str, payload: bytes) -> dict[str, Any]:
    return {"kind": "direct", "uri": uri, "sha256": _sha(payload)}


def _manifest(*candidates: dict[str, Any]) -> dict[str, Any]:
    seed = "docragenslate-ocr-v1"
    selected = []
    for source in candidates:
        candidate = json.loads(json.dumps(source))
        candidate["selection_hash"] = _sha(
            "\0".join(
                (seed, candidate["dataset"], candidate["stratum"], candidate["canonical_id"])
            ).encode()
        )
        selected.append(candidate)
    datasets = {candidate["dataset"] for candidate in candidates}
    dataset_metadata: dict[str, dict[str, Any]] = {}
    summary_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted(datasets):
        items = [candidate for candidate in selected if candidate["dataset"] == dataset]
        strata = sorted({candidate["stratum"] for candidate in items})
        quotas = {
            stratum: sum(candidate["stratum"] == stratum for candidate in items)
            for stratum in strata
        }
        dataset_metadata[dataset] = {**_dataset_metadata(dataset), "quotas": quotas}
        summary_by_dataset[dataset] = {
            "selected_units": len(items),
            "input_count": sum(len(candidate["inputs"]) for candidate in items),
            "by_stratum": quotas,
        }
    return {
        "manifest_version": 1,
        "verification_state": "metadata_only_unverified",
        "datasets": dataset_metadata,
        "selection": {
            "algorithm": "sha256-nul-v1",
            "seed": seed,
            "catalog_sha256": _sha(_canonical_bytes(list(candidates))),
            "canonical_json_algorithm": "sha256-json-sort-keys-ascii-v1",
            "group_policy": "claimed_identity_only",
        },
        "materialization_requirements": {
            "verify_every_referenced_object_bytes_against_sha256": True,
            "verify_group_id_against_trusted_physical_identity": True,
            "reject_manifest_on_any_mismatch": True,
        },
        "summary": {
            "candidate_count": len(selected),
            "selected_units": len(selected),
            "input_count": sum(len(candidate["inputs"]) for candidate in selected),
            "by_dataset": summary_by_dataset,
        },
        "selected": selected,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> bytes:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return payload


def _zip_payload(files: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
        if symlink is not None:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
    return output.getvalue()


def _tar_payload(files: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if symlink is not None:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            archive.addfile(info)
    return output.getvalue()


class FakeHttpClient:
    def __init__(
        self,
        payloads: dict[str, bytes],
        *,
        redirects: dict[str, str] | None = None,
    ) -> None:
        self.payloads = payloads
        self.redirects = redirects or {}
        self.opened: list[str] = []

    @contextmanager
    def open(self, uri: str, *, timeout_s: float, redirect_validator: Any):  # type: ignore[no-untyped-def]
        assert timeout_s > 0
        self.opened.append(uri)
        if uri in self.redirects:
            redirect_validator(self.redirects[uri])
        yield io.BytesIO(self.payloads[uri])


class FakeParquetAdapter:
    def __init__(self, fields: dict[tuple[int, str], bytes], rows: dict[int, dict[str, Any]]) -> None:
        self.fields = fields
        self.rows = rows
        self.field_calls: list[tuple[int, str]] = []
        self.row_calls: list[tuple[int, tuple[str, ...]]] = []

    def read_field(self, container: Path, *, row_index: int, field: str) -> bytes:
        assert container.is_file()
        self.field_calls.append((row_index, field))
        return self.fields[(row_index, field)]

    def read_row(
        self,
        container: Path,
        *,
        row_index: int,
        fields: list[str],
    ) -> dict[str, Any]:
        assert container.is_file()
        self.row_calls.append((row_index, tuple(fields)))
        return {field: self.rows[row_index][field] for field in fields}


def _mws_candidate(image: bytes, rows: list[dict[str, Any]], *, row_index: int = 0) -> dict[str, Any]:
    dataset = "mws_vision_bench"
    record = rows[row_index]
    metadata_payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    image_path = record["file_name"]
    image_uri = (
        "https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench/resolve/"
        f"{_PINNED_DATASETS[dataset]['revision']}/{image_path}"
    )
    source_record = {
        "kind": "jsonl_row",
        "container": _container(_MWS_METADATA_URI, metadata_payload, "jsonl"),
        "row_index": row_index,
        "fields": list(_MWS_RECORD_FIELDS),
        "sha256": _canonical_hash(record),
    }
    return {
        "dataset": dataset,
        "stratum": "document_parsing_ru",
        "canonical_id": record["id"],
        "group_id": image_path,
        "inputs": [{"role": "image", **_direct(image_uri, image)}],
        "metadata": {
            "image_path": image_path,
            "task_type": "document_parsing_ru",
            "dataset_name": record["dataset_name"],
            "question": record["question"],
            "answers": record["answers"],
            "source_record": source_record,
            "source_record_canonical_sha256": _canonical_hash(record),
        },
    }


def _mws_row(suffix: str = "1") -> dict[str, Any]:
    return {
        "answers": ["Answer"],
        "dataset_name": "business",
        "file_name": f"images/{suffix}.png",
        "id": f"qa-{suffix}",
        "question": "Question",
        "type": "document parsing ru",
    }


def _varex_candidate(
    container_payload: bytes,
    image_200: bytes,
    image_50: bytes,
    row: dict[str, Any],
    *,
    row_index: int = 0,
) -> dict[str, Any]:
    dataset = "varex"
    uri = next(uri for uri in _PINNED_CONTAINER_SHA256 if "VAREX" in uri)
    container = _container(uri, container_payload, "parquet")
    record = {field: row[field] for field in _VAREX_RECORD_FIELDS}
    return {
        "dataset": dataset,
        "stratum": row["split"],
        "canonical_id": row["doc_id"],
        "group_id": row["doc_id"],
        "inputs": [
            {
                "role": "image_200dpi",
                "kind": "parquet_field",
                "container": container,
                "row_index": row_index,
                "field": "image",
                "sha256": _sha(image_200),
            },
            {
                "role": "image_50dpi",
                "kind": "parquet_field",
                "container": container,
                "row_index": row_index,
                "field": "image_50dpi",
                "sha256": _sha(image_50),
            },
        ],
        "metadata": {
            "doc_id": row["doc_id"],
            "schema": row["schema"],
            "ground_truth": row["ground_truth"],
            "schema_canonical_sha256": _canonical_hash(row["schema"]),
            "ground_truth_canonical_sha256": _canonical_hash(row["ground_truth"]),
            "source_record": {
                "kind": "parquet_row",
                "container": container,
                "row_index": row_index,
                "fields": list(_VAREX_RECORD_FIELDS),
                "sha256": _canonical_hash(record),
            },
        },
    }


def _varex_row() -> dict[str, Any]:
    return {
        "doc_id": "form-1",
        "ground_truth": {"field": "value"},
        "schema": {"type": "object", "properties": {"field": {"type": "string"}}},
        "split": "Flat",
    }


def _ai2d_candidate(
    image: bytes,
    annotation: bytes,
    image_zip: bytes,
    annotation_zip: bytes,
) -> dict[str, Any]:
    diagram_id = "diagram-1"
    return {
        "dataset": "ai2d_rst",
        "stratum": "diagram",
        "canonical_id": diagram_id,
        "group_id": diagram_id,
        "inputs": [
            {
                "role": "image",
                "kind": "archive_member",
                "container": _container(_AI2D_IMAGE_ARCHIVE, image_zip, "zip"),
                "member": f"ai2d/images/{diagram_id}.png",
                "sha256": _sha(image),
            }
        ],
        "metadata": {
            "diagram_id": diagram_id,
            "annotation": {
                "kind": "archive_member",
                "container": _container(_AI2D_ANNOTATION_ARCHIVE, annotation_zip, "zip"),
                "member": f"ai2d-rst-v1-1/json/ai2d-rst/{diagram_id}.png.json",
                "sha256": _sha(annotation),
            },
        },
    }


def _pubtables_candidate(
    pages: list[bytes],
    annotation: bytes,
    image_tar: bytes,
    table_tar: bytes,
) -> dict[str, Any]:
    document_id = "PMC1"
    image_container = _container(_PUBTABLES_IMAGES_URI, image_tar, "tar.gz")
    return {
        "dataset": "pubtables_v2",
        "stratum": "pages_2",
        "canonical_id": "PMC1-table-0",
        "group_id": document_id,
        "inputs": [
            {
                "role": f"page_{index + 1}",
                "kind": "archive_member",
                "container": image_container,
                "member": f"Full Documents/test/images/{document_id}_page_{index}.jpg",
                "sha256": _sha(page),
            }
            for index, page in enumerate(pages)
        ],
        "metadata": {
            "document_id": document_id,
            "table_id": "table-0",
            "page_indices": [0, 1],
            "annotation": {
                "kind": "archive_member",
                "container": _container(_PUBTABLES_TABLES_URI, table_tar, "tar.gz"),
                "member": f"Full Documents/test/tables/{document_id}_tables.json",
                "sha256": _sha(annotation),
            },
        },
    }


def _pin_container(monkeypatch: pytest.MonkeyPatch, uri: str, payload: bytes) -> None:
    monkeypatch.setitem(_PINNED_CONTAINER_SHA256, uri, _sha(payload))


def test_direct_and_jsonl_row_materialize_with_canonical_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"mws-image"
    rows = [_mws_row("0"), _mws_row("1")]
    candidate = _mws_candidate(image, rows, row_index=1)
    metadata_payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    source_bytes = _write_manifest(source, _manifest(candidate))
    client = FakeHttpClient(
        {
            candidate["inputs"][0]["uri"]: image,
            _MWS_METADATA_URI: metadata_payload,
        }
    )

    verified_path = materialize_manifest(source, tmp_path / "out", http_client=client)

    verified = json.loads(verified_path.read_text())
    selected = verified["selected"][0]
    assert verified["verification_state"] == "bytes_verified"
    assert verified["source_manifest_sha256"] == _sha(source_bytes)
    assert selected["inputs"][0]["derivation"] == {"kind": "direct"}
    record_ref = selected["metadata"]["source_record"]
    assert record_ref["derivation"]["kind"] == "jsonl_row"
    assert record_ref["derivation"]["fields"] == list(_MWS_RECORD_FIELDS)
    assert (verified_path.parent / record_ref["materialized_path"]).read_bytes() == _canonical_bytes(
        rows[1]
    )


def test_parquet_fields_and_row_share_one_verified_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container_payload = b"fake-parquet-container"
    image_200 = b"image-200"
    image_50 = b"image-50"
    row = _varex_row()
    candidate = _varex_candidate(container_payload, image_200, image_50, row)
    uri = candidate["inputs"][0]["container"]["uri"]
    _pin_container(monkeypatch, uri, container_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    client = FakeHttpClient({uri: container_payload})
    adapter = FakeParquetAdapter(
        {(0, "image"): image_200, (0, "image_50dpi"): image_50},
        {
            0: {
                **row,
                "schema": json.dumps(row["schema"]),
                "ground_truth": json.dumps(row["ground_truth"]),
            }
        },
    )

    verified_path = materialize_manifest(
        source,
        tmp_path / "out",
        http_client=client,
        parquet_adapter=adapter,
    )

    verified = json.loads(verified_path.read_text())
    assert client.opened == [uri]
    assert adapter.field_calls == [(0, "image"), (0, "image_50dpi")]
    assert adapter.row_calls == [(0, tuple(_VAREX_RECORD_FIELDS))]
    assert verified["byte_verification"]["verified_unique_containers"] == 1
    assert verified["byte_verification"]["verified_references"] == 3
    assert (
        verified["selected"][0]["metadata"]["source_record"]["derivation"]["row_normalization"]
        == "varex-json-fields-v1"
    )


def test_tar_members_are_streamed_and_container_is_cached_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [b"page-0", b"page-1"]
    annotation = b'{"tables":[]}'
    image_tar = _tar_payload(
        {
            "Full Documents/test/images/PMC1_page_0.jpg": pages[0],
            "Full Documents/test/images/PMC1_page_1.jpg": pages[1],
        }
    )
    table_tar = _tar_payload({"Full Documents/test/tables/PMC1_tables.json": annotation})
    _pin_container(monkeypatch, _PUBTABLES_IMAGES_URI, image_tar)
    _pin_container(monkeypatch, _PUBTABLES_TABLES_URI, table_tar)
    candidate = _pubtables_candidate(pages, annotation, image_tar, table_tar)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    client = FakeHttpClient(
        {_PUBTABLES_IMAGES_URI: image_tar, _PUBTABLES_TABLES_URI: table_tar}
    )

    verified_path = materialize_manifest(source, tmp_path / "out", http_client=client)

    verified = json.loads(verified_path.read_text())
    assert client.opened.count(_PUBTABLES_IMAGES_URI) == 1
    assert client.opened.count(_PUBTABLES_TABLES_URI) == 1
    assert verified["byte_verification"]["verified_unique_containers"] == 2
    assert all(item["derivation"]["kind"] == "archive_member" for item in verified["selected"][0]["inputs"])


def test_zip_members_materialize_from_two_exact_ai2d_containers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"ai2d-image"
    annotation = b'{"nodes":[]}'
    image_zip = _zip_payload({"ai2d/images/diagram-1.png": image})
    annotation_zip = _zip_payload(
        {"ai2d-rst-v1-1/json/ai2d-rst/diagram-1.png.json": annotation}
    )
    _pin_container(monkeypatch, _AI2D_IMAGE_ARCHIVE, image_zip)
    _pin_container(monkeypatch, _AI2D_ANNOTATION_ARCHIVE, annotation_zip)
    candidate = _ai2d_candidate(image, annotation, image_zip, annotation_zip)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))

    verified_path = materialize_manifest(
        source,
        tmp_path / "out",
        http_client=FakeHttpClient(
            {_AI2D_IMAGE_ARCHIVE: image_zip, _AI2D_ANNOTATION_ARCHIVE: annotation_zip}
        ),
    )

    verified = json.loads(verified_path.read_text())
    assert verified["datasets"]["ai2d_rst"]["license"] == "CC-BY-4.0 AND CC-BY-SA-4.0"
    assert verified["selected"][0]["metadata"]["annotation"]["derivation"]["archive_format"] == "zip"


@pytest.mark.parametrize("bad_name", ["../evil", "/absolute", "safe/../../evil"])
def test_zip_rejects_traversal_anywhere_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str
) -> None:
    image = b"image"
    annotation = b"annotation"
    image_zip = _zip_payload({"ai2d/images/diagram-1.png": image, bad_name: b"bad"})
    annotation_zip = _zip_payload(
        {"ai2d-rst-v1-1/json/ai2d-rst/diagram-1.png.json": annotation}
    )
    _pin_container(monkeypatch, _AI2D_IMAGE_ARCHIVE, image_zip)
    _pin_container(monkeypatch, _AI2D_ANNOTATION_ARCHIVE, annotation_zip)
    candidate = _ai2d_candidate(image, annotation, image_zip, annotation_zip)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))

    with pytest.raises(ValueError, match="archive member"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient(
                {_AI2D_IMAGE_ARCHIVE: image_zip, _AI2D_ANNOTATION_ARCHIVE: annotation_zip}
            ),
        )
    assert not (tmp_path / "out").exists()


def test_zip_rejects_links_and_total_uncompressed_bomb_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    annotation = b"annotation"
    image_zip = _zip_payload({"ai2d/images/diagram-1.png": image}, symlink="link")
    annotation_zip = _zip_payload(
        {"ai2d-rst-v1-1/json/ai2d-rst/diagram-1.png.json": annotation}
    )
    _pin_container(monkeypatch, _AI2D_IMAGE_ARCHIVE, image_zip)
    _pin_container(monkeypatch, _AI2D_ANNOTATION_ARCHIVE, annotation_zip)
    candidate = _ai2d_candidate(image, annotation, image_zip, annotation_zip)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    client = FakeHttpClient(
        {_AI2D_IMAGE_ARCHIVE: image_zip, _AI2D_ANNOTATION_ARCHIVE: annotation_zip}
    )

    with pytest.raises(ValueError, match="link, special, or encrypted"):
        materialize_manifest(source, tmp_path / "links", http_client=client)
    clean_image_zip = _zip_payload({"ai2d/images/diagram-1.png": image})
    _pin_container(monkeypatch, _AI2D_IMAGE_ARCHIVE, clean_image_zip)
    clean_candidate = _ai2d_candidate(image, annotation, clean_image_zip, annotation_zip)
    _write_manifest(source, _manifest(clean_candidate))
    with pytest.raises(ValueError, match="total uncompressed bytes cap"):
        materialize_manifest(
            source,
            tmp_path / "bomb",
            http_client=FakeHttpClient(
                {
                    _AI2D_IMAGE_ARCHIVE: clean_image_zip,
                    _AI2D_ANNOTATION_ARCHIVE: annotation_zip,
                }
            ),
            max_archive_uncompressed_bytes=4,
        )


def test_tar_rejects_links_even_after_requested_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [b"page-0", b"page-1"]
    annotation = b"annotation"
    image_tar = _tar_payload(
        {
            "Full Documents/test/images/PMC1_page_0.jpg": pages[0],
            "Full Documents/test/images/PMC1_page_1.jpg": pages[1],
        },
        symlink="late-link",
    )
    table_tar = _tar_payload({"Full Documents/test/tables/PMC1_tables.json": annotation})
    _pin_container(monkeypatch, _PUBTABLES_IMAGES_URI, image_tar)
    _pin_container(monkeypatch, _PUBTABLES_TABLES_URI, table_tar)
    candidate = _pubtables_candidate(pages, annotation, image_tar, table_tar)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))

    with pytest.raises(ValueError, match="forbidden member type"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient(
                {_PUBTABLES_IMAGES_URI: image_tar, _PUBTABLES_TABLES_URI: table_tar}
            ),
        )
    assert not (tmp_path / "out").exists()


def test_container_cap_is_separate_and_default_exceeds_3_2gb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _DEFAULT_MAX_CONTAINER_BYTES >= 3_200_000_000
    rows = [_mws_row()]
    candidate = _mws_candidate(b"image", rows)
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))

    with pytest.raises(ValueError, match="exceeds max bytes"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient(
                {
                    candidate["inputs"][0]["uri"]: b"image",
                    _MWS_METADATA_URI: metadata_payload,
                }
            ),
            max_container_bytes=4,
        )


def test_member_cap_is_separate_from_large_container_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image-too-large"
    annotation = b"annotation"
    image_zip = _zip_payload({"ai2d/images/diagram-1.png": image})
    annotation_zip = _zip_payload(
        {"ai2d-rst-v1-1/json/ai2d-rst/diagram-1.png.json": annotation}
    )
    _pin_container(monkeypatch, _AI2D_IMAGE_ARCHIVE, image_zip)
    _pin_container(monkeypatch, _AI2D_ANNOTATION_ARCHIVE, annotation_zip)
    candidate = _ai2d_candidate(image, annotation, image_zip, annotation_zip)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))

    with pytest.raises(ValueError, match="member exceeds max member bytes"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient(
                {_AI2D_IMAGE_ARCHIVE: image_zip, _AI2D_ANNOTATION_ARCHIVE: annotation_zip}
            ),
            max_container_bytes=1024,
            max_member_bytes=4,
        )


def test_local_container_cache_is_rehashed_and_avoids_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(image, rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / _sha(metadata_payload)).write_bytes(metadata_payload)
    image_uri = candidate["inputs"][0]["uri"]
    client = FakeHttpClient({image_uri: image})

    verified_path = materialize_manifest(
        source,
        tmp_path / "out",
        http_client=client,
        container_source_dir=cache,
    )

    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    assert client.opened == [image_uri]
    assert verified["byte_verification"]["verified_unique_containers"] == 1


def test_local_container_cache_sha_mismatch_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(image, rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / _sha(metadata_payload)).write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient({candidate["inputs"][0]["uri"]: image}),
            container_source_dir=cache,
        )
    assert not (tmp_path / "out").exists()


def test_local_container_cache_rejects_symlink_without_following_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(image, rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    cache = tmp_path / "cache"
    cache.mkdir()
    target = tmp_path / "metadata.jsonl"
    target.write_bytes(metadata_payload)
    (cache / _sha(metadata_payload)).symlink_to(target)

    with pytest.raises(ValueError, match="cannot be opened safely"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient({candidate["inputs"][0]["uri"]: image}),
            container_source_dir=cache,
        )
    assert not (tmp_path / "out").exists()


def test_total_byte_limit_stops_local_container_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    annotation = b"annotation"
    image_zip = _zip_payload({"ai2d/images/diagram-1.png": image})
    annotation_zip = _zip_payload(
        {"ai2d-rst-v1-1/json/ai2d-rst/diagram-1.png.json": annotation}
    )
    _pin_container(monkeypatch, _AI2D_IMAGE_ARCHIVE, image_zip)
    _pin_container(monkeypatch, _AI2D_ANNOTATION_ARCHIVE, annotation_zip)
    source = tmp_path / "source.json"
    _write_manifest(
        source,
        _manifest(_ai2d_candidate(image, annotation, image_zip, annotation_zip)),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / _sha(image_zip)).write_bytes(image_zip)

    with pytest.raises(ValueError, match="object exceeds max bytes: 1"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient({}),
            container_source_dir=cache,
            max_total_bytes=1,
        )
    assert not (tmp_path / "out").exists()


def test_hf_redirect_allows_explicit_cdn_and_rejects_other_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(image, rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    image_uri = candidate["inputs"][0]["uri"]
    payloads = {image_uri: image, _MWS_METADATA_URI: metadata_payload}

    materialize_manifest(
        source,
        tmp_path / "allowed",
        http_client=FakeHttpClient(
            payloads,
            redirects={image_uri: "https://us.aws.cdn.hf.co/xet-bridge-us/object?signature=x"},
        ),
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        materialize_manifest(
            source,
            tmp_path / "blocked",
            http_client=FakeHttpClient(
                payloads,
                redirects={image_uri: "https://evil.example/object"},
            ),
        )


def test_mws_metadata_allows_exact_same_host_resolve_cache_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = b"image"
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(image, rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    image_uri = candidate["inputs"][0]["uri"]
    cache_uri = (
        "https://huggingface.co/api/resolve-cache/datasets/MTSAIR/MWS-Vision-Bench/"
        "e204166bde25f7dcaaffb9313b855de67b516e5d/metadata.jsonl?etag=pinned"
    )

    materialize_manifest(
        source,
        tmp_path / "out",
        http_client=FakeHttpClient(
            {image_uri: image, _MWS_METADATA_URI: metadata_payload},
            redirects={_MWS_METADATA_URI: cache_uri},
        ),
    )


def test_pubtables_rejects_page_members_from_another_document_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [b"page-0", b"page-1"]
    annotation = b"tables"
    image_tar = _tar_payload(
        {
            "Full Documents/test/images/OTHER_page_7.jpg": pages[0],
            "Full Documents/test/images/OTHER_page_8.jpg": pages[1],
        }
    )
    table_tar = _tar_payload({"Full Documents/test/tables/PMC1_tables.json": annotation})
    _pin_container(monkeypatch, _PUBTABLES_IMAGES_URI, image_tar)
    _pin_container(monkeypatch, _PUBTABLES_TABLES_URI, table_tar)
    candidate = _pubtables_candidate(pages, annotation, image_tar, table_tar)
    manifest = _manifest(candidate)
    manifest["selected"][0]["metadata"]["page_indices"] = []
    for index, item in enumerate(manifest["selected"][0]["inputs"], start=7):
        item["member"] = f"Full Documents/test/images/OTHER_page_{index}.jpg"
    source = tmp_path / "source.json"
    _write_manifest(source, manifest)
    client = FakeHttpClient({})

    with pytest.raises(ValueError, match="page_indices must match"):
        materialize_manifest(source, tmp_path / "out", http_client=client)
    assert client.opened == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["selection"].update(algorithm="wrong"), "provenance"),
        (
            lambda manifest: manifest["selected"][0].update(selection_hash="0" * 64),
            "selection_hash",
        ),
        (lambda manifest: manifest["summary"].update(input_count=999), "summary"),
        (
            lambda manifest: manifest["datasets"]["mws_vision_bench"]["quotas"].update(
                document_parsing_ru=2
            ),
            "quotas",
        ),
    ],
)
def test_rejects_corrupted_selection_provenance_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    row = _mws_row()
    metadata_payload = _canonical_bytes(row) + b"\n"
    candidate = _mws_candidate(b"image", [row])
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    manifest = _manifest(candidate)
    mutation(manifest)
    source = tmp_path / "source.json"
    _write_manifest(source, manifest)
    client = FakeHttpClient({})

    with pytest.raises(ValueError, match=message):
        materialize_manifest(source, tmp_path / "out", http_client=client)
    assert client.opened == []


@pytest.mark.parametrize("field", ["license_components", "source_components"])
def test_rejects_tampered_component_provenance_before_download(
    tmp_path: Path,
    field: str,
) -> None:
    image = b"image"
    annotation = b"annotation"
    image_zip = _zip_payload({"ai2d/images/diagram-1.png": image})
    annotation_zip = _zip_payload(
        {"ai2d-rst-v1-1/json/ai2d-rst/diagram-1.png.json": annotation}
    )
    manifest = _manifest(_ai2d_candidate(image, annotation, image_zip, annotation_zip))
    manifest["datasets"]["ai2d_rst"][field] = {"tampered": "MIT"}
    source = tmp_path / "source.json"
    _write_manifest(source, manifest)
    client = FakeHttpClient({})

    with pytest.raises(ValueError, match=field):
        materialize_manifest(source, tmp_path / "out", http_client=client)
    assert client.opened == []


def test_rejects_non_finite_json_in_unrecognized_metadata_before_download(
    tmp_path: Path,
) -> None:
    candidate = _mws_candidate(b"image", [_mws_row()])
    manifest = _manifest(candidate)
    manifest["datasets"]["mws_vision_bench"]["unrecognized"] = float("nan")
    source = tmp_path / "source.json"
    _write_manifest(source, manifest)
    client = FakeHttpClient({})

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        materialize_manifest(source, tmp_path / "out", http_client=client)
    assert client.opened == []


def test_rejects_declared_quota_above_pinned_design_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_mws_row(str(index)) for index in range(7)]
    metadata_payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    candidates = [
        _mws_candidate(f"image-{index}".encode(), rows, row_index=index)
        for index in range(7)
    ]
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(*candidates))
    client = FakeHttpClient({})

    with pytest.raises(ValueError, match="quotas exceed the pinned corpus design"):
        materialize_manifest(source, tmp_path / "out", http_client=client)
    assert client.opened == []


def test_source_manifest_must_be_a_regular_file_without_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source.fifo"
    os.mkfifo(source)

    with pytest.raises(ValueError, match="must be a regular file"):
        materialize_manifest(source, tmp_path / "out", http_client=FakeHttpClient({}))


def test_parquet_default_adapter_has_precise_optional_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container_payload = b"fake parquet"
    row = _varex_row()
    candidate = _varex_candidate(container_payload, b"a", b"b", row)
    uri = candidate["inputs"][0]["container"]["uri"]
    _pin_container(monkeypatch, uri, container_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    real_import = importlib.import_module

    def missing_pyarrow(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == "pyarrow.parquet":
            error = ModuleNotFoundError("No module named 'pyarrow'")
            error.name = "pyarrow"
            raise error
        return real_import(name, package)

    monkeypatch.setattr(_SCRIPT["importlib"], "import_module", missing_pyarrow)
    with pytest.raises(RuntimeError, match="install pyarrow or inject a ParquetAdapter"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient({uri: container_payload}),
        )
    assert not (tmp_path / "out").exists()


def test_rejects_unsorted_row_fields_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(b"image", rows)
    candidate["metadata"]["source_record"]["fields"] = list(reversed(_MWS_RECORD_FIELDS))
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))
    client = FakeHttpClient({})

    with pytest.raises(ValueError, match="sorted unique"):
        materialize_manifest(source, tmp_path / "out", http_client=client)
    assert client.opened == []


def test_derived_sha_mismatch_rolls_back_without_verified_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_mws_row()]
    metadata_payload = _canonical_bytes(rows[0]) + b"\n"
    candidate = _mws_candidate(b"expected", rows)
    _pin_container(monkeypatch, _MWS_METADATA_URI, metadata_payload)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(candidate))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=FakeHttpClient(
                {
                    candidate["inputs"][0]["uri"]: b"tampered",
                    _MWS_METADATA_URI: metadata_payload,
                }
            ),
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_verified_cross_dataset_digest_duplicate_rejects_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = b"same-image-bytes"
    mws_rows = [_mws_row()]
    mws_metadata = _canonical_bytes(mws_rows[0]) + b"\n"
    mws = _mws_candidate(duplicate, mws_rows)
    parquet = b"parquet-container"
    varex_row = _varex_row()
    varex = _varex_candidate(parquet, duplicate, b"small-image", varex_row)
    varex_uri = varex["inputs"][0]["container"]["uri"]
    _pin_container(monkeypatch, _MWS_METADATA_URI, mws_metadata)
    _pin_container(monkeypatch, varex_uri, parquet)
    source = tmp_path / "source.json"
    _write_manifest(source, _manifest(mws, varex))
    client = FakeHttpClient(
        {
            mws["inputs"][0]["uri"]: duplicate,
            _MWS_METADATA_URI: mws_metadata,
            varex_uri: parquet,
        }
    )
    adapter = FakeParquetAdapter(
        {(0, "image"): duplicate, (0, "image_50dpi"): b"small-image"},
        {
            0: {
                **varex_row,
                "schema": json.dumps(varex_row["schema"]),
                "ground_truth": json.dumps(varex_row["ground_truth"]),
            }
        },
    )

    with pytest.raises(ValueError, match="verified digest occurs across datasets"):
        materialize_manifest(
            source,
            tmp_path / "out",
            http_client=client,
            parquet_adapter=adapter,
        )
    assert not (tmp_path / "out").exists()
