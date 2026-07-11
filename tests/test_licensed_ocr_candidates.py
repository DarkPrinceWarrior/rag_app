from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "build_licensed_ocr_candidates.py")
)
SafetyLimits = _SCRIPT["SafetyLimits"]
SourcePins = _SCRIPT["SourcePins"]
build_candidates = _SCRIPT["build_candidates"]
write_candidates = _SCRIPT["write_candidates"]
_varex_candidates = _SCRIPT["_varex_candidates"]

PUB_IMAGES = "PubTables-v2_Full-Documents_test_images.tar.gz"
PUB_TABLES = "PubTables-v2_Full-Documents_test_tables.tar.gz"
VAREX_SHARD = "data/benchmark-00000-of-00004.parquet"
AI2D_IMAGES = "ai2d-all.zip"
AI2D_ANNOTATIONS = "ai2d-rst-v1-1.zip"
MWS_METADATA = "metadata.jsonl"
MWS_TYPES = [
    "document parsing ru",
    "full-page OCR ru",
    "key information extraction ru",
    "reasoning VQA ru",
    "text grounding ru",
]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


class FakeParquetAdapter:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def iter_rows(self, path: Path, columns: tuple[str, ...]):
        self.calls.append((path, tuple(columns)))
        return list(self.rows)


def _varex_rows() -> list[dict[str, Any]]:
    flat_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    nested_schema = {
        "type": "object",
        "properties": {"person": {"type": "object", "properties": {"name": {}}}},
    }
    table_schema = {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
    }
    return [
        {
            "doc_id": "v-flat",
            "split": "Flat",
            "image": {"bytes": b"varex-flat-200"},
            "image_50dpi": {"bytes": b"varex-flat-50"},
            "schema": json.dumps(flat_schema),
            "ground_truth": json.dumps({"name": "A"}),
        },
        {
            "doc_id": "v-nested",
            "split": "Nested",
            "image": b"varex-nested-200",
            "image_50dpi": b"varex-nested-50",
            "schema": json.dumps(nested_schema),
            "ground_truth": json.dumps({"person": {"name": "B"}}),
        },
        {
            "doc_id": "v-table",
            "split": "Table",
            "image": memoryview(b"varex-table-200"),
            "image_50dpi": bytearray(b"varex-table-50"),
            "schema": table_schema,
            "ground_truth": {"rows": [{"value": "C"}]},
        },
    ]


def _fixture_source(tmp_path: Path) -> tuple[Path, Path, Any, FakeParquetAdapter]:
    root = tmp_path / "source"
    root.mkdir()
    pub_images = {
        f"Full Documents/test/images/doc-a_page_{page}.jpg": f"page-{page}".encode()
        for page in range(4)
    }
    pub_tables = {
        "Full Documents/test/tables/doc-a_tables.json": json.dumps(
            [{"table_id": "table-a", "page_num": page} for page in range(4)]
        ).encode()
    }
    _write_tar(root / PUB_IMAGES, pub_images)
    _write_tar(root / PUB_TABLES, pub_tables)
    parquet_path = root / VAREX_SHARD
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"fake-parquet-container")
    _write_zip(
        root / AI2D_IMAGES,
        {
            "ai2d/images/1.png": b"ai2d-image-1",
            "ai2d/images/2.png": b"ai2d-image-2",
            "ai2d/questions/1.json": b"{}",
        },
    )
    _write_zip(
        root / AI2D_ANNOTATIONS,
        {
            "ai2d-rst-v1-1/json/ai2d-rst/1.png.json": b'{"id":"1"}',
            "ai2d-rst-v1-1/json/ai2d-rst/2.png.json": b'{"id":"2"}',
            "ai2d-rst-v1-1/json/ai2d-rst/3.png.json": b'{"id":"3"}',
        },
    )
    mws_rows = []
    image_index = []
    for index, type_ in enumerate(MWS_TYPES, start=1):
        image_path = f"images/{index}.png"
        mws_rows.append(
            {
                "answers": [f"answer-{index}"],
                "dataset_name": "fixture",
                "file_name": image_path,
                "id": str(index),
                "question": f"question-{index}",
                "type": type_,
            }
        )
        image_index.append(
            {
                "path": image_path,
                "sha256": _sha(f"image-{index}".encode()),
                "size_bytes": len(f"image-{index}".encode()),
            }
        )
    (root / MWS_METADATA).write_text(
        "".join(json.dumps(row) + "\n" for row in mws_rows), encoding="utf-8"
    )
    index_path = tmp_path / "mws-images.json"
    index_path.write_text(
        json.dumps(
            {
                "dataset": "MTSAIR/MWS-Vision-Bench",
                "revision": "b8d473734b79343cac2b74f692a29ab191c7d11d",
                "images": image_index,
            }
        ),
        encoding="utf-8",
    )
    pins = SourcePins(
        pub_images_sha256=_sha((root / PUB_IMAGES).read_bytes()),
        pub_tables_sha256=_sha((root / PUB_TABLES).read_bytes()),
        varex_shards={VAREX_SHARD: _sha(parquet_path.read_bytes())},
        ai2d_images_sha256=_sha((root / AI2D_IMAGES).read_bytes()),
        ai2d_annotations_sha256=_sha((root / AI2D_ANNOTATIONS).read_bytes()),
        mws_metadata_sha256=_sha((root / MWS_METADATA).read_bytes()),
    )
    return root, index_path, pins, FakeParquetAdapter(_varex_rows())


def _generate(tmp_path: Path) -> tuple[list[dict[str, Any]], FakeParquetAdapter]:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    candidates = build_candidates(
        root,
        index_path,
        pins=pins,
        parquet_adapter=adapter,
        require_quotas=False,
    )
    return candidates, adapter


def test_builds_deterministic_offline_catalog_with_exact_reference_contract(tmp_path: Path) -> None:
    candidates, adapter = _generate(tmp_path)
    assert len(candidates) == 13
    assert candidates == sorted(
        candidates, key=lambda item: (item["dataset"], item["stratum"], item["canonical_id"])
    )
    assert adapter.calls[0][1] == (
        "doc_id",
        "split",
        "image",
        "image_50dpi",
        "schema",
        "ground_truth",
    )

    pub = next(item for item in candidates if item["canonical_id"] == "doc-a:pages_4")
    assert [item["role"] for item in pub["inputs"]] == ["page_1", "page_2", "page_3", "page_4"]
    assert pub["inputs"][0]["kind"] == "archive_member"
    assert pub["metadata"]["annotation"]["member"].endswith("doc-a_tables.json")

    varex = next(item for item in candidates if item["canonical_id"] == "v-flat")
    assert [item["field"] for item in varex["inputs"]] == ["image", "image_50dpi"]
    assert varex["metadata"]["source_record"]["kind"] == "parquet_row"
    assert varex["metadata"]["source_record"]["row_index"] == 0

    ai2d = next(item for item in candidates if item["dataset"] == "ai2d_rst")
    assert ai2d["inputs"][0]["member"].startswith("ai2d/images/")
    assert ai2d["metadata"]["annotation"]["member"].startswith(
        "ai2d-rst-v1-1/json/ai2d-rst/"
    )

    mws = next(item for item in candidates if item["dataset"] == "mws_vision_bench")
    assert mws["inputs"][0]["kind"] == "direct"
    assert mws["metadata"]["source_record"]["kind"] == "jsonl_row"
    assert mws["metadata"]["source_record"]["fields"] == [
        "answers",
        "dataset_name",
        "file_name",
        "id",
        "question",
        "type",
    ]


def test_member_and_field_hashes_are_computed_from_raw_bytes(tmp_path: Path) -> None:
    candidates, _ = _generate(tmp_path)
    pub = next(item for item in candidates if item["canonical_id"] == "doc-a:pages_2")
    assert pub["inputs"][0]["sha256"] == _sha(b"page-0")
    ai2d = next(item for item in candidates if item["canonical_id"] == "1")
    assert ai2d["inputs"][0]["sha256"] == _sha(b"ai2d-image-1")
    varex = next(item for item in candidates if item["canonical_id"] == "v-flat")
    assert varex["inputs"][0]["sha256"] == _sha(b"varex-flat-200")


def test_zip_intersection_excludes_unpaired_images_and_annotations(tmp_path: Path) -> None:
    candidates, _ = _generate(tmp_path)
    assert [
        item["canonical_id"] for item in candidates if item["dataset"] == "ai2d_rst"
    ] == ["1", "2"]


def test_rejects_container_sha_mismatch_before_catalog_generation(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    bad_pins = SourcePins(
        pub_images_sha256="0" * 64,
        pub_tables_sha256=pins.pub_tables_sha256,
        varex_shards=pins.varex_shards,
        ai2d_images_sha256=pins.ai2d_images_sha256,
        ai2d_annotations_sha256=pins.ai2d_annotations_sha256,
        mws_metadata_sha256=pins.mws_metadata_sha256,
    )
    with pytest.raises(ValueError, match="container SHA-256 mismatch"):
        build_candidates(
            root,
            index_path,
            pins=bad_pins,
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_rejects_archive_traversal_even_for_irrelevant_member(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    _write_zip(
        root / AI2D_IMAGES,
        {"ai2d/images/1.png": b"image", "ai2d/images/../../escape": b"bad"},
    )
    pins = SourcePins(
        pub_images_sha256=pins.pub_images_sha256,
        pub_tables_sha256=pins.pub_tables_sha256,
        varex_shards=pins.varex_shards,
        ai2d_images_sha256=_sha((root / AI2D_IMAGES).read_bytes()),
        ai2d_annotations_sha256=pins.ai2d_annotations_sha256,
        mws_metadata_sha256=pins.mws_metadata_sha256,
    )
    with pytest.raises(ValueError, match="unsafe archive member path"):
        build_candidates(
            root,
            index_path,
            pins=pins,
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_rejects_duplicate_mws_index_paths_or_hashes(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    duplicate = {"path": "images/1.png", "sha256": _sha(b"duplicate")}
    index_path.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate paths or hashes"):
        build_candidates(
            root,
            index_path,
            pins=pins,
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_rejects_mws_index_envelope_for_another_revision(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["revision"] = "0" * 40
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned dataset revision"):
        build_candidates(
            root,
            index_path,
            pins=pins,
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_rejects_member_size_limit(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    with pytest.raises(ValueError, match="member exceeds size limit"):
        build_candidates(
            root,
            index_path,
            pins=pins,
            limits=SafetyLimits(max_member_bytes=5),
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_varex_fields_obey_member_size_limit(tmp_path: Path) -> None:
    root, _, pins, adapter = _fixture_source(tmp_path)

    with pytest.raises(ValueError, match="image field exceeds size limit"):
        _varex_candidates(
            root,
            pins,
            adapter,
            SafetyLimits(max_member_bytes=8),
        )


def test_rejects_special_zip_members_even_when_unrelated(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    with zipfile.ZipFile(root / AI2D_IMAGES, "a") as archive:
        info = zipfile.ZipInfo("unrelated-link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target")
    pins = SourcePins(
        pub_images_sha256=pins.pub_images_sha256,
        pub_tables_sha256=pins.pub_tables_sha256,
        varex_shards=pins.varex_shards,
        ai2d_images_sha256=_sha((root / AI2D_IMAGES).read_bytes()),
        ai2d_annotations_sha256=pins.ai2d_annotations_sha256,
        mws_metadata_sha256=pins.mws_metadata_sha256,
    )

    with pytest.raises(ValueError, match="special.*ZIP members"):
        build_candidates(
            root,
            index_path,
            pins=pins,
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_mws_image_index_obeys_file_size_limit(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)

    with pytest.raises(ValueError, match="image index exceeds file size limit"):
        build_candidates(
            root,
            index_path,
            pins=pins,
            limits=SafetyLimits(max_image_index_bytes=1),
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_mws_image_index_must_be_a_regular_file(tmp_path: Path) -> None:
    root, _, pins, adapter = _fixture_source(tmp_path)
    fifo = tmp_path / "mws-images.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="must be a regular file"):
        build_candidates(
            root,
            fifo,
            pins=pins,
            parquet_adapter=adapter,
            require_quotas=False,
        )


def test_default_generation_rejects_catalog_below_required_quotas(tmp_path: Path) -> None:
    root, index_path, pins, adapter = _fixture_source(tmp_path)
    with pytest.raises(ValueError, match="lacks quota capacity"):
        build_candidates(root, index_path, pins=pins, parquet_adapter=adapter)


def test_atomic_jsonl_output_is_deterministic_and_preserves_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates, _ = _generate(tmp_path)
    output = tmp_path / "candidates.jsonl"
    write_candidates(output, candidates)
    first = output.read_bytes()
    write_candidates(output, candidates)
    assert output.read_bytes() == first
    assert [json.loads(line) for line in first.splitlines()] == candidates

    output.write_bytes(b"old")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        write_candidates(output, candidates)
    assert output.read_bytes() == b"old"
    assert not list(tmp_path.glob(".candidates.jsonl.*.tmp"))
