from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import postgresql

from rag_app.db.models import DocumentKind, DocumentStatus, SegmentKind
from rag_app.pipeline.segments import SegmentDraft
from rag_app.workers import tasks


class _ScalarResult:
    def __init__(self, value: object = None, *, rowcount: int = 0) -> None:
        self.value = value
        self.rowcount = rowcount

    def scalars(self) -> _ScalarResult:
        return self

    def scalar_one_or_none(self) -> object:
        return self.value

    def all(self) -> list[object]:
        return list(self.value or [])


class _Session:
    def __init__(self, results: list[_ScalarResult] | None = None) -> None:
        self.results = list(results or [])
        self.statements: list[object] = []
        self.added: list[object] = []
        self.commits = 0

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    async def execute(self, statement: object, *args: object) -> _ScalarResult:
        self.statements.append(statement)
        return self.results.pop(0) if self.results else _ScalarResult()

    async def commit(self) -> None:
        self.commits += 1

    def add_all(self, objects: object) -> None:
        self.added.extend(objects)  # type: ignore[arg-type]


class _Storage:
    def __init__(self) -> None:
        self.puts: list[tuple[object, ...]] = []

    async def download_to(self, bucket: str, key: str, path: Path) -> None:
        path.write_bytes(b"%PDF-1.7\n")

    async def put_bytes(self, *args: object, **kwargs: object) -> None:
        self.puts.append((*args, kwargs))


class _Arq:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def enqueue_job(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return object()


def _scan_doc(doc_id: uuid.UUID, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=doc_id,
        filename="scan.pdf",
        s3_key_original="original/scan.pdf",
        owner_sub="user-a",
        parser_backend=backend,
        parse_force_ocr=False,
        ocr_lang="east_slavic",
    )


def _disable_quality_router(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks.settings, "parser_quality_shadow_enabled", False)
    monkeypatch.setattr(tasks, "page_router_allowed", lambda *args, **kwargs: False)


def _compiled(statement: object) -> tuple[str, dict[str, object]]:
    compiled = statement.compile(dialect=postgresql.dialect())  # type: ignore[attr-defined]
    return str(compiled), compiled.params


def test_image_only_scan_uses_explicit_paddle_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    session = _Session([_ScalarResult(), _ScalarResult(), _ScalarResult(rowcount=1)])
    storage = _Storage()
    redis = _Arq()
    draft = SegmentDraft(
        idx=0,
        kind=SegmentKind.paragraph,
        source_text="MEETING TOPICS",
        page_idx=0,
    )
    vlm_segments = AsyncMock(return_value=[draft])
    mineru = AsyncMock()

    monkeypatch.setattr(tasks, "_claim_parse", AsyncMock(return_value=7))
    monkeypatch.setattr(tasks, "_get_doc", AsyncMock(return_value=_scan_doc(doc_id, "paddle_vl")))
    monkeypatch.setattr(tasks, "_vlm_segments", vlm_segments)
    monkeypatch.setattr(tasks, "run_mineru", mineru)
    monkeypatch.setattr(tasks, "pdf_info", lambda path: (1, False))
    monkeypatch.setattr(tasks, "_upload_segment_images", AsyncMock())
    _disable_quality_router(monkeypatch)

    result = asyncio.run(
        tasks.parse_document(
            {
                "storage": storage,
                "sessionmaker": lambda: session,
                "redis": redis,
            },
            str(doc_id),
            7,
        )
    )

    assert result == "parsed [pdf_scan]: 1 segments, 1 pages"
    assert vlm_segments.await_args.args[0] == "paddle_vl"
    mineru.assert_not_awaited()
    assert len(session.added) == 1
    assert session.added[0].kind == SegmentKind.paragraph
    _, update_params = _compiled(session.statements[2])
    assert DocumentKind.pdf_scan.value in update_params.values()
    assert redis.calls[0][0][:3] == ("translate_document", str(doc_id), 7)


def test_office_parse_always_uses_native_ooxml_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    session = _Session([_ScalarResult(), _ScalarResult(), _ScalarResult(rowcount=1)])
    storage = _Storage()
    redis = _Arq()
    document = SimpleNamespace(
        id=doc_id,
        filename="annex.docx",
        s3_key_original="original/annex.docx",
        owner_sub="user-a",
        parser_backend="paddle_vl",
        parse_force_ocr=False,
        ocr_lang="east_slavic",
    )
    draft = SegmentDraft(
        idx=0,
        kind=SegmentKind.paragraph,
        source_text="MEETING TOPICS",
        meta={"location": {"p": 0}},
    )
    extract = Mock(return_value=[draft])
    vlm_segments = AsyncMock()

    monkeypatch.setattr(tasks, "_claim_parse", AsyncMock(return_value=10))
    monkeypatch.setattr(tasks, "_get_doc", AsyncMock(return_value=document))
    monkeypatch.setattr(tasks.ooxml, "extract", extract)
    monkeypatch.setattr(tasks, "_vlm_segments", vlm_segments)
    monkeypatch.setattr(tasks, "_upload_segment_images", AsyncMock())

    result = asyncio.run(
        tasks.parse_document(
            {
                "storage": storage,
                "sessionmaker": lambda: session,
                "redis": redis,
            },
            str(doc_id),
            10,
        )
    )

    assert result == "parsed [docx]: 1 segments, None pages"
    extract.assert_called_once()
    assert extract.call_args.args[0] == "docx"
    assert isinstance(extract.call_args.args[1], Path)
    assert isinstance(extract.call_args.args[2], Path)
    vlm_segments.assert_not_awaited()
    _, update_params = _compiled(session.statements[2])
    assert DocumentKind.docx.value in update_params.values()
    assert "native_ooxml" in update_params.values()
    assert redis.calls[0][0][:3] == ("translate_document", str(doc_id), 10)


def test_ooxml_export_fails_when_no_translation_is_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage()
    inject = Mock(return_value=0)
    monkeypatch.setattr(tasks.ooxml, "inject", inject)
    document = SimpleNamespace(
        id=uuid.uuid4(),
        filename="annex.docx",
        kind=DocumentKind.docx,
        s3_key_original="original/annex.docx",
    )
    segments = [
        SimpleNamespace(
            source_text="Meeting topics",
            translated_text="Темы встречи",
            meta={"location": {"p": 0}},
        )
    ]

    with pytest.raises(RuntimeError, match="не применил ни одного перевода"):
        asyncio.run(
            tasks._export_translation(
                {"storage": storage},
                document,
                segments,
                "ru",
            )
        )

    inject.assert_called_once()
    assert storage.puts == []


def test_primary_ooxml_export_fails_before_upload_when_nothing_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    storage = _Storage()
    document = SimpleNamespace(
        id=doc_id,
        filename="annex.docx",
        kind=DocumentKind.docx,
        s3_key_original="original/annex.docx",
    )
    segment = SimpleNamespace(
        source_text="Meeting topics",
        translated_text="Темы встречи",
        meta={"location": {"p": 0}},
    )
    session = _Session([_ScalarResult([segment])])
    inject = Mock(return_value=0)
    status = AsyncMock()
    monkeypatch.setattr(tasks, "_get_doc", AsyncMock(return_value=document))
    monkeypatch.setattr(tasks, "_claim_document_stage", AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, "_set_status", status)
    monkeypatch.setattr(tasks.ooxml, "inject", inject)

    with pytest.raises(RuntimeError, match="не применил ни одного перевода"):
        asyncio.run(
            tasks.export_document(
                {
                    "storage": storage,
                    "sessionmaker": lambda: session,
                    "redis": _Arq(),
                },
                str(doc_id),
                4,
            )
        )

    inject.assert_called_once()
    assert storage.puts == []
    assert status.await_args.args[2] == DocumentStatus.error
    assert "не применил ни одного перевода" in status.await_args.args[3]


def test_empty_scan_result_fails_without_overwriting_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    session = _Session()
    storage = _Storage()
    redis = _Arq()
    set_status = AsyncMock()

    monkeypatch.setattr(tasks, "_claim_parse", AsyncMock(return_value=8))
    monkeypatch.setattr(tasks, "_get_doc", AsyncMock(return_value=_scan_doc(doc_id, "paddle_vl")))
    monkeypatch.setattr(tasks, "_vlm_segments", AsyncMock(return_value=[]))
    monkeypatch.setattr(tasks, "pdf_info", lambda path: (1, False))
    monkeypatch.setattr(tasks, "_upload_segment_images", AsyncMock())
    monkeypatch.setattr(tasks, "_set_status", set_status)
    _disable_quality_router(monkeypatch)

    with pytest.raises(RuntimeError, match="OCR-парсер не извлёк"):
        asyncio.run(
            tasks.parse_document(
                {
                    "storage": storage,
                    "sessionmaker": lambda: session,
                    "redis": redis,
                },
                str(doc_id),
                8,
            )
        )

    assert session.statements == []
    assert session.added == []
    assert storage.puts == []
    assert redis.calls == []
    assert set_status.await_args.args[2] == DocumentStatus.error
    assert set_status.await_args.kwargs["parse_revision"] == 8


def test_empty_mineru_scan_retries_with_forced_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    session = _Session([_ScalarResult(), _ScalarResult(), _ScalarResult(rowcount=1)])
    storage = _Storage()
    redis = _Arq()
    draft = SegmentDraft(
        idx=0,
        kind=SegmentKind.paragraph,
        source_text="Recognized scan text",
        page_idx=0,
    )
    mineru_calls: list[dict[str, object]] = []

    async def run_mineru(
        input_pdf: Path,
        out_dir: Path,
        **kwargs: object,
    ) -> Path:
        mineru_calls.append(kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = out_dir / "scan_content_list.json"
        result.write_text("[]", encoding="utf-8")
        return result

    class _Geometry:
        page_sizes = {0: (595.0, 842.0)}

        def pop_typed(self, page_idx: int | None, kind: str) -> None:
            return None

        def match_text(self, page_idx: int | None, text: str) -> None:
            return None

        def reflow(self, page_idx: int | None, text: str) -> None:
            return None

    monkeypatch.setattr(tasks, "_claim_parse", AsyncMock(return_value=9))
    monkeypatch.setattr(tasks, "_get_doc", AsyncMock(return_value=_scan_doc(doc_id, "mineru")))
    monkeypatch.setattr(tasks, "pdf_info", lambda path: (1, False))
    monkeypatch.setattr(tasks, "run_mineru", run_mineru)
    monkeypatch.setattr(tasks, "load_content_list", Mock(side_effect=[[], []]))
    monkeypatch.setattr(
        tasks,
        "content_list_to_segments",
        Mock(side_effect=[[], [draft]]),
    )
    monkeypatch.setattr(tasks, "load_block_geometry", lambda path: _Geometry())
    monkeypatch.setattr(tasks, "_upload_segment_images", AsyncMock())
    _disable_quality_router(monkeypatch)

    result = asyncio.run(
        tasks.parse_document(
            {
                "storage": storage,
                "sessionmaker": lambda: session,
                "redis": redis,
            },
            str(doc_id),
            9,
        )
    )

    assert result == "parsed [pdf_scan]: 1 segments, 1 pages"
    assert len(mineru_calls) == 2
    assert mineru_calls[0] == {}
    assert mineru_calls[1]["method"] == "ocr"
    assert mineru_calls[1]["lang"] == "east_slavic"


def test_index_failure_removes_chunks_from_previous_parse_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    session = _Session([_ScalarResult([object()]), _ScalarResult(4)])
    monkeypatch.setattr(tasks, "segments_to_chunks", lambda segments: [])

    with pytest.raises(RuntimeError, match="нет чанков"):
        asyncio.run(
            tasks.index_document(
                {
                    "sessionmaker": lambda: session,
                    "embedder": object(),
                },
                str(doc_id),
                4,
            )
        )

    assert len(session.statements) == 4
    delete_sql, _ = _compiled(session.statements[2])
    update_sql, update_params = _compiled(session.statements[3])
    assert delete_sql.startswith("DELETE FROM chunks")
    assert "chunk_count" in update_sql
    assert "indexed_at" in update_sql
    assert 0 in update_params.values()
    assert "нет чанков (документ пуст?)" in update_params.values()
    assert session.commits == 1


def test_post_index_enrichment_failure_keeps_committed_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    segment = SimpleNamespace(meta={})
    draft = SimpleNamespace(
        idx=0,
        kind="paragraph",
        heading_path=[],
        page_start=0,
        page_end=0,
        text_en="Meeting topics",
        text_ru="Темы встречи",
        meta={},
    )
    session = _Session([_ScalarResult([segment]), _ScalarResult(4)])
    embedder = SimpleNamespace(embed=AsyncMock(side_effect=[[[0.1, 0.2]], [[0.3, 0.4]]]))
    monkeypatch.setattr(tasks, "segments_to_chunks", lambda segments: [draft])
    monkeypatch.setattr(tasks.settings, "vl_enabled", True)
    monkeypatch.setattr(tasks.settings, "visual_enabled", False)
    monkeypatch.setattr(
        tasks,
        "_get_doc",
        AsyncMock(side_effect=RuntimeError("temporary document lookup failure")),
    )

    result = asyncio.run(
        tasks.index_document(
            {
                "sessionmaker": lambda: session,
                "embedder": embedder,
                "redis": _Arq(),
            },
            str(doc_id),
            4,
        )
    )

    assert result == "indexed: 1 chunks"
    deletes = [
        statement
        for statement in session.statements
        if _compiled(statement)[0].startswith("DELETE FROM chunks")
    ]
    assert len(deletes) == 1
    assert len(session.added) == 1
    assert session.commits == 1


def test_legacy_index_job_without_revision_is_safely_skipped() -> None:
    doc_id = uuid.uuid4()

    result = asyncio.run(tasks.index_document({}, str(doc_id)))

    assert result == "skipped index: missing parse_revision"


def test_stale_index_revision_does_not_replace_current_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    segment = SimpleNamespace(meta={})
    draft = SimpleNamespace(
        idx=0,
        kind="paragraph",
        heading_path=[],
        page_start=0,
        page_end=0,
        text_en="Old revision",
        text_ru="Старая ревизия",
        meta={},
    )
    session = _Session([_ScalarResult([segment]), _ScalarResult(12)])
    embedder = SimpleNamespace(embed=AsyncMock(side_effect=[[[0.1, 0.2]], [[0.3, 0.4]]]))
    monkeypatch.setattr(tasks, "segments_to_chunks", lambda segments: [draft])

    result = asyncio.run(
        tasks.index_document(
            {
                "sessionmaker": lambda: session,
                "embedder": embedder,
                "redis": _Arq(),
            },
            str(doc_id),
            11,
        )
    )

    assert result == "skipped index revision=11: current=12"
    assert session.added == []
    assert all(
        not _compiled(statement)[0].startswith("DELETE FROM chunks") for statement in session.statements
    )
    assert session.commits == 0


def test_failed_stale_index_revision_does_not_delete_newer_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    segment = SimpleNamespace(meta={})
    draft = SimpleNamespace(text_en="Old revision", text_ru="Старая ревизия")
    session = _Session([_ScalarResult([segment]), _ScalarResult(12)])
    embedder = SimpleNamespace(embed=AsyncMock(side_effect=RuntimeError("embedding endpoint unavailable")))
    monkeypatch.setattr(tasks, "segments_to_chunks", lambda segments: [draft])

    result = asyncio.run(
        tasks.index_document(
            {
                "sessionmaker": lambda: session,
                "embedder": embedder,
                "redis": _Arq(),
            },
            str(doc_id),
            11,
        )
    )

    assert result == "skipped failed index revision=11: current=12"
    cleanup_select, _ = _compiled(session.statements[1])
    assert "FOR UPDATE" in cleanup_select
    assert all(
        not _compiled(statement)[0].startswith("DELETE FROM chunks") for statement in session.statements
    )
    assert session.commits == 0


def test_legacy_describe_images_job_without_revision_is_safely_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    monkeypatch.setattr(tasks.settings, "vl_enabled", True)

    result = asyncio.run(tasks.describe_images({}, str(doc_id)))

    assert result == "skipped VL: missing parse_revision"


def test_stale_describe_images_revision_is_skipped_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    storage = _Storage()
    download = AsyncMock()
    storage.download_to = download
    monkeypatch.setattr(tasks.settings, "vl_enabled", True)
    monkeypatch.setattr(
        tasks,
        "_get_doc",
        AsyncMock(
            return_value=SimpleNamespace(
                parse_revision=12,
                kind=DocumentKind.pdf_scan,
                s3_key_original="original/scan.pdf",
            )
        ),
    )

    result = asyncio.run(
        tasks.describe_images(
            {"storage": storage},
            str(doc_id),
            11,
        )
    )

    assert result == "skipped VL revision=11: current=12"
    download.assert_not_awaited()


def test_describe_images_revision_change_during_inference_skips_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    session = _Session([_ScalarResult(12)])
    redis = _Arq()
    vision = SimpleNamespace(describe=AsyncMock(return_value="Описание страницы"))
    monkeypatch.setattr(tasks.settings, "vl_enabled", True)
    monkeypatch.setattr(
        tasks,
        "_get_doc",
        AsyncMock(
            return_value=SimpleNamespace(
                parse_revision=11,
                kind=DocumentKind.pdf_scan,
                s3_key_original="original/scan.pdf",
            )
        ),
    )
    monkeypatch.setattr(tasks, "_render_pdf_pages", lambda *args: [(0, b"png")])
    monkeypatch.setattr(tasks, "VisionClient", lambda: vision)

    result = asyncio.run(
        tasks.describe_images(
            {
                "sessionmaker": lambda: session,
                "storage": _Storage(),
                "redis": redis,
            },
            str(doc_id),
            11,
        )
    )

    assert result == "skipped VL revision=11: current=12"
    vision.describe.assert_awaited_once_with(b"png")
    assert len(session.statements) == 1
    lock_select, _ = _compiled(session.statements[0])
    assert "FOR UPDATE" in lock_select
    assert "documents.parse_revision" in lock_select
    assert session.added == []
    assert session.commits == 0
    assert redis.calls == []
