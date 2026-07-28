from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from rag_app.api.routes import segments as segment_routes
from rag_app.db.models import (
    Document,
    DocumentKind,
    Segment,
    SegmentKind,
    document_segment_filter,
    is_document_segment,
)
from rag_app.rag.chunking import segments_to_chunks
from rag_app.workers import tasks


class _Result:
    def __init__(self, value: object = None, *, rowcount: int = 0) -> None:
        self.value = value
        self.rowcount = rowcount

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[object]:
        return list(self.value or [])

    def scalar_one(self) -> object:
        return self.value

    def scalar_one_or_none(self) -> object:
        return self.value


class _Session:
    def __init__(
        self,
        results: list[_Result] | None = None,
        *,
        document: object | None = None,
        segment: object | None = None,
    ) -> None:
        self.results = list(results or [])
        self.document = document
        self.segment = segment
        self.statements: list[object] = []
        self.added: list[object] = []
        self.commits = 0

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    async def get(self, model: object, key: object) -> object | None:
        if model is Document:
            return self.document
        if model is Segment:
            return self.segment
        return None

    async def execute(self, statement: object, *args: object) -> _Result:
        self.statements.append(statement)
        return self.results.pop(0) if self.results else _Result()

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1


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


def _segment(
    document_id: uuid.UUID,
    *,
    idx: int,
    source_text: str,
    translated_text: str | None,
    meta: dict[str, object] | None = None,
    kind: SegmentKind = SegmentKind.paragraph,
) -> Segment:
    return Segment(
        id=uuid.uuid4(),
        document_id=document_id,
        idx=idx,
        page_idx=0,
        kind=kind,
        heading_level=None,
        source_text=source_text,
        translated_text=translated_text,
        needs_review=False,
        validation=None,
        meta=meta or {},
    )


def _compiled(statement: object) -> tuple[str, dict[str, object]]:
    compiled = statement.compile(dialect=postgresql.dialect())  # type: ignore[attr-defined]
    return str(compiled), compiled.params


def test_public_segments_api_hides_internal_vl_context() -> None:
    document_id = uuid.uuid4()
    public = _segment(
        document_id,
        idx=0,
        source_text="MEETING TOPICS",
        translated_text="ТЕМЫ ВСТРЕЧИ",
    )
    internal = _segment(
        document_id,
        idx=1,
        source_text="Внутреннее описание страницы",
        translated_text=None,
        meta={"vl_describe": True},
        kind=SegmentKind.image,
    )
    session = _Session(
        [_Result([public, internal])],
        document=SimpleNamespace(owner_sub="ruslan"),
        segment=internal,
    )
    request = SimpleNamespace(
        state=SimpleNamespace(user=SimpleNamespace(sub="ruslan", is_admin=False)),
        app=SimpleNamespace(state=SimpleNamespace(sessionmaker=lambda: session)),
    )

    response = asyncio.run(segment_routes.list_segments(request, document_id, limit=4000))

    assert [item.id for item in response] == [public.id]
    sql, params = _compiled(session.statements[0])
    assert "IS DISTINCT FROM" in sql
    assert "vl_describe" in params.values()
    assert is_document_segment(public)
    assert not is_document_segment(internal)
    filter_sql, _ = _compiled(document_segment_filter())
    assert "IS DISTINCT FROM" in filter_sql

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            segment_routes._segment_or_404(
                session,
                internal.id,
                request.state.user,
            )
        )
    assert exc_info.value.status_code == 404


def test_vl_context_is_indexable_but_not_counted_as_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.uuid4()
    session = _Session(
        [
            _Result(6),
            _Result(),
            _Result(35),
            _Result(35),
            _Result(35),
            _Result(rowcount=1),
        ]
    )
    redis = _Arq()
    vision = SimpleNamespace(describe=AsyncMock(return_value="Схема установки и её узлы"))
    monkeypatch.setattr(tasks.settings, "vl_enabled", True)
    monkeypatch.setattr(
        tasks,
        "_get_doc",
        AsyncMock(
            return_value=SimpleNamespace(
                parse_revision=6,
                kind=DocumentKind.pdf_scan.value,
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
            str(document_id),
            6,
        )
    )

    assert result == "vl: 1 описаний на 1 стр."
    assert len(session.added) == 1
    description = session.added[0]
    assert isinstance(description, Segment)
    assert description.meta == {"vl_describe": True}
    assert description.translated_text is None
    assert not is_document_segment(description)

    count_sql, _ = _compiled(session.statements[3])
    translated_count_sql, _ = _compiled(session.statements[4])
    assert "IS DISTINCT FROM" in count_sql
    assert "IS DISTINCT FROM" in translated_count_sql
    update_sql, update_params = _compiled(session.statements[5])
    assert "segment_count" in update_sql
    assert "translated_count" in update_sql
    assert list(update_params.values()).count(35) >= 2

    description.id = uuid.uuid4()
    chunks = segments_to_chunks([description])
    assert len(chunks) == 1
    assert chunks[0].kind == "image"
    assert chunks[0].text_en == "Схема установки и её узлы"
    assert chunks[0].text_ru == "Схема установки и её узлы"
    assert redis.calls[0][0][:3] == ("index_document", str(document_id), 6)


def test_primary_export_excludes_internal_vl_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = uuid.uuid4()
    public = _segment(
        document_id,
        idx=0,
        source_text="MEETING TOPICS",
        translated_text="ТЕМЫ ВСТРЕЧИ",
    )
    internal = _segment(
        document_id,
        idx=1,
        source_text="Внутреннее описание страницы",
        translated_text=None,
        meta={"vl_describe": True},
        kind=SegmentKind.image,
    )
    document = SimpleNamespace(
        id=document_id,
        filename="scan.txt",
        kind=DocumentKind.text,
        s3_key_original="original/scan.txt",
    )
    session = _Session([_Result([public, internal]), _Result(rowcount=1)])
    build_docx = Mock(return_value=b"docx")
    monkeypatch.setattr(tasks, "_get_doc", AsyncMock(return_value=document))
    monkeypatch.setattr(tasks, "_claim_document_stage", AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, "build_docx", build_docx)

    result = asyncio.run(
        tasks.export_document(
            {
                "sessionmaker": lambda: session,
                "storage": _Storage(),
                "redis": _Arq(),
            },
            str(document_id),
            6,
        )
    )

    assert result == "exported: s3_key_export_docx"
    assert build_docx.call_args.args[1] == [public]
