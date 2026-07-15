from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from rag_app.api.auth import User
from rag_app.api.routes import documents, segments
from rag_app.db.models import DocumentKind, DocumentStatus, SegmentKind
from rag_app.workers import tasks
from rag_app.workers.main import WorkerSettings
from rag_app.workers.recovery import (
    STATUS_RECOVERY_GRACE_S,
    recover_stale_documents,
    stale_status_cutoff,
)


class _ScalarResult:
    def __init__(self, value: object = None, *, rowcount: int = 0) -> None:
        self.value = value
        self.rowcount = rowcount

    def scalar_one_or_none(self) -> object:
        return self.value

    def scalars(self) -> _ScalarResult:
        return self

    def all(self) -> list[object]:
        return list(self.value or [])


class _Session:
    def __init__(self, results: list[_ScalarResult] | None = None, *, document: object = None) -> None:
        self.results = list(results or [])
        self.document = document
        self.statements: list[object] = []
        self.commits = 0
        self.added: list[object] = []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    async def execute(self, statement: object, *args: object) -> _ScalarResult:
        self.statements.append(statement)
        return self.results.pop(0) if self.results else _ScalarResult()

    async def commit(self) -> None:
        self.commits += 1

    async def get(self, model: object, object_id: object) -> object:
        return self.document

    async def delete(self, obj: object) -> None:
        return None

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def add_all(self, objects: object) -> None:
        self.added.extend(objects)  # type: ignore[arg-type]


class _Arq:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def enqueue_job(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def _compiled(statement: object) -> tuple[str, dict[str, object]]:
    compiled = statement.compile(dialect=postgresql.dialect())  # type: ignore[attr-defined]
    return str(compiled), compiled.params


def _request(sessionmaker, arq: _Arq, *, storage: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(user=User(sub="user-a", username="user-a", roles={"user"})),
        app=SimpleNamespace(
            state=SimpleNamespace(
                sessionmaker=sessionmaker,
                arq=arq,
                storage=storage,
            )
        ),
    )


def test_parse_claim_reclaims_only_same_revision_and_parse_failures() -> None:
    session = _Session([_ScalarResult(7)])

    claimed = asyncio.run(tasks._claim_parse({"sessionmaker": lambda: session}, uuid.uuid4(), 7))

    assert claimed == 7
    sql_text, params = _compiled(session.statements[0])
    assert "documents.parse_revision =" in sql_text
    assert "documents.status IN" in sql_text
    assert "documents.error LIKE" in sql_text
    reclaim_statuses = next(
        value
        for value in params.values()
        if isinstance(value, list) and DocumentStatus.uploaded in value
    )
    assert reclaim_statuses == [DocumentStatus.uploaded, DocumentStatus.parsing]
    assert DocumentStatus.error in params.values()
    assert "парсинг:%" in params.values()


def test_legacy_parse_claim_does_not_reclaim_in_progress_document() -> None:
    session = _Session([_ScalarResult(None)])

    asyncio.run(tasks._claim_parse({"sessionmaker": lambda: session}, uuid.uuid4(), None))

    sql_text, params = _compiled(session.statements[0])
    assert "documents.parse_revision =" not in sql_text
    assert "documents.status =" in sql_text
    assert DocumentStatus.uploaded in params.values()


def test_downstream_stage_claim_rejects_stale_revision() -> None:
    session = _Session([_ScalarResult(None)])

    claimed = asyncio.run(
        tasks._claim_document_stage(
            {"sessionmaker": lambda: session},
            uuid.uuid4(),
            9,
            ready=DocumentStatus.parsed,
            running=DocumentStatus.translating,
            error_prefix="перевод:",
        )
    )

    assert claimed is False
    sql_text, params = _compiled(session.statements[0])
    assert "documents.parse_revision =" in sql_text
    assert "documents.status IN" in sql_text
    assert "documents.error LIKE" in sql_text
    assert 9 in params.values()
    assert "перевод:%" in params.values()


def test_stale_cutoff_is_later_than_arq_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks.settings, "job_timeout_s", 120)
    now = datetime(2026, 7, 15, tzinfo=UTC)

    assert stale_status_cutoff(now) == now - timedelta(seconds=120 + STATUS_RECOVERY_GRACE_S)


def test_watchdog_marks_documents_and_additional_translations_error() -> None:
    session = _Session([_ScalarResult(rowcount=3), _ScalarResult(rowcount=2)])

    result = asyncio.run(recover_stale_documents({"sessionmaker": lambda: session}))

    assert result == {"documents": 3, "translations": 2}
    assert session.commits == 1
    doc_sql, doc_params = _compiled(session.statements[0])
    translation_sql, translation_params = _compiled(session.statements[1])
    assert "documents.updated_at <" in doc_sql
    assert DocumentStatus.error in doc_params.values()
    assert "document_translations.updated_at <" in translation_sql
    assert "error" in translation_params.values()


def test_watchdog_is_registered_on_startup_and_every_five_minutes() -> None:
    watchdog = next(job for job in WorkerSettings.cron_jobs if job.coroutine is recover_stale_documents)

    assert watchdog.run_at_startup is True
    assert watchdog.minute == set(range(0, 60, 5))
    assert recover_stale_documents in WorkerSettings.functions


def test_parse_cancelled_error_is_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid.uuid4()
    status = AsyncMock()

    async def claim(ctx: dict, value: uuid.UUID, revision: int | None) -> int:
        return 4

    async def get_doc(ctx: dict, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(
            filename="cancelled.txt",
            s3_key_original="original",
            owner_sub="user-a",
            parser_backend=None,
            parse_force_ocr=False,
        )

    class _Storage:
        async def download_to(self, bucket: str, key: str, path: object) -> None:
            raise asyncio.CancelledError

    monkeypatch.setattr(tasks, "_claim_parse", claim)
    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks, "_set_status", status)
    monkeypatch.setattr(tasks, "page_router_allowed", lambda *args, **kwargs: False)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            tasks.parse_document(
                {"storage": _Storage(), "sessionmaker": lambda: _Session()}, str(doc_id), 4
            )
        )

    assert status.await_args.args[2] == DocumentStatus.error
    assert status.await_args.kwargs["parse_revision"] == 4
    assert "отменена" in status.await_args.args[3]


def test_translate_cancelled_error_is_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid.uuid4()
    status = AsyncMock()
    segment = SimpleNamespace(
        id=uuid.uuid4(),
        idx=0,
        kind=SegmentKind.paragraph,
        source_text="Текст",
        translated_text=None,
    )

    async def get_doc(ctx: dict, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(
            source_lang="ru",
            target_lang="ru",
            filename="cancelled.txt",
            kind=DocumentKind.text,
        )

    async def cancel(*args: object, **kwargs: object) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks, "_set_status", status)
    monkeypatch.setattr(tasks, "_translate_segment", cancel)
    session = _Session([_ScalarResult([segment])])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            tasks.translate_document(
                {
                    "translator": object(),
                    "sessionmaker": lambda: session,
                },
                str(doc_id),
            )
        )

    assert status.await_args.args[2] == DocumentStatus.error
    assert "отменена" in status.await_args.args[3]


def test_additional_translation_cancelled_error_is_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    set_translation_status = AsyncMock()
    segment = SimpleNamespace(
        id=uuid.uuid4(),
        idx=0,
        kind=SegmentKind.paragraph,
        source_text="Текст",
        translated_text=None,
    )

    async def get_doc(ctx: dict, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(source_lang="ru")

    async def cancel(*args: object, **kwargs: object) -> object:
        raise asyncio.CancelledError

    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks, "_set_translation_status", set_translation_status)
    monkeypatch.setattr(tasks, "_translate_segment", cancel)
    session = _Session([_ScalarResult([segment])])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            tasks.translate_to_language(
                {"translator": object(), "sessionmaker": lambda: session}, str(doc_id), "en"
            )
        )

    assert set_translation_status.await_args.args[3] == "error"
    assert "отменена" in set_translation_status.await_args.kwargs["error"]


def test_export_cancelled_error_is_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid.uuid4()
    status = AsyncMock()

    async def get_doc(ctx: dict, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(filename="cancelled.txt", kind=DocumentKind.text)

    class _CancelledSession(_Session):
        async def execute(self, statement: object, *args: object) -> _ScalarResult:
            raise asyncio.CancelledError

    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks, "_set_status", status)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            tasks.export_document(
                {
                    "storage": object(),
                    "sessionmaker": lambda: _CancelledSession(),
                },
                str(doc_id),
            )
        )

    assert status.await_args.args[2] == DocumentStatus.error
    assert "отменена" in status.await_args.args[3]


def test_parse_enqueue_refusal_sets_error_and_returns_503() -> None:
    session = _Session()
    request = _request(lambda: session, _Arq(result=None))

    with pytest.raises(HTTPException) as caught:
        asyncio.run(documents._enqueue_parse_job(request, uuid.uuid4(), 8))

    assert caught.value.status_code == 503
    sql_text, params = _compiled(session.statements[0])
    assert "documents.parse_revision =" in sql_text
    assert DocumentStatus.error in params.values()


def test_parse_stage_enqueue_refusal_moves_document_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    status = AsyncMock()
    session = _Session([_ScalarResult(), _ScalarResult(rowcount=1)])

    async def claim(ctx: dict, value: uuid.UUID, revision: int | None) -> int:
        return 5

    async def get_doc(ctx: dict, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(
            filename="document.txt",
            s3_key_original="original",
            owner_sub="user-a",
            parser_backend=None,
            parse_force_ocr=False,
        )

    class _Storage:
        async def download_to(self, bucket: str, key: str, path: object) -> None:
            path.write_text("Первый абзац", encoding="utf-8")  # type: ignore[attr-defined]

    monkeypatch.setattr(tasks, "_claim_parse", claim)
    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks, "_set_status", status)
    monkeypatch.setattr(tasks, "page_router_allowed", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="отклонила задачу перевода"):
        asyncio.run(
            tasks.parse_document(
                {
                    "storage": _Storage(),
                    "sessionmaker": lambda: session,
                    "redis": _Arq(result=None),
                },
                str(doc_id),
                5,
            )
        )

    assert status.await_args.args[2] == DocumentStatus.error
    assert "очередь отклонила" in status.await_args.args[3]
    assert status.await_args.kwargs["parse_revision"] == 5


def test_translate_stage_enqueue_refusal_moves_document_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    status = AsyncMock()

    async def get_doc(ctx: dict, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(
            source_lang="ru",
            target_lang="ru",
            filename="document.txt",
            kind=DocumentKind.text,
        )

    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks, "_set_status", status)

    with pytest.raises(RuntimeError, match="отклонила задачу экспорта"):
        asyncio.run(
            tasks.translate_document(
                {
                    "translator": object(),
                    "sessionmaker": lambda: _Session([_ScalarResult([])]),
                    "redis": _Arq(result=None),
                },
                str(doc_id),
            )
        )

    assert status.await_args.args[2] == DocumentStatus.error
    assert "очередь отклонила" in status.await_args.args[3]


def test_reexport_refusal_restores_previous_status() -> None:
    doc_id = uuid.uuid4()
    doc = SimpleNamespace(
        status=DocumentStatus.done,
        error="old",
        owner_sub="user-a",
        parse_revision=6,
    )
    session = _Session([_ScalarResult(rowcount=1)], document=doc)
    request = _request(lambda: session, _Arq(result=None))

    with pytest.raises(HTTPException) as caught:
        asyncio.run(segments.reexport_document(request, doc_id))

    assert caught.value.status_code == 503
    assert session.commits == 2
    assert request.app.state.arq.calls[0][0] == ("export_document", str(doc_id), 6)
    _, params = _compiled(session.statements[-1])
    assert DocumentStatus.done in params.values()
    assert "old" in params.values()


def test_additional_translation_uses_unique_job_id_and_restores_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    row = SimpleNamespace(
        status="done",
        error=None,
        data={"segment": {"text": "old"}},
        segment_count=2,
        translated_count=2,
        needs_review_count=1,
        s3_key_docx="old.docx",
        s3_key_source="old.src",
        updated_at=datetime.now(UTC),
    )
    first = _Session([_ScalarResult(row)])
    rollback = _Session([_ScalarResult(row)])
    sessions = iter((first, rollback))
    arq = _Arq(result=None)
    request = _request(lambda: next(sessions), arq)

    async def get_doc(request: object, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(source_lang="ru", status=DocumentStatus.done)

    monkeypatch.setattr(documents, "_get_or_404", get_doc)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(documents.create_translation(request, doc_id, documents.TranslationIn(target_lang="en")))

    assert caught.value.status_code == 503
    job_id = arq.calls[0][1]["_job_id"]
    assert isinstance(job_id, str)
    assert job_id.startswith(f"translate_lang:{doc_id}:en:")
    assert row.status == "done"
    assert row.data == {"segment": {"text": "old"}}
    assert row.s3_key_docx == "old.docx"


def test_active_additional_translation_cannot_be_started_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid.uuid4()
    row = SimpleNamespace(status="translating", updated_at=datetime.now(UTC))
    session = _Session([_ScalarResult(row)])
    arq = _Arq(result=object())
    request = _request(lambda: session, arq)

    async def get_doc(request: object, value: uuid.UUID) -> SimpleNamespace:
        return SimpleNamespace(source_lang="ru", status=DocumentStatus.done)

    monkeypatch.setattr(documents, "_get_or_404", get_doc)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(documents.create_translation(request, doc_id, documents.TranslationIn(target_lang="en")))

    assert caught.value.status_code == 409
    assert arq.calls == []


def test_delete_document_cleans_complete_export_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid.uuid4()
    doc = SimpleNamespace(
        filename="doc.pdf",
        owner_sub="user-a",
        s3_key_original=f"{doc_id}/original.pdf",
        s3_key_content_list=f"{doc_id}/content_list.json",
        s3_key_export_docx=None,
        s3_key_export_pdf=None,
        s3_key_export_pdf_dual=None,
        s3_key_export_source=None,
    )

    class _Storage:
        def __init__(self) -> None:
            self.prefixes: list[tuple[str, uuid.UUID]] = []

        async def remove_object(self, bucket: str, key: str) -> None:
            return None

        async def remove_document_objects(self, bucket: str, value: uuid.UUID) -> int:
            self.prefixes.append((bucket, value))
            return 0

    storage = _Storage()
    session = _Session(document=None)
    request = _request(lambda: session, _Arq(), storage=storage)
    request.app.state.memory = SimpleNamespace(
        scope_for=lambda *args, **kwargs: None,
    )

    async def get_doc(request: object, value: uuid.UUID) -> SimpleNamespace:
        return doc

    monkeypatch.setattr(documents, "_get_or_404", get_doc)
    monkeypatch.setattr(documents, "audit", AsyncMock())

    asyncio.run(documents.delete_document(request, doc_id))

    assert (documents.settings.bucket_artifacts, doc_id) in storage.prefixes
    assert (documents.settings.bucket_exports, doc_id) in storage.prefixes
