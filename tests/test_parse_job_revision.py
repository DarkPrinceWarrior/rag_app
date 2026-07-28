from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from rag_app.api.auth import User
from rag_app.api.routes.documents import _queue_reparse
from rag_app.db.models import Document, DocumentKind, DocumentStatus
from rag_app.workers.tasks import _claim_parse, _parser_backend


class _ScalarResult:
    def __init__(self, value: int | None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> int | None:
        return self.value


class _Session:
    def __init__(self, result: int | None) -> None:
        self.result = result
        self.statements: list[object] = []
        self.commits = 0

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    async def execute(self, statement: object) -> _ScalarResult:
        self.statements.append(statement)
        return _ScalarResult(self.result)

    async def commit(self) -> None:
        self.commits += 1


class _Arq:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def enqueue_job(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return object()


def _compiled(statement: object) -> tuple[str, dict[str, object]]:
    compiled = statement.compile(dialect=postgresql.dialect())  # type: ignore[attr-defined]
    return str(compiled), compiled.params


def test_worker_claim_is_guarded_by_status_and_revision() -> None:
    session = _Session(result=7)
    ctx = {"sessionmaker": lambda: session}
    doc_id = uuid.uuid4()

    claimed = asyncio.run(_claim_parse(ctx, doc_id, 7))

    assert claimed == 7
    assert session.commits == 1
    sql_text, params = _compiled(session.statements[0])
    assert "documents.status =" in sql_text
    assert "documents.parse_revision =" in sql_text
    reclaim_statuses = next(value for value in params.values() if isinstance(value, list))
    assert reclaim_statuses == [DocumentStatus.uploaded, DocumentStatus.parsing]
    assert DocumentStatus.error in params.values()
    assert "парсинг:%" in params.values()
    assert 7 in params.values()


def test_reparse_uses_monotonic_revision_as_job_identity() -> None:
    session = _Session(result=4)
    arq = _Arq()
    request = SimpleNamespace(
        state=SimpleNamespace(user=User(sub="user-a", username="user-a", roles={"user"})),
        app=SimpleNamespace(state=SimpleNamespace(sessionmaker=lambda: session, arq=arq)),
    )
    doc_id = uuid.uuid4()

    revision = asyncio.run(_queue_reparse(request, doc_id, parser_backend="mineru"))

    assert revision == 4
    assert session.commits == 1
    assert arq.calls == [
        (
            ("parse_document", str(doc_id), 4),
            {"_job_id": f"parse:{doc_id}:4"},
        )
    ]
    sql_text, params = _compiled(session.statements[0])
    assert "documents.status IN" in sql_text
    assert "documents.owner_sub =" in sql_text
    assert "documents.parse_revision +" in sql_text
    status_values = next(value for value in params.values() if isinstance(value, list))
    assert status_values == [DocumentStatus.error, DocumentStatus.done]


def test_legacy_dots_rows_are_normalized_without_backend_execution() -> None:
    scan = Document(kind=DocumentKind.pdf_scan, parser_backend="dots_mocr")
    text = Document(kind=DocumentKind.pdf_text, parser_backend="dots_mocr")

    assert _parser_backend(scan) == "paddle_vl"
    assert _parser_backend(text) == "mineru"
