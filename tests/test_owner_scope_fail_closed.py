"""App-level regressions for fail-closed owner scoping."""

from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from rag_app.api.auth import User
from rag_app.api.routes import chat as chat_routes
from rag_app.api.routes import documents as document_routes
from rag_app.api.routes import extract as extract_routes
from rag_app.api.routes import library as library_routes
from rag_app.api.routes import segments as segment_routes
from rag_app.api.routes.chat import ChatIn
from rag_app.api.routes.extract import ExtractIn
from rag_app.db.models import DocumentStatus
from rag_app.rag import retrieve as retrieve_module
from rag_app.rag.tools import AgentTools


class _ScalarValues:
    def __init__(self, values: tuple[Any, ...]) -> None:
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def all(self) -> list[Any]:
        return list(self._values)


class _Result:
    def __init__(self, *, scalar: Any = None, scalars: tuple[Any, ...] = ()) -> None:
        self._scalar = scalar
        self._scalars = scalars

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalars(self) -> _ScalarValues:
        return _ScalarValues(self._scalars)

    def all(self) -> list[Any]:
        return list(self._scalars)


class _Session:
    def __init__(
        self,
        *,
        get_value: Any = None,
        execute_results: tuple[_Result, ...] = (),
    ) -> None:
        self.get_value = get_value
        self.execute_results = list(execute_results)
        self.statements: list[Any] = []
        self.committed = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        if self.execute_results:
            return self.execute_results.pop(0)
        return _Result()

    async def get(self, _model: Any, _identifier: Any) -> Any:
        return self.get_value

    async def commit(self) -> None:
        self.committed = True


class _SessionMaker:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _Session:
        return self.session


class _Arq:
    def __init__(self) -> None:
        self.jobs: list[tuple[Any, ...]] = []

    async def enqueue_job(self, *args: Any, **kwargs: Any) -> None:
        self.jobs.append((*args, kwargs))


def _user(*, admin: bool = False) -> User:
    roles = {"admin"} if admin else {"user"}
    return User(sub="user-a", username="user-a", roles=roles)


def _request(session: _Session, *, admin: bool = False) -> Any:
    arq = _Arq()
    state = SimpleNamespace(
        sessionmaker=_SessionMaker(session),
        arq=arq,
        chat_engine=SimpleNamespace(client=object()),
        retriever=object(),
    )
    return SimpleNamespace(
        state=SimpleNamespace(user=_user(admin=admin)),
        app=SimpleNamespace(state=state),
    )


def _compiled(statement: Any) -> tuple[str, dict[str, Any]]:
    compiled = statement.compile(dialect=postgresql.dialect())
    return str(compiled), compiled.params


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_sub", [None, "user-b"])
async def test_queue_reparse_rejects_null_and_foreign_owner(owner_sub: str | None) -> None:
    document_id = uuid.uuid4()
    document = SimpleNamespace(owner_sub=owner_sub, status=DocumentStatus.done)
    session = _Session(get_value=document, execute_results=(_Result(scalar=None),))
    request = _request(session)

    with pytest.raises(HTTPException) as exc_info:
        await document_routes._queue_reparse(request, document_id, parser_backend="mineru")

    assert exc_info.value.status_code == 404
    assert session.committed is False
    assert request.app.state.arq.jobs == []
    sql, params = _compiled(session.statements[0])
    assert "documents.owner_sub =" in sql
    assert "owner_sub IS NULL" not in sql
    assert "user-a" in params.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_sub", [None, "user-b"])
async def test_reexport_rejects_null_and_foreign_owner(owner_sub: str | None) -> None:
    document = SimpleNamespace(owner_sub=owner_sub, status=DocumentStatus.done)
    session = _Session(get_value=document)
    request = _request(session)

    with pytest.raises(HTTPException) as exc_info:
        await segment_routes.reexport_document(request, uuid.uuid4())

    assert exc_info.value.status_code == 404
    assert session.statements == []
    assert session.committed is False
    assert request.app.state.arq.jobs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("admin", "expected_owner"), [(False, "user-a"), (True, None)])
async def test_extract_route_forwards_exact_owner_scope(
    monkeypatch: pytest.MonkeyPatch,
    admin: bool,
    expected_owner: str | None,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_extract_table(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"rows": []}

    monkeypatch.setattr(extract_routes, "extract_table", fake_extract_table)
    request = _request(_Session(), admin=admin)

    result = await extract_routes.extract_table_route(request, ExtractIn(query="specification"))

    assert result == {"rows": []}
    assert captured["owner_sub"] == expected_owner


@pytest.mark.parametrize("owner_sub", [None, "user-b"])
def test_chat_session_owner_check_rejects_null_and_foreign_owner(owner_sub: str | None) -> None:
    chat_session = SimpleNamespace(owner_sub=owner_sub)

    assert chat_routes._owner_ok(chat_session, _user()) is False
    assert chat_routes._owner_ok(chat_session, _user(admin=True)) is True


@pytest.mark.asyncio
async def test_chat_document_scope_uses_exact_owner_filter() -> None:
    document_id = uuid.uuid4()
    session = _Session(execute_results=(_Result(scalars=()),))
    request = _request(session)

    with pytest.raises(HTTPException) as exc_info:
        await chat_routes._validate_chat_scope(
            request,
            ChatIn(message="question", document_id=document_id),
            request.state.user,
        )

    assert exc_info.value.status_code == 404
    sql, params = _compiled(session.statements[0])
    assert "documents.owner_sub =" in sql
    assert "owner_sub IS NULL" not in sql
    assert "user-a" in params.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_sub", [None, "user-b"])
async def test_chat_folder_scope_rejects_null_and_foreign_owner(owner_sub: str | None) -> None:
    session = _Session(get_value=SimpleNamespace(owner_sub=owner_sub))
    request = _request(session)

    with pytest.raises(HTTPException) as exc_info:
        await chat_routes._validate_chat_scope(
            request,
            ChatIn(message="question", folder_id=uuid.uuid4()),
            request.state.user,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_agent_tools_catalog_uses_exact_owner_filter() -> None:
    session = _Session(execute_results=(_Result(scalars=()),))
    tools = AgentTools(
        _SessionMaker(session),
        retriever=object(),  # type: ignore[arg-type]
        document_id=None,
        folder_id=None,
        session_id=uuid.uuid4(),
        owner_sub="user-a",
    )

    assert await tools.list_documents() == "list_documents: в библиотеке нет готовых документов."
    sql, params = _compiled(session.statements[0])
    assert "documents.owner_sub =" in sql
    assert "owner_sub IS NULL" not in sql
    assert "user-a" in params.values()


@pytest.mark.asyncio
async def test_agent_tools_find_figure_stays_inside_document_set() -> None:
    selected = [uuid.uuid4(), uuid.uuid4()]
    outside = uuid.uuid4()
    session = _Session(execute_results=(_Result(scalars=()),))
    tools = AgentTools(
        _SessionMaker(session),
        retriever=object(),  # type: ignore[arg-type]
        document_id=None,
        document_ids=selected,
        folder_id=None,
        session_id=uuid.uuid4(),
        owner_sub="user-a",
    )

    await tools.find_figure("рисунок 3", str(outside))

    sql, params = _compiled(session.statements[0])
    assert "chunks.document_id =" in sql
    assert "chunks.document_id IN" in sql
    assert "documents.owner_sub =" in sql
    assert selected in params.values()
    assert outside in params.values()


def test_library_retriever_and_agent_sources_have_no_null_owner_allowance() -> None:
    for sql in (
        retrieve_module._SCOPE,
        retrieve_module._VISUAL_PAGES_SQL,
        retrieve_module._DENSE_SQL,
        retrieve_module._SPARSE_SQL,
    ):
        assert "d.owner_sub = :owner" in sql
        assert "d.owner_sub IS NULL" not in sql

    for source in (inspect.getsource(library_routes), inspect.getsource(AgentTools)):
        assert ".owner_sub.is_(None)" not in source
        assert "owner_sub IS NULL" not in source
