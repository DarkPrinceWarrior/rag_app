from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from rag_app.rag.retrieve import _SCOPE, _VISUAL_PAGES_SQL, Retriever
from rag_app.rag.tools import AgentTools


class _Embedder:
    async def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2]


class _Reranker:
    async def rerank(self, _query: str, _texts: list[str]) -> list[float]:
        raise AssertionError("reranker must not run for an empty candidate set")


class _Result:
    def all(self) -> list[Any]:
        return []

    def scalars(self) -> _Result:
        return self


class _RetrievalSession:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.parameters: list[dict[str, Any]] = []

    async def execute(self, statement: Any, parameters: dict[str, Any]) -> _Result:
        self.statements.append(str(statement))
        self.parameters.append(parameters)
        return _Result()


class _AgentSession:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def __aenter__(self) -> _AgentSession:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        return _Result()


class _SessionMaker:
    def __init__(self, session: _AgentSession) -> None:
        self.session = session

    def __call__(self) -> _AgentSession:
        return self.session


class _RecordingRetriever:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def retrieve(self, *_args: Any, **kwargs: Any) -> list[Any]:
        self.kwargs = kwargs
        return []


def _compiled(statement: Any) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


@pytest.mark.asyncio
async def test_retriever_uses_document_or_folder_union_in_every_primary_query() -> None:
    document_id = uuid.uuid4()
    folder_id = uuid.uuid4()
    session = _RetrievalSession()

    await Retriever(_Embedder(), _Reranker()).retrieve_with_trace(  # type: ignore[arg-type]
        session,  # type: ignore[arg-type]
        "pressure",
        document_ids=[document_id],
        folder_ids=[folder_id],
        hierarchical_mode="off",
    )

    assert len(session.parameters) == 2
    for parameters in session.parameters:
        assert parameters["doc_ids"] == [document_id]
        assert parameters["folder_ids"] == [folder_id]
    for statement in session.statements:
        assert "c.document_id = ANY(CAST(:doc_ids AS uuid[]))" in statement
        assert "d.folder_id = ANY(CAST(:folder_ids AS uuid[]))" in statement


@pytest.mark.asyncio
async def test_retriever_keeps_explicit_empty_folder_selection_fail_closed() -> None:
    session = _RetrievalSession()

    await Retriever(_Embedder(), _Reranker()).retrieve_with_trace(  # type: ignore[arg-type]
        session,  # type: ignore[arg-type]
        "pressure",
        folder_ids=[],
        hierarchical_mode="off",
    )

    assert session.parameters
    assert all(parameters["doc_ids"] is None for parameters in session.parameters)
    assert all(parameters["folder_ids"] == [] for parameters in session.parameters)


def test_text_and_visual_scope_sql_share_union_semantics() -> None:
    assert "CAST(:doc_ids AS uuid[]) IS NULL AND CAST(:folder_ids AS uuid[]) IS NULL" in _SCOPE
    assert "OR c.document_id = ANY(CAST(:doc_ids AS uuid[]))" in _SCOPE
    assert "OR d.folder_id = ANY(CAST(:folder_ids AS uuid[]))" in _SCOPE
    assert "CAST(:doc_ids AS uuid[]) IS NULL AND CAST(:folder_ids AS uuid[]) IS NULL" in (_VISUAL_PAGES_SQL)
    assert "OR p.document_id = ANY(CAST(:doc_ids AS uuid[]))" in _VISUAL_PAGES_SQL
    assert "OR d.folder_id = ANY(CAST(:folder_ids AS uuid[]))" in _VISUAL_PAGES_SQL


@pytest.mark.asyncio
async def test_agent_search_forwards_folder_ids_without_normalizing_empty_scope() -> None:
    session = _AgentSession()
    retriever = _RecordingRetriever()
    tools = AgentTools(
        _SessionMaker(session),
        retriever,  # type: ignore[arg-type]
        document_id=None,
        folder_id=None,
        folder_ids=[],
        session_id=uuid.uuid4(),
        owner_sub="user-a",
    )

    assert await tools.search_chunks("pressure") == "search_chunks('pressure'): ничего не найдено."
    assert retriever.kwargs["folder_ids"] == []


@pytest.mark.asyncio
async def test_agent_catalog_unites_selected_documents_and_folders() -> None:
    selected_document = uuid.uuid4()
    selected_folder = uuid.uuid4()
    session = _AgentSession()
    tools = AgentTools(
        _SessionMaker(session),
        _RecordingRetriever(),  # type: ignore[arg-type]
        document_id=None,
        document_ids=[selected_document],
        folder_id=None,
        folder_ids=[selected_folder],
        session_id=uuid.uuid4(),
        owner_sub="user-a",
    )

    await tools.list_documents()

    statement = _compiled(session.statements[0])
    assert "documents.id IN" in statement
    assert "documents.folder_id IN" in statement
    assert " OR " in statement
    assert "documents.owner_sub =" in statement


@pytest.mark.asyncio
async def test_agent_catalog_keeps_empty_folder_selection_fail_closed() -> None:
    session = _AgentSession()
    tools = AgentTools(
        _SessionMaker(session),
        _RecordingRetriever(),  # type: ignore[arg-type]
        document_id=None,
        folder_id=None,
        folder_ids=[],
        session_id=uuid.uuid4(),
        owner_sub="user-a",
    )

    await tools.list_documents()

    assert "false" in _compiled(session.statements[0]).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_kind", ["folder_id", "folder_ids"])
@pytest.mark.parametrize("method_name", ["get_chapter", "get_tables", "find_figure"])
async def test_agent_manual_document_tools_cannot_escape_folder_scope(
    scope_kind: str,
    method_name: str,
) -> None:
    selected_folder = uuid.uuid4()
    outside_document = uuid.uuid4()
    session = _AgentSession()
    tools = AgentTools(
        _SessionMaker(session),
        _RecordingRetriever(),  # type: ignore[arg-type]
        document_id=None,
        folder_id=selected_folder if scope_kind == "folder_id" else None,
        folder_ids=[selected_folder] if scope_kind == "folder_ids" else None,
        session_id=uuid.uuid4(),
        owner_sub="user-a",
    )

    method = getattr(tools, method_name)
    if method_name == "get_tables":
        await method(str(outside_document))
    else:
        await method("3", str(outside_document))

    assert session.statements
    for raw_statement in session.statements:
        statement = _compiled(raw_statement)
        assert "chunks.document_id =" in statement
        assert "documents.folder_id" in statement
        assert "documents.owner_sub =" in statement
