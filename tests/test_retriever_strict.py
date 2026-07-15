from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from rag_app.config import Settings
from rag_app.rag.retrieve import Retriever, dense_query_plan, sparse_query_plan


class FakeEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class FailingReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("synthetic reranker outage")


class EqualReranker:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return [0.5] * len(texts)


class NearEqualReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [0.50003, 0.50004]


class Result:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows

    def all(self) -> list[SimpleNamespace]:
        return self.rows


class Session:
    def __init__(self) -> None:
        self.calls = 0
        self.row = SimpleNamespace(
            id=uuid.UUID(int=1),
            document_id=uuid.UUID(int=2),
            filename="synthetic.pdf",
            heading_path="Section",
            kind="section",
            page_start=0,
            page_end=0,
            text_en="Synthetic source text",
            text_ru="Синтетический текст",
            meta={},
        )

    async def execute(self, statement, parameters) -> Result:
        self.calls += 1
        return Result([self.row] if self.calls % 2 else [])

    @asynccontextmanager
    async def begin_nested(self) -> AsyncIterator[None]:
        yield


class TieSession:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, statement, parameters) -> Result:
        self.calls += 1
        chunk_id = uuid.UUID(int=2 if self.calls == 1 else 1)
        return Result(
            [
                SimpleNamespace(
                    id=chunk_id,
                    document_id=uuid.UUID(int=10),
                    filename="synthetic.pdf",
                    heading_path="Section",
                    kind="section",
                    page_start=0,
                    page_end=0,
                    text_en=f"Synthetic source text {chunk_id}",
                    text_ru=f"Синтетический текст {chunk_id}",
                    meta={},
                )
            ]
        )

    @asynccontextmanager
    async def begin_nested(self) -> AsyncIterator[None]:
        yield


class HnswSession(Session):
    def __init__(self) -> None:
        super().__init__()
        self.statements: list[str] = []
        self.parameters: list[dict] = []

    async def execute(self, statement, parameters) -> Result:
        self.statements.append(str(statement))
        self.parameters.append(parameters)
        if len(self.statements) == 1:
            return Result([])
        return Result([self.row] if len(self.statements) == 2 else [])


class HierarchicalSession(TieSession):
    def __init__(self) -> None:
        super().__init__()
        self.statements: list[str] = []

    async def execute(self, statement, parameters) -> Result:
        self.statements.append(str(statement))
        if len(self.statements) <= 2:
            return await super().execute(statement, parameters)
        return Result(
            [
                SimpleNamespace(
                    id=uuid.UUID(int=3),
                    document_id=uuid.UUID(int=10),
                    filename="synthetic.pdf",
                    heading_path="Section",
                    kind="table",
                    page_start=1,
                    page_end=1,
                    text_en="Related table continuation",
                    text_ru="Связанное продолжение таблицы",
                    meta={"table_merge_group": "table-1"},
                )
            ]
        )


class _RecordingSavepoint:
    def __init__(self, session: FailingHierarchicalSession) -> None:
        self.session = session

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.session.savepoint_rolled_back = exc_type is not None
        return False


class FailingHierarchicalSession(TieSession):
    def __init__(self) -> None:
        super().__init__()
        self.savepoint_rolled_back = False

    async def execute(self, statement, parameters) -> Result:
        if self.calls >= 2:
            raise RuntimeError("synthetic PostgreSQL statement failure")
        return await super().execute(statement, parameters)

    def begin_nested(self) -> Any:
        return _RecordingSavepoint(self)


def test_retriever_fails_closed_when_baseline_disables_fallback() -> None:
    retriever = Retriever(FakeEmbedder(), FailingReranker())

    with pytest.raises(RuntimeError, match="fallback was disabled"):
        asyncio.run(
            retriever.retrieve(
                Session(),
                "pressure",
                top_k=10,
                allow_rerank_fallback=False,
            )
        )


def test_retriever_keeps_production_rrf_fallback_by_default() -> None:
    retriever = Retriever(FakeEmbedder(), FailingReranker())

    result = asyncio.run(retriever.retrieve(Session(), "pressure", top_k=10))

    assert [item.id for item in result] == [uuid.UUID(int=1)]
    assert result[0].score > 0


def test_retriever_breaks_equal_rrf_and_reranker_scores_by_chunk_id() -> None:
    retriever = Retriever(FakeEmbedder(), EqualReranker())

    result = asyncio.run(retriever.retrieve(TieSession(), "pressure", top_k=10))

    assert [item.id for item in result] == [uuid.UUID(int=1), uuid.UUID(int=2)]


def test_retriever_quantizes_numerical_noise_before_tie_break() -> None:
    retriever = Retriever(FakeEmbedder(), NearEqualReranker())

    result = asyncio.run(retriever.retrieve(TieSession(), "pressure", top_k=10))

    assert [item.id for item in result] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert {item.score for item in result} == {0.5}


def test_retrieval_trace_keeps_every_ranked_stage_and_timings() -> None:
    retriever = Retriever(FakeEmbedder(), EqualReranker())

    trace = asyncio.run(
        retriever.retrieve_with_trace(
            TieSession(),
            "pressure",
            top_k=10,
            sparse_backend="postgres_fts",
            dense_top_k=11,
            sparse_top_k=12,
            rrf_k=30,
            rerank_top_k=10,
            rerank_min_score=0.1,
        )
    )

    assert trace.requested_sparse_backend == "postgres_fts"
    assert trace.sparse_engine == "postgres_fts"
    assert trace.dense_backend == "exact"
    assert [item.id for item in trace.dense] == [uuid.UUID(int=2)]
    assert [item.id for item in trace.sparse] == [uuid.UUID(int=1)]
    assert [item.id for item in trace.hybrid_pre_rerank] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert [item.id for item in trace.reranked] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert [item.id for item in trace.final] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert trace.reranker_fallback is False
    assert trace.hierarchical_mode == "off"
    assert trace.hierarchical_pre_rerank == trace.hybrid_pre_rerank
    assert trace.hierarchical_added == 0
    assert trace.hierarchical_fallback is False
    assert set(trace.stage_latency_ms) == {
        "embedding",
        "dense_sql",
        "sparse_sql",
        "fusion",
        "hierarchy_sql",
        "hierarchy_rerank",
        "rerank",
        "visual",
        "total",
    }
    assert all(value >= 0.0 for value in trace.stage_latency_ms.values())


def test_retrieval_trace_rejects_invalid_local_sweep_override() -> None:
    retriever = Retriever(FakeEmbedder(), EqualReranker())

    with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
        asyncio.run(retriever.retrieve_with_trace(Session(), "pressure", rrf_k=0))


@pytest.mark.parametrize(
    ("query", "expected_engine", "expected_index"),
    [
        ("испытательное давление ГОСТ 32569", "pg_textsearch_ru", "ix_chunks_bm25_ru_v1"),
        ("pipeline pressure API 5L", "pg_textsearch_en", "ix_chunks_bm25_en_v1"),
        ("API 5L X65", "pg_textsearch_en", "ix_chunks_bm25_en_v1"),
    ],
)
def test_pg_textsearch_routes_supported_query_scripts(
    query: str,
    expected_engine: str,
    expected_index: str,
) -> None:
    plan = sparse_query_plan(query, "pg_textsearch")

    assert plan.engine == expected_engine
    assert expected_index in plan.statement
    assert "to_bm25query(:q" in plan.statement
    assert ")) < 0" in plan.statement
    assert "ORDER BY" in plan.statement
    assert "c.id" in plan.statement


def test_pg_textsearch_keeps_chinese_on_existing_fts_until_segmented() -> None:
    plan = sparse_query_plan("管道压力试验要求", "pg_textsearch")

    assert plan.requested_backend == "pg_textsearch"
    assert plan.engine == "postgres_fts"
    assert "websearch_to_tsquery" in plan.statement
    assert "to_bm25query" not in plan.statement


def test_pg_textsearch_indexes_use_distinct_matching_expressions() -> None:
    russian = sparse_query_plan("испытательное давление", "pg_textsearch").statement
    english = sparse_query_plan("test pressure", "pg_textsearch").statement

    assert "coalesce(c.text_ru, '') || E'\\n' || coalesce(c.text_en, '')" in russian
    assert "coalesce(c.text_en, '') || E'\\n' || coalesce(c.text_ru, '')" in english


def test_sparse_backend_rejects_unknown_runtime_value() -> None:
    with pytest.raises(ValueError, match="unsupported sparse backend"):
        sparse_query_plan("pressure", "unknown")  # type: ignore[arg-type]


def test_sparse_backend_and_rrf_use_public_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_SPARSE_BACKEND", "pg_textsearch")
    monkeypatch.setenv("RAG_DENSE_BACKEND", "hnsw")
    monkeypatch.setenv("RAG_HNSW_ITERATIVE_SCAN", "strict_order")
    monkeypatch.setenv("RAG_HNSW_EF_SEARCH", "120")
    monkeypatch.setenv("RAG_HNSW_MAX_SCAN_TUPLES", "30000")
    monkeypatch.setenv("RAG_HNSW_SCAN_MEM_MULTIPLIER", "3.5")
    monkeypatch.setenv("RAG_RRF_K", "77")
    monkeypatch.setenv("RAG_HIERARCHICAL_MODE", "shadow")
    monkeypatch.setenv("RAG_HIERARCHICAL_ANCHOR_TOP_K", "6")
    monkeypatch.setenv("RAG_HIERARCHICAL_PER_ANCHOR_K", "3")
    monkeypatch.setenv("RAG_HIERARCHICAL_MAX_CANDIDATES", "32")
    monkeypatch.setenv("RAG_HIERARCHICAL_PAGE_RADIUS", "2")

    configured = Settings(_env_file=None)

    assert configured.rag_sparse_backend == "pg_textsearch"
    assert configured.rag_dense_backend == "hnsw"
    assert configured.rag_hnsw_iterative_scan == "strict_order"
    assert configured.rag_hnsw_ef_search == 120
    assert configured.rag_hnsw_max_scan_tuples == 30_000
    assert configured.rag_hnsw_scan_mem_multiplier == 3.5
    assert configured.rag_rrf_k == 77
    assert configured.rag_hierarchical_mode == "shadow"
    assert configured.rag_hierarchical_anchor_top_k == 6
    assert configured.rag_hierarchical_per_anchor_k == 3
    assert configured.rag_hierarchical_max_candidates == 32
    assert configured.rag_hierarchical_page_radius == 2


def test_hnsw_dense_plan_has_two_indexable_filtered_language_branches() -> None:
    statement = dense_query_plan("hnsw").statement

    assert "ORDER BY c.emb_en <=> CAST(:qe AS vector)" in statement
    assert "ORDER BY c.emb_ru <=> CAST(:qe AS vector)" in statement
    assert statement.count("CAST(:owner AS text) IS NULL OR d.owner_sub = :owner") == 3
    assert statement.count("CAST(:doc_id AS uuid) IS NULL OR c.document_id = :doc_id") == 3
    assert "GROUP BY candidate.id" in statement
    assert "ORDER BY MIN(candidate.dist), candidate.id" in statement
    assert "LEAST(" not in statement


def test_exact_dense_plan_stays_default_and_rejects_unknown_backend() -> None:
    configured = Settings(_env_file=None)

    assert configured.rag_dense_backend == "exact"
    assert "LEAST(" in dense_query_plan("exact").statement
    assert "c.emb_en IS NOT NULL OR c.emb_ru IS NOT NULL" in dense_query_plan("exact").statement
    with pytest.raises(ValueError, match="unsupported dense backend"):
        dense_query_plan("unknown")  # type: ignore[arg-type]


def test_hnsw_backend_applies_transaction_local_settings_before_dense_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_app.rag.retrieve.settings.rag_dense_backend", "hnsw")
    monkeypatch.setattr("rag_app.rag.retrieve.settings.rag_hnsw_iterative_scan", "strict_order")
    session = HnswSession()

    trace = asyncio.run(Retriever(FakeEmbedder(), EqualReranker()).retrieve_with_trace(session, "pressure"))

    assert trace.dense_backend == "hnsw"
    assert "set_config('hnsw.iterative_scan', :iterative_scan, true)" in session.statements[0]
    assert session.parameters[0] == {
        "iterative_scan": "strict_order",
        "ef_search": "100",
        "max_scan_tuples": "20000",
        "scan_mem_multiplier": "2.0",
    }
    assert "WITH en_scan AS MATERIALIZED" in session.statements[1]
    assert "websearch_to_tsquery" in session.statements[2]


def test_hierarchical_shadow_collects_related_chunks_without_changing_result() -> None:
    session = HierarchicalSession()
    reranker = EqualReranker()

    trace = asyncio.run(
        Retriever(FakeEmbedder(), reranker).retrieve_with_trace(
            session,
            "pressure",
            top_k=10,
            hierarchical_mode="shadow",
            allow_hierarchical_fallback=False,
        )
    )

    assert [item.id.int for item in trace.final] == [1, 2]
    assert [item.id.int for item in trace.hierarchical_pre_rerank] == [1, 2, 3]
    assert [item.id.int for item in trace.hierarchical_final] == [1, 2, 3]
    assert trace.hierarchical_added == 1
    assert reranker.calls == 2
    assert reranker.batch_sizes == [3, 2]
    assert "WITH anchor_input AS MATERIALIZED" in session.statements[2]
    assert session.statements[2].count("CAST(:owner AS text) IS NULL OR d.owner_sub = :owner") == 3


def test_hierarchical_active_reranks_baseline_and_expansion_once() -> None:
    session = HierarchicalSession()
    reranker = EqualReranker()

    trace = asyncio.run(
        Retriever(FakeEmbedder(), reranker).retrieve_with_trace(
            session,
            "pressure",
            top_k=10,
            hierarchical_mode="active",
            allow_hierarchical_fallback=False,
        )
    )

    assert [item.id.int for item in trace.final] == [1, 2, 3]
    assert trace.hierarchical_mode == "active"
    assert trace.hierarchical_added == 1
    assert trace.reranker_fallback is False
    assert reranker.calls == 1
    assert reranker.batch_sizes == [3]


def test_hierarchical_mode_rejects_unknown_runtime_value() -> None:
    with pytest.raises(ValueError, match="unsupported hierarchical mode"):
        asyncio.run(
            Retriever(FakeEmbedder(), EqualReranker()).retrieve_with_trace(
                Session(),
                "pressure",
                hierarchical_mode="unknown",  # type: ignore[arg-type]
            )
        )


def test_hierarchical_sql_failure_rolls_back_savepoint_and_returns_baseline() -> None:
    session = FailingHierarchicalSession()

    trace = asyncio.run(
        Retriever(FakeEmbedder(), EqualReranker()).retrieve_with_trace(
            session,
            "pressure",
            top_k=10,
            hierarchical_mode="active",
        )
    )

    assert session.savepoint_rolled_back is True
    assert trace.hierarchical_fallback is True
    assert trace.hierarchical_added == 0
    assert [item.id.int for item in trace.final] == [1, 2]
