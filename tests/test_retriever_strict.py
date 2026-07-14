from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from rag_app.config import Settings
from rag_app.rag.retrieve import Retriever, sparse_query_plan


class FakeEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class FailingReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("synthetic reranker outage")


class EqualReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [0.5] * len(texts)


class NearEqualReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [0.5000003, 0.5000004]


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
    assert [item.id for item in trace.dense] == [uuid.UUID(int=2)]
    assert [item.id for item in trace.sparse] == [uuid.UUID(int=1)]
    assert [item.id for item in trace.hybrid_pre_rerank] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert [item.id for item in trace.reranked] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert [item.id for item in trace.final] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert trace.reranker_fallback is False
    assert set(trace.stage_latency_ms) == {
        "embedding",
        "dense_sql",
        "sparse_sql",
        "fusion",
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
    monkeypatch.setenv("RAG_RRF_K", "77")

    configured = Settings(_env_file=None)

    assert configured.rag_sparse_backend == "pg_textsearch"
    assert configured.rag_rrf_k == 77
