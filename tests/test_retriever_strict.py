from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from rag_app.rag.retrieve import Retriever


class FakeEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class FailingReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("synthetic reranker outage")


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
