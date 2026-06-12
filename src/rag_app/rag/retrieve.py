"""Гибридный поиск (roadmap § 5 п.2): dense (BGE-M3) + BM25 (tsvector)
→ RRF → reranker → top-K.

Гибрид критичен для технички: артикулы, номера ГОСТ/ISO, аббревиатуры
dense-поиском ловятся плохо. Права/фильтры — обычный SQL в том же запросе.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.config import settings
from rag_app.llm.embeddings import Embedder, Reranker

logger = logging.getLogger(__name__)

_RRF_K = 60


@dataclass
class RetrievedChunk:
    id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    heading_path: str
    kind: str
    page_start: int | None
    page_end: int | None
    text_en: str
    text_ru: str
    meta: dict[str, Any]
    score: float = 0.0


_BASE_FIELDS = """
    c.id, c.document_id, d.filename, c.heading_path, c.kind,
    c.page_start, c.page_end, c.text_en, c.text_ru, c.meta
"""

_DENSE_SQL = f"""
SELECT {_BASE_FIELDS},
       LEAST(c.emb_en <=> CAST(:qe AS vector), c.emb_ru <=> CAST(:qe AS vector)) AS dist
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE (CAST(:doc_id AS uuid) IS NULL OR c.document_id = :doc_id)
  AND (CAST(:folder_id AS uuid) IS NULL OR d.folder_id = :folder_id)
ORDER BY dist
LIMIT :k
"""

_SPARSE_SQL = f"""
SELECT {_BASE_FIELDS},
       ts_rank(c.tsv, q) AS rank
FROM chunks c JOIN documents d ON d.id = c.document_id,
     LATERAL (SELECT websearch_to_tsquery('russian', :q)
                  || websearch_to_tsquery('english', :q) AS q) tsq
WHERE c.tsv @@ q
  AND (CAST(:doc_id AS uuid) IS NULL OR c.document_id = :doc_id)
  AND (CAST(:folder_id AS uuid) IS NULL OR d.folder_id = :folder_id)
ORDER BY rank DESC
LIMIT :k
"""


def _row_to_chunk(row: Any) -> RetrievedChunk:
    return RetrievedChunk(
        id=row.id,
        document_id=row.document_id,
        filename=row.filename,
        heading_path=row.heading_path,
        kind=row.kind,
        page_start=row.page_start,
        page_end=row.page_end,
        text_en=row.text_en,
        text_ru=row.text_ru,
        meta=row.meta,
    )


class Retriever:
    def __init__(self, embedder: Embedder, reranker: Reranker) -> None:
        self.embedder = embedder
        self.reranker = reranker

    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
        document_id: uuid.UUID | None = None,
        folder_id: uuid.UUID | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.rag_context_top_k
        params = {"doc_id": document_id, "folder_id": folder_id}

        q_emb = await self.embedder.embed_query(query)
        dense_rows = (
            await session.execute(
                sql(_DENSE_SQL),
                {**params, "qe": str(q_emb), "k": settings.rag_dense_top_k},
            )
        ).all()
        sparse_rows = (
            await session.execute(
                sql(_SPARSE_SQL), {**params, "q": query, "k": settings.rag_sparse_top_k}
            )
        ).all()

        # RRF-слияние двух ранжировок
        fused: dict[uuid.UUID, RetrievedChunk] = {}
        scores: dict[uuid.UUID, float] = {}
        for rows in (dense_rows, sparse_rows):
            for rank, row in enumerate(rows):
                chunk = fused.setdefault(row.id, _row_to_chunk(row))
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        candidates = sorted(fused.values(), key=lambda c: -scores[c.id])[: settings.rag_rerank_top_k]
        if not candidates:
            return []

        # reranker: считаем релевантность по RU-тексту (вопросы по-русски),
        # для нераспознанных RU — EN (BGE-reranker-v2-m3 мультиязычный)
        try:
            rr = await self.reranker.rerank(query, [c.text_ru or c.text_en for c in candidates])
            for c, s in zip(candidates, rr, strict=True):
                c.score = s
            candidates.sort(key=lambda c: -c.score)
        except Exception as exc:  # reranker недоступен → порядок RRF
            logger.warning("reranker недоступен (%s) — отдаю RRF-порядок", exc)
            for c in candidates:
                c.score = scores[c.id]
        return candidates[:top_k]
