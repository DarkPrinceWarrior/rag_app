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
from rag_app.llm.visual import VisualEmbedder
from rag_app.llm.visual_reranker import VisualReranker
from rag_app.storage.s3 import Storage

logger = logging.getLogger(__name__)

_RRF_K = 60

# Визуальный recall: страницы по эмбеддингу страницы-картинки (Qwen3-VL-Embedding)
_VISUAL_PAGES_SQL = """
SELECT p.document_id, p.page_idx, 1 - (p.emb <=> CAST(:qe AS vector)) AS vscore
FROM page_embeddings p JOIN documents d ON d.id = p.document_id
WHERE (CAST(:doc_id AS uuid) IS NULL OR p.document_id = :doc_id)
  AND (CAST(:folder_id AS uuid) IS NULL OR d.folder_id = :folder_id)
ORDER BY p.emb <=> CAST(:qe AS vector)
LIMIT :k
"""


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


_IMG_CHUNKS_SQL = f"""
SELECT {_BASE_FIELDS}
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.kind = 'image' AND (c.meta ? 'img_s3') AND c.document_id = ANY(:doc_ids)
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
    def __init__(
        self,
        embedder: Embedder,
        reranker: Reranker,
        visual_embedder: VisualEmbedder | None = None,
        visual_reranker: VisualReranker | None = None,
        storage: Storage | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.visual_embedder = visual_embedder
        self.visual_reranker = visual_reranker
        self.storage = storage

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
        result = candidates[:top_k]
        # Визуальный контур (§ 12.1 шаг 4): релевантные страницы-рисунки по
        # page_embeddings → их image-чанки → визуальный реранк кропов. Добавляем к
        # тексту — vision-on-demand подаст кропы в Qwen3.5 (chat.stream_answer).
        if settings.visual_enabled and self.visual_embedder is not None:
            result = await self._visual_augment(session, query, result, document_id, folder_id)
        return result

    async def _visual_augment(
        self,
        session: AsyncSession,
        query: str,
        result: list[RetrievedChunk],
        document_id: uuid.UUID | None,
        folder_id: uuid.UUID | None,
    ) -> list[RetrievedChunk]:
        """Визуальный recall (Qwen3-VL-Embedding) + реранк (Qwen3-VL-Reranker) →
        добавить релевантные image-чанки страниц, которых текстовый поиск не поднял."""
        try:
            q_emb = await self.visual_embedder.embed_text_query(query)
            rows = (
                await session.execute(
                    sql(_VISUAL_PAGES_SQL),
                    {
                        "qe": str(q_emb),
                        "doc_id": document_id,
                        "folder_id": folder_id,
                        "k": settings.rag_visual_pages_k,
                    },
                )
            ).all()
        except Exception as exc:  # визуальный контур необязателен
            logger.warning("visual recall недоступен (%s)", exc)
            return result
        if not rows:
            return result
        visual_pages = {(r.document_id, r.page_idx) for r in rows}
        doc_ids = list({r.document_id for r in rows})
        img_rows = (await session.execute(sql(_IMG_CHUNKS_SQL), {"doc_ids": doc_ids})).all()
        img_chunks = [
            _row_to_chunk(r) for r in img_rows if (r.document_id, r.page_start) in visual_pages
        ]
        if not img_chunks:
            return result
        # визуальный реранк: query × вырезанный рисунок страницы (кроп из img_s3)
        if self.visual_reranker is not None and self.storage is not None:
            try:
                crops: list[bytes] = []
                for c in img_chunks:
                    key = (c.meta or {}).get("img_s3")
                    crops.append(
                        await self.storage.get_bytes(settings.bucket_artifacts, key)
                        if key
                        else b""
                    )
                vs = await self.visual_reranker.rerank(query, crops)
                for c, s in zip(img_chunks, vs, strict=True):
                    c.score = s
                img_chunks.sort(key=lambda c: -c.score)
            except Exception as exc:  # реранк необязателен — порядок по page_embeddings
                logger.warning("visual rerank недоступен (%s)", exc)
        have = {c.id for c in result}
        extra = [c for c in img_chunks if c.id not in have][: settings.rag_visual_top_k]
        return result + extra
