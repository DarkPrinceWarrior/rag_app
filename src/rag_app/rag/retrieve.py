"""Гибридный поиск (roadmap § 5 п.2): dense + переключаемый lexical-контур
→ RRF → reranker → top-K.

Гибрид критичен для технички: артикулы, номера ГОСТ/ISO, аббревиатуры
dense-поиском ловятся плохо. Права/фильтры — обычный SQL в том же запросе.
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.config import settings
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.llm.visual import VisualEmbedder
from rag_app.llm.visual_reranker import VisualReranker
from rag_app.storage.s3 import Storage

logger = logging.getLogger(__name__)

type SparseBackend = Literal["postgres_fts", "pg_textsearch"]
type SparseEngine = Literal["postgres_fts", "pg_textsearch_ru", "pg_textsearch_en"]

_CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True, slots=True)
class SparseQueryPlan:
    """Закрепляет фактически выбранный lexical-движок для запроса."""

    requested_backend: SparseBackend
    engine: SparseEngine
    statement: str


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Полный порядок стадий одного retrieval-вызова для воспроизводимого A/B."""

    requested_sparse_backend: SparseBackend
    sparse_engine: SparseEngine
    dense: tuple[RetrievedChunk, ...]
    sparse: tuple[RetrievedChunk, ...]
    hybrid_pre_rerank: tuple[RetrievedChunk, ...]
    reranked: tuple[RetrievedChunk, ...]
    final: tuple[RetrievedChunk, ...]
    stage_latency_ms: dict[str, float]
    reranker_fallback: bool

# Визуальный recall: страницы по эмбеддингу страницы-картинки (Qwen3-VL-Embedding)
_VISUAL_PAGES_SQL = """
SELECT p.document_id, p.page_idx, 1 - (p.emb <=> CAST(:qe AS vector)) AS vscore
FROM page_embeddings p JOIN documents d ON d.id = p.document_id
WHERE (CAST(:doc_id AS uuid) IS NULL OR p.document_id = :doc_id)
  AND (CAST(:doc_ids AS uuid[]) IS NULL OR p.document_id = ANY(CAST(:doc_ids AS uuid[])))
  AND (CAST(:folder_id AS uuid) IS NULL OR d.folder_id = :folder_id)
  AND (CAST(:owner AS text) IS NULL OR d.owner_sub = :owner)
  AND d.status = 'done'
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

# Фильтр области: одиночный документ ИЛИ набор документов (мульти-док выбор
# в чате) ИЛИ папка. Каждый клауз — no-op, если параметр NULL (как :doc_id).
_SCOPE = """
  AND (CAST(:doc_id AS uuid) IS NULL OR c.document_id = :doc_id)
  AND (CAST(:doc_ids AS uuid[]) IS NULL OR c.document_id = ANY(CAST(:doc_ids AS uuid[])))
  AND (CAST(:folder_id AS uuid) IS NULL OR d.folder_id = :folder_id)
  AND (CAST(:owner AS text) IS NULL OR d.owner_sub = :owner)
  AND d.status = 'done'
"""

_DENSE_SQL = f"""
SELECT {_BASE_FIELDS},
       LEAST(c.emb_en <=> CAST(:qe AS vector), c.emb_ru <=> CAST(:qe AS vector)) AS dist
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE TRUE{_SCOPE}
ORDER BY dist, c.id
LIMIT :k
"""

_SPARSE_SQL = f"""
SELECT {_BASE_FIELDS},
       ts_rank(c.tsv, q) AS rank
FROM chunks c JOIN documents d ON d.id = c.document_id,
     LATERAL (SELECT websearch_to_tsquery('russian', :q)
                  || websearch_to_tsquery('english', :q) AS q) tsq
WHERE c.tsv @@ q{_SCOPE}
ORDER BY rank DESC, c.id
LIMIT :k
"""

_BM25_TEXT_RU = "(coalesce(c.text_ru, '') || E'\\n' || coalesce(c.text_en, ''))"
_BM25_TEXT_EN = "(coalesce(c.text_en, '') || E'\\n' || coalesce(c.text_ru, ''))"
_BM25_INDEX_RU = "ix_chunks_bm25_ru_v1"
_BM25_INDEX_EN = "ix_chunks_bm25_en_v1"


def _bm25_sql(index_name: str, text_expression: str) -> str:
    # index_name приходит только из двух констант выше. Литерал обязателен:
    # planner pg_textsearch должен определить expression index при подготовке SQL.
    return f"""
SELECT {_BASE_FIELDS},
       -({text_expression} <@> to_bm25query(:q, '{index_name}')) AS rank
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE TRUE{_SCOPE}
  AND ({text_expression} <@> to_bm25query(:q, '{index_name}')) < 0
ORDER BY {text_expression} <@> to_bm25query(:q, '{index_name}'), c.id
LIMIT :k
"""


_BM25_RU_SQL = _bm25_sql(_BM25_INDEX_RU, _BM25_TEXT_RU)
_BM25_EN_SQL = _bm25_sql(_BM25_INDEX_EN, _BM25_TEXT_EN)


def sparse_query_plan(query: str, backend: SparseBackend) -> SparseQueryPlan:
    """Select BM25 by query script; keep the proven FTS path for Chinese.

    PostgreSQL's built-in parser has no Chinese word segmentation, while the
    currently reviewed pg_textsearch indexes cover Russian and English only.
    Routing Han queries to the existing path avoids claiming unsupported BM25.
    """

    if backend not in {"postgres_fts", "pg_textsearch"}:
        raise ValueError(f"unsupported sparse backend: {backend}")
    if backend == "postgres_fts" or _HAN_RE.search(query):
        return SparseQueryPlan(backend, "postgres_fts", _SPARSE_SQL)
    if _CYRILLIC_RE.search(query):
        return SparseQueryPlan(backend, "pg_textsearch_ru", _BM25_RU_SQL)
    return SparseQueryPlan(backend, "pg_textsearch_en", _BM25_EN_SQL)


_IMG_CHUNKS_SQL = f"""
SELECT {_BASE_FIELDS}
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.kind = 'image' AND (c.meta ? 'img_s3') AND c.document_id = ANY(:doc_ids)
  AND d.status = 'done'
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
        document_ids: list[uuid.UUID] | None = None,
        owner_sub: str | None = None,
        allow_rerank_fallback: bool = True,
        sparse_backend: SparseBackend | None = None,
    ) -> list[RetrievedChunk]:
        trace = await self.retrieve_with_trace(
            session,
            query,
            document_id=document_id,
            folder_id=folder_id,
            top_k=top_k,
            document_ids=document_ids,
            owner_sub=owner_sub,
            allow_rerank_fallback=allow_rerank_fallback,
            sparse_backend=sparse_backend,
        )
        return list(trace.final)

    async def retrieve_with_trace(
        self,
        session: AsyncSession,
        query: str,
        document_id: uuid.UUID | None = None,
        folder_id: uuid.UUID | None = None,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
        owner_sub: str | None = None,
        allow_rerank_fallback: bool = True,
        sparse_backend: SparseBackend | None = None,
        *,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
        rerank_top_k: int | None = None,
        rerank_min_score: float | None = None,
    ) -> RetrievalTrace:
        """Run production retrieval and retain content-bearing stages in memory.

        Qualification serializes only stable IDs, timings and hashes. Explicit
        overrides make parameter sweeps local to one call and cannot mutate the
        process-wide settings used by concurrent production requests.
        """

        total_start = time.perf_counter()
        latencies: dict[str, float] = {}
        top_k = settings.rag_context_top_k if top_k is None else top_k
        dense_top_k = settings.rag_dense_top_k if dense_top_k is None else dense_top_k
        sparse_top_k = settings.rag_sparse_top_k if sparse_top_k is None else sparse_top_k
        rrf_k = settings.rag_rrf_k if rrf_k is None else rrf_k
        rerank_top_k = settings.rag_rerank_top_k if rerank_top_k is None else rerank_top_k
        rerank_min_score = (
            settings.rag_rerank_min_score if rerank_min_score is None else rerank_min_score
        )
        for name, value in (
            ("top_k", top_k),
            ("dense_top_k", dense_top_k),
            ("sparse_top_k", sparse_top_k),
            ("rrf_k", rrf_k),
            ("rerank_top_k", rerank_top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(rerank_min_score) or not 0.0 <= rerank_min_score <= 1.0:
            raise ValueError("rerank_min_score must be finite and between 0 and 1")

        # пустой список = нет фильтра (трактуем как None)
        document_ids = document_ids or None
        # RBAC (ТЗ §4.7.1): owner_sub=None — admin/dev (без фильтра по владельцу);
        # иначе только свои документы + dev-документы (owner NULL). Закрывает утечку
        # чужого контента через поиск/чат — фильтр в том же SQL, что и область.
        params = {
            "doc_id": document_id,
            "doc_ids": document_ids,
            "folder_id": folder_id,
            "owner": owner_sub,
        }

        stage_start = time.perf_counter()
        q_emb = await self.embedder.embed_query(query)
        latencies["embedding"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        dense_rows = (
            await session.execute(
                sql(_DENSE_SQL),
                {**params, "qe": str(q_emb), "k": dense_top_k},
            )
        ).all()
        latencies["dense_sql"] = (time.perf_counter() - stage_start) * 1000
        requested_backend = settings.rag_sparse_backend if sparse_backend is None else sparse_backend
        sparse_plan = sparse_query_plan(query, requested_backend)
        stage_start = time.perf_counter()
        sparse_rows = (
            await session.execute(
                sql(sparse_plan.statement),
                {**params, "q": query, "k": sparse_top_k},
            )
        ).all()
        latencies["sparse_sql"] = (time.perf_counter() - stage_start) * 1000
        dense_chunks = tuple(_row_to_chunk(row) for row in dense_rows)
        sparse_chunks = tuple(_row_to_chunk(row) for row in sparse_rows)

        # RRF-слияние двух ранжировок
        stage_start = time.perf_counter()
        fused: dict[uuid.UUID, RetrievedChunk] = {}
        scores: dict[uuid.UUID, float] = {}
        for rows in (dense_rows, sparse_rows):
            for rank, row in enumerate(rows):
                chunk = fused.setdefault(row.id, _row_to_chunk(row))
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank + 1)
        candidates = sorted(
            fused.values(),
            key=lambda chunk: (-scores[chunk.id], chunk.id.int),
        )[:rerank_top_k]
        hybrid_pre_rerank = tuple(candidates)
        latencies["fusion"] = (time.perf_counter() - stage_start) * 1000

        def build_trace(
            *,
            reranked: tuple[RetrievedChunk, ...],
            final: tuple[RetrievedChunk, ...],
            reranker_fallback: bool,
        ) -> RetrievalTrace:
            return RetrievalTrace(
                requested_sparse_backend=requested_backend,
                sparse_engine=sparse_plan.engine,
                dense=dense_chunks,
                sparse=sparse_chunks,
                hybrid_pre_rerank=hybrid_pre_rerank,
                reranked=reranked,
                final=final,
                stage_latency_ms={
                    **latencies,
                    "total": (time.perf_counter() - total_start) * 1000,
                },
                reranker_fallback=reranker_fallback,
            )

        if not candidates:
            latencies["rerank"] = 0.0
            latencies["visual"] = 0.0
            return build_trace(reranked=(), final=(), reranker_fallback=False)

        # reranker: считаем релевантность по RU-тексту (вопросы по-русски),
        # для нераспознанных RU — EN (BGE-reranker-v2-m3 мультиязычный)
        reranker_fallback = False
        stage_start = time.perf_counter()
        try:
            rr = await self.reranker.rerank(query, [c.text_ru or c.text_en for c in candidates])
            for c, s in zip(candidates, rr, strict=True):
                c.score = s
            candidates.sort(key=lambda c: (-c.score, c.id.int))
            # Порог релевантности: если даже лучший фрагмент почти нерелевантен
            # (запрос — не про эти документы), не вываливаем случайные чанки —
            # пусто → чат честно скажет, что релевантного не нашлось. Только в ветке
            # успешного реранка (у RRF-фолбэка иная шкала скоров).
            if candidates and candidates[0].score < rerank_min_score:
                logger.info(
                    "retrieve: топ-реранк %.4f < порога %.4f — релевантных фрагментов нет",
                    candidates[0].score,
                    rerank_min_score,
                )
                latencies["rerank"] = (time.perf_counter() - stage_start) * 1000
                latencies["visual"] = 0.0
                return build_trace(
                    reranked=tuple(candidates),
                    final=(),
                    reranker_fallback=False,
                )
        except Exception as exc:  # reranker недоступен → порядок RRF
            if not allow_rerank_fallback:
                raise RuntimeError("reranker failed while fallback was disabled") from None
            logger.warning("reranker недоступен (%s) — отдаю RRF-порядок", exc)
            reranker_fallback = True
            for c in candidates:
                c.score = scores[c.id]
        latencies["rerank"] = (time.perf_counter() - stage_start) * 1000
        reranked = tuple(candidates)
        result = candidates[:top_k]
        # Визуальный контур (§ 12.1 шаг 4): релевантные страницы-рисунки по
        # page_embeddings → их image-чанки → визуальный реранк кропов. Добавляем к
        # тексту — vision-on-demand подаст кропы в Qwen3.5 (chat.stream_answer).
        stage_start = time.perf_counter()
        if settings.visual_enabled and self.visual_embedder is not None:
            result = await self._visual_augment(
                session, query, result, document_id, folder_id, document_ids, owner_sub
            )
        latencies["visual"] = (time.perf_counter() - stage_start) * 1000
        return build_trace(
            reranked=reranked,
            final=tuple(result),
            reranker_fallback=reranker_fallback,
        )

    async def _visual_augment(
        self,
        session: AsyncSession,
        query: str,
        result: list[RetrievedChunk],
        document_id: uuid.UUID | None,
        folder_id: uuid.UUID | None,
        document_ids: list[uuid.UUID] | None = None,
        owner_sub: str | None = None,
    ) -> list[RetrievedChunk]:
        """Визуальный recall (Qwen3-VL-Embedding) + реранк (Qwen3-VL-Reranker) →
        добавить релевантные image-чанки страниц, которых текстовый поиск не поднял."""
        if self.visual_embedder is None:
            return result
        try:
            q_emb = await self.visual_embedder.embed_text_query(query)
            rows = (
                await session.execute(
                    sql(_VISUAL_PAGES_SQL),
                    {
                        "qe": str(q_emb),
                        "doc_id": document_id,
                        "doc_ids": document_ids or None,
                        "folder_id": folder_id,
                        "owner": owner_sub,
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
