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
from dataclasses import dataclass, replace
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
type DenseBackend = Literal["exact", "hnsw"]
type HnswIterativeScan = Literal["off", "strict_order", "relaxed_order"]
type HierarchicalMode = Literal["off", "shadow", "active"]

_CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_RERANK_SCORE_DIGITS = 4


@dataclass(frozen=True, slots=True)
class SparseQueryPlan:
    """Закрепляет фактически выбранный lexical-движок для запроса."""

    requested_backend: SparseBackend
    engine: SparseEngine
    statement: str


@dataclass(frozen=True, slots=True)
class DenseQueryPlan:
    """Закрепляет exact baseline или индексируемый dual-language HNSW."""

    backend: DenseBackend
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
    dense_backend: DenseBackend = "exact"
    hierarchical_mode: HierarchicalMode = "off"
    hierarchical_pre_rerank: tuple[RetrievedChunk, ...] = ()
    hierarchical_reranked: tuple[RetrievedChunk, ...] = ()
    hierarchical_final: tuple[RetrievedChunk, ...] = ()
    hierarchical_added: int = 0
    hierarchical_fallback: bool = False


# Визуальный recall: страницы по эмбеддингу страницы-картинки (Qwen3-VL-Embedding)
_VISUAL_PAGES_SQL = """
SELECT p.document_id, p.page_idx, 1 - (p.emb <=> CAST(:qe AS vector)) AS vscore
FROM page_embeddings p JOIN documents d ON d.id = p.document_id
WHERE (CAST(:doc_id AS uuid) IS NULL OR p.document_id = :doc_id)
  AND (
    (CAST(:doc_ids AS uuid[]) IS NULL AND CAST(:folder_ids AS uuid[]) IS NULL)
    OR p.document_id = ANY(CAST(:doc_ids AS uuid[]))
    OR d.folder_id = ANY(CAST(:folder_ids AS uuid[]))
  )
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

# Массивы document_ids/folder_ids образуют единую selection-область:
# документ входит в неё, если явно выбран ИЛИ лежит в выбранной папке.
# NULL/NULL означает отсутствие selection-фильтра, но явный пустой folder_ids=[]
# остаётся пустой областью, а не превращается в поиск по всей библиотеке.
_SCOPE = """
  AND (CAST(:doc_id AS uuid) IS NULL OR c.document_id = :doc_id)
  AND (
    (CAST(:doc_ids AS uuid[]) IS NULL AND CAST(:folder_ids AS uuid[]) IS NULL)
    OR c.document_id = ANY(CAST(:doc_ids AS uuid[]))
    OR d.folder_id = ANY(CAST(:folder_ids AS uuid[]))
  )
  AND (CAST(:folder_id AS uuid) IS NULL OR d.folder_id = :folder_id)
  AND (CAST(:owner AS text) IS NULL OR d.owner_sub = :owner)
  AND d.status = 'done'
"""

_DENSE_SQL = f"""
SELECT {_BASE_FIELDS},
       LEAST(c.emb_en <=> CAST(:qe AS vector), c.emb_ru <=> CAST(:qe AS vector)) AS dist
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE (c.emb_en IS NOT NULL OR c.emb_ru IS NOT NULL){_SCOPE}
ORDER BY dist, c.id
LIMIT :k
"""

# LEAST(emb_en, emb_ru) не соответствует ни одному operator-class индексу и
# планируется как exact scan. Для HNSW каждый язык обязан иметь собственный
# прямой ORDER BY distance; внешний CTE затем дедуплицирует кандидатов.
_HNSW_DENSE_SQL = f"""
WITH en_scan AS MATERIALIZED (
    SELECT c.id, c.emb_en <=> CAST(:qe AS vector) AS dist
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.emb_en IS NOT NULL{_SCOPE}
    ORDER BY c.emb_en <=> CAST(:qe AS vector)
    LIMIT :k
),
ru_scan AS MATERIALIZED (
    SELECT c.id, c.emb_ru <=> CAST(:qe AS vector) AS dist
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.emb_ru IS NOT NULL{_SCOPE}
    ORDER BY c.emb_ru <=> CAST(:qe AS vector)
    LIMIT :k
),
nearest AS MATERIALIZED (
    SELECT candidate.id, MIN(candidate.dist) AS dist
    FROM (
        SELECT id, dist FROM en_scan
        UNION ALL
        SELECT id, dist FROM ru_scan
    ) candidate
    GROUP BY candidate.id
    ORDER BY MIN(candidate.dist), candidate.id
    LIMIT :k
)
SELECT {_BASE_FIELDS}, nearest.dist
FROM nearest
JOIN chunks c ON c.id = nearest.id
JOIN documents d ON d.id = c.document_id
WHERE TRUE{_SCOPE}
ORDER BY nearest.dist, c.id
LIMIT :k
"""

_HNSW_LOCAL_SETTINGS_SQL = """
SELECT
    set_config('hnsw.iterative_scan', :iterative_scan, true),
    set_config('hnsw.ef_search', CAST(:ef_search AS text), true),
    set_config('hnsw.max_scan_tuples', CAST(:max_scan_tuples AS text), true),
    set_config('hnsw.scan_mem_multiplier', CAST(:scan_mem_multiplier AS text), true)
"""


def dense_query_plan(backend: DenseBackend) -> DenseQueryPlan:
    if backend == "exact":
        return DenseQueryPlan(backend, _DENSE_SQL)
    if backend == "hnsw":
        return DenseQueryPlan(backend, _HNSW_DENSE_SQL)
    raise ValueError(f"unsupported dense backend: {backend}")


async def configure_hnsw_transaction(
    session: AsyncSession,
    *,
    iterative_scan: HnswIterativeScan | None = None,
    ef_search: int | None = None,
    max_scan_tuples: int | None = None,
    scan_mem_multiplier: float | None = None,
) -> None:
    """Apply candidate HNSW knobs only until the current transaction ends."""

    iterative_scan = settings.rag_hnsw_iterative_scan if iterative_scan is None else iterative_scan
    ef_search = settings.rag_hnsw_ef_search if ef_search is None else ef_search
    max_scan_tuples = settings.rag_hnsw_max_scan_tuples if max_scan_tuples is None else max_scan_tuples
    scan_mem_multiplier = (
        settings.rag_hnsw_scan_mem_multiplier if scan_mem_multiplier is None else scan_mem_multiplier
    )
    if iterative_scan not in {"off", "strict_order", "relaxed_order"}:
        raise ValueError("unsupported HNSW iterative scan mode")
    if ef_search < 1 or max_scan_tuples < 1 or not math.isfinite(scan_mem_multiplier):
        raise ValueError("HNSW search limits must be finite and positive")
    if scan_mem_multiplier < 1.0:
        raise ValueError("HNSW scan memory multiplier must be at least one")
    await session.execute(
        sql(_HNSW_LOCAL_SETTINGS_SQL),
        {
            "iterative_scan": iterative_scan,
            "ef_search": str(ef_search),
            "max_scan_tuples": str(max_scan_tuples),
            "scan_mem_multiplier": str(scan_mem_multiplier),
        },
    )


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

# Первый уровень задают RRF-якоря (документ/раздел/страница). Второй уровень
# поднимает точные чанки внутри этих узлов: тот же раздел, соседний ordinal/страница,
# продолжение таблицы или связанный continuation-group. Все scope/RBAC-фильтры
# применяются в том же SQL; RLS в PostgreSQL остается дополнительной защитой.
_HIERARCHICAL_EXPANSION_SQL = f"""
WITH anchor_input AS MATERIALIZED (
    SELECT anchor_id, anchor_rank
    FROM unnest(CAST(:anchor_ids AS uuid[])) WITH ORDINALITY AS t(anchor_id, anchor_rank)
),
anchors AS MATERIALIZED (
    SELECT c.id, c.document_id, c.idx, c.heading_path, c.page_start, c.page_end,
           c.meta, anchor_input.anchor_rank,
           COALESCE(
             NULLIF(c.meta->>'section_id', ''),
             'legacy:' || COALESCE(c.heading_path, '')
           ) AS section_key,
           COUNT(*) OVER (
             PARTITION BY c.document_id, COALESCE(
               NULLIF(c.meta->>'section_id', ''),
               'legacy:' || COALESCE(c.heading_path, '')
             )
           ) AS section_anchor_count
    FROM anchor_input
    JOIN chunks c ON c.id = anchor_input.anchor_id
    JOIN documents d ON d.id = c.document_id
    WHERE TRUE{_SCOPE}
),
related_ranked AS MATERIALIZED (
    SELECT c.id,
           anchors.anchor_rank,
           CASE
             WHEN NULLIF(c.meta->>'logical_table_id', '') IS NOT NULL
              AND c.meta->>'logical_table_id' = anchors.meta->>'logical_table_id' THEN 0
             WHEN NULLIF(c.meta->>'table_merge_group', '') IS NOT NULL
              AND c.meta->>'table_merge_group' = anchors.meta->>'table_merge_group' THEN 1
             WHEN NULLIF(c.meta->>'continuation_group', '') IS NOT NULL
              AND c.meta->>'continuation_group' = anchors.meta->>'continuation_group' THEN 1
             WHEN COALESCE(
                NULLIF(c.meta->>'section_id', ''),
                'legacy:' || COALESCE(c.heading_path, '')
              ) = anchors.section_key
              AND ABS(
                COALESCE(NULLIF(c.meta->>'ordinal_in_section', '')::int, c.idx)
                - COALESCE(NULLIF(anchors.meta->>'ordinal_in_section', '')::int, anchors.idx)
              ) = 1 THEN 2
             ELSE 3
           END AS relation_priority,
           ABS(
             COALESCE(NULLIF(c.meta->>'ordinal_in_section', '')::int, c.idx)
             - COALESCE(NULLIF(anchors.meta->>'ordinal_in_section', '')::int, anchors.idx)
           ) AS idx_distance,
           row_number() OVER (
               PARTITION BY anchors.id
               ORDER BY
                 CASE
                   WHEN NULLIF(c.meta->>'logical_table_id', '') IS NOT NULL
                    AND c.meta->>'logical_table_id' = anchors.meta->>'logical_table_id' THEN 0
                   WHEN NULLIF(c.meta->>'table_merge_group', '') IS NOT NULL
                    AND c.meta->>'table_merge_group' = anchors.meta->>'table_merge_group' THEN 1
                   WHEN NULLIF(c.meta->>'continuation_group', '') IS NOT NULL
                    AND c.meta->>'continuation_group' = anchors.meta->>'continuation_group' THEN 1
                   WHEN COALESCE(
                      NULLIF(c.meta->>'section_id', ''),
                      'legacy:' || COALESCE(c.heading_path, '')
                    ) = anchors.section_key
                    AND ABS(
                      COALESCE(NULLIF(c.meta->>'ordinal_in_section', '')::int, c.idx)
                      - COALESCE(NULLIF(anchors.meta->>'ordinal_in_section', '')::int, anchors.idx)
                    ) = 1 THEN 2
                   ELSE 3
                 END,
                 ABS(
                   COALESCE(NULLIF(c.meta->>'ordinal_in_section', '')::int, c.idx)
                   - COALESCE(NULLIF(anchors.meta->>'ordinal_in_section', '')::int, anchors.idx)
                 ),
                 c.id
           ) AS relation_rank
    FROM anchors
    JOIN chunks c ON c.document_id = anchors.document_id AND c.id <> anchors.id
    JOIN documents d ON d.id = c.document_id
    WHERE TRUE{_SCOPE}
      AND (
        (
          COALESCE(
            NULLIF(c.meta->>'section_id', ''),
            'legacy:' || COALESCE(c.heading_path, '')
          ) = anchors.section_key
          AND (
            ABS(
              COALESCE(NULLIF(c.meta->>'ordinal_in_section', '')::int, c.idx)
              - COALESCE(NULLIF(anchors.meta->>'ordinal_in_section', '')::int, anchors.idx)
            ) = 1
            OR anchors.section_anchor_count >= 2
            OR (
              c.page_start IS NOT NULL AND c.page_end IS NOT NULL
              AND anchors.page_start IS NOT NULL AND anchors.page_end IS NOT NULL
              AND c.page_start <= anchors.page_end + :page_radius
              AND c.page_end >= anchors.page_start - :page_radius
            )
          )
        )
        OR (
          NULLIF(c.meta->>'logical_table_id', '') IS NOT NULL
          AND c.meta->>'logical_table_id' = anchors.meta->>'logical_table_id'
        )
        OR (
          NULLIF(c.meta->>'table_merge_group', '') IS NOT NULL
          AND c.meta->>'table_merge_group' = anchors.meta->>'table_merge_group'
        )
        OR (
          NULLIF(c.meta->>'continuation_group', '') IS NOT NULL
          AND c.meta->>'continuation_group' = anchors.meta->>'continuation_group'
        )
      )
),
related AS MATERIALIZED (
    SELECT id,
           MIN(anchor_rank) AS anchor_rank,
           MIN(relation_priority) AS relation_priority,
           MIN(idx_distance) AS idx_distance
    FROM related_ranked
    WHERE relation_rank <= :per_anchor_k
    GROUP BY id
    ORDER BY MIN(anchor_rank), MIN(relation_priority), MIN(idx_distance), id
    LIMIT :expansion_k
)
SELECT {_BASE_FIELDS}
FROM related
JOIN chunks c ON c.id = related.id
JOIN documents d ON d.id = c.document_id
WHERE TRUE{_SCOPE}
ORDER BY related.anchor_rank, related.relation_priority, related.idx_distance, c.id
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


def _deduplicate_chunks(chunks: list[RetrievedChunk], *, protected_prefix: int = 0) -> list[RetrievedChunk]:
    """Keep ordering while removing UUID and wholly overlapping segment payloads."""

    have_ids: set[uuid.UUID] = set()
    seen_segments: dict[uuid.UUID, set[str]] = {}
    deduplicated: list[RetrievedChunk] = []
    for index, chunk in enumerate(chunks):
        if chunk.id in have_ids:
            continue
        segment_ids = {str(value) for value in (chunk.meta or {}).get("segment_ids", []) if value}
        document_segments = seen_segments.setdefault(chunk.document_id, set())
        if index >= protected_prefix and segment_ids and segment_ids <= document_segments:
            continue
        have_ids.add(chunk.id)
        document_segments.update(segment_ids)
        deduplicated.append(chunk)
    return deduplicated


async def _rerank_candidates(
    reranker: Reranker,
    query: str,
    candidates: list[RetrievedChunk],
    fallback_scores: dict[uuid.UUID, float],
    *,
    min_score: float,
    allow_fallback: bool,
    failure_message: str,
) -> tuple[list[RetrievedChunk], bool, bool]:
    """Rerank one complete pool and report fallback/relevance without hiding errors."""

    try:
        scores = await reranker.rerank(
            query, [candidate.text_ru or candidate.text_en for candidate in candidates]
        )
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.score = round(float(score), _RERANK_SCORE_DIGITS)
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.id.int))
        relevant = not candidates or candidates[0].score >= min_score
        return candidates, False, relevant
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(failure_message) from None
        logger.warning("reranker недоступен (%s) — отдаю RRF-порядок", exc)
        for candidate in candidates:
            candidate.score = fallback_scores[candidate.id]
        return candidates, True, True


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
        folder_ids: list[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        trace = await self.retrieve_with_trace(
            session,
            query,
            document_id=document_id,
            folder_id=folder_id,
            top_k=top_k,
            document_ids=document_ids,
            folder_ids=folder_ids,
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
        folder_ids: list[uuid.UUID] | None = None,
        *,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
        rerank_top_k: int | None = None,
        rerank_min_score: float | None = None,
        hierarchical_mode: HierarchicalMode | None = None,
        allow_hierarchical_fallback: bool = True,
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
        rerank_min_score = settings.rag_rerank_min_score if rerank_min_score is None else rerank_min_score
        hierarchical_mode = settings.rag_hierarchical_mode if hierarchical_mode is None else hierarchical_mode
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
        if hierarchical_mode not in {"off", "shadow", "active"}:
            raise ValueError("unsupported hierarchical mode")
        if hierarchical_mode != "off" and settings.rag_hierarchical_max_candidates < rerank_top_k:
            raise ValueError("hierarchical max candidates must cover the baseline rerank pool")

        # Сохраняем прежнюю семантику document_ids=[] как «нет фильтра».
        # Для folder_ids явный [] важен: это пустая выбранная область,
        # которая должна fail-closed дать ноль результатов.
        document_ids = document_ids or None
        # RBAC (ТЗ §4.7.1): owner_sub=None — admin/dev (без фильтра по владельцу);
        # иначе только свои документы + dev-документы (owner NULL). Закрывает утечку
        # чужого контента через поиск/чат — фильтр в том же SQL, что и область.
        params = {
            "doc_id": document_id,
            "doc_ids": document_ids,
            "folder_ids": folder_ids,
            "folder_id": folder_id,
            "owner": owner_sub,
        }

        stage_start = time.perf_counter()
        q_emb = await self.embedder.embed_query(query)
        latencies["embedding"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        dense_plan = dense_query_plan(settings.rag_dense_backend)
        if dense_plan.backend == "hnsw":
            await configure_hnsw_transaction(session)
        dense_rows = (
            await session.execute(
                sql(dense_plan.statement),
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

        hierarchical_fallback = False
        hierarchical_candidates = list(candidates)
        latencies["hierarchy_sql"] = 0.0
        if candidates and hierarchical_mode != "off":
            stage_start = time.perf_counter()
            anchor_ids = [candidate.id for candidate in candidates[: settings.rag_hierarchical_anchor_top_k]]
            expansion_k = settings.rag_hierarchical_max_candidates - len(candidates)
            try:
                async with session.begin_nested():
                    related_rows = (
                        await session.execute(
                            sql(_HIERARCHICAL_EXPANSION_SQL),
                            {
                                **params,
                                "anchor_ids": anchor_ids,
                                "page_radius": settings.rag_hierarchical_page_radius,
                                "per_anchor_k": settings.rag_hierarchical_per_anchor_k,
                                "expansion_k": expansion_k,
                            },
                        )
                    ).all()
                have = {candidate.id for candidate in hierarchical_candidates}
                for row in related_rows:
                    if row.id in have:
                        continue
                    hierarchical_candidates.append(_row_to_chunk(row))
                    have.add(row.id)
                    if len(hierarchical_candidates) >= settings.rag_hierarchical_max_candidates:
                        break
            except Exception as exc:
                if not allow_hierarchical_fallback:
                    raise RuntimeError("hierarchical expansion failed while fallback was disabled") from None
                logger.warning("hierarchical expansion недоступен (%s) — оставляю baseline", exc)
                hierarchical_fallback = True
                hierarchical_candidates = list(candidates)
            latencies["hierarchy_sql"] = (time.perf_counter() - stage_start) * 1000
        hierarchical_candidates = _deduplicate_chunks(
            hierarchical_candidates, protected_prefix=len(candidates)
        )
        hierarchical_pre_rerank = tuple(hierarchical_candidates)
        hierarchical_added = len(hierarchical_candidates) - len(candidates)
        for rank, candidate in enumerate(hierarchical_candidates):
            scores.setdefault(candidate.id, 1.0 / (rrf_k + rerank_top_k + rank + 1))

        shadow_reranked: tuple[RetrievedChunk, ...] | None = None
        shadow_final: tuple[RetrievedChunk, ...] | None = None
        latencies["hierarchy_rerank"] = 0.0
        if hierarchical_mode == "shadow" and hierarchical_added:
            stage_start = time.perf_counter()
            shadow_pool = [
                replace(candidate, meta=dict(candidate.meta or {})) for candidate in hierarchical_candidates
            ]
            shadow_pool, shadow_fallback, shadow_relevant = await _rerank_candidates(
                self.reranker,
                query,
                shadow_pool,
                scores,
                min_score=rerank_min_score,
                allow_fallback=allow_hierarchical_fallback,
                failure_message=("hierarchical reranker failed while fallback was disabled"),
            )
            hierarchical_fallback = hierarchical_fallback or shadow_fallback
            shadow_reranked = tuple(shadow_pool)
            shadow_final = tuple(shadow_pool[:top_k]) if shadow_relevant else ()
            latencies["hierarchy_rerank"] = (time.perf_counter() - stage_start) * 1000
        if hierarchical_mode == "active":
            candidates = hierarchical_candidates

        def build_trace(
            *,
            reranked: tuple[RetrievedChunk, ...],
            final: tuple[RetrievedChunk, ...],
            reranker_fallback: bool,
        ) -> RetrievalTrace:
            return RetrievalTrace(
                requested_sparse_backend=requested_backend,
                sparse_engine=sparse_plan.engine,
                dense_backend=dense_plan.backend,
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
                hierarchical_mode=hierarchical_mode,
                hierarchical_pre_rerank=hierarchical_pre_rerank,
                hierarchical_reranked=(
                    shadow_reranked
                    if shadow_reranked is not None
                    else reranked
                    if hierarchical_mode == "active"
                    else ()
                ),
                hierarchical_final=(
                    shadow_final
                    if shadow_final is not None
                    else final
                    if hierarchical_mode == "active"
                    else ()
                ),
                hierarchical_added=hierarchical_added,
                hierarchical_fallback=hierarchical_fallback,
            )

        if not candidates:
            latencies["rerank"] = 0.0
            latencies["visual"] = 0.0
            return build_trace(reranked=(), final=(), reranker_fallback=False)

        # reranker: считаем релевантность по RU-тексту (вопросы по-русски),
        # для нераспознанных RU — EN (BGE-reranker-v2-m3 мультиязычный)
        stage_start = time.perf_counter()
        candidates, reranker_fallback, relevant = await _rerank_candidates(
            self.reranker,
            query,
            candidates,
            scores,
            min_score=rerank_min_score,
            allow_fallback=allow_rerank_fallback,
            failure_message="reranker failed while fallback was disabled",
        )
        # Порог действует только при успешном реранке; RRF использует другую шкалу.
        if not relevant:
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
        latencies["rerank"] = (time.perf_counter() - stage_start) * 1000
        reranked = tuple(candidates)
        result = candidates[:top_k]
        # Визуальный контур (§ 12.1 шаг 4): релевантные страницы-рисунки по
        # page_embeddings → их image-чанки → визуальный реранк кропов. Добавляем к
        # тексту — vision-on-demand подаст кропы в Qwen3.5 (chat.stream_answer).
        stage_start = time.perf_counter()
        if settings.visual_enabled and self.visual_embedder is not None:
            result = await self._visual_augment(
                session,
                query,
                result,
                document_id,
                folder_id,
                document_ids,
                folder_ids,
                owner_sub,
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
        folder_ids: list[uuid.UUID] | None = None,
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
                        "folder_ids": folder_ids,
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
        img_chunks = [_row_to_chunk(r) for r in img_rows if (r.document_id, r.page_start) in visual_pages]
        if not img_chunks:
            return result
        # визуальный реранк: query × вырезанный рисунок страницы (кроп из img_s3)
        if self.visual_reranker is not None and self.storage is not None:
            try:
                crops: list[bytes] = []
                for c in img_chunks:
                    key = (c.meta or {}).get("img_s3")
                    crops.append(await self.storage.get_bytes(settings.bucket_artifacts, key) if key else b"")
                vs = await self.visual_reranker.rerank(query, crops)
                for c, s in zip(img_chunks, vs, strict=True):
                    c.score = s
                img_chunks.sort(key=lambda c: -c.score)
            except Exception as exc:  # реранк необязателен — порядок по page_embeddings
                logger.warning("visual rerank недоступен (%s)", exc)
        have = {c.id for c in result}
        extra = [c for c in img_chunks if c.id not in have][: settings.rag_visual_top_k]
        return result + extra
