"""Проверяемая translation memory: exact reuse и scoped hybrid hints."""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.config import settings
from rag_app.db.models import TranslationMemory


class EmbedderLike(Protocol):
    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class TranslationMemoryScope:
    owner_sub: str
    folder_id: uuid.UUID | None
    project: str | None
    domain: str = "technical"


@dataclass(frozen=True, slots=True)
class TranslationMemoryExample:
    entry_id: uuid.UUID
    source_text: str
    translation: str
    score: float


@dataclass(frozen=True, slots=True)
class TranslationMemoryMatch:
    exact: TranslationMemoryExample | None = None
    nearest: tuple[TranslationMemoryExample, ...] = ()


def normalize_source(text: str) -> str:
    """Unicode/whitespace canonicalization без case folding: exact остаётся строгим."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def source_hash(text: str) -> str:
    return hashlib.sha256(normalize_source(text).encode("utf-8")).hexdigest()


def _scope_predicates(scope: TranslationMemoryScope) -> tuple:
    folder = (
        TranslationMemory.folder_id.is_(None)
        if scope.folder_id is None
        else TranslationMemory.folder_id == scope.folder_id
    )
    project = (
        TranslationMemory.project.is_(None)
        if scope.project is None
        else TranslationMemory.project == scope.project
    )
    return (
        TranslationMemory.owner_sub == scope.owner_sub,
        folder,
        project,
        TranslationMemory.domain == scope.domain,
        TranslationMemory.status == "approved",
        TranslationMemory.revoked_at.is_(None),
    )


class TranslationMemoryService:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        embedder: EmbedderLike,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._embedder = embedder

    async def lookup_batch(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
        scope: TranslationMemoryScope,
    ) -> dict[str, TranslationMemoryMatch]:
        unique = list(dict.fromkeys(text for text in texts if text.strip()))
        if not unique:
            return {}
        hashes = {text: source_hash(text) for text in unique}
        base = (
            *_scope_predicates(scope),
            TranslationMemory.source_lang == source_lang,
            TranslationMemory.target_lang == target_lang,
        )
        exact_by_hash: dict[str, TranslationMemory] = {}
        async with self._sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        select(TranslationMemory).where(
                            *base,
                            TranslationMemory.source_hash.in_(set(hashes.values())),
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                exact_by_hash[row.source_hash] = row

        matches: dict[str, TranslationMemoryMatch] = {}
        remaining: list[str] = []
        for text in unique:
            exact_row = exact_by_hash.get(hashes[text])
            if exact_row is not None and exact_row.source_normalized == normalize_source(text):
                matches[text] = TranslationMemoryMatch(
                    exact=TranslationMemoryExample(
                        exact_row.id,
                        exact_row.source_text,
                        exact_row.approved_translation,
                        1.0,
                    )
                )
            else:
                remaining.append(text)

        if not remaining or settings.translation_memory_nearest_top_k == 0:
            return matches

        vectors = await self._embedder.embed(remaining)
        async with self._sessionmaker() as session:
            for text, vector in zip(remaining, vectors, strict=True):
                nearest = await self._nearest(session, text, vector, base)
                matches[text] = TranslationMemoryMatch(nearest=nearest)
        return matches

    async def _nearest(
        self,
        session: AsyncSession,
        text: str,
        vector: list[float],
        base: tuple,
    ) -> tuple[TranslationMemoryExample, ...]:
        pool = settings.translation_memory_candidate_pool
        normalized = normalize_source(text)
        lexical_score = func.similarity(TranslationMemory.source_normalized, normalized).label(
            "score"
        )
        lexical_rows = (
            await session.execute(
                select(TranslationMemory, lexical_score)
                .where(*base, lexical_score >= settings.translation_memory_lexical_min_similarity)
                .order_by(desc(lexical_score))
                .limit(pool)
            )
        ).all()

        cosine_distance = TranslationMemory.source_embedding.cosine_distance(vector)
        dense_score = (1.0 - cosine_distance).label("score")
        dense_rows = (
            await session.execute(
                select(TranslationMemory, dense_score)
                .where(
                    *base,
                    TranslationMemory.source_embedding.is_not(None),
                    cosine_distance <= 1.0 - settings.translation_memory_dense_min_similarity,
                )
                .order_by(cosine_distance)
                .limit(pool)
            )
        ).all()

        candidates: dict[uuid.UUID, tuple[TranslationMemory, float, float]] = {}
        for row, score in lexical_rows:
            candidates[row.id] = (row, float(score), 0.0)
        for row, score in dense_rows:
            previous = candidates.get(row.id)
            candidates[row.id] = (
                row,
                previous[1] if previous else 0.0,
                float(score),
            )
        ranked = sorted(
            candidates.values(),
            key=lambda item: (0.35 * item[1] + 0.65 * item[2], str(item[0].id)),
            reverse=True,
        )
        return tuple(
            TranslationMemoryExample(
                entry_id=row.id,
                source_text=row.source_text,
                translation=row.approved_translation,
                score=round(0.35 * lexical + 0.65 * dense, 6),
            )
            for row, lexical, dense in ranked[: settings.translation_memory_nearest_top_k]
        )
