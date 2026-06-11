"""Клиенты эмбеддингов (vLLM --task embed) и reranker'а (vLLM --task score).

Оба OpenAI/Cohere-совместимые HTTP-сервисы на GPU4 — замена движка
(TEI, SGLang) сводится к смене base_url.
"""

from __future__ import annotations

import logging

import httpx
from openai import AsyncOpenAI

from rag_app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.embed_base_url, api_key="local", timeout=120.0
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Батчами; пустые строки заменяются точкой (vLLM не принимает '')."""
        out: list[list[float]] = []
        batch = settings.embed_batch_size
        for i in range(0, len(texts), batch):
            chunk = [t.strip()[:8000] or "." for t in texts[i : i + batch]]
            resp = await self.client.embeddings.create(model=settings.embed_model, input=chunk)
            out.extend(d.embedding for d in resp.data)
        return out


class Reranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Релевантности (в исходном порядке texts) через /v1/rerank (Cohere-совместимый)."""
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.rerank_base_url}/v1/rerank",
                json={
                    "model": settings.rerank_model,
                    "query": query[:2000],
                    "documents": [t[:4000] for t in texts],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        scores = [0.0] * len(texts)
        for item in data["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores
