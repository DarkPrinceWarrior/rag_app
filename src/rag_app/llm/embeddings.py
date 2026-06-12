"""Клиенты эмбеддингов (vLLM --runner pooling) и reranker'а (vLLM /v1/rerank).

§ 12.1 шаг 1: Qwen3-Embedding-0.6B + Qwen3-Reranker-4B. Серия instruction-aware:
- эмбеддинг ЗАПРОСА — с инструкцией («Instruct: …\nQuery: …»), документов — без;
- reranker получает запрос в формате «<Instruct>: …\n<Query>: …».
Замена движка (TEI, SGLang) или отказ от инструкций — сменой base_url/конфига.
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

    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]:
        """Эмбеддинги ДОКУМЕНТОВ (без инструкции). Пустые строки → точка."""
        out: list[list[float]] = []
        batch = batch or settings.embed_batch_size
        for i in range(0, len(texts), batch):
            chunk = [t.strip()[:8000] or "." for t in texts[i : i + batch]]
            resp = await self.client.embeddings.create(model=settings.embed_model, input=chunk)
            out.extend(d.embedding for d in resp.data)
        return out

    async def embed_query(self, query: str) -> list[float]:
        """Эмбеддинг ЗАПРОСА — с инструкцией (instruction-aware серии теряют
        1–5% без неё; пустая настройка отключает префикс)."""
        text = query.strip()[:8000] or "."
        if settings.embed_query_instruction:
            text = f"Instruct: {settings.embed_query_instruction}\nQuery: {text}"
        resp = await self.client.embeddings.create(model=settings.embed_model, input=[text])
        return resp.data[0].embedding


class Reranker:
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Релевантности (в исходном порядке texts) через /v1/rerank (Cohere-совместимый)."""
        if not texts:
            return []
        q = query[:2000]
        if settings.rerank_model.startswith("qwen3-reranker") and settings.rerank_instruction:
            q = f"<Instruct>: {settings.rerank_instruction}\n<Query>: {q}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.rerank_base_url}/v1/rerank",
                json={
                    "model": settings.rerank_model,
                    "query": q,
                    "documents": [t[:4000] for t in texts],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        scores = [0.0] * len(texts)
        for item in data["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores
