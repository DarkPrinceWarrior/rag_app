"""Визуальный реранкер Qwen3-VL-Reranker-2B (§ визуальный контур).

Cross-encoder: вход — (query, документ-страница-картинка), выход — relevance
score (сигмоид 0..1, выше = релевантнее). Раздаётся отдельным FastAPI-сервисом
`scripts/visual_rerank_server.py` через transformers на GPU2 (:8009).

Реранкер НЕ поднимается через vLLM: vllm#35412 даёт реверсивные скоры, а
vLLM 0.22/0.23 не знают архитектуру Qwen3VLForSequenceClassification. Поэтому
здесь — простой HTTP-клиент к собственному сервису (как llm/visual.py).
"""

from __future__ import annotations

import base64
import logging

import httpx

from rag_app.config import settings

logger = logging.getLogger(__name__)


class VisualReranker:
    async def rerank(self, query: str, pages: list[bytes]) -> list[float]:
        """Релевантности страниц (в исходном порядке `pages`) запросу `query`.

        Каждая страница — JPEG-байты → base64 → документ {image_b64} сервиса.
        Возвращает сигмоид-скоры 0..1 (выше = релевантнее). Пустой вход → [].
        """
        if not pages:
            return []
        documents = [
            {"image_b64": base64.b64encode(jpeg).decode()} for jpeg in pages
        ]
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.visual_rerank_base_url}/rerank",
                json={"query": query.strip()[:2000], "documents": documents},
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
        return [float(s) for s in scores]
