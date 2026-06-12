"""Визуальный эмбеддер (§ 12.1 шаг 4): Qwen3-VL-Embedding-8B через vLLM.

Изображения и текст живут в одном пространстве: страница-картинка
эмбеддится без текста, запрос — текстом с инструкцией. Мультимодальный
вход у pooling-моделей vLLM идёт через endpoint /pooling (messages-формат).
Вектора MRL-усекаются до settings.visual_embed_dim (4096 → 1024) и
L2-нормируются на клиенте.
"""

from __future__ import annotations

import base64
import logging
import math

import httpx

from rag_app.config import settings

logger = logging.getLogger(__name__)


def _norm(vec: list[float]) -> list[float]:
    vec = vec[: settings.visual_embed_dim]
    s = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / s for x in vec]


class VisualEmbedder:
    async def _pool(self, messages: list[dict]) -> list[float]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.visual_embed_base_url}/pooling",
                json={"model": settings.visual_embed_model, "messages": messages},
            )
            resp.raise_for_status()
            data = resp.json()["data"][0]["data"]
        return _norm(data)

    async def embed_page(self, jpeg: bytes) -> list[float]:
        b64 = base64.b64encode(jpeg).decode()
        return await self._pool(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ],
                }
            ]
        )

    async def embed_text_query(self, query: str) -> list[float]:
        text = f"Instruct: {settings.visual_query_instruction}\nQuery: {query.strip()[:2000]}"
        return await self._pool([{"role": "user", "content": [{"type": "text", "text": text}]}])
