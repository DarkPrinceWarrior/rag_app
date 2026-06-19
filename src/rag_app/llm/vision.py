"""Генеративный VL-клиент (Qwen3.5-35B-A3B, мультимодальный, vLLM :8006).

В отличие от visual.py (эмбеддинги страниц для поиска) — этот РАСКРЫВАЕТ СМЫСЛ
изображения текстом: чертежи, P&ID, схемы, графики, фото оборудования, сканы.
Описание сразу на русском (домен нефтегаз/стройка), чтобы попасть в индекс/чат
без отдельного перевода. Использует воркхорс Qwen3.5 (отдельный Qwen3-VL-8B
ретайрнут 2026-06-19); картинка капается до vl_max_side (GPU3 тесная, ctx 8192).
"""

from __future__ import annotations

import base64
import io
import logging

from openai import AsyncOpenAI
from PIL import Image

from rag_app.config import settings

logger = logging.getLogger(__name__)


def _cap_image(data: bytes) -> bytes:
    """Уменьшить картинку до vl_max_side по большей стороне и пережать в JPEG —
    бьёт число vision-токенов (большой чертёж иначе переполняет контекст 8192)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((settings.vl_max_side, settings.vl_max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

_VL_SYSTEM = """\
Ты — инженер технической документации (нефтегаз, строительство). Тебе дают
изображение: чертёж, P&ID/технологическую схему, график, таблицу-картинку,
фото оборудования или скан страницы. Опиши и ОБЪЯСНИ его по-русски: что это,
что изображено, ключевые элементы/обозначения/связи, и какой инженерный смысл.
Если на изображении есть надписи/позиции/номера — перечисли важные. Пиши по делу,
без воды. Если это просто страница текста — кратко передай, о чём она."""

_DEFAULT_PROMPT = "Опиши и объясни, что изображено на этом техническом изображении."


class VisionClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.vl_base_url, api_key=settings.llm_api_key, timeout=180.0
        )
        self.model = settings.vl_model

    async def describe(self, image_png: bytes, prompt: str | None = None) -> str:
        """Картинка (PNG-байты) → текстовое описание/объяснение на русском."""
        b64 = base64.b64encode(_cap_image(image_png)).decode("ascii")
        content = [
            {"type": "text", "text": prompt or _DEFAULT_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _VL_SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            max_tokens=settings.vl_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()
