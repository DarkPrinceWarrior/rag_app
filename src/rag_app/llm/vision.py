"""Генеративный VL-клиент (Qwen3.5-35B-A3B, мультимодальный, vLLM :8006).

В отличие от visual.py (эмбеддинги страниц для поиска) — этот РАСКРЫВАЕТ СМЫСЛ
изображения текстом: чертежи, P&ID, схемы, графики, фото оборудования, сканы.
Описание сразу на русском (домен-нейтрально, любая отрасль), чтобы попасть в индекс/чат
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
Ты — специалист по технической документации. Тебе дают изображение: чертёж,
принципиальную/технологическую схему, диаграмму, график, таблицу-картинку,
фото оборудования или скан страницы. Опиши и ОБЪЯСНИ его по-русски: что это,
что изображено, ключевые элементы/обозначения/связи, и какой технический смысл.
Если на изображении есть надписи/позиции/номера — перечисли важные. Пиши по делу,
без воды. Если это просто страница текста — кратко передай, о чём она."""

_DEFAULT_PROMPT = "Опиши и объясни, что изображено на этом техническом изображении."

# figure-sweep (pdf_text/docx/pptx): описать ТОЛЬКО рисунок на странице, иначе EMPTY.
_FIG_SYSTEM = """\
Ты — специалист по технической документации. Тебе дают изображение СТРАНИЦЫ
технического документа (любая отрасль). Опиши ТОЛЬКО визуальные объекты
(рисунок, схему, диаграмму, граф, график, иллюстрацию, фото оборудования), если
они на странице есть. Текст, заголовки, формулы и таблицы — игнорируй."""

_FIG_PROMPT = (
    "Если на странице есть рисунок / схема / диаграмма / граф / график / "
    "иллюстрация / фото — кратко опиши ПО-РУССКИ только этот объект: что "
    "изображено, ключевые элементы, связи, инженерный/смысловой смысл; если есть "
    "подпись (Figure N / Рис. N) — укажи её номер. Если страница содержит ТОЛЬКО "
    "текст, заголовки, формулы и таблицы без рисунков — ответь РОВНО одним словом: EMPTY"
)

_EMPTY_TOKENS = {"EMPTY", "ПУСТО", "НЕТ"}


class VisionClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.vl_base_url, api_key=settings.llm_api_key, timeout=180.0
        )
        self.model = settings.vl_model

    async def _complete(self, system: str, prompt: str, image_png: bytes) -> str:
        b64 = base64.b64encode(_cap_image(image_png)).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
            max_tokens=settings.vl_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()

    async def describe(self, image_png: bytes, prompt: str | None = None) -> str:
        """Картинка (PNG-байты) → описание/объяснение на русском. Описывает изображение
        ЦЕЛИКОМ (режим скана/чертежа: вся страница — рисунок)."""
        return await self._complete(_VL_SYSTEM, prompt or _DEFAULT_PROMPT, image_png)

    async def describe_figure(self, image_png: bytes) -> str:
        """Страница текстового документа → описание ТОЛЬКО рисунка/схемы/графика на ней,
        если он есть; для чисто текстовой страницы вернёт '' (VL отвечает EMPTY)."""
        desc = await self._complete(_FIG_SYSTEM, _FIG_PROMPT, image_png)
        if len(desc) < 8 or desc.upper().strip(".!? \n") in _EMPTY_TOKENS:
            return ""
        return desc
