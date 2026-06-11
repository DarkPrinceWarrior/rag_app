"""Клиент перевода через vLLM (OpenAI-совместимый API).

Qwen3 — reasoning-модель; для перевода thinking отключается через
chat_template_kwargs (enable_thinking=False), иначе в ответ попадает
<think>-блок и латентность растёт в разы.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from rag_app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Ты — профессиональный переводчик технической документации с английского на русский.
Домен: нефтегазовая отрасль, строительство, договоры и спецификации.

Правила:
1. Числа, единицы измерения, артикулы, обозначения стандартов (ГОСТ, ISO, API, ASTM),
   номера пунктов и формулы переноси без изменений.
2. Сохраняй разметку исходного текста (Markdown, LaTeX), если она есть.
3. Имена собственные компаний и продуктов не переводи.
4. Выводи ТОЛЬКО перевод — без пояснений, комментариев и кавычек-обёрток.
5. Если текст уже на русском или переводить нечего (число, код, обозначение) —
   верни его без изменений."""

# Есть ли в тексте латинские буквы — иначе переводить нечего (числа, кириллица).
_HAS_LATIN = re.compile(r"[A-Za-z]")


def needs_translation(text: str) -> bool:
    return bool(text.strip()) and bool(_HAS_LATIN.search(text))


@dataclass
class SegmentContext:
    heading: str | None = None  # заголовок текущего раздела
    prev_text: str | None = None  # предыдущий абзац (оригинал)


class Translator:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=300.0,
        )
        self.model = settings.llm_model

    async def translate(self, text: str, context: SegmentContext | None = None) -> str:
        if not needs_translation(text):
            return text

        parts: list[str] = []
        if context and context.heading:
            parts.append(f"Текущий раздел документа: {context.heading.strip()[:300]}")
        if context and context.prev_text:
            prev = context.prev_text.strip()[:1000]
            parts.append(f"Предыдущий абзац (только контекст, НЕ переводить):\n{prev}")
        parts.append(f"Переведи на русский:\n{text}")
        user_prompt = "\n\n".join(parts)

        last_err: Exception | None = None
        for attempt in range(settings.translate_max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    top_p=0.8,
                    max_tokens=settings.llm_max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out
                raise ValueError("пустой ответ модели")
            except Exception as exc:  # сеть/перегрузка/пустой ответ → ретрай
                last_err = exc
                logger.warning("перевод: попытка %d не удалась: %s", attempt + 1, exc)
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"перевод не удался после {settings.translate_max_retries} попыток: {last_err}")
