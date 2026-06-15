"""Быстрый контур перевода для виджета (roadmap § 3.3.C, § 4.1, § 12.1 п.5).

HY-MT1.5-7B (преемник WMT25-чемпиона Hunyuan-MT-7B) на GPU1 :8005 — низкая
латентность для перевода выделения и страниц; изолирован от batch-нагрузки
воркхорса. При недоступности — фолбэк на основной контур (без глоссария).
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.llm.client import needs_translation

logger = logging.getLogger(__name__)

# Официальный промпт-формат HY-MT / Hunyuan-MT (XX→XX, кроме китайских пар) — общий
_HY_PROMPT = "Translate the following segment into {lang}, without additional explanation.\n\n{text}"
_LANG_NAMES = {"ru": "Russian", "en": "English"}


class FastTranslator:
    def __init__(self) -> None:
        self.fast = AsyncOpenAI(
            base_url=settings.fast_llm_base_url, api_key="local", timeout=60.0
        )
        self.fallback = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0
        )

    async def translate(self, text: str, target_lang: str = "ru") -> tuple[str, str]:
        """→ (перевод, движок). Пустые/нелатинские тексты возвращаются как есть."""
        if target_lang == "ru" and not needs_translation(text):
            return text, "none"
        prompt = _HY_PROMPT.format(lang=_LANG_NAMES.get(target_lang, "Russian"), text=text)

        if settings.fast_llm_enabled:
            try:
                resp = await self.fast.chat.completions.create(
                    model=settings.fast_llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    # рекомендованные параметры Hunyuan-MT
                    temperature=0.7,
                    top_p=0.6,
                    extra_body={"top_k": 20, "repetition_penalty": 1.05},
                    max_tokens=2048,
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out, settings.fast_llm_model
            except Exception as exc:
                logger.warning("быстрый контур недоступен (%s) — фолбэк на Qwen3", exc)

        resp = await self.fallback.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты переводчик. Выводи только перевод, без пояснений.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            top_p=0.8,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip(), settings.llm_model
