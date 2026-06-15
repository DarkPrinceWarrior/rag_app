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
# Терминологическая интервенция HY-MT (term-anchored префикс): принудительный
# перевод терминов глоссария. Формат проверен на :8005 (sour service →
# «сероводородная среда»; без префикса HY-MT даёт «агрессивная среда»).
_HY_TERM_PREFIX = "Refer to the following terminology:\n{terms}\n"
_LANG_NAMES = {"ru": "Russian", "en": "English"}


def build_fast_prompt(text: str, lang: str, glossary: list[tuple[str, str]] | None) -> str:
    """Промпт HY-MT с опциональной терминологической интервенцией."""
    prompt = _HY_PROMPT.format(lang=lang, text=text)
    if glossary:
        terms = "\n".join(f"{en} translates to {ru}" for en, ru in glossary)
        prompt = _HY_TERM_PREFIX.format(terms=terms) + prompt
    return prompt


class FastTranslator:
    def __init__(self) -> None:
        self.fast = AsyncOpenAI(
            base_url=settings.fast_llm_base_url, api_key="local", timeout=60.0
        )
        self.fallback = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0
        )

    async def translate(
        self,
        text: str,
        target_lang: str = "ru",
        glossary: list[tuple[str, str]] | None = None,
    ) -> tuple[str, str]:
        """→ (перевод, движок). Пустые/нелатинские тексты возвращаются как есть.

        glossary — термины для терминологической интервенции HY-MT (опц.).
        """
        if target_lang == "ru" and not needs_translation(text):
            return text, "none"
        prompt = build_fast_prompt(text, _LANG_NAMES.get(target_lang, "Russian"), glossary)

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
