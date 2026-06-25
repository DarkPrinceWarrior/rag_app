"""Контур перевода Hy-MT2 (roadmap § 3.3.C, § 4.1, § 12.1 п.5).

Hy-MT2-7B (спец-MT, преемник HY-MT1.5/Hunyuan-MT) на GPU1 :8005 — ВЕСЬ перевод:
быстрый контур виджета (`FastTranslator`, низкая латентность) и документы
(`HyMTDocTranslator`, нативный шаблон + глоссарий). Qwen3.5 переводчиком не
работает — фолбэка на него нет.
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.llm.client import SegmentContext, has_cjk, needs_translation

logger = logging.getLogger(__name__)

# Официальные промпт-форматы Hunyuan-MT: для пар БЕЗ китайского — английская
# инструкция; для пар С китайским (zh↔ru/en) — китайская инструкция (модель
# обучена именно так, иначе китайский контур деградирует — см. model card).
_HY_PROMPT = "Translate the following segment into {lang}, without additional explanation.\n\n{text}"
_HY_PROMPT_ZH = "把下面的文本翻译成{lang}，不要额外解释。\n\n{text}"
# Терминологическая интервенция HY-MT (term-anchored префикс): принудительный
# перевод терминов глоссария. Формат проверен на :8005 (sour service →
# «сероводородная среда»; без префикса HY-MT даёт «агрессивная среда»).
_HY_TERM_PREFIX = "Refer to the following terminology:\n{terms}\n"
_LANG_NAMES = {"ru": "Russian", "en": "English", "zh": "Chinese"}
# Названия языков для китайского шаблона (целевой язык — по-китайски).
_LANG_NAMES_ZH = {"ru": "俄语", "en": "英语", "zh": "中文"}


def build_fast_prompt(
    text: str,
    target_lang: str,
    glossary: list[tuple[str, str]] | None,
    source_lang: str = "en",
) -> str:
    """Промпт Hunyuan-MT по направлению source→target (+ опц. глоссарий).
    target_lang/source_lang — коды (en|ru|zh). Шаблон выбираем по фактическому
    скрипту СЕГМЕНТА: текст с иероглифами → китайская инструкция Hunyuan; чисто
    латинская/английская вставка (даже внутри zh-документа) → обычная инструкция
    — иначе китайский шаблон поверх английского текста даёт деградацию."""
    if has_cjk(text) or target_lang == "zh":
        prompt = _HY_PROMPT_ZH.format(lang=_LANG_NAMES_ZH.get(target_lang, "中文"), text=text)
    else:
        prompt = _HY_PROMPT.format(lang=_LANG_NAMES.get(target_lang, "Russian"), text=text)
    if glossary:
        terms = "\n".join(f"{en} translates to {ru}" for en, ru in glossary)
        prompt = _HY_TERM_PREFIX.format(terms=terms) + prompt
    return prompt


class FastTranslator:
    """Быстрый контур (виджет/выделение/страница) — ТОЛЬКО Hy-MT2 (:8005).
    Qwen3.5-фолбэка нет (Qwen3.5 переводчиком не работает): при недоступности
    Hy-MT2 — ретраи, затем RuntimeError (контур вернёт ошибку, не подменит движок)."""

    def __init__(self) -> None:
        self.fast = AsyncOpenAI(
            base_url=settings.fast_llm_base_url, api_key="local", timeout=60.0
        )

    async def translate(
        self,
        text: str,
        target_lang: str = "ru",
        glossary: list[tuple[str, str]] | None = None,
        source_lang: str = "en",
    ) -> tuple[str, str]:
        """→ (перевод, движок). Если в тексте нет букв НЕцелевого скрипта —
        возвращаем как есть. glossary — терминологическая интервенция Hy-MT (опц.)."""
        if not needs_translation(text, source_lang, target_lang):
            return text, "none"
        prompt = build_fast_prompt(text, target_lang, glossary, source_lang)
        last_err: Exception | None = None
        for attempt in range(settings.translate_max_retries):
            try:
                resp = await self.fast.chat.completions.create(
                    model=settings.fast_llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # рекомендованные параметры Hy-MT
                    top_p=0.6,
                    extra_body={"top_k": 20, "repetition_penalty": 1.05},
                    max_tokens=2048,
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out, settings.fast_llm_model
                raise ValueError("пустой ответ Hy-MT2")
            except Exception as exc:
                last_err = exc
                logger.warning("быстрый контур Hy-MT2: попытка %d не удалась: %s", attempt + 1, exc)
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"быстрый перевод Hy-MT2 не удался: {last_err}")


class HyMTDocTranslator:
    """Перевод ДОКУМЕНТОВ спец-MT-моделью Hy-MT2 (нативный шаблон + term-anchored
    глоссарий). БЕЗ scaffolding-лесов воркхорса (system-prompt, «Текущий раздел»,
    <doc>-маркеры — спец-MT эхо-копирует их в перевод) и БЕЗ Qwen3.5-фолбэка:
    перевод документов целиком на Hy-MT2 (выиграл COMET-A/B 2026-06-19). Сигнатура
    .translate(text, context, feedback) совпадает с Translator → цикл воркера и
    числовая валидация работают без изменений. При сбое — RuntimeError (как у
    Translator: сегмент в ошибку, документ не молчит)."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.fast_llm_base_url, api_key="local", timeout=300.0
        )
        self.model = settings.fast_llm_model

    async def translate(
        self,
        text: str,
        context: SegmentContext | None = None,
        feedback: str | None = None,
    ) -> str:
        src = context.source_lang if context else "en"
        tgt = context.target_lang if context else "ru"
        if not needs_translation(text, src, tgt):
            return text
        glossary = context.glossary if context else None
        prompt = build_fast_prompt(text, tgt, glossary, src)
        if feedback:
            # числовая валидация отклонила перевод → просим сохранить числа
            # (нативной инструкцией Hy-MT, без scaffolding'а воркхорса)
            prompt = "Keep all numbers, units and symbols exactly as in the source.\n" + prompt
        last_err: Exception | None = None
        for attempt in range(settings.translate_max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # детерминизм/консистентность под документы
                    top_p=0.8,
                    max_tokens=settings.llm_max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out
                raise ValueError("пустой ответ Hy-MT2")
            except Exception as exc:
                last_err = exc
                logger.warning("док-перевод Hy-MT2: попытка %d не удалась: %s", attempt + 1, exc)
                await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"перевод Hy-MT2 не удался после {settings.translate_max_retries} попыток: {last_err}"
        )
