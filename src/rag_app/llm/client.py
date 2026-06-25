"""Клиент перевода через vLLM (OpenAI-совместимый API).

Qwen3 — reasoning-модель; для перевода thinking отключается через
chat_template_kwargs (enable_thinking=False), иначе в ответ попадает
<think>-блок и латентность растёт в разы.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

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
4. Выводи ТОЛЬКО перевод — без пояснений, комментариев, кавычек-обёрток
   и без маркеров <doc>/</doc>. Контекст (раздел, предыдущий абзац) в ответ
   не включай никогда.
5. Если текст уже на русском или переводить нечего (число, код, обозначение) —
   верни его без изменений."""

# Буквы соответствующего скрипта — иначе переводить нечего (числа, символы).
# Скрипт зависит от ЯЗЫКА-ИСТОЧНИКА: en→латиница, ru→кириллица, zh→CJK.
_HAS_LATIN = re.compile(r"[A-Za-z]")
_HAS_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_HAS_CJK = re.compile(r"[㐀-鿿]")
_SOURCE_SCRIPT = {"en": _HAS_LATIN, "ru": _HAS_CYRILLIC, "zh": _HAS_CJK}


def needs_translation(text: str, source_lang: str = "en", target_lang: str = "ru") -> bool:
    """Нужен ли перевод сегмента в целевой язык. Цель задаёт «свой» скрипт:
    переводим, если в тексте есть буквы ЛЮБОГО иного скрипта (для ru — латиница
    ИЛИ CJK). Так в смешанном китайско-английском документе переводятся и
    иероглифы, и английские вставки. Чистый целевой текст и сегменты без букв
    (числа/коды/символы) — пропускаем. `source_lang` оставлен для совместимости
    сигнатуры; идентичность source==target отсекается выше по стеку."""
    if not text.strip():
        return False
    others = [pat for lang, pat in _SOURCE_SCRIPT.items() if lang != target_lang]
    return any(pat.search(text) for pat in others)


def has_cjk(text: str) -> bool:
    """Есть ли в тексте иероглифы CJK — для выбора китайского шаблона Hunyuan-MT
    по фактическому скрипту сегмента (а не по направлению документа)."""
    return bool(_HAS_CJK.search(text))


def detect_lang(text: str) -> str:
    """Определить язык-источник по преобладающему скрипту: ``ru`` | ``en`` | ``zh``.
    Домен фиксирован тремя языками (ТЗ §4.3), поэтому хватает счёта букв:
    русский/английский документы CJK не содержат вовсе → любая заметная доля
    иероглифов однозначно означает китайский; иначе кириллица vs латиница."""
    cjk = len(_HAS_CJK.findall(text))
    cyr = len(_HAS_CYRILLIC.findall(text))
    lat = len(_HAS_LATIN.findall(text))
    total = cjk + cyr + lat
    if total == 0:
        return "en"
    if cjk >= 0.10 * total:  # для ru/en это 0; китайский техтекст — много CJK + латинские термины
        return "zh"
    return "ru" if cyr >= lat else "en"


@dataclass
class SegmentContext:
    heading: str | None = None  # заголовок текущего раздела
    prev_text: str | None = None  # предыдущий абзац (оригинал)
    # утверждённые термины (EN, RU), найденные в этом сегменте — roadmap § 3.4 п.1
    glossary: list[tuple[str, str]] = field(default_factory=list)
    # направление перевода документа (по умолчанию EN→RU — текущий MVP)
    source_lang: str = "en"
    target_lang: str = "ru"


def pick_glossary_terms(
    text: str, terms: list[tuple[str, str]], limit: int = 10
) -> list[tuple[str, str]]:
    """Термины глоссария, встречающиеся в тексте (без учёта регистра, по границам слов)."""
    found: list[tuple[str, str]] = []
    low = text.lower()
    for en, ru in terms:
        en_low = en.lower()
        pos = low.find(en_low)
        if pos < 0:
            continue
        before_ok = pos == 0 or not low[pos - 1].isalnum()
        end = pos + len(en_low)
        after_ok = end >= len(low) or not low[end].isalnum()
        if before_ok and after_ok:
            found.append((en, ru))
            if len(found) >= limit:
                break
    return found


class Translator:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        """base_url/model/temperature переопределяются для A/B-стендов (§ 12.1)."""
        self.client = AsyncOpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=300.0,
        )
        self.model = model or settings.llm_model
        self.temperature = temperature

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

        parts: list[str] = []
        if context and context.heading:
            parts.append(f"Текущий раздел документа: {context.heading.strip()[:300]}")
        if context and context.prev_text:
            prev = context.prev_text.strip()[:1000]
            parts.append(f"Предыдущий абзац (только контекст, НЕ переводить):\n{prev}")
        if context and context.glossary:
            terms = "\n".join(f"- {en} → {ru}" for en, ru in context.glossary)
            parts.append(f"ОБЯЗАТЕЛЬНАЯ терминология (использовать именно эти переводы):\n{terms}")
        if feedback:
            parts.append(f"Предыдущая попытка перевода была отклонена. Причина: {feedback}")
        parts.append(
            f"Переведи на русский ТОЛЬКО текст между маркерами <doc> и </doc>:\n<doc>\n{text}\n</doc>"
        )
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
                    temperature=self.temperature,
                    top_p=0.8,
                    max_tokens=settings.llm_max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                out = (resp.choices[0].message.content or "").strip()
                if out.startswith("<doc>"):
                    out = out[len("<doc>"):].strip()
                if out.endswith("</doc>"):
                    out = out[: -len("</doc>")].strip()
                if out:
                    return out
                raise ValueError("пустой ответ модели")
            except Exception as exc:  # сеть/перегрузка/пустой ответ → ретрай
                last_err = exc
                logger.warning("перевод: попытка %d не удалась: %s", attempt + 1, exc)
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"перевод не удался после {settings.translate_max_retries} попыток: {last_err}")
