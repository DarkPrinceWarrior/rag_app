"""RAG-чат с документом/библиотекой (roadmap § 5): single-hop hybrid+rerank,
стрим токенов, обязательные цитаты [n].

Agentic-уровень (§ 5 п.7, multi-hop tool-цикл) — следующая итерация этапа;
все стоп-условия дизайна будут там.
"""

from __future__ import annotations

import base64
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.llm.vision import _cap_image
from rag_app.rag.retrieve import RetrievedChunk
from rag_app.storage.s3 import Storage

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
Ты — ассистент по корпоративной технической документации (нефтегаз, строительство, договоры).
Отвечай на русском языке, точно и по делу.

Правила:
1. Отвечай ТОЛЬКО на основании приведённых фрагментов документов. Если ответа
   в них нет — прямо скажи «В документах ответа не нашлось» и ничего не выдумывай.
2. После каждого утверждения ставь ссылку на фрагмент в виде [n], где n — номер
   фрагмента. Ссылки обязательны.
3. Числа, единицы измерения и обозначения стандартов переноси без изменений.
4. Если фрагменты противоречат друг другу — отметь это явно.
5. Когда просят «свести в таблицу», «сравнить», «перечислить параметры» — оформляй
   ответ Markdown-таблицей (строка заголовка `| Колонка | Колонка |`, разделитель
   `|---|---|`, затем строки). Для сравнений колонки — это сравниваемые объекты.
6. Если фрагмент-рисунок приложен изображением (схема, чертёж, график, P&ID) —
   рассмотри его САМ: элементы, формулы, обозначения, числа внутри — и используй
   в ответе со ссылкой [n] на этот фрагмент."""

# Маршрут memory_only (§2.3.1): документного контекста нет — отвечаем из памяти
# о пользователе/проекте и истории диалога; цитаты [n] не требуются.
MEMORY_ONLY_SYSTEM_PROMPT = """\
Ты — ассистент по корпоративной технической документации (нефтегаз, строительство, договоры).
Отвечай на русском языке, кратко и по делу.

Этот вопрос — о пользователе/проекте, не о содержимом документов. Отвечай на
основании раздела «Память о пользователе и проекте» и истории диалога.

Важно: приложение АВТОМАТИЧЕСКИ запоминает устойчивые факты и предпочтения
пользователя между сессиями. Если пользователь просит что-то запомнить
(имя, предпочтение, правило) — подтверди, что запомнил (это сохранится
автоматически), НЕ говори, что не умеешь сохранять данные. Если же у тебя
спрашивают факт, которого в памяти ещё нет, — честно скажи, что пока не
располагаешь им. Не выдумывай. Ссылки [n] не нужны (фрагментов документов нет)."""


def source_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Цитируемые источники: всё, КРОМЕ синтетического чанка-каталога
    (list_documents, kind='catalog'). Только они получают номер [n] и
    становятся кликабельными цитатами — иначе каталог занимал бы номер, на
    который модель ссылается, но клик-цитаты для него нет («висячая» ссылка).
    Нумерация [n] обязана совпадать здесь, в build_context_block,
    extract_citations и при вложении картинок — поэтому единый источник истины."""
    return [c for c in chunks if c.kind != "catalog"]


def _catalog_text(chunks: list[RetrievedChunk]) -> str | None:
    cat = next((c for c in chunks if c.kind == "catalog"), None)
    return (cat.text_ru or cat.text_en or "").strip() if cat else None


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    # Бюджет контекста: multi-hop собирает много фрагментов — без лимита промпт
    # перерастает окно модели. Клеим по убыванию ранга, пока влезает; хвост
    # (низкоранговые) отбрасываем. Нумеруем ТОЛЬКО источники (без каталога).
    parts = []
    total = 0
    for n, c in enumerate(source_chunks(chunks), 1):
        header = f"[{n}] {c.filename}"
        if c.heading_path:
            header += f" · {c.heading_path}"
        if c.page_start is not None:
            pages = f"стр. {c.page_start + 1}" + (
                f"–{c.page_end + 1}" if c.page_end is not None and c.page_end != c.page_start else ""
            )
            header += f" · {pages}"
        body = c.text_ru or c.text_en
        seg = f"{header}\n{body[:3000]}"
        if parts and total + len(seg) > settings.rag_context_max_chars:
            break
        parts.append(seg)
        total += len(seg)
    block = "\n\n---\n\n".join(parts)
    # Каталог библиотеки — справочно, БЕЗ номера [n]: модель использует его для
    # ответов «какие документы есть», но не должна ставить на него ссылку.
    catalog = _catalog_text(chunks)
    if catalog:
        head = f"Каталог библиотеки (справочно, НЕ источник для ссылок [n]):\n{catalog}"
        block = f"{head}\n\n===\n\n{block}" if block else head
    return block


_CITATION = re.compile(r"\[(\d{1,2})\]")


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Цитаты из ответа → метаданные чанков (страница, bbox для подсветки).
    Нумерация [n] — по source_chunks (без каталога), идентично build_context_block,
    поэтому каждый номер [n] из ответа имеет кликабельную цитату (нет «висячих»)."""
    sources = source_chunks(chunks)
    seen: list[dict[str, Any]] = []
    used: set[int] = set()
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if n in used or not (1 <= n <= len(sources)):
            continue
        used.add(n)
        c = sources[n - 1]
        seen.append(
            {
                "n": n,
                "chunk_id": str(c.id),
                "document_id": str(c.document_id),
                "filename": c.filename,
                "heading_path": c.heading_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "bboxes": (c.meta or {}).get("bboxes", [])[:20],
                "segment_ids": (c.meta or {}).get("segment_ids", [])[:30],
            }
        )
    return sorted(seen, key=lambda x: x["n"])


class ChatEngine:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=300.0
        )
        self.storage = Storage()  # вырезанные кропы рисунков → vision on-demand

    async def summarize_history(self, prior_summary: str | None, messages: list[Any]) -> str:
        """Инкрементальная сводка вытесненных из окна реплик (§ 5 п.5)."""
        convo = "\n".join(f"{m.role}: {m.content[:600]}" for m in messages)
        head = f"Текущая сводка диалога:\n{prior_summary}\n\n" if prior_summary else ""
        prompt = (
            f"{head}Новые реплики диалога:\n{convo}\n\n"
            "Обнови краткую сводку диалога на русском (до 6 пунктов: что спрашивал "
            "пользователь и ключевые факты/числа из ответов). Только сводка, без вступлений."
        )
        resp = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()

    async def stream_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
        summary: str | None = None,
        memory_block: str | None = None,
    ) -> AsyncIterator[str]:
        # Нет документного контекста (маршрут memory_only / пустой поиск, но есть
        # память) → отвечаем из памяти, не из строгого doc-only промпта.
        system = CHAT_SYSTEM_PROMPT if chunks else MEMORY_ONLY_SYSTEM_PROMPT
        if summary:
            system += f"\n\nКраткое содержание более ранней части диалога:\n{summary}"
        # Блок памяти — ОТДЕЛЬНО от фрагментов документов и как contextual hints
        # (§1, §6.2): не переопределяет факты/числа/цитаты документа.
        if memory_block:
            system += f"\n\n=== Память о пользователе и проекте ===\n{memory_block}"
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history[-settings.rag_history_messages :])
        if chunks:
            text_block = (
                f"Фрагменты документов:\n\n{build_context_block(chunks)}\n\n"
                f"Вопрос: {question}\n\n"
                "Ответь по правилам (цитаты [n] обязательны)."
            )
            # vision on-demand: к найденным рисункам прикладываем вырезанные кропы —
            # Qwen3.5 мультимодален, рассмотрит схему/формулы/обозначения на картинке
            content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
            attached = 0
            # та же нумерация, что в build_context_block/extract_citations (без каталога)
            for n, c in enumerate(source_chunks(chunks), 1):
                img_key = (c.meta or {}).get("img_s3")
                if not img_key or attached >= settings.rag_vision_max_images:
                    continue
                try:
                    data = await self.storage.get_bytes(settings.bucket_artifacts, img_key)
                    b64 = base64.b64encode(_cap_image(data)).decode("ascii")
                except Exception as exc:  # noqa: BLE001 — рисунок необязателен
                    logger.warning("vision attach [%d] %s: %s", n, img_key, exc)
                    continue
                content.append({"type": "text", "text": f"Изображение фрагмента [{n}]:"})
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
                attached += 1
            user_content: Any = content if attached else text_block
        else:
            user_content = (
                f"Вопрос: {question}\n\n"
                "Ответь на основании памяти о пользователе/проекте и истории диалога."
            )
        messages.append({"role": "user", "content": user_content})
        stream = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.2,
            top_p=0.8,
            max_tokens=2048,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def make_session_title(question: str) -> str:
    title = " ".join(question.split())
    return title[:77] + "…" if len(title) > 78 else title or "Новый чат"


def new_id() -> uuid.UUID:
    return uuid.uuid4()
