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
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from rag_app.config import settings
from rag_app.llm.vision import _cap_image
from rag_app.rag.grounding import (
    CitationGuardResult,
    CitationVerifier,
    ContextBudgetAudit,
    GroundingError,
    GroundingMode,
    compress_low_priority_chunks,
    count_chat_tokens,
)
from rag_app.rag.retrieve import RetrievedChunk
from rag_app.rag.selective_citations import LocalHttpClaimScoreBackend
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
6. Если вопрос явно о схеме/рисунке/чертеже/изображении и такой фрагмент приложен
   картинкой — рассмотри его САМ (элементы, формулы, обозначения, числа) и сошлись
   [n] на него. Если же спрашивают про текстовый пункт/раздел или «распиши пункт N»
   — отвечай по тексту фрагментов и истории диалога; НЕ своди ответ к рисунку,
   если о рисунке не спрашивали."""

# Маршрут out_of_scope: запрос НЕ про содержание документов (перевод, действие над
# присланным текстом, общий вопрос/приветствие). Документный поиск пропускаем;
# отвечаем как ассистент — без цитат [n] и без «в документах ответа не нашлось».
GENERAL_SYSTEM_PROMPT = """\
Ты — ассистент DocRAGenslate (корпоративный инструмент перевода и анализа документации).
Этот запрос — не вопрос к содержанию документов библиотеки.

- Если пользователь прислал текст и просит перевести, сократить, переформулировать
  или сделать саммари — выполни это с присланным текстом. Перевод по умолчанию на
  русский; если просят на другой язык — на него.
- Если просят перевести ВЕСЬ открытый документ (а не присланный фрагмент) на другой
  язык — объясни, что это делается на карточке документа: «Действия» → «Перевести на
  язык» → «English» или «китайский (упрощённый)». Чат переводит только присланный
  или выделенный текст, а не весь документ целиком.
- На приветствие или вопрос о возможностях ответь кратко и по делу.

НЕ говори «в документах ответа не нашлось» — это не поиск по документам.
Отвечай по-русски (если не просят на другом языке); ссылки [n] не нужны."""

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


def build_context_block(chunks: list[RetrievedChunk], *, legacy_chunk_chars: int | None = 3000) -> str:
    # Бюджет контекста: multi-hop собирает много фрагментов — без лимита промпт
    # перерастает окно модели. Клеим по убыванию ранга, пока влезает; хвост
    # (низкоранговые) отбрасываем. Нумеруем ТОЛЬКО источники (без каталога).
    parts: list[str] = []
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
        if legacy_chunk_chars is not None:
            body = body[:legacy_chunk_chars]
        seg = f"{header}\n{body}"
        if legacy_chunk_chars is not None and parts and total + len(seg) > settings.rag_context_max_chars:
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


def _content_chars(content: Any) -> int:
    """Объём текста сообщения (мультимодальный list — суммируем только текст)."""
    if isinstance(content, str):
        return len(content)
    return sum(len(p.get("text", "")) for p in content if p.get("type") == "text")


def _fit_context_window(messages: list[dict[str, Any]], n_images: int) -> None:
    """§4.5: не дать входу переполнить окно модели. Бюджет символов = (окно −
    ответ − запас − стоимость картинок) × символов-на-токен. При переборе режем
    историю (старейшие реплики, индексы между system и текущим вопросом), затем
    усекаем текст контекста — graceful-деградация вместо ошибки vLLM."""
    budget = (
        settings.chat_context_window
        - settings.chat_output_tokens
        - 400
        - n_images * settings.chat_image_tokens
    ) * settings.chat_chars_per_token
    budget = max(budget, 4000)

    def total() -> int:
        return sum(_content_chars(m["content"]) for m in messages)

    dropped = 0
    # messages[0] — system, messages[-1] — текущий вопрос; между ними история
    while total() > budget and len(messages) > 2:
        del messages[1]
        dropped += 1
    if total() > budget:  # даже без истории перебор — усечь текст контекста
        over = total() - budget
        last = messages[-1]["content"]
        if isinstance(last, str):
            messages[-1]["content"] = last[: max(500, len(last) - over)]
        else:
            for part in last:
                if part.get("type") == "text" and len(part["text"]) > over + 500:
                    part["text"] = part["text"][: len(part["text"]) - over]
                    break
    if dropped:
        logger.info("chat: окно близко к пределу — отброшено %d старых реплик истории", dropped)


@dataclass(frozen=True)
class PreparedChat:
    messages: list[dict[str, Any]]
    chunks: list[RetrievedChunk]
    budget_audit: ContextBudgetAudit | None


def _system_text(
    *,
    chunks: list[RetrievedChunk],
    route: str,
    summary: str | None,
    memory_block: str | None,
) -> str:
    if chunks:
        system = CHAT_SYSTEM_PROMPT
    elif route == "out_of_scope":
        system = GENERAL_SYSTEM_PROMPT
    else:
        system = MEMORY_ONLY_SYSTEM_PROMPT
    if summary:
        system += f"\n\nКраткое содержание более ранней части диалога:\n{summary}"
    if memory_block:
        system += f"\n\n=== Память о пользователе и проекте ===\n{memory_block}"
    return system


def _text_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]],
    *,
    summary: str | None,
    memory_block: str | None,
    route: str,
    legacy_context: bool,
) -> list[dict[str, Any]]:
    system = _system_text(
        chunks=chunks,
        route=route,
        summary=summary,
        memory_block=memory_block,
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history[-settings.rag_history_messages :])
    if chunks:
        context = build_context_block(
            chunks,
            legacy_chunk_chars=3000 if legacy_context else None,
        )
        user_content = (
            f"Фрагменты документов:\n\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            "Ответь по правилам (цитаты [n] обязательны)."
        )
    else:
        user_content = (
            f"Вопрос: {question}\n\nОтветь на основании памяти о пользователе/проекте и истории диалога."
        )
    messages.append({"role": "user", "content": user_content})
    return messages


def _potential_image_count(chunks: list[RetrievedChunk]) -> int:
    return min(
        sum(bool((chunk.meta or {}).get("img_s3")) for chunk in source_chunks(chunks)),
        settings.rag_vision_max_images,
    )


def _drop_lowest_source(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    for index in range(len(chunks) - 1, -1, -1):
        if chunks[index].kind != "catalog":
            return chunks[:index] + chunks[index + 1 :]
    return chunks


class ChatEngine:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=300.0
        )
        self.storage = Storage()  # вырезанные кропы рисунков → vision on-demand
        self.citation_verifier = CitationVerifier(
            self.client,
            model=settings.llm_model,
            max_tokens=settings.rag_citation_verifier_max_tokens,
            selective_backend=LocalHttpClaimScoreBackend(
                settings.rag_citation_verifier_url,
                model=settings.rag_citation_verifier_model,
                adapter=settings.rag_citation_verifier_backend,
                timeout_s=settings.rag_citation_verifier_timeout_s,
            ),
            selective_threshold=settings.rag_citation_verifier_threshold,
        )

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

    async def _fit_exact_context(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
        *,
        summary: str | None,
        memory_block: str | None,
        route: str,
        max_tokens: int,
        mode: GroundingMode,
    ) -> PreparedChat:
        candidate_chunks, compressed = compress_low_priority_chunks(
            chunks,
            question,
            after_rank=settings.rag_context_compress_after_rank,
            max_chars=settings.rag_context_compressed_chars,
        )
        candidate_history = list(history[-settings.rag_history_messages :])
        dropped_history = 0
        dropped_sources = 0
        while True:
            messages = _text_messages(
                question,
                candidate_chunks,
                candidate_history,
                summary=summary,
                memory_block=memory_block,
                route=route,
                legacy_context=False,
            )
            token_count = await count_chat_tokens(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                messages=messages,
            )
            max_model_len = min(token_count.max_model_len, settings.chat_context_window)
            input_limit = (
                max_model_len
                - max_tokens
                - settings.rag_context_reserve_tokens
                - (
                    settings.rag_citation_verifier_max_tokens
                    if settings.rag_citation_verification_mode != "off"
                    else 0
                )
                - _potential_image_count(candidate_chunks) * settings.chat_image_tokens
            )
            if input_limit < 1:
                raise GroundingError("резервы ответа и изображений исчерпали окно модели")
            if token_count.count <= input_limit:
                return PreparedChat(
                    messages=messages,
                    chunks=candidate_chunks,
                    budget_audit=ContextBudgetAudit(
                        mode=mode,
                        exact_tokens=token_count.count,
                        input_limit=input_limit,
                        max_model_len=max_model_len,
                        dropped_history=dropped_history,
                        dropped_sources=dropped_sources,
                        compressed_sources=compressed,
                    ),
                )
            if candidate_history:
                candidate_history.pop(0)
                dropped_history += 1
                continue
            if any(chunk.kind == "catalog" for chunk in candidate_chunks):
                candidate_chunks = [chunk for chunk in candidate_chunks if chunk.kind != "catalog"]
                continue
            if len(source_chunks(candidate_chunks)) > 1:
                candidate_chunks = _drop_lowest_source(candidate_chunks)
                dropped_sources += 1
                continue
            raise GroundingError("даже один целый источник не помещается в окно модели")

    async def _attach_images(
        self,
        messages: list[dict[str, Any]],
        chunks: list[RetrievedChunk],
    ) -> int:
        if not chunks:
            return 0
        text_block = messages[-1]["content"]
        content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
        attached = 0
        for number, chunk in enumerate(source_chunks(chunks), 1):
            img_key = (chunk.meta or {}).get("img_s3")
            if not img_key or attached >= settings.rag_vision_max_images:
                continue
            try:
                data = await self.storage.get_bytes(settings.bucket_artifacts, img_key)
                b64 = base64.b64encode(_cap_image(data)).decode("ascii")
            except Exception as exc:  # noqa: BLE001 - image evidence is optional
                logger.warning("vision attach [%d] %s: %s", number, img_key, exc)
                continue
            content.append({"type": "text", "text": f"Изображение фрагмента [{number}]:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            attached += 1
        if attached:
            messages[-1]["content"] = content
        return attached

    async def prepare_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
        *,
        summary: str | None = None,
        memory_block: str | None = None,
        route: str = "doc_only",
        max_tokens: int = 2048,
        budget_mode: GroundingMode | None = None,
    ) -> PreparedChat:
        mode = budget_mode or settings.rag_context_budget_mode
        exact: PreparedChat | None = None
        if mode in {"shadow", "enforce"}:
            try:
                exact = await self._fit_exact_context(
                    question,
                    chunks,
                    history,
                    summary=summary,
                    memory_block=memory_block,
                    route=route,
                    max_tokens=max_tokens,
                    mode=mode,
                )
            except GroundingError as exc:
                if mode == "enforce":
                    raise
                logger.warning("exact context budget shadow failed: %s", exc)
                exact = PreparedChat(
                    messages=[],
                    chunks=chunks,
                    budget_audit=ContextBudgetAudit(
                        mode=mode,
                        exact_tokens=None,
                        input_limit=None,
                        max_model_len=None,
                        tokenizer_error=str(exc),
                    ),
                )
        if mode == "enforce":
            assert exact is not None
            prepared = exact
        else:
            messages = _text_messages(
                question,
                chunks,
                history,
                summary=summary,
                memory_block=memory_block,
                route=route,
                legacy_context=True,
            )
            _fit_context_window(messages, _potential_image_count(chunks))
            prepared = PreparedChat(
                messages=messages,
                chunks=chunks,
                budget_audit=exact.budget_audit if exact else None,
            )
        await self._attach_images(prepared.messages, prepared.chunks)
        if prepared.budget_audit is not None:
            audit = prepared.budget_audit
            logger.info(
                "exact context budget: mode=%s tokens=%s limit=%s history_dropped=%d "
                "sources_dropped=%d sources_compressed=%d error=%s",
                audit.mode,
                audit.exact_tokens,
                audit.input_limit,
                audit.dropped_history,
                audit.dropped_sources,
                audit.compressed_sources,
                audit.tokenizer_error,
            )
        return prepared

    async def stream_prepared(
        self,
        prepared: PreparedChat,
        *,
        temperature: float = 0.2,
        top_p: float = 0.8,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=cast(list[ChatCompletionMessageParam], prepared.messages),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def verify_answer(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        *,
        mode: GroundingMode | None = None,
    ) -> CitationGuardResult:
        return await self.citation_verifier.guard(
            answer,
            chunks,
            mode=mode or settings.rag_citation_verification_mode,
        )

    async def stream_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
        summary: str | None = None,
        memory_block: str | None = None,
        route: str = "doc_only",
        temperature: float = 0.2,
        top_p: float = 0.8,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        prepared = await self.prepare_answer(
            question,
            chunks,
            history,
            summary=summary,
            memory_block=memory_block,
            route=route,
            max_tokens=max_tokens,
        )
        mode = settings.rag_citation_verification_mode
        if mode in {"enforce", "selective"} and prepared.chunks:
            parts: list[str] = []
            async for delta in self.stream_prepared(
                prepared,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            ):
                parts.append(delta)
            guarded = await self.verify_answer("".join(parts).strip(), prepared.chunks, mode=mode)
            yield guarded.answer
            return
        shadow_parts: list[str] = []
        async for delta in self.stream_prepared(
            prepared,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        ):
            if mode == "shadow" and prepared.chunks:
                shadow_parts.append(delta)
            yield delta
        if shadow_parts:
            await self.verify_answer("".join(shadow_parts).strip(), prepared.chunks, mode=mode)


def make_session_title(question: str) -> str:
    title = " ".join(question.split())
    return title[:77] + "…" if len(title) > 78 else title or "Новый чат"


def new_id() -> uuid.UUID:
    return uuid.uuid4()
