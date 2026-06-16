"""Adaptive agentic-RAG (roadmap § 5 п.7): роутинг по сложности + multi-hop
tool-цикл.

- Классификатор (structured output) делит запросы: single_hop — обычный
  hybrid+rerank; multi_hop — агентный сбор контекста несколькими tool-вызовами.
- Цикл «руками» (без LangChain), function calling эмулируется через
  response_format=json_schema (vLLM не поднят с --enable-auto-tool-choice).
- Стоп-условия обязательны (главный провал agentic RAG — незавершающийся цикл):
  ≤ agent_max_iters итераций, ≤ agent_token_budget токенов, ≤ agent_timeout_s
  wall-clock. Финальный ответ генерит существующий ChatEngine.stream_answer —
  контур цитат [n] не меняется.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.rag.retrieve import RetrievedChunk
from rag_app.rag.tools import AgentTools

logger = logging.getLogger(__name__)

_THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

# --- Классификатор сложности -------------------------------------------------

# QueryRouter (§2.3.1, §15.2): один проход решает и глубину документного поиска
# (mode), и нужна ли память. route — пять маршрутов спеки; mode выводится из него
# (agentic_multi_step → multi_hop), needs_memory включает retrieve_memories.
_ROUTES = ("doc_only", "memory_only", "doc_plus_memory", "agentic_multi_step", "clarification")

_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(_ROUTES)},
        "needs_memory": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["route", "needs_memory", "reason"],
    "additionalProperties": False,
}

_CLASSIFY_SYSTEM = """\
Ты — маршрутизатор запросов к технической документации с долговременной памятью
о пользователе. Определи маршрут:
- "doc_only": ответ целиком в документах одним поиском, память не нужна (один
  факт/раздел). Примеры: «что сказано про испытания», «какое расчётное давление».
- "doc_plus_memory": нужны документы И контекст прошлых обсуждений/предпочтений.
  Примеры: «а что там с поставкой, которую обсуждали», «продолжи по тому договору».
- "memory_only": ответ только из памяти о пользователе/проекте, без поиска по
  документам. Примеры: «какой формат отчёта я просил», «что ты обо мне помнишь».
- "agentic_multi_step": нужно несколько поисков / сбор по всему документу /
  сравнение мест / экстракция списка-таблицы, ЛИБО вопрос о составе библиотеки
  (какие документы загружены — отвечается инструментом list_documents, не
  поиском по тексту). Примеры: «вытащи все спецификации в таблицу», «сравни
  требования раздела 4 и 5», «сведи всё про коррозию», «какие документы
  загружены», «назови документ из библиотеки», «что есть в библиотеке».
- "clarification": вопрос неоднозначен, нужно уточнение.
needs_memory=true, если для ответа полезен контекст прошлых реплик/предпочтений
пользователя. Верни строгий JSON {route, needs_memory, reason}. reason — одна
короткая фраза по-русски."""

# эвристический fallback, если LLM-маршрутизатор недоступен
_MULTI_HINTS = (
    "сравн", "сопостав", "все ", "всех", "всю", "кажд", "список", "перечисл",
    "таблиц", "сведи", "свод", "вытащ", "извлек", "собери", "сколько всего", "динамик",
    "какие документ", "список документ", "в библиотеке", "назови документ",
)
_MEMORY_HINTS = (
    "обсужда", "ранее", "раньше", "помн", "запомн", "я говорил", "я просил",
    "как обычно", "мой формат", "моих", "предпочит", "тот договор", "тот документ",
    "выше по чату", "продолжи",
)


@dataclass
class Route:
    mode: str  # single_hop | multi_hop — глубина документного поиска
    reason: str
    route: str  # один из _ROUTES
    needs_memory: bool


def _route_to_mode(route: str) -> str:
    return "multi_hop" if route == "agentic_multi_step" else "single_hop"


async def classify(client: AsyncOpenAI, question: str, history: list[dict[str, str]]) -> Route:
    """Маршрутизация запроса: глубина документного поиска + нужна ли память."""
    if not settings.agent_enabled:
        return Route("single_hop", "agent выключен", "doc_only", False)
    hist = ""
    if history:
        hist = "Контекст диалога:\n" + "\n".join(
            f"{m['role']}: {m['content'][:160]}" for m in history[-3:]
        ) + "\n\n"
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"{hist}Вопрос: {question}"},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "classify", "strict": True, "schema": _CLASSIFY_SCHEMA},
            },
            extra_body=_THINK_OFF,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        route = data.get("route")
        if route in _ROUTES:
            return Route(
                _route_to_mode(route),
                data.get("reason", ""),
                route,
                bool(data.get("needs_memory", route in ("memory_only", "doc_plus_memory"))),
            )
    except Exception as exc:
        logger.warning("classify: LLM недоступен (%s) — эвристика", exc)
    ql = question.lower()
    multi = any(h in ql for h in _MULTI_HINTS)
    mem = any(h in ql for h in _MEMORY_HINTS)
    route = "agentic_multi_step" if multi else ("doc_plus_memory" if mem else "doc_only")
    return Route(_route_to_mode(route), "fallback-эвристика", route, mem or multi)


# --- Агентный цикл -----------------------------------------------------------

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "string",
            "enum": [
                "search_chunks", "get_section", "get_tables",
                "get_chat_history", "list_documents", "finish",
            ],
        },
        "query": {"type": "string"},
        "ref": {"type": "string"},
        "document_id": {"type": "string"},
    },
    "required": ["thought", "action"],
    "additionalProperties": False,
}

_AGENT_SYSTEM = """\
Ты — исследовательский агент по технической документации (нефтегаз, стройка,
договоры). Задача — СОБРАТЬ достаточный контекст, чтобы потом дать полный ответ.
Вызывай инструменты по одному за шаг:
- search_chunks(query): гибридный поиск фрагментов (делай разные запросы для
  сравнения и сбора по всему документу);
- get_section(ref): полный текст фрагмента по ref из результатов поиска;
- get_tables(document_id): все таблицы документа (для «вытащи спецификации» и т.п.);
- get_chat_history(): предыдущие реплики этого чата (для «как обсуждали выше»);
- list_documents(): каталог библиотеки — какие документы загружены (НЕ их
  содержимое). Для «какие документы есть», «назови документ из библиотеки»;
- finish: когда собранного достаточно для полного ответа.
Правила: ровно один инструмент за шаг (поле action); query/ref/document_id —
только нужный; thought — одна короткая фраза. Не зацикливайся: как только
фрагментов хватает — action=finish."""


class AgentLoop:
    def __init__(self, client: AsyncOpenAI, tools: AgentTools) -> None:
        self.client = client
        self.tools = tools
        self.chunks: list[RetrievedChunk] = []
        self.tokens = 0
        self.steps: list[dict[str, Any]] = []
        self.stop_reason = "finish"

    async def gather(self, question: str, history: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
        """Гоняет tool-цикл, по ходу yield'ит шаги; в конце наполняет self.chunks."""
        start = time.monotonic()
        messages: list[dict[str, str]] = [{"role": "system", "content": _AGENT_SYSTEM}]
        if history:
            hist = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-4:])
            messages.append({"role": "user", "content": f"Прошлые реплики чата:\n{hist}"})
        messages.append({"role": "user", "content": f"Вопрос: {question}\nНачинай сбор контекста."})

        self.stop_reason = "max_iters"
        for it in range(settings.agent_max_iters):
            if time.monotonic() - start > settings.agent_timeout_s:
                self.stop_reason = "timeout"
                break
            if self.tokens > settings.agent_token_budget:
                self.stop_reason = "token_budget"
                break
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=400,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "agent_action", "strict": True, "schema": _ACTION_SCHEMA},
                    },
                    extra_body=_THINK_OFF,
                )
            except Exception as exc:
                logger.warning("agent: шаг LLM упал (%s)", exc)
                self.stop_reason = "llm_error"
                break
            if resp.usage:
                self.tokens += resp.usage.total_tokens
            content = resp.choices[0].message.content or "{}"
            try:
                act = json.loads(content)
            except json.JSONDecodeError:
                self.stop_reason = "parse_error"
                break
            messages.append({"role": "assistant", "content": content})
            action = act.get("action", "")
            if action == "finish":
                self.stop_reason = "finish"
                break
            obs = await self.tools.dispatch(
                action,
                query=act.get("query", ""),
                ref=act.get("ref", ""),
                document_id=act.get("document_id", ""),
            )
            arg = act.get("query") or act.get("ref") or act.get("document_id") or ""
            thought = str(act.get("thought", ""))
            self.steps.append({"iter": it + 1, "tool": action, "arg": arg, "thought": thought})
            yield {"type": "step", "tool": action, "arg": str(arg)[:80], "thought": thought[:140]}
            messages.append(
                {"role": "user", "content": f"Наблюдение:\n{obs[:4000]}\n\nПродолжай или action=finish."}
            )

        evidence = list(self.tools.evidence.values())
        if not evidence:
            # подстраховка: ни один поиск не дал результата — прямой single-hop,
            # чтобы ответу было на что опереться
            async with self.tools.sessionmaker() as db:
                evidence = await self.tools.retriever.retrieve(
                    db, question, document_id=self.tools.document_id,
                    folder_id=self.tools.folder_id, top_k=settings.rag_context_top_k,
                )
        self.chunks = await self._rerank_top(question, evidence)
        yield {
            "type": "agent_summary",
            "iters": len(self.steps),
            "chunks": len(self.chunks),
            "stop": self.stop_reason,
            "tokens": self.tokens,
        }

    async def _rerank_top(self, question: str, evidence: list[RetrievedChunk]) -> list[RetrievedChunk]:
        cap = settings.agent_max_context_chunks
        if len(evidence) <= cap:
            return evidence
        try:
            scores = await self.tools.retriever.reranker.rerank(
                question, [c.text_ru or c.text_en for c in evidence]
            )
            for c, s in zip(evidence, scores, strict=True):
                c.score = s
            evidence.sort(key=lambda c: -c.score)
        except Exception as exc:
            logger.warning("agent: финальный rerank упал (%s)", exc)
        return evidence[:cap]
