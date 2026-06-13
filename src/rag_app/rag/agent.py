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
from typing import Any

from openai import AsyncOpenAI

from rag_app.config import settings
from rag_app.rag.retrieve import RetrievedChunk
from rag_app.rag.tools import AgentTools

logger = logging.getLogger(__name__)

_THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

# --- Классификатор сложности -------------------------------------------------

_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["single_hop", "multi_hop"]},
        "reason": {"type": "string"},
    },
    "required": ["mode", "reason"],
    "additionalProperties": False,
}

_CLASSIFY_SYSTEM = """\
Ты — классификатор сложности запросов к технической документации. Определи, как
отвечать на вопрос пользователя:
- "single_hop": ответ находится в одном месте одним поиском (один факт, один
  раздел). Примеры: «что сказано про испытания», «найди раздел про ответственность»,
  «какое расчётное давление».
- "multi_hop": нужно несколько поисков / сбор по всему документу / сравнение
  разных мест / экстракция списка или таблицы / опора на прошлые реплики чата.
  Примеры: «вытащи все спецификации в таблицу», «сравни требования раздела 4 и 5»,
  «перечисли все сроки и штрафы», «сведи воедино всё про коррозию», «а как это
  соотносится с тем, что обсуждали выше».
Верни строгий JSON {mode, reason}. reason — одна короткая фраза по-русски."""

# эвристический fallback, если LLM-классификатор недоступен
_MULTI_HINTS = (
    "сравн", "сопостав", "все ", "всех", "всю", "кажд", "список", "перечисл",
    "таблиц", "сведи", "свод", "вытащ", "извлек", "собери", "сколько всего", "динамик",
)


async def classify(client: AsyncOpenAI, question: str, history: list[dict[str, str]]) -> tuple[str, str]:
    """Возвращает (mode, reason). multi_hop включает агентный цикл."""
    if not settings.agent_enabled:
        return "single_hop", "agent выключен"
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
        mode = data.get("mode")
        if mode in ("single_hop", "multi_hop"):
            return mode, data.get("reason", "")
    except Exception as exc:
        logger.warning("classify: LLM недоступен (%s) — эвристика", exc)
    ql = question.lower()
    return ("multi_hop" if any(h in ql for h in _MULTI_HINTS) else "single_hop"), "fallback-эвристика"


# --- Агентный цикл -----------------------------------------------------------

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "string",
            "enum": ["search_chunks", "get_section", "get_tables", "get_chat_history", "finish"],
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
