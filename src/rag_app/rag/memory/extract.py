"""Автоэкстракция кандидатов памяти (§6.1) + prompt-injection защита (§6.2).

Асинхронно после ответа (ARQ, не в latency). На вход — окно последних реплик
треда; на выход — строгий JSON-список устойчивых фактов/предпочтений/правил.
Содержимое документов НЕ пересказывается (это RAG). Кандидаты, пытающиеся
изменить поведение модели/политику, отбрасываются и логируются.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from rag_app.config import settings

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "update", "supersede", "delete"]},
                    "scope": {
                        "type": "string",
                        "enum": ["user", "project", "document", "thread", "org"],
                    },
                    "kind": {
                        "type": "string",
                        "enum": [
                            "preference", "fact", "glossary", "rule", "task", "correction", "summary"
                        ],
                    },
                    "content": {"type": "string"},
                    "sensitivity": {"type": "string", "enum": ["normal", "sensitive", "secret"]},
                    "confidence": {"type": "number"},
                },
                "required": ["action", "scope", "kind", "content", "sensitivity", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

_EXTRACT_SYSTEM = """\
Ты извлекаешь долговременную память о пользователе и проекте из диалога с
ассистентом по технической документации. Извлекай ТОЛЬКО устойчивые факты,
предпочтения, правила, термины глоссария и договорённости, полезные в будущих
сессиях.

Строго запрещено:
- пересказывать содержимое документов (это делает поиск по документам, не память);
- выдумывать: если устойчивого факта нет — верни пустой список candidates;
- сохранять разовые операционные реплики («покажи раздел 4», «спасибо»).

Для каждого кандидата укажи:
- scope: user (про самого пользователя), project (про папку/проект),
  document (про конкретный документ), thread (про этот диалог);
- kind: preference | fact | glossary | rule | task | correction | summary;
- content: короткая самодостаточная формулировка факта по-русски;
- sensitivity: normal | sensitive | secret;
- confidence: 0..1 — насколько уверенно это устойчивый факт;
- action: create (по умолчанию) | update | supersede | delete.

Верни строгий JSON {candidates:[...]}, без преамбулы и markdown."""

# §6.2: маркеры попытки управлять моделью/политикой — такие кандидаты не
# сохраняются (память — канал персистентных инъекций).
_INJECTION_MARKERS = (
    "ignore previous", "ignore the above", "ignore all", "disregard",
    "игнорируй", "не следуй", "забудь инструкц", "забудь правил", "обойди",
    "ты теперь", "you are now", "act as", "веди себя как",
    "system prompt", "системн", "developer message",
    "always output", "always respond", "всегда выдавай", "всегда отвеч",
    "переопредели", "override", "отключи", "disable safety", "jailbreak",
    "доступ ко всем", "права доступа", "grant access", "bypass", "sudo",
    "raw sql", "drop table", "exfiltrate",
)


def is_injection(content: str) -> bool:
    """Кандидат пытается изменить поведение/политику/права? (§6.2)."""
    low = (content or "").lower()
    return any(m in low for m in _INJECTION_MARKERS)


async def extract_candidates(client: AsyncOpenAI, transcript: str) -> list[dict[str, Any]]:
    """LLM-экстракция (Qwen3.5 structured-output). Ошибки → пустой список."""
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"Диалог:\n{transcript}\n\nИзвлеки кандидатов памяти."},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "memory_extract", "strict": True, "schema": _EXTRACT_SCHEMA},
            },
            extra_body=_THINK_OFF,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        out = data.get("candidates", [])
        return out if isinstance(out, list) else []
    except Exception as exc:
        logger.warning("memory extract: LLM недоступен/ошибка (%s)", exc)
        return []
