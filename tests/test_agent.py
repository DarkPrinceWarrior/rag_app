"""Agentic-RAG (§ 5 п.7): эвристический fallback классификатора и dispatch.

Сетевые шаги (LLM, поиск) здесь не дёргаются — проверяем чистую логику:
fallback-эвристику при недоступном LLM и роутинг неизвестного инструмента."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from rag_app.rag.agent import classify
from rag_app.rag.tools import AgentTools, _ref


async def _raise(**_kw):  # имитация недоступного vLLM
    raise RuntimeError("llm offline")


_FAKE_CLIENT = SimpleNamespace(
    chat=SimpleNamespace(completions=SimpleNamespace(create=_raise))
)


def test_classify_fallback_multi_hop() -> None:
    r = asyncio.run(classify(_FAKE_CLIENT, "Сравни раздел 4 и 5 и сведи всё в таблицу", []))
    assert r.mode == "multi_hop"
    assert r.route == "agentic_multi_step"
    assert "эвристика" in r.reason


def test_classify_fallback_single_hop() -> None:
    r = asyncio.run(classify(_FAKE_CLIENT, "Что такое расчётное давление", []))
    assert r.mode == "single_hop"
    assert r.route == "doc_only"
    assert r.needs_memory is False


def test_classify_fallback_memory_hint() -> None:
    # упоминание прошлых обсуждений → нужна память (doc_plus_memory)
    r = asyncio.run(classify(_FAKE_CLIENT, "А что там с тем договором, что обсуждали ранее", []))
    assert r.needs_memory is True
    assert r.route == "doc_plus_memory"


def test_dispatch_unknown_tool() -> None:
    tools = AgentTools(None, None, document_id=None, folder_id=None, session_id=uuid.uuid4())
    out = asyncio.run(tools.dispatch("nonsense"))
    assert out.startswith("неизвестный инструмент")


def test_ref_is_eight_chars() -> None:
    assert len(_ref(uuid.uuid4())) == 8
