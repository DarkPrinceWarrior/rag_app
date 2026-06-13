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
    mode, reason = asyncio.run(
        classify(_FAKE_CLIENT, "Сравни раздел 4 и 5 и сведи всё в таблицу", [])
    )
    assert mode == "multi_hop"
    assert "эвристика" in reason


def test_classify_fallback_single_hop() -> None:
    mode, _ = asyncio.run(classify(_FAKE_CLIENT, "Что такое расчётное давление", []))
    assert mode == "single_hop"


def test_dispatch_unknown_tool() -> None:
    tools = AgentTools(None, None, document_id=None, folder_id=None, session_id=uuid.uuid4())
    out = asyncio.run(tools.dispatch("nonsense"))
    assert out.startswith("неизвестный инструмент")


def test_ref_is_eight_chars() -> None:
    assert len(_ref(uuid.uuid4())) == 8
