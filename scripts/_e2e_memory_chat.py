"""Живой e2e слоя памяти ЧЕРЕЗ реальный чат-эндпоинт (не мимо него).

Поднимает FastAPI in-process (ASGITransport), подменяет ТОЛЬКО auth-зависимость
(сам 401-контур проверен отдельно), и гоняет POST /api/chat дважды для одного
пользователя:
  Сессия 1 (thread A): пользователь сообщает устойчивый факт/предпочтение.
  → ждём, пока live-воркер извлечёт и сконсолидирует кандидата в memory_items.
  Сессия 2 (thread B, новый): вопрос про тот факт → проверяем, что
    (а) пришло SSE-событие memory (блок инжектнут), (б) ответ содержит факт.
Это покрывает интеграцию chat.py: запись событий → classify needs_memory →
retrieve_block+gate → memory_block в промпт → enqueue extract. По завершении —
purge тестового пользователя.

Запуск на сервере: uv run python scripts/_e2e_memory_chat.py
"""

from __future__ import annotations

import asyncio
import json

import httpx
from fastapi import Request
from sqlalchemy import text as sql

from rag_app.api.auth import User, get_current_user
from rag_app.api.main import app

_TEST_SUB = "e2e-memory-test"


def _fake_user(request: Request) -> User:
    u = User(sub=_TEST_SUB, username="e2e", roles={"user"})
    request.state.user = u  # chat.py читает request.state.user
    return u


async def _chat(client: httpx.AsyncClient, message: str, *, session_id=None) -> dict:
    """POST /api/chat, собрать SSE-поток в структуру."""
    body = {"message": message, "session_id": session_id, "document_id": None}
    events: list[dict] = []
    answer = []
    async with client.stream("POST", "/api/chat", json=body, timeout=180) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if ev.get("type") == "delta":
                    answer.append(ev["text"])
    sid = next((e["session_id"] for e in events if e.get("type") == "session"), None)
    return {
        "sid": sid,
        "answer": "".join(answer).strip(),
        "memory_event": next((e for e in events if e.get("type") == "memory"), None),
        "mode": next((e for e in events if e.get("type") == "mode"), None),
        "events": events,
    }


async def _wait_for_memory(sm, timeout: float = 60.0) -> int:
    """Ждём, пока live-воркер создаст memory_items для тестового пользователя."""
    waited = 0.0
    while waited < timeout:
        async with sm() as db:
            n = (
                await db.execute(
                    sql(
                        "SELECT count(*) FROM memory_items WHERE user_id=:u"
                        " AND status='active' AND kind IN ('preference','fact','rule','task')"
                    ),
                    {"u": _TEST_SUB},
                )
            ).scalar_one()
        if n > 0:
            return n
        await asyncio.sleep(3)
        waited += 3
    return 0


async def main() -> None:
    app.dependency_overrides[get_current_user] = _fake_user
    async with app.router.lifespan_context(app):
        sessionmaker = app.state.sessionmaker  # lifespan поднял app.state
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://e2e") as client:
            # чистим прошлый прогон
            async with sessionmaker() as db:
                await app.state.memory.purge_user(db, _TEST_SUB)
                await db.commit()

            print("── Сессия 1 (сообщаем факт) ──")
            s1 = await _chat(
                client,
                "Запомни моё предпочтение: технические отчёты всегда присылай в формате XLSX. "
                "И зафиксируй факт: срок поставки винтовых насосов по договору 3086 — 90 дней.",
            )
            print(f"  sid={s1['sid']} mode={s1['mode']}")
            print(f"  ответ: {s1['answer'][:160]}")

            print("── Ждём извлечение памяти live-воркером ──")
            n = await _wait_for_memory(sessionmaker, timeout=75)
            print(f"  memory_items создано: {n}")
            async with sessionmaker() as db:
                items = (
                    await db.execute(
                        sql(
                            "SELECT kind, scope, content FROM memory_items"
                            " WHERE user_id=:u AND status='active' ORDER BY kind"
                        ),
                        {"u": _TEST_SUB},
                    )
                ).all()
            for it in items:
                print(f"    [{it.kind}/{it.scope}] {it.content[:90]}")

            print("── Сессия 2 (новый тред, тот же пользователь) ──")
            s2 = await _chat(
                client,
                "Напомни, в каком формате я просил присылать технические отчёты?",
            )
            print(f"  sid={s2['sid']} mode={s2['mode']}")
            print(f"  memory-событие: {s2['memory_event']}")
            print(f"  ответ: {s2['answer'][:200]}")

            ok_mem = s2["memory_event"] is not None
            ok_ans = "xlsx" in s2["answer"].lower()
            print("\n=== ВЕРДИКТ ===")
            print(f"  память извлечена воркером:        {'✓' if n > 0 else '✗'}")
            print(f"  сессия 2 инжектнула блок памяти:  {'✓' if ok_mem else '✗'}")
            print(f"  ответ сессии 2 содержит факт XLSX: {'✓' if ok_ans else '✗'}")
            verdict = (
                "ПАМЯТЬ РАБОТАЕТ КРОСС-СЕССИОННО"
                if (n > 0 and ok_mem and ok_ans)
                else "ЕСТЬ ПРОБЕЛ — см. выше"
            )
            print(f"  ИТОГ: {verdict}")

            # уборка
            async with sessionmaker() as db:
                await app.state.memory.purge_user(db, _TEST_SUB)
                await db.commit()
    app.dependency_overrides.clear()


if __name__ == "__main__":
    asyncio.run(main())
