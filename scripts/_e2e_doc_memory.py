"""ТЗ §4.5 e2e: кросс-сессионный контекст по ОДНОМУ документу.

Сценарий заказчика: «вчера спросил про сроки [чат A по документу X], сегодня в
новом чате B по тому же документу — система помнит, о чём спрашивали». Проверяем
механизм напрямую (без 8+ реплик для свёртки): write_summary в треде A пишет
document-scoped сводку → новый тред B по тому же документу её достаёт; другой
документ Y её НЕ видит; повторная свёртка не плодит дубль.

Запуск на сервере: set -a && . ./.env.api.local && uv run python scripts/_e2e_doc_memory.py
"""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import text as sql

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.memory import MemoryService
from rag_app.rag.memory.rls import apply_scope_guc

_USER = "doc-mem-test"
_NEEDLE = "сроки поставки"


async def main() -> None:
    eng = create_engine()
    sm = create_sessionmaker(eng)
    mem = MemoryService(Embedder(), Reranker())
    doc_x, doc_y = uuid.uuid4(), uuid.uuid4()
    thread_a, thread_b = uuid.uuid4(), uuid.uuid4()

    async def purge() -> None:
        async with sm() as s:
            await mem.purge_user(s, _USER)
            await s.commit()

    await purge()

    # Чат A по документу X: свёртка истории → thread- и document-summary
    scope_a = mem.scope_for(_USER, document_id=doc_x, thread_id=thread_a)
    async with sm() as s:
        await mem.write_summary(
            s, scope_a,
            "Пользователь спрашивал про сроки поставки оборудования по этому документу; "
            "в ответе фигурировал срок 90 дней.")
        await s.commit()

    # Новый чат B по ТОМУ ЖЕ документу X (другой тред) — должен достать сводку
    scope_b = mem.scope_for(_USER, document_id=doc_x, thread_id=thread_b)
    async with sm() as s:
        _, hits_b = await mem.retrieve_block(s, "что я раньше спрашивал по этому документу?", scope_b)
    got = [h for h in hits_b if h.scope == "document" and _NEEDLE in h.content.lower()]

    # Другой документ Y — НЕ должен видеть сводку документа X (изоляция)
    scope_c = mem.scope_for(_USER, document_id=doc_y, thread_id=uuid.uuid4())
    async with sm() as s:
        _, hits_c = await mem.retrieve_block(s, "что я раньше спрашивал по этому документу?", scope_c)
    leak = [h for h in hits_c if h.scope == "document" and _NEEDLE in h.content.lower()]

    # Повторная свёртка в треде A — не плодит дубль document-summary (дедуп по ключу)
    async with sm() as s:
        await mem.write_summary(
            s, scope_a, "Обновлённая сводка: обсуждали сроки поставки и комплектацию.")
        await s.commit()
    async with sm() as s:
        await apply_scope_guc(s, mem.scope_for(_USER))
        n_doc = (
            await s.execute(
                sql("SELECT count(*) FROM memory_items WHERE user_id=:u"
                    " AND scope='document' AND kind='summary' AND status='active'"),
                {"u": _USER},
            )
        ).scalar_one()

    print(f"B (новый тред, тот же документ) достал сводку: {'✓' if got else '✗'}"
          + (f"  → {got[0].content[:60]!r}" if got else ""))
    print(f"документ Y не видит сводку X (изоляция):       {'✓' if not leak else '✗ УТЕЧКА'}")
    print(f"дедуп: одна document-summary на документ:       {'✓' if n_doc == 1 else f'✗ ({n_doc})'}")
    ok = bool(got) and not leak and n_doc == 1
    print("ИТОГ:", "ТЗ §4.5 РАБОТАЕТ (кросс-сессионный контекст по документу)" if ok else "ЕСТЬ ПРОБЕЛ")

    await purge()
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
