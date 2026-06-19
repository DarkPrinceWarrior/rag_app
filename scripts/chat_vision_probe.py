"""Проба vision-on-demand в чате: вопрос про рисунок → кроп в Qwen3.5 → ответ."""

from __future__ import annotations

import asyncio

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.chat import ChatEngine
from rag_app.rag.retrieve import Retriever

Q = "Какое уравнение записано в блоках выходного тензора на рисунке 9.1?"


async def main() -> None:
    eng = create_engine()
    sm = create_sessionmaker(eng)
    retr = Retriever(Embedder(), Reranker())
    ce = ChatEngine()
    async with sm() as db:
        chunks = await retr.retrieve(db, Q)
    imgs = [c for c in chunks if (c.meta or {}).get("img_s3")]
    print(f"ВОПРОС: {Q}")
    print(f"чанков: {len(chunks)}, с кропом (пойдут в vision): {len(imgs)}")
    parts: list[str] = []
    async for d in ce.stream_answer(Q, chunks, []):
        parts.append(d)
    ans = "".join(parts)
    has = "aw" in ans and "fz" in ans
    print(f"\nОТВЕТ СОДЕРЖИТ ФОРМУЛЫ: {'✓ ДА' if has else '✗ нет'}\n")
    print(ans[:1400])


if __name__ == "__main__":
    asyncio.run(main())
