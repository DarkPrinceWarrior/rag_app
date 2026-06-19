"""Проба retrieval: находит ли чат описание рисунка с формулами (crop-режим)."""

from __future__ import annotations

import asyncio
import re

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.rag.retrieve import Retriever

QS = [
    "Какое уравнение записано в блоках выходного тензора на рисунке 9.1?",
    "Чему равен верхний левый элемент output при свёртке (рисунок 9.1)?",
]


async def main() -> None:
    eng = create_engine()
    sm = create_sessionmaker(eng)
    retr = Retriever(Embedder(), Reranker())
    for q in QS:
        async with sm() as db:
            chunks = await retr.retrieve(db, q)
        print("\nВОПРОС:", q)
        for c in chunks[:3]:
            txt = (getattr(c, "text_ru", "") or "").replace("\n", " ")
            has = "aw" in txt and "fz" in txt
            mark = "  <<< ФОРМУЛЫ aw...fz" if has else ""
            pg = (getattr(c, "page_start", 0) or 0) + 1
            print(f"  [{getattr(c,'kind','?')} стр.{pg}] {txt[:70]}{mark}")


if __name__ == "__main__":
    asyncio.run(main())
