"""Проба визуального контура в Retriever: визуальный recall + реранк → image-чанки."""

from __future__ import annotations

import asyncio
import uuid

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.llm.visual import VisualEmbedder
from rag_app.llm.visual_reranker import VisualReranker
from rag_app.rag.retrieve import Retriever
from rag_app.storage.s3 import Storage

DOC = uuid.UUID("5352cf83-9a3d-47ae-991b-5f7e653b4281")
QS = [
    "схема разреженной связности нейронной сети",
    "операция 2D-свёртки",
    "инвариантность max-pooling",
]


async def main() -> None:
    eng = create_engine()
    sm = create_sessionmaker(eng)
    r = Retriever(Embedder(), Reranker(), VisualEmbedder(), VisualReranker(), Storage())
    for q in QS:
        async with sm() as db:
            ch = await r.retrieve(db, q, document_id=DOC)
        nimg = sum(1 for c in ch if (c.meta or {}).get("img_s3"))
        print(f"\nQ: {q}  ->  {len(ch)} чанков, image с кропом: {nimg}")
        for c in ch:
            tag = "IMG" if (c.meta or {}).get("img_s3") else "txt"
            txt = (c.text_ru or "").replace("\n", " ")[:48]
            print(f"  [{c.kind}/{tag} стр.{(c.page_start or 0) + 1}] {c.score:.3f} {txt}")


if __name__ == "__main__":
    asyncio.run(main())
