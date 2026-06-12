"""A/B перевода (§ 12.1 п.2): модель B переводит ТЕ ЖЕ сегменты тем же
пайплайном (контекст раздела + глоссарий), что и прод-модель A.

Выход: два jsonl для eval_comet.py --jsonl (одинаковый набор source → числа
сравнимы напрямую) + замер скорости обеих моделей живьём на подвыборке.

Запуск (на сервере):
  uv run python scripts/ab_translate.py \
      --b-url http://127.0.0.1:8005/v1 --b-model qwen36-27b-awq [--n 300]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import GlossaryTerm, Segment, SegmentKind
from rag_app.llm.client import SegmentContext, Translator, pick_glossary_terms


async def load_segments(n: int) -> tuple[list[Segment], dict]:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    async with sessionmaker() as session:
        rows = (
            (
                await session.execute(
                    select(Segment)
                    .where(
                        Segment.kind.in_([SegmentKind.paragraph, SegmentKind.heading]),
                        Segment.translated_text.is_not(None),
                        Segment.translated_text != Segment.source_text,
                    )
                    .order_by(Segment.id)
                    .limit(n)
                )
            )
            .scalars()
            .all()
        )
        # все сегменты их документов — для восстановления контекстов как в проде
        doc_ids = {s.document_id for s in rows}
        all_rows = (
            (
                await session.execute(
                    select(Segment).where(Segment.document_id.in_(doc_ids)).order_by(Segment.idx)
                )
            )
            .scalars()
            .all()
        )
        terms = sorted(
            (await session.execute(select(GlossaryTerm.en_term, GlossaryTerm.ru_term))).all(),
            key=lambda t: -len(t[0]),
        )
    await engine.dispose()

    by_doc: dict = defaultdict(list)
    for s in all_rows:
        by_doc[s.document_id].append(s)
    contexts: dict = {}
    for doc_segments in by_doc.values():
        cur_heading = prev_text = None
        for seg in doc_segments:
            terms_found = pick_glossary_terms(seg.source_text, terms)
            if seg.kind == SegmentKind.heading:
                contexts[seg.id] = SegmentContext(glossary=terms_found)
                cur_heading, prev_text = seg.source_text, None
                continue
            contexts[seg.id] = SegmentContext(
                heading=cur_heading, prev_text=prev_text, glossary=terms_found
            )
            if seg.kind == SegmentKind.paragraph:
                prev_text = seg.source_text
    return list(rows), contexts


async def translate_all(
    translator: Translator, segments: list[Segment], contexts: dict, concurrency: int
) -> tuple[dict, float]:
    sem = asyncio.Semaphore(concurrency)
    out: dict = {}

    async def work(seg: Segment) -> None:
        async with sem:
            out[seg.id] = await translator.translate(seg.source_text, contexts[seg.id])

    t0 = time.monotonic()
    await asyncio.gather(*(work(s) for s in segments))
    return out, time.monotonic() - t0


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--b-url", required=True)
    p.add_argument("--b-model", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--speed-sample", type=int, default=30)
    p.add_argument("--out-dir", default="/tmp")
    args = p.parse_args()

    segments, contexts = await load_segments(args.n)
    print(f"сегментов: {len(segments)}")

    model_b = Translator(base_url=args.b_url, model=args.b_model)
    mt_b, sec_b = await translate_all(model_b, segments, contexts, concurrency=12)
    print(f"B ({args.b_model}): {len(segments)} сегментов за {sec_b:.0f} с")

    # скорость A живьём на подвыборке (качество A берём из БД — прод-переводы)
    model_a = Translator()
    sample = segments[: args.speed_sample]
    _, sec_a = await translate_all(model_a, sample, contexts, concurrency=12)
    print(f"A (прод, живой замер): {len(sample)} сегментов за {sec_a:.0f} с")
    print(f"скорость B на той же подвыборке: ~{sec_b * len(sample) / len(segments):.0f} с (пропорция)")

    out = Path(args.out_dir)
    with (out / "ab_a.jsonl").open("w", encoding="utf-8") as fa:
        for s in segments:
            fa.write(json.dumps({"src": s.source_text, "mt": s.translated_text}, ensure_ascii=False) + "\n")
    with (out / "ab_b.jsonl").open("w", encoding="utf-8") as fb:
        for s in segments:
            fb.write(json.dumps({"src": s.source_text, "mt": mt_b[s.id]}, ensure_ascii=False) + "\n")
    print(f"пары: {out}/ab_a.jsonl (прод) и {out}/ab_b.jsonl ({args.b_model})")


if __name__ == "__main__":
    asyncio.run(main())
