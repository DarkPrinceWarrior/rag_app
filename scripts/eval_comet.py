"""COMET-бейзлайн качества перевода (roadmap § 3.4 п.4, § 11 этап 5).

Reference-free QE: wmt22-cometkiwi-da по EN→RU парам сегментов из БД.
Бейзлайн фиксируется для будущего A/B (Qwen3-32B vs Qwen3.6-27B, § 12.1).

Запуск на сервере (модель ~2.3 ГБ, GPU5):
  CUDA_VISIBLE_DEVICES=5 uv run python scripts/eval_comet.py [лимит=500]

Зависимость ставится отдельно (тяжёлая): uv add --dev unbabel-comet
"""

from __future__ import annotations

import asyncio
import statistics
import sys

from sqlalchemy import select

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Segment, SegmentKind


async def fetch_pairs(limit: int) -> list[dict]:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    async with sessionmaker() as session:
        rows = (
            (
                await session.execute(
                    select(Segment.source_text, Segment.translated_text)
                    .where(
                        Segment.kind.in_([SegmentKind.paragraph, SegmentKind.heading]),
                        Segment.translated_text.is_not(None),
                        Segment.translated_text != Segment.source_text,
                    )
                    .limit(limit)
                )
            )
            .all()
        )
    await engine.dispose()
    return [{"src": en, "mt": ru} for en, ru in rows]


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    pairs = asyncio.run(fetch_pairs(limit))
    if not pairs:
        print("нет переведённых сегментов")
        return
    print(f"пар EN→RU: {len(pairs)}")

    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    out = model.predict(pairs, batch_size=32, gpus=1)
    scores = out["scores"]
    print(
        f"\nCOMETKiwi (reference-free, 0..1):\n"
        f"  средний: {out['system_score']:.4f}\n"
        f"  медиана: {statistics.median(scores):.4f}\n"
        f"  p10:     {sorted(scores)[int(len(scores) * 0.1)]:.4f}\n"
        f"  доля < 0.5 (подозрительные): {sum(s < 0.5 for s in scores) / len(scores):.1%}"
    )


if __name__ == "__main__":
    main()
