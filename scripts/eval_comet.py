"""COMET-бейзлайн качества перевода (roadmap § 3.4 п.4, § 11 этап 5).

Reference-free QE: wmt22-cometkiwi-da по EN→RU парам сегментов из БД.
Бейзлайн фиксируется для будущего A/B (Qwen3-32B vs Qwen3.6-27B, § 12.1).

Зависимости конфликтуют с mineru (transformers) → отдельный venv:
  uv venv /root/services/comet-eval/.venv --python 3.12
  uv pip install --python /root/services/comet-eval/.venv/bin/python \
      unbabel-comet "sqlalchemy[asyncio]" asyncpg pydantic-settings pgvector python-dotenv

Запуск на сервере (модель ~2.3 ГБ, GPU5):
  cd /root/projects/rag_app && PYTHONPATH=src CUDA_VISIBLE_DEVICES=5 \
      /root/services/comet-eval/.venv/bin/python scripts/eval_comet.py [лимит=500]
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
    arg = sys.argv[1] if len(sys.argv) > 1 else "500"
    if arg.endswith(".jsonl"):
        # режим A/B (§ 12.1): готовые пары из scripts/ab_translate.py
        import json

        with open(arg, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        print(f"файл: {arg}")
    else:
        pairs = asyncio.run(fetch_pairs(int(arg)))
    if not pairs:
        print("нет переведённых сегментов")
        return
    print(f"пар EN→RU: {len(pairs)}")

    from comet import download_model, load_from_checkpoint

    # cometkiwi-22 — gated (нужен HF-токен с принятой CC-BY-NC лицензией);
    # фолбэк — открытая wmt20-comet-qe-da (тоже reference-free). Для A/B важна
    # консистентность метрики между прогонами, не её абсолютные значения.
    model_path = None
    for name in ("Unbabel/wmt22-cometkiwi-da", "wmt20-comet-qe-da-v2", "wmt20-comet-qe-da"):
        try:
            model_path = download_model(name)
            print(f"метрика: {name}")
            break
        except Exception as exc:
            print(f"  {name}: недоступна ({str(exc)[:80]})")
    if model_path is None:
        raise SystemExit("ни одна QE-модель не скачалась")
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
