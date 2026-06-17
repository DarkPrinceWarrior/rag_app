"""dots.mocr (rednote-hilab/dots.mocr) как альтернативный парсер pdf_text.

Запускаем штатный `dots_mocr/parser.py` (CLI клиент) против постоянного
vLLM-сервиса (deploy/dots-mocr.service, GPU4:8120); он кладёт на каждую
страницу `*_page_N.json` — список элементов `{bbox, category, text}` в
reading-order. Адаптер `dots_to_segments` переводит их в наши SegmentDraft
(тот же контракт, что у MinerU content_list), так что перевод/экспорт/индекс
работают без изменений. Таблицы dots отдаёт готовым HTML — разбираем тем же
`parse_table`, что и MinerU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pypdfium2 as pdfium

from rag_app.config import settings
from rag_app.db.models import SegmentKind
from rag_app.pipeline.parse import PDFIUM_LOCK
from rag_app.pipeline.segments import SegmentDraft, parse_table

logger = logging.getLogger(__name__)

_DPI = 200  # parser.py рендерит страницы в 200 dpi → px·72/200 = pt

# категория dots → наш вид сегмента (Page-header/footer — колонтитулы, выбрасываем)
_HEADING = {"Title", "Section-header"}
_PARAGRAPH = {"Text", "List-item", "Caption", "Footnote"}
_SKIP = {"Page-header", "Page-footer"}


async def run_dots(pdf_path: Path, out_dir: Path) -> Path:
    """Прогон dots.mocr CLI; возвращает каталог с постраничными *_page_N.json."""
    parsed = urlparse(settings.dots_url)
    cmd = [
        settings.dots_venv_python,
        str(Path(settings.dots_repo) / "dots_mocr" / "parser.py"),
        str(pdf_path),
        "--output", str(out_dir),
        "--ip", parsed.hostname or "127.0.0.1",
        "--port", str(parsed.port or 8120),
        "--model_name", settings.dots_model_name,
        "--prompt", settings.dots_prompt,
        "--num_thread", str(settings.dots_num_thread),
        "--max_completion_tokens", "16384",
    ]
    env = dict(os.environ, PYTHONPATH=settings.dots_repo)
    logger.info("dots.mocr: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=settings.dots_repo, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.dots_timeout_s)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"dots.mocr: таймаут {settings.dots_timeout_s}s") from None
    if proc.returncode != 0:
        raise RuntimeError(f"dots.mocr: код {proc.returncode}\n{out.decode(errors='replace')[-3000:]}")
    # parser кладёт в <out>/<stem>/<stem>_page_N.json
    page_dir = out_dir / pdf_path.stem
    if not any(page_dir.glob("*_page_*.json")):
        # на всякий случай ищем глубже
        found = sorted(out_dir.rglob("*_page_*.json"))
        if not found:
            raise RuntimeError(f"dots.mocr: нет *_page_*.json в {out_dir}\n{out.decode(errors='replace')[-1500:]}")
        page_dir = found[0].parent
    return page_dir


def _page_sizes_pt(pdf_path: Path) -> list[tuple[float, float]]:
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return [tuple(pdf[i].get_size()) for i in range(len(pdf))]
        finally:
            pdf.close()


def dots_to_segments(page_dir: Path, pdf_path: Path) -> list[SegmentDraft]:
    """Постраничные JSON dots → SegmentDraft (reading-order, idx по порядку)."""
    sizes = _page_sizes_pt(pdf_path)
    files = sorted(
        page_dir.glob("*_page_*.json"),
        key=lambda p: int(re.search(r"_page_(\d+)\.json$", p.name).group(1)),
    )
    drafts: list[SegmentDraft] = []
    scale = 72.0 / _DPI
    for f in files:
        pidx = int(re.search(r"_page_(\d+)\.json$", f.name).group(1))
        try:
            elements = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        psize = sizes[pidx] if pidx < len(sizes) else None
        for el in elements:
            cat = el.get("category")
            text = (el.get("text") or "").strip()
            if cat in _SKIP or not text:
                continue
            bbox = el.get("bbox")
            meta: dict = {}
            if bbox and psize:
                meta["bbox_pt"] = [c * scale for c in bbox]
                meta["page_size_pt"] = list(psize)
            if cat == "Table":
                cells, rows = parse_table(text)
                if not cells:
                    continue
                preview = "\n".join(" | ".join(c["text"] for c in row) for row in cells)
                drafts.append(SegmentDraft(
                    0, SegmentKind.table, source_text=preview, page_idx=pidx,
                    meta={**meta, "table_cells": cells, "table_rows": rows, "caption": ""},
                ))
            elif cat == "Formula":
                drafts.append(SegmentDraft(0, SegmentKind.equation, text, pidx, meta=meta))
            elif cat == "Picture":
                # dots не вырезает картинку в файл — оставляем плейсхолдер с bbox
                drafts.append(SegmentDraft(0, SegmentKind.image, "", pidx, meta=meta))
            elif cat in _HEADING:
                clean = re.sub(r"^#+\s*", "", text)
                level = 1 if cat == "Title" else min(text.count("#") or 2, 6)
                drafts.append(SegmentDraft(
                    0, SegmentKind.heading, clean, pidx, heading_level=level, meta=meta
                ))
            else:  # _PARAGRAPH и всё прочее текстовое
                drafts.append(SegmentDraft(0, SegmentKind.paragraph, text, pidx, meta=meta))
    for i, d in enumerate(drafts):
        d.idx = i
    return drafts
