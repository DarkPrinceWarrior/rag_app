"""Оверлей-PDF для сканов: фон — исходный скан (печати/штампы видны),
текстовые блоки закрыты плашками с переводом по bbox из MinerU.

Это «запасной самописный re-render» из roadmap § 9: BabelDOC сканы
не переводит в принципе (нет текстового слоя), для них этот путь — основной.
"""

from __future__ import annotations

import io
import logging
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from rag_app.config import settings
from rag_app.db.models import Segment, SegmentKind

logger = logging.getLogger(__name__)

_OVERLAY_KINDS = {SegmentKind.heading, SegmentKind.paragraph, SegmentKind.table}
_RENDER_SCALE = 2.0  # 144 DPI — компромисс размер/читаемость


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        words = raw_line.split()
        cur = ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if draw.textlength(cand, font=font) <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def _fit_text(
    draw: ImageDraw.ImageDraw, text: str, box_w: float, box_h: float
) -> tuple[ImageFont.FreeTypeFont, list[str], float]:
    max_size = max(10, min(40, int(box_h * 0.85)))
    for size in range(max_size, 9, -1):
        font = ImageFont.truetype(settings.scan_font_path, size)
        lines = _wrap(draw, text, font, box_w)
        line_h = size * 1.18
        if len(lines) * line_h <= box_h * 1.12:
            return font, lines, line_h
    font = ImageFont.truetype(settings.scan_font_path, 10)
    return font, _wrap(draw, text, font, box_w), 11.8


def build_scan_overlay(original_pdf: Path, segments: list[Segment]) -> tuple[bytes, bytes]:
    """Возвращает (mono_pdf, dual_pdf): перевод поверх скана / чередование EN-RU."""
    by_page: dict[int, list[Segment]] = defaultdict(list)
    skipped = 0
    for seg in segments:
        if (
            seg.kind in _OVERLAY_KINDS
            and seg.translated_text
            and seg.page_idx is not None
            and seg.meta.get("bbox_pt")
            and seg.meta.get("page_size_pt")
        ):
            by_page[seg.page_idx].append(seg)
        elif seg.kind in _OVERLAY_KINDS and seg.translated_text:
            skipped += 1
    if skipped:
        logger.warning("оверлей: %d сегментов без геометрии (останутся фоном)", skipped)

    from rag_app.pipeline.parse import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(original_pdf))
        originals: list[Image.Image] = []
        try:
            for i in range(len(doc)):
                originals.append(doc[i].render(scale=_RENDER_SCALE).to_pil().convert("RGB"))
        finally:
            doc.close()

    # наложение — чистый PIL, замок не нужен
    overlaid: list[Image.Image] = []
    for i, base in enumerate(originals):
        img = base.copy()
        draw = ImageDraw.Draw(img)
        for seg in sorted(by_page.get(i, []), key=lambda s: s.idx):
            pw, ph = seg.meta["page_size_pt"]
            fx, fy = img.width / pw, img.height / ph
            x0, y0, x1, y1 = seg.meta["bbox_pt"]
            px0, py0, px1, py1 = x0 * fx, y0 * fy, x1 * fx, y1 * fy
            draw.rectangle([px0 - 2, py0 - 2, px1 + 2, py1 + 2], fill=(255, 255, 255))
            font, lines, line_h = _fit_text(draw, seg.translated_text, px1 - px0, py1 - py0)
            y = py0
            for line in lines:
                if y > py1 + line_h:  # лёгкий выход за низ бокса допустим
                    break
                draw.text((px0, y), line, fill=(20, 24, 33), font=font)
                y += line_h
        overlaid.append(img)

    def to_pdf(pages: list[Image.Image]) -> bytes:
        buf = io.BytesIO()
        pages[0].save(
            buf, format="PDF", save_all=True, append_images=pages[1:], resolution=72 * _RENDER_SCALE
        )
        return buf.getvalue()

    dual_pages = [p for pair in zip(originals, overlaid, strict=True) for p in pair]
    return to_pdf(overlaid), to_pdf(dual_pages)
