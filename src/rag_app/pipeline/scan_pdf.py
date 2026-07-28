"""Оверлей-PDF для сканов: фон — исходный скан (печати/штампы видны),
текстовые блоки закрыты плашками с переводом по bbox из MinerU.

Это «запасной самописный re-render» из roadmap § 9: BabelDOC сканы
не переводит в принципе (нет текстового слоя), для них этот путь — основной.
"""

from __future__ import annotations

import io
import logging
import re
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from rag_app.config import settings
from rag_app.db.models import Segment, SegmentKind

logger = logging.getLogger(__name__)

_OVERLAY_KINDS = {SegmentKind.heading, SegmentKind.paragraph, SegmentKind.table}
_RENDER_SCALE = 2.0  # 144 DPI — компромисс размер/читаемость
_LATEX_WRAPPER_RE = re.compile(r"\\(?:underline|text|mathrm|mathbf)\{([^{}]*)\}")
_LATEX_MATH_RE = re.compile(r"\$\s*(\\(?:underline|text|mathrm|mathbf)\{(?:[^{}]|\{[^{}]*\})*\})\s*\$")


def _overlay_groups(
    segments: list[Segment],
) -> tuple[dict[int, list[list[Segment]]], int]:
    """Сгруппировать переводы по геоблокам, сохранив порядок сегментов.

    PaddleOCR-VL может разбить один layout-блок на несколько Markdown-сегментов.
    Их bbox совпадают, поэтому рисовать каждый отдельно нельзя: поздний сегмент
    закрасит предыдущий. Группа отрисовывается одной плашкой с объединённым
    переводом.
    """
    grouped: dict[int, dict[tuple[tuple[float, ...], tuple[float, ...]], list[Segment]]] = defaultdict(dict)
    skipped = 0
    for seg in sorted(segments, key=lambda item: item.idx):
        bbox = seg.meta.get("bbox_pt")
        page_size = seg.meta.get("page_size_pt")
        if (
            seg.kind in _OVERLAY_KINDS
            and seg.translated_text
            and seg.page_idx is not None
            and isinstance(bbox, list)
            and len(bbox) == 4
            and isinstance(page_size, list)
            and len(page_size) == 2
        ):
            try:
                key = (
                    tuple(float(value) for value in bbox),
                    tuple(float(value) for value in page_size),
                )
            except (TypeError, ValueError):
                skipped += 1
                continue
            grouped[seg.page_idx].setdefault(key, []).append(seg)
        elif seg.kind in _OVERLAY_KINDS and seg.translated_text:
            skipped += 1
    return {page: list(groups.values()) for page, groups in grouped.items()}, skipped


def _plain_overlay_text(text: str) -> str:
    """Убрать узкий набор Paddle/LaTeX-маркеров, который PIL не умеет рисовать."""
    # Убираем только math-разделители вокруг поддержанного wrapper-выражения.
    # Обычные валютные значения (`$100`) и произвольные формулы не трогаем.
    plain = _LATEX_MATH_RE.sub(r"\1", text)
    for _ in range(4):
        replaced = _LATEX_WRAPPER_RE.sub(r"\1", plain)
        if replaced == plain:
            break
        plain = replaced
    plain = plain.replace(r"\_", "_")
    plain = plain.replace("《", "«").replace("》", "»")
    return "\n".join(" ".join(line.split()) for line in plain.splitlines() if line.strip())


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
    by_page, skipped = _overlay_groups(segments)
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
        for group in by_page.get(i, []):
            seg = group[0]
            pw, ph = seg.meta["page_size_pt"]
            fx, fy = img.width / pw, img.height / ph
            x0, y0, x1, y1 = seg.meta["bbox_pt"]
            px0, py0, px1, py1 = x0 * fx, y0 * fy, x1 * fx, y1 * fy
            draw.rectangle([px0 - 2, py0 - 2, px1 + 2, py1 + 2], fill=(255, 255, 255))
            translated = "\n".join(
                _plain_overlay_text(item.translated_text)
                for item in group
                if item.translated_text and item.translated_text.strip()
            )
            font, lines, line_h = _fit_text(draw, translated, px1 - px0, py1 - py0)
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
