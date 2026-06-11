"""Генерация тестового СКАНА с печатями для e2e этапа 2.

Критерий этапа 2 (roadmap § 11): скан договора с печатями → переведённый PDF
с сохранённой вёрсткой + редактируемый DOCX.

Текстовый PDF (reportlab) → растеризация 150 DPI (pypdfium2) → печати/штампы
(PIL) → image-only PDF без текстового слоя.

Запуск: uv run python scripts/make_scan_pdf.py [страниц] [выход.pdf]
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

# генератор текстового PDF переиспользуем из соседнего скрипта
sys.path.insert(0, str(Path(__file__).parent))
from make_test_pdf import PARAGRAPHS, SPEC_ROWS  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)


def build_text_pdf(n_pages: int) -> bytes:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=14, spaceAfter=8)
    story = [Paragraph("CONTRACT AGREEMENT No. 3086-CA-007", styles["Title"])]
    for page in range(1, n_pages + 1):
        story.append(Paragraph(f"Article {page}. Terms for Unit {200 + page}", styles["Heading1"]))
        for j, text in enumerate(PARAGRAPHS[:4]):
            story.append(Paragraph(f"{page}.{j + 1} {text}", body))
        if page % 2 == 0:
            t = Table(SPEC_ROWS, colWidths=[1.5 * cm, 6 * cm, 3 * cm, 2 * cm])
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
        story.append(PageBreak())
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()


def draw_round_seal(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    """Синяя круглая печать: два кольца + текст по окружности + звезда в центре."""
    blue = (28, 60, 170, 170)
    for rr, w in ((r, 4), (r - 14, 2)):
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=blue, width=w)
    text = "ООО «СЕВЕРГАЗСТРОЙ» * ДЛЯ ДОКУМЕНТОВ * "
    for i, ch in enumerate(text):
        ang = 2 * math.pi * i / len(text) - math.pi / 2
        tx = cx + int((r - 30) * math.cos(ang))
        ty = cy + int((r - 30) * math.sin(ang))
        draw.text((tx, ty), ch, fill=blue, anchor="mm")
    draw.text((cx, cy), "№ 3086", fill=blue, anchor="mm")


def draw_stamp(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """Прямоугольный штамп «APPROVED»."""
    violet = (90, 30, 150, 180)
    draw.rectangle([x, y, x + 360, y + 90], outline=violet, width=4)
    draw.text((x + 180, y + 28), "APPROVED / СОГЛАСОВАНО", fill=violet, anchor="mm")
    draw.text((x + 180, y + 62), "19.03.2026", fill=violet, anchor="mm")


def main() -> None:
    n_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/test_scan.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf_bytes = build_text_pdf(n_pages)
    doc = pdfium.PdfDocument(pdf_bytes)
    pages: list[Image.Image] = []
    for i in range(len(doc)):
        bitmap = doc[i].render(scale=150 / 72)  # 150 DPI — типичный «офисный скан»
        img = bitmap.to_pil().convert("RGB")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        w, h = img.size
        draw_round_seal(d, int(w * 0.78), int(h * 0.82), 130)
        if i == 0:
            draw_stamp(d, int(w * 0.08), int(h * 0.86))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        pages.append(img)
    doc.close()

    pages[0].save(out, save_all=True, append_images=pages[1:], format="PDF", resolution=150)
    print(f"OK: {out} ({n_pages} стр., image-only)")


if __name__ == "__main__":
    main()
