"""Сборка редактируемого DOCX из переведённых сегментов (python-docx)."""

from __future__ import annotations

import io
import re

from docx import Document as DocxDocument
from docx.shared import Pt

from rag_app.db.models import Segment, SegmentKind

# Символы, недопустимые в XML 1.0 (python-docx/lxml роняет «All strings must be
# XML compatible … no NULL bytes»). Источник — OCR'нутые txt/сканы с битыми
# control-байтами. Чистим перед записью в DOCX.
_XML_INVALID = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")


def xml_safe(s: str | None) -> str:
    return _XML_INVALID.sub("", s) if s else (s or "")


def _seg_text(seg: Segment) -> str:
    return xml_safe(seg.translated_text if seg.translated_text is not None else seg.source_text)


def build_docx(filename: str, segments: list[Segment]) -> bytes:
    doc = DocxDocument()
    doc.core_properties.title = filename

    for seg in segments:
        if seg.kind == SegmentKind.heading:
            level = min(max(seg.heading_level or 1, 1), 6)
            doc.add_heading(_seg_text(seg), level=level)

        elif seg.kind == SegmentKind.paragraph:
            doc.add_paragraph(_seg_text(seg))

        elif seg.kind == SegmentKind.equation:
            p = doc.add_paragraph()
            run = p.add_run(xml_safe(seg.source_text))  # LaTeX переносим как есть
            run.italic = True
            run.font.size = Pt(10)

        elif seg.kind == SegmentKind.image:
            caption = _seg_text(seg).strip()
            p = doc.add_paragraph()
            run = p.add_run(f"[Рисунок]{' ' + caption if caption else ''}")
            run.italic = True

        elif seg.kind == SegmentKind.table:
            rows = seg.meta.get("table_rows_ru") or seg.meta.get("table_rows") or []
            caption = xml_safe((seg.meta.get("caption_ru") or seg.meta.get("caption") or "").strip())
            if caption:
                p = doc.add_paragraph()
                p.add_run(caption).bold = True
            if rows:
                n_cols = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=n_cols)
                table.style = "Table Grid"
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        table.cell(i, j).text = xml_safe(cell)
            doc.add_paragraph()

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
