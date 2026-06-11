"""Сборка редактируемого DOCX из переведённых сегментов (python-docx)."""

from __future__ import annotations

import io

from docx import Document as DocxDocument
from docx.shared import Pt

from rag_app.db.models import Segment, SegmentKind


def _seg_text(seg: Segment) -> str:
    return seg.translated_text if seg.translated_text is not None else seg.source_text


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
            run = p.add_run(seg.source_text)  # LaTeX переносим как есть
            run.italic = True
            run.font.size = Pt(10)

        elif seg.kind == SegmentKind.image:
            caption = _seg_text(seg).strip()
            p = doc.add_paragraph()
            run = p.add_run(f"[Рисунок]{' ' + caption if caption else ''}")
            run.italic = True

        elif seg.kind == SegmentKind.table:
            rows = seg.meta.get("table_rows_ru") or seg.meta.get("table_rows") or []
            caption = (seg.meta.get("caption_ru") or seg.meta.get("caption") or "").strip()
            if caption:
                p = doc.add_paragraph()
                p.add_run(caption).bold = True
            if rows:
                n_cols = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=n_cols)
                table.style = "Table Grid"
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        table.cell(i, j).text = cell
            doc.add_paragraph()

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
