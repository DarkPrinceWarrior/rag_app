"""OOXML-транслятор (roadmap § 3.3.B): DOCX/XLSX/PPTX → тот же формат.

Вёрстку не трогаем — она в XML. Извлекаем текстовые узлы с адресом
(location в meta сегмента), переводим сегментно, записываем обратно
в копию оригинала.

Адресация:
- DOCX:  {"p": i} — абзац body; {"t": ti, "r": ri, "c": ci, "p": pi} — абзац ячейки
- XLSX:  {"sheet": name, "cell": "A1"}
- PPTX:  {"slide": si, "shape": id, "para": pi}; заметки — {"slide": si, "notes": true, "para": pi}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from openpyxl import load_workbook
from pptx import Presentation

from rag_app.db.models import SegmentKind
from rag_app.pipeline.segments import SegmentDraft


def location_key(location: dict[str, Any]) -> str:
    return json.dumps(location, sort_keys=True, ensure_ascii=False)


# ------------------------------------------------------------------ DOCX

def _docx_set_paragraph_text(paragraph: Any, text: str) -> None:
    """Перевод в первый run (его формат — доминирующий), остальные очищаем."""
    if not paragraph.runs:
        if text:
            paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def extract_docx(path: Path) -> list[SegmentDraft]:
    doc = DocxDocument(str(path))
    drafts: list[SegmentDraft] = []

    def add(text: str, location: dict[str, Any], style_name: str) -> None:
        text = text.strip()
        if not text:
            return
        kind = SegmentKind.paragraph
        level = None
        if style_name.startswith("Heading"):
            kind = SegmentKind.heading
            try:
                level = int(style_name.split()[-1])
            except ValueError:
                level = 1
        drafts.append(
            SegmentDraft(
                idx=len(drafts),
                kind=kind,
                source_text=text,
                heading_level=level,
                meta={"location": location},
            )
        )

    for i, p in enumerate(doc.paragraphs):
        add(p.text, {"p": i}, p.style.name if p.style else "")
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                for pi, p in enumerate(cell.paragraphs):
                    add(p.text, {"t": ti, "r": ri, "c": ci, "p": pi}, "")
    return drafts


def inject_docx(src: Path, dst: Path, translations: dict[str, str]) -> int:
    doc = DocxDocument(str(src))
    applied = 0

    def apply(paragraph: Any, location: dict[str, Any]) -> None:
        nonlocal applied
        text = translations.get(location_key(location))
        if text is not None:
            _docx_set_paragraph_text(paragraph, text)
            applied += 1

    for i, p in enumerate(doc.paragraphs):
        apply(p, {"p": i})
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                for pi, p in enumerate(cell.paragraphs):
                    apply(p, {"t": ti, "r": ri, "c": ci, "p": pi})
    doc.save(str(dst))
    return applied


# ------------------------------------------------------------------ XLSX

def extract_xlsx(path: Path) -> list[SegmentDraft]:
    wb = load_workbook(str(path))  # data_only=False: формулы остаются формулами
    drafts: list[SegmentDraft] = []
    for s_i, ws in enumerate(wb.worksheets):
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                # только строковые значения; формулы ("=...") и числа не трогаем по построению
                if not isinstance(v, str) or not v.strip() or v.startswith("="):
                    continue
                drafts.append(
                    SegmentDraft(
                        idx=len(drafts),
                        kind=SegmentKind.paragraph,
                        source_text=v,
                        page_idx=s_i,
                        meta={"location": {"sheet": ws.title, "cell": cell.coordinate}},
                    )
                )
    return drafts


def inject_xlsx(src: Path, dst: Path, translations: dict[str, str]) -> int:
    wb = load_workbook(str(src))
    applied = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if not isinstance(cell.value, str):
                    continue
                text = translations.get(location_key({"sheet": ws.title, "cell": cell.coordinate}))
                if text is not None:
                    cell.value = text
                    applied += 1
    wb.save(str(dst))
    return applied


# ------------------------------------------------------------------ PPTX

def _pptx_paragraphs(prs: Any):
    """(paragraph, location) для всех текстовых фреймов и заметок."""
    for s_i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for p_i, p in enumerate(shape.text_frame.paragraphs):
                yield p, {"slide": s_i, "shape": shape.shape_id, "para": p_i}
        if slide.has_notes_slide:
            for p_i, p in enumerate(slide.notes_slide.notes_text_frame.paragraphs):
                yield p, {"slide": s_i, "notes": True, "para": p_i}


def _pptx_paragraph_text(p: Any) -> str:
    return "".join(run.text for run in p.runs)


def extract_pptx(path: Path) -> list[SegmentDraft]:
    prs = Presentation(str(path))
    drafts: list[SegmentDraft] = []
    for p, location in _pptx_paragraphs(prs):
        text = _pptx_paragraph_text(p).strip()
        if not text:
            continue
        drafts.append(
            SegmentDraft(
                idx=len(drafts),
                kind=SegmentKind.paragraph,
                source_text=text,
                page_idx=location["slide"],
                meta={"location": location},
            )
        )
    return drafts


def inject_pptx(src: Path, dst: Path, translations: dict[str, str]) -> int:
    prs = Presentation(str(src))
    applied = 0
    for p, location in _pptx_paragraphs(prs):
        text = translations.get(location_key(location))
        if text is None:
            continue
        if p.runs:
            p.runs[0].text = text
            for run in p.runs[1:]:
                run.text = ""
        applied += 1
    prs.save(str(dst))
    return applied


# ------------------------------------------------------------------ единый вход

EXTRACTORS = {"docx": extract_docx, "xlsx": extract_xlsx, "pptx": extract_pptx}
INJECTORS = {"docx": inject_docx, "xlsx": inject_xlsx, "pptx": inject_pptx}


def extract(kind: str, path: Path) -> list[SegmentDraft]:
    return EXTRACTORS[kind](path)


def inject(kind: str, src: Path, dst: Path, translations: dict[str, str]) -> int:
    return INJECTORS[kind](src, dst, translations)
