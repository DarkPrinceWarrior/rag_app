from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from openpyxl import Workbook, load_workbook
from pptx import Presentation
from pptx.util import Inches

from rag_app.db.models import SegmentKind
from rag_app.pipeline import ooxml


def test_docx_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src.docx"
    doc = DocxDocument()
    doc.add_heading("Scope", level=1)
    p = doc.add_paragraph()
    p.add_run("Design pressure is ")
    bold = p.add_run("16.5 MPa")
    bold.bold = True
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Value"
    doc.save(str(src))

    drafts = ooxml.extract_docx(src)
    texts = [d.source_text for d in drafts]
    assert "Scope" in texts
    assert "Design pressure is 16.5 MPa" in texts
    assert "Item" in texts
    heading = next(d for d in drafts if d.source_text == "Scope")
    assert heading.kind == SegmentKind.heading and heading.heading_level == 1

    translations = {
        ooxml.location_key(d.meta["location"]): f"RU:{d.source_text}" for d in drafts
    }
    dst = tmp_path / "dst.docx"
    applied = ooxml.inject_docx(src, dst, translations)
    assert applied == len(drafts)

    out = DocxDocument(str(dst))
    out_texts = [p.text for p in out.paragraphs if p.text.strip()]
    assert "RU:Scope" in out_texts
    assert "RU:Design pressure is 16.5 MPa" in out_texts
    assert out.tables[0].cell(0, 0).text == "RU:Item"


def test_xlsx_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src.xlsx"
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Design pressure"
    ws["B1"] = 16.5  # число — не трогаем
    ws["C1"] = "=B1*2"  # формула — не трогаем
    ws["A2"] = "Test pressure"
    ws["A3"] = "Design pressure"  # дубль текста A1 → один сегмент, оба перевода
    wb.save(str(src))

    drafts = ooxml.extract_xlsx(src)
    # дедуп по тексту: "Design pressure" один раз
    assert sorted(d.source_text for d in drafts) == ["Design pressure", "Test pressure"]

    # inject теперь по ИСХОДНОМУ ТЕКСТУ ячейки (а не по location)
    translations = {d.source_text: f"RU:{d.source_text}" for d in drafts}
    dst = tmp_path / "dst.xlsx"
    # 3 ячейки переведены (A1, A2, A3 — дубль тоже), несмотря на 2 сегмента
    assert ooxml.inject_xlsx(src, dst, translations) == 3

    out = load_workbook(str(dst))
    ws2 = out.active
    assert ws2["A1"].value == "RU:Design pressure"
    assert ws2["A3"].value == "RU:Design pressure"  # дубль получил тот же перевод
    assert ws2["B1"].value == 16.5
    assert ws2["C1"].value == "=B1*2"


def test_xlsx_skips_data_dump(tmp_path: Path) -> None:
    """Числовой/кодовый дамп НЕ плодит сегменты; проза — переводится."""
    src = tmp_path / "dump.xlsx"
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Compound name"  # проза (есть пробел) → переводим
    ws["B1"] = "Melting point, C"  # проза → переводим
    ws["A2"] = "DMFA"  # короткий код-токен → НЕ переводим
    ws["B2"] = "0.43"  # число-строка → НЕ переводим
    ws["A3"] = "130/130/300"  # код → НЕ переводим
    ws["B3"] = "pH"  # короткий токен → НЕ переводим
    ws["A4"] = "Eo"  # короткий токен → НЕ переводим
    ws["B4"] = "BTC"  # короткий код-токен → НЕ переводим
    ws["A5"] = "Hydrostatic test"  # проза → переводим
    wb.save(str(src))

    texts = sorted(d.source_text for d in ooxml.extract_xlsx(src))
    assert texts == ["Compound name", "Hydrostatic test", "Melting point, C"]


def test_pptx_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    box.text_frame.text = "Hydrostatic testing"
    prs.save(str(src))

    drafts = ooxml.extract_pptx(src)
    texts = [d.source_text for d in drafts]
    assert "Hydrostatic testing" in texts

    translations = {ooxml.location_key(d.meta["location"]): f"RU:{d.source_text}" for d in drafts}
    dst = tmp_path / "dst.pptx"
    applied = ooxml.inject_pptx(src, dst, translations)
    assert applied == len(drafts)

    out = Presentation(str(dst))
    out_texts = [
        sh.text_frame.text
        for s in out.slides
        for sh in s.shapes
        if getattr(sh, "has_text_frame", False)
    ]
    assert any("RU:Hydrostatic testing" in t for t in out_texts)


def test_pick_glossary_terms() -> None:
    from rag_app.llm.client import pick_glossary_terms

    terms = [
        ("maximum allowable working pressure", "максимально допустимое рабочее давление"),
        ("pressure vessel", "сосуд под давлением"),
        ("weld", "сварной шов"),
    ]
    found = pick_glossary_terms("The Pressure Vessel shall be welded.", terms)
    assert ("pressure vessel", "сосуд под давлением") in found
    # "weld" не должен находиться внутри "welded" (границы слов)
    assert ("weld", "сварной шов") not in found
