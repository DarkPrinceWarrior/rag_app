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
import logging
import re
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_AUTO_SIZE

from rag_app.config import settings
from rag_app.db.models import SegmentKind
from rag_app.pipeline.segments import SegmentDraft

logger = logging.getLogger(__name__)


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


def _docx_paragraph_images(paragraph: Any, images_dir: Path, drafts: list[SegmentDraft]) -> None:
    """Встроенные картинки абзаца → файлы в images_dir + сегменты kind=image.

    Картинка лежит в part'ах документа, ссылка — `a:blip r:embed`. Кладём байты
    в images_dir (img_path в meta), парс-задача потом грузит их в MinIO для
    вставки в MD-просмотр. Сегмент-картинка идёт сразу за своим абзацем."""
    for blip in paragraph._p.iterfind(".//" + qn("a:blip")):
        rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
        if not rid:
            continue
        try:
            part = paragraph.part.related_parts[rid]
        except KeyError:
            continue
        name = Path(str(part.partname)).name
        try:
            (images_dir / name).write_bytes(part.blob)
        except Exception:
            continue
        drafts.append(
            SegmentDraft(idx=len(drafts), kind=SegmentKind.image, source_text="", meta={"img_path": name})
        )


def extract_docx(path: Path, images_dir: Path | None = None) -> list[SegmentDraft]:
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

    # идём по детям body В ПОРЯДКЕ ДОКУМЕНТА (абзацы, таблицы, картинки на своих
    # местах). Индексы p_idx/t_idx совпадают с doc.paragraphs[i]/doc.tables[ti],
    # поэтому location-ключи те же, что у inject_docx — экспорт не ломается.
    p_idx = 0
    t_idx = 0
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, doc)
            add(para.text, {"p": p_idx}, para.style.name if para.style else "")
            if images_dir is not None:
                _docx_paragraph_images(para, images_dir, drafts)
            p_idx += 1
        elif child.tag == qn("w:tbl"):
            table = Table(child, doc)
            for ri, row in enumerate(table.rows):
                for ci, cell in enumerate(row.cells):
                    for pi, cp in enumerate(cell.paragraphs):
                        add(cp.text, {"t": t_idx, "r": ri, "c": ci, "p": pi}, "")
            t_idx += 1
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

# Слово = ≥2 подряд идущих буквы (латиница/кириллица). Чисто-числовой/кодовый
# дамп (0.43, 130/130/300, DMFA, pH, Eo) слов в этом смысле не даёт «прозы».
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")


def _is_translatable_xlsx(v: str) -> bool:
    """Ячейка переводима, если в ней есть осмысленный текст (слово/фраза), а не
    голый код/число/идентификатор.

    Условие: есть «буквенная» группа ≥2 символов И выполнено одно из:
      - в строке есть пробел (фраза из нескольких токенов: «Lift method»), либо
      - длина ≥12 («North Kudrinskoye»), либо
      - буквы составляют ≥50% непробельных символов (одиночное слово-подпись:
        «Operating», «Status», «Field:», «Normal», «Inflow», «Shutdown»).
    Последнее правило ловит короткие однословные заголовки/статусы, которые
    раньше терялись. Смешанные коды («42/713», «ES-0517», «01.05.2026») при этом
    отсекаются — в них мало букв либо их нет вовсе. Объём по-прежнему ограничен
    `xlsx_max_segments` + дедуп, поэтому крупные data-дампы не раздуваются."""
    s = v.strip()
    if not s or s.startswith("="):
        return False
    if not _WORD_RE.search(s):
        return False
    if " " in s or len(s) >= 12:
        return True
    letters = sum(c.isalpha() for c in s)
    nonspace = sum(not c.isspace() for c in s) or 1
    return letters / nonspace >= 0.5


def extract_xlsx(path: Path) -> list[SegmentDraft]:
    wb = load_workbook(str(path))  # data_only=False: формулы остаются формулами
    drafts: list[SegmentDraft] = []
    seen: set[str] = set()  # дедуп по исходному тексту: одинаковые строки = 1 сегмент
    capped = False
    skipped_dup = 0
    for s_i, ws in enumerate(wb.worksheets):
        # название листа — тоже переводим (вкладки листов показываем и на русском)
        title = (ws.title or "").strip()
        if title and title not in seen:
            seen.add(title)
            drafts.append(
                SegmentDraft(
                    idx=len(drafts),
                    kind=SegmentKind.paragraph,
                    source_text=title,
                    page_idx=s_i,
                    meta={"location": {"sheet": ws.title, "cell": "__sheet_title__"}, "sheet_title": True},
                )
            )
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                # только строковые значения; формулы и числа не трогаем по построению
                if not isinstance(v, str) or not _is_translatable_xlsx(v):
                    continue
                if v in seen:
                    skipped_dup += 1
                    continue
                if len(drafts) >= settings.xlsx_max_segments:
                    capped = True
                    continue
                seen.add(v)
                drafts.append(
                    SegmentDraft(
                        idx=len(drafts),
                        kind=SegmentKind.paragraph,
                        source_text=v,
                        page_idx=s_i,
                        # location — первой встреченной ячейки с этим текстом; inject
                        # применяет перевод ПО ТЕКСТУ ко всем ячейкам-дубликатам.
                        meta={"location": {"sheet": ws.title, "cell": cell.coordinate}},
                    )
                )
    if capped:
        logger.warning(
            "extract_xlsx %s: достигнут потолок xlsx_max_segments=%d — часть прозовых "
            "ячеек отброшена (перевод неполный); скрытых дублей-ячеек: %d",
            path.name,
            settings.xlsx_max_segments,
            skipped_dup,
        )
    return drafts


def inject_xlsx(src: Path, dst: Path, translations: dict[str, str]) -> int:
    """Записать перевод обратно в .xlsx ПО ИСХОДНОМУ ТЕКСТУ ячейки.

    translations: {исходный_текст_ячейки: перевод}. Перевод применяется ко ВСЕМ
    ячейкам с тем же исходным текстом (дедуп на extract → один перевод
    раскладывается на все дубликаты)."""
    wb = load_workbook(str(src))
    applied = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if not isinstance(cell.value, str):
                    continue
                text = translations.get(cell.value)
                if text is not None:
                    cell.value = text
                    applied += 1
    wb.save(str(dst))
    return applied


# ------------------------------------------------------------------ PPTX

_CITATION_RE = re.compile(r"^\s*\[\d+\]")  # элемент списка литературы: «[1] Stevenson…»


def is_pptx_citation(text: str) -> bool:
    """Запись библиографии вида «[1] …» — список литературы не переводим."""
    return bool(_CITATION_RE.match(text or ""))


def _para_text(p: Any) -> str:
    return "".join(run.text for run in p.runs)


def _set_para(p: Any, text: str) -> None:
    """Записать перевод в абзац, сохранив форматирование первого run'а."""
    if p.runs:
        p.runs[0].text = text
        for run in p.runs[1:]:
            run.text = ""
    elif text:
        p.text = text


def _iter_shape_units(shapes: Any, s_i: int):
    """Рекурсивный обход фигур слайда (С ЗАХОДОМ В ГРУППЫ) → текстовые единицы.

    Yields (location, get_text, set_text) для:
    - абзацев текстовых фреймов  → {"slide", "shape", "para"}
    - ячеек таблиц               → {"slide", "shape", "row", "col"}
    Группы (MSO GROUP) рекурсивно разворачиваются — иначе текст в сгруппированных
    фигурах теряется (слайды-«только заголовок»). Таблицы (GraphicFrame.has_table)
    раньше вообще не извлекались."""
    for shape in shapes:
        try:
            is_group = shape.shape_type == MSO_SHAPE_TYPE.GROUP
        except Exception:
            is_group = False
        if is_group:
            yield from _iter_shape_units(shape.shapes, s_i)
            continue
        if getattr(shape, "has_table", False):
            tbl = shape.table
            for r, row in enumerate(tbl.rows):
                for c, cell in enumerate(row.cells):
                    loc = {"slide": s_i, "shape": shape.shape_id, "row": r, "col": c}

                    def _get(cell=cell) -> str:
                        return cell.text

                    def _set(text: str, cell=cell) -> None:
                        cell.text = text

                    yield loc, _get, _set
            continue
        if getattr(shape, "has_text_frame", False):
            for p_i, p in enumerate(shape.text_frame.paragraphs):
                loc = {"slide": s_i, "shape": shape.shape_id, "para": p_i}

                def _get(p=p) -> str:
                    return _para_text(p)

                def _set(text: str, p=p) -> None:
                    _set_para(p, text)

                yield loc, _get, _set


def _pptx_units(prs: Any):
    """Все переводимые единицы презентации (фигуры+группы+таблицы и заметки)."""
    for s_i, slide in enumerate(prs.slides):
        yield from _iter_shape_units(slide.shapes, s_i)
        if slide.has_notes_slide:
            tf = slide.notes_slide.notes_text_frame
            for p_i, p in enumerate(tf.paragraphs):
                loc = {"slide": s_i, "notes": True, "para": p_i}

                def _get(p=p) -> str:
                    return _para_text(p)

                def _set(text: str, p=p) -> None:
                    _set_para(p, text)

                yield loc, _get, _set


def extract_pptx(path: Path) -> list[SegmentDraft]:
    prs = Presentation(str(path))
    drafts: list[SegmentDraft] = []
    for location, get_text, _ in _pptx_units(prs):
        text = (get_text() or "").strip()
        if not text or is_pptx_citation(text):  # список литературы не переводим
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
    for location, _, set_text in _pptx_units(prs):
        text = translations.get(location_key(location))
        if text is None:
            continue
        set_text(text)
        applied += 1
    prs.save(str(dst))
    return applied


def _autofit_shapes(shapes: Any) -> None:
    for shape in shapes:
        try:
            is_group = shape.shape_type == MSO_SHAPE_TYPE.GROUP
        except Exception:
            is_group = False
        if is_group:
            _autofit_shapes(shape.shapes)
            continue
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    try:
                        cell.text_frame.word_wrap = True
                    except Exception:
                        pass
            continue
        if getattr(shape, "has_text_frame", False):
            tf = shape.text_frame
            try:
                tf.word_wrap = True
            except Exception:
                pass
            try:
                # normAutofit: «ужать текст до фигуры» — LibreOffice уменьшит кегль,
                # чтобы длинный (особенно переведённый) текст не вылезал за рамку.
                tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            except Exception:
                pass


def pptx_autofit(src: Path, dst: Path) -> None:
    """Копия pptx, где у фигур включены перенос слов и «ужать текст до фигуры».

    Только для РЕНДЕРА-ПРОСМОТРА (office-PDF): LibreOffice иначе выпускает длинный
    текст за границы фигур и наезжает на картинки (русский текст длиннее
    английского). Оригинальный .pptx-экспорт не трогаем."""
    prs = Presentation(str(src))
    for slide in prs.slides:
        _autofit_shapes(slide.shapes)
    prs.save(str(dst))


# ------------------------------------------------------------------ единый вход

EXTRACTORS = {"docx": extract_docx, "xlsx": extract_xlsx, "pptx": extract_pptx}
INJECTORS = {"docx": inject_docx, "xlsx": inject_xlsx, "pptx": inject_pptx}


def extract(kind: str, path: Path, images_dir: Path | None = None) -> list[SegmentDraft]:
    if kind == "docx":
        return extract_docx(path, images_dir)
    return EXTRACTORS[kind](path)


def inject(kind: str, src: Path, dst: Path, translations: dict[str, str]) -> int:
    return INJECTORS[kind](src, dst, translations)
