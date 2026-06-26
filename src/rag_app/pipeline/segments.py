"""content_list (MinerU) → сегменты документа.

Формат элементов content_list (стабильный контракт MinerU 2.x):
- text:     {"type": "text", "text": ..., "text_level": 1..N (только заголовки), "page_idx": N}
- table:    {"type": "table", "table_body": "<html>...", "table_caption": [...], "table_footnote": [...]}
- image:    {"type": "image", "img_path": ..., "image_caption": [...], "image_footnote": [...]}
- equation: {"type": "equation", "text": "\\[...\\]", "text_format": "latex"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from rag_app.db.models import SegmentKind


@dataclass
class SegmentDraft:
    idx: int
    kind: SegmentKind
    source_text: str
    page_idx: int | None = None
    heading_level: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
    for k, v in attrs:
        if k == name:
            try:
                return max(1, int(v or 1))
            except (TypeError, ValueError):
                return 1
    return 1


class _TableHTMLParser(HTMLParser):
    """<table> HTML → ровная сетка ячеек с разворотом rowspan/colspan.

    MinerU размечает объединённые ячейки шапок через colspan/rowspan; без их
    разворота строки получаются разной длины (колонки «съезжают»). Здесь каждая
    строка приводится к одинаковой ширине: текст ставится в левую-верхнюю ячейку
    спана, ячейки-продолжения остаются пустыми.
    """

    def __init__(self) -> None:
        super().__init__()
        # сырые строки: список (текст, colspan, rowspan) до разворота
        self._raw: list[list[tuple[str, int, int]]] = []
        self._cell: list[str] | None = None
        self._cs = 1
        self._rs = 1

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._raw.append([])
        elif tag in ("td", "th"):
            self._cell = []
            self._cs = _span(attrs, "colspan")
            self._rs = _span(attrs, "rowspan")
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if not self._raw:
                self._raw.append([])
            self._raw[-1].append(("".join(self._cell).strip(), self._cs, self._rs))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def grid(self) -> list[list[str]]:
        out: list[list[str]] = []
        carry: dict[int, int] = {}  # колонка → сколько строк ниже занято rowspan'ом
        for raw_row in self._raw:
            row: list[str] = []
            col = 0
            for text, cs, rs in raw_row:
                while carry.get(col, 0) > 0:  # колонка занята спаном сверху
                    row.append("")
                    carry[col] -= 1
                    col += 1
                for k in range(cs):
                    row.append(text if k == 0 else "")
                    if rs > 1:
                        carry[col] = rs - 1
                    col += 1
            while carry.get(col, 0) > 0:  # хвостовые rowspan-колонки справа
                row.append("")
                carry[col] -= 1
                col += 1
            out.append(row)
        return out

    def cells(self) -> list[list[dict[str, Any]]]:
        """Сырые строки с сохранением спанов — для рендера merged-ячеек в UI.

        Текст ячейки приводится к одной строке (внутренние переводы строк —
        в пробел), чтобы число строк превью совпадало с числом строк таблицы
        (перевод сохраняет разметку построчно).
        """
        out: list[list[dict[str, Any]]] = []
        for raw_row in self._raw:
            row = [
                {"text": " ".join(t.split()), "colspan": cs, "rowspan": rs}
                for (t, cs, rs) in raw_row
            ]
            if any(c["text"] for c in row):
                out.append(row)
        return out


def parse_table(html: str) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    """(сырые ячейки со спанами, ровная сетка) из <table> HTML."""
    parser = _TableHTMLParser()
    parser.feed(html or "")
    grid = [row for row in parser.grid() if any(cell.strip() for cell in row)]
    return parser.cells(), grid


def parse_table_html(html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(html or "")
    return [row for row in parser.grid() if any(cell.strip() for cell in row)]


def _join_captions(*caption_lists: Any) -> str:
    parts: list[str] = []
    for captions in caption_lists:
        if isinstance(captions, list):
            parts.extend(str(c).strip() for c in captions if str(c).strip())
    return "\n".join(parts)


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_header(text: Any) -> str:
    """Нормализация текста колонтитула для подсчёта повторов (бегущий vs титул)."""
    return " ".join(str(text or "").split()).lower()


def _is_degenerate(text: str) -> bool:
    """Парсер изредка уходит в repetition-collapse: блок из одного повторяющегося
    символа (`!!!!…`) или почти без уникальных слов. Такой сегмент — мусор: ломает
    перевод (выходит за контекст) и засоряет вьювер/RAG. Реальный текст так не
    выглядит. Короткие строки не трогаем (могут быть «———», «...»)."""
    t = (text or "").strip()
    if len(t) < 40:
        return False
    body = t.replace(" ", "").replace("\n", "")
    if body:
        top = max((body.count(c) for c in set(body)), default=0)
        if top / len(body) > 0.8:  # один символ доминирует (>80%)
            return True
    if len(t) > 8000:
        return True
    words = t.split()
    if len(words) >= 60 and len({w.lower() for w in words}) / len(words) < 0.35:
        return True
    return False


def content_list_to_segments(items: list[dict[str, Any]]) -> list[SegmentDraft]:
    # MinerU помечает колонтитулы type=header/footer и НЕ кладёт их в основной
    # поток text. Бегущий колонтитул повторяется на ≥2 страницах — это шум для
    # перевода/RAG (выбрасываем). Уникальный header — титульная шапка документа
    # (первая страница / одностраничник): терять её нельзя, поэтому промотируем
    # в heading и ставим в начало её страницы (MinerU кладёт header в конец
    # content_list). footer отбрасываем всегда.
    header_counts: dict[str, int] = {}
    for item in items:
        if item.get("type") == "header":
            norm = _norm_header(item.get("text"))
            if norm:
                header_counts[norm] = header_counts.get(norm, 0) + 1

    drafts: list[SegmentDraft] = []
    titles: list[SegmentDraft] = []  # уникальные header'ы → вставим в начало их страницы
    for item in items:
        itype = item.get("type")
        page_idx = _to_int(item.get("page_idx"))
        # bbox нужен на этапе 3: подсветка цитат RAG в оригинале (roadmap § 5)
        base_meta = {"bbox": item.get("bbox")} if item.get("bbox") else {}

        if itype == "text":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            level = _to_int(item.get("text_level"))
            if level:
                drafts.append(
                    SegmentDraft(0, SegmentKind.heading, text, page_idx, heading_level=level, meta=base_meta)
                )
            else:
                drafts.append(SegmentDraft(0, SegmentKind.paragraph, text, page_idx, meta=base_meta))

        elif itype == "header":
            text = (item.get("text") or "").strip()
            # бегущий колонтитул (повтор) или пусто — пропускаем; уникальный — титул
            if not text or header_counts.get(_norm_header(text), 0) != 1:
                continue
            titles.append(
                SegmentDraft(0, SegmentKind.heading, text, page_idx, heading_level=1, meta=base_meta)
            )

        elif itype == "table":
            cells, rows = parse_table(item.get("table_body") or "")
            caption = _join_captions(item.get("table_caption"), item.get("table_footnote"))
            if not cells and not caption:
                continue
            # превью для перевода/RAG — по СЫРЫМ ячейкам (одна строка на строку
            # таблицы): так перевод построчно совпадает с table_cells, и UI может
            # восстановить объединённые ячейки (colspan/rowspan) с переводом.
            preview = "\n".join(" | ".join(c["text"] for c in row) for row in cells)
            drafts.append(
                SegmentDraft(
                    0,
                    SegmentKind.table,
                    source_text=(caption + "\n" + preview).strip(),
                    page_idx=page_idx,
                    # table_cells — для merged-рендера; table_rows — ровная сетка для RAG/экспорта
                    meta={**base_meta, "table_cells": cells, "table_rows": rows, "caption": caption},
                )
            )

        elif itype == "image":
            caption = _join_captions(item.get("image_caption"), item.get("image_footnote"))
            drafts.append(
                SegmentDraft(
                    0,
                    SegmentKind.image,
                    caption,
                    page_idx,
                    meta={**base_meta, "img_path": item.get("img_path")},
                )
            )

        elif itype == "equation":
            text = (item.get("text") or "").strip()
            if text:
                drafts.append(SegmentDraft(0, SegmentKind.equation, text, page_idx, meta=base_meta))

    # выбросить вырожденные сегменты (repetition-collapse парсера: `!!!!`-блоб и т.п.)
    drafts = [
        d
        for d in drafts
        if d.kind not in (SegmentKind.paragraph, SegmentKind.table)
        or not _is_degenerate(d.source_text)
    ]

    # титульные шапки — в начало своей страницы (перед первым блоком той же страницы)
    for title in titles:
        pos = next((i for i, d in enumerate(drafts) if d.page_idx == title.page_idx), len(drafts))
        drafts.insert(pos, title)

    for i, draft in enumerate(drafts):
        draft.idx = i
    return drafts
