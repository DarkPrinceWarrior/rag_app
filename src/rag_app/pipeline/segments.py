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


class _TableHTMLParser(HTMLParser):
    """<table> HTML → список строк с текстами ячеек (rowspan/colspan игнорируем)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_table_html(html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(html or "")
    return [row for row in parser.rows if any(cell.strip() for cell in row)]


def _join_captions(*caption_lists: Any) -> str:
    parts: list[str] = []
    for captions in caption_lists:
        if isinstance(captions, list):
            parts.extend(str(c).strip() for c in captions if str(c).strip())
    return "\n".join(parts)


def content_list_to_segments(items: list[dict[str, Any]]) -> list[SegmentDraft]:
    drafts: list[SegmentDraft] = []
    for item in items:
        itype = item.get("type")
        page_idx = item.get("page_idx")
        idx = len(drafts)

        if itype == "text":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            level = item.get("text_level")
            if level:
                drafts.append(
                    SegmentDraft(idx, SegmentKind.heading, text, page_idx, heading_level=int(level))
                )
            else:
                drafts.append(SegmentDraft(idx, SegmentKind.paragraph, text, page_idx))

        elif itype == "table":
            rows = parse_table_html(item.get("table_body") or "")
            caption = _join_captions(item.get("table_caption"), item.get("table_footnote"))
            if not rows and not caption:
                continue
            preview = "\n".join(" | ".join(row) for row in rows)
            drafts.append(
                SegmentDraft(
                    idx,
                    SegmentKind.table,
                    source_text=(caption + "\n" + preview).strip(),
                    page_idx=page_idx,
                    meta={"table_rows": rows, "caption": caption},
                )
            )

        elif itype == "image":
            caption = _join_captions(item.get("image_caption"), item.get("image_footnote"))
            drafts.append(
                SegmentDraft(
                    idx, SegmentKind.image, caption, page_idx, meta={"img_path": item.get("img_path")}
                )
            )

        elif itype == "equation":
            text = (item.get("text") or "").strip()
            if text:
                drafts.append(SegmentDraft(idx, SegmentKind.equation, text, page_idx))

    return drafts
