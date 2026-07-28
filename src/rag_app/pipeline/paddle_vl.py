"""PaddleOCR-VL 1.6 как альтернативный парсер pdf_text.

On-demand: воркер запускает `deploy/parsers/run_paddle_cli.py` из изолированного
paddle-venv (грузит модель на GPU4 на время парса, потом освобождает). Скрипт
кладёт постраничный Markdown `<stem>_<page>.md`; адаптер `paddle_to_segments`
разбирает md → SegmentDraft (заголовки `#`, таблицы `<table>` через parse_table,
остальное — абзацы, с капом длины под лимит перевода).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_app.config import settings
from rag_app.db.models import SegmentKind
from rag_app.pipeline.parse import _cap
from rag_app.pipeline.segments import SegmentDraft, parse_table

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"<table.*?</table>", re.S | re.I)
_IMG_RE = re.compile(r"^!\[[^\]]*\]\(([^)]*)\)$")  # markdown ![](path) (старый формат)
_IMG_HTML = re.compile(r'<img[^>]*\bsrc="([^"]+)"[^>]*>', re.I)  # PaddleOCR-VL: <img src="imgs/...">
_DIV_TAG = re.compile(r"</?div[^>]*>", re.I)  # PaddleOCR-VL центрирует картинки/подписи в <div>
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)")
_PAGE_FILE_RE = re.compile(r"_(\d+)\.md$")
_IMAGE_LABELS = {"image", "chart", "seal", "header_image", "footer_image"}
_TABLE_LABELS = {"table"}
_EQUATION_LABELS = {"formula", "display_formula", "inline_formula", "equation"}


def _img_meta(rel: str) -> dict:
    rel = (rel or "").strip().split()[0] if (rel or "").strip() else ""
    return {"img_path": rel} if rel and not rel.startswith(("http://", "https://", "data:")) else {}


async def run_paddle(pdf_path: Path, out_dir: Path, *, timeout_s: int | None = None) -> Path:
    """Прогон PaddleOCR-VL; возвращает каталог с постраничным Markdown (doc_N.md)."""
    timeout = settings.paddle_timeout_s if timeout_s is None else timeout_s
    if timeout <= 0:
        raise ValueError("paddle timeout must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "doc.pdf"  # чистое имя (без пробелов/скобок) для стабильных doc_N.md
    shutil.copy(pdf_path, clean)
    cmd = [settings.paddle_venv_python, settings.paddle_runner, str(clean), str(out_dir)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=settings.paddle_device,
        PADDLE_VL_SERVER_URL=settings.paddle_vl_server_url,  # VLM на genai vLLM-сервер
    )
    logger.info("paddle-vl: %s (GPU=%s)", " ".join(cmd), settings.paddle_device)
    proc = await asyncio.create_subprocess_exec(
        *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"paddle-vl: таймаут {timeout}s") from None
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"paddle-vl: код {proc.returncode}\n{out.decode(errors='replace')[-3000:]}")
    if not any(out_dir.glob("doc_*.md")):
        raise RuntimeError(f"paddle-vl: нет doc_*.md в {out_dir}\n{out.decode(errors='replace')[-1500:]}")
    return out_dir


def _text_blocks(text: str, pidx: int, drafts: list[SegmentDraft]) -> None:
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        # вырезанные картинки PaddleOCR-VL — HTML <img src="imgs/..."> (часто в <div>);
        # путь относителен out_dir → _upload_segment_images зальёт в img_s3 и рисунок
        # появится в текст-просмотре. Каждый <img> — отдельный image-сегмент.
        for m in _IMG_HTML.finditer(block):
            drafts.append(SegmentDraft(0, SegmentKind.image, "", pidx, meta=_img_meta(m.group(1))))
        # чистый текст блока без html-обёрток (<img>, <div style=center> у подписей)
        clean = _DIV_TAG.sub("", _IMG_HTML.sub("", block)).strip()
        m_md = _IMG_RE.fullmatch(clean)  # старый markdown-формат ![](path)
        if m_md:
            drafts.append(SegmentDraft(0, SegmentKind.image, "", pidx, meta=_img_meta(m_md.group(1))))
            continue
        if not clean:
            continue
        hm = _HEAD_RE.match(clean)
        if hm and "\n" not in clean:
            drafts.append(
                SegmentDraft(
                    0, SegmentKind.heading, hm.group(2).strip(), pidx, heading_level=len(hm.group(1))
                )
            )
            continue
        for piece in _cap([clean]):
            drafts.append(SegmentDraft(0, SegmentKind.paragraph, piece, pidx))


def _page_index(path: Path) -> int:
    match = _PAGE_FILE_RE.search(path.name)
    if match is None:
        raise ValueError(f"unexpected Paddle page filename: {path.name}")
    return int(match.group(1))


def _layout_text(text: Any) -> str:
    """Нормализовать текст только для привязки Markdown-фрагмента к layout-блоку."""
    if not isinstance(text, str):
        return ""
    return "".join(char.casefold() for char in text if char.isalnum())


def _layout_tokens(text: Any) -> tuple[str, ...]:
    """Unicode-alnum токены для boundary-aware сопоставления Markdown с layout."""

    if not isinstance(text, str):
        return ()
    tokens: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _token_subsequence(
    haystack: tuple[str, ...],
    needle: tuple[str, ...],
    start: int,
) -> int | None:
    """Начало целой token-последовательности; внутри ``Subtotal`` нет ``Total``."""

    if not needle or start < 0 or len(needle) > len(haystack) - start:
        return None
    stop = len(haystack) - len(needle) + 1
    for index in range(start, stop):
        if haystack[index : index + len(needle)] == needle:
            return index
    return None


def _markdown_ignored_labels(page: Mapping[str, Any]) -> frozenset[str]:
    """Метки native-блоков, которые сам Paddle не включил в Markdown."""

    model_settings = page.get("model_settings")
    if not isinstance(model_settings, Mapping):
        return frozenset()
    labels = model_settings.get("markdown_ignore_labels")
    if isinstance(labels, str):
        return frozenset({labels.casefold()})
    if not isinstance(labels, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(label).casefold() for label in labels if label is not None)


def _native_geometry(
    block: dict[str, Any],
    page: dict[str, Any],
    page_size_pt: tuple[float, float] | None,
) -> dict[str, Any]:
    """Перевести bbox Paddle из пикселей в физические пункты исходного PDF."""

    width = page.get("width")
    height = page.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return {}
    page_size_px = [float(width), float(height)]
    boxes: list[list[float]] = []
    candidates = [block]
    group_id = block.get("group_id")
    label = block.get("block_label")
    if group_id is not None:
        siblings = [
            sibling
            for sibling in page.get("parsing_res_list", [])
            if isinstance(sibling, dict)
            and sibling.get("group_id") == group_id
            and sibling.get("block_label") == label
        ]
        populated = [sibling for sibling in siblings if _layout_text(sibling.get("block_content"))]
        # Empty sibling boxes safely complete a split layout region only when
        # the group has one content owner. With two populated blocks, sharing
        # the same empty box would create overlapping translated plaques.
        if len(populated) == 1 and populated[0] is block:
            candidates.extend(
                sibling
                for sibling in siblings
                if sibling is not block and not _layout_text(sibling.get("block_content"))
            )
    for candidate in candidates:
        bbox = candidate.get("block_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values) and values[2] > values[0] and values[3] > values[1]:
            boxes.append(values)
    if (
        not boxes
        or not all(math.isfinite(value) for value in page_size_px)
        or page_size_px[0] <= 0
        or page_size_px[1] <= 0
    ):
        return {}
    union = [
        max(0.0, min(box[0] for box in boxes)),
        max(0.0, min(box[1] for box in boxes)),
        min(page_size_px[0], max(box[2] for box in boxes)),
        min(page_size_px[1], max(box[3] for box in boxes)),
    ]
    if union[2] <= union[0] or union[3] <= union[1]:
        return {}
    # Without the source PDF page size this geometry is useful for diagnostics,
    # but must never masquerade as canonical points consumed by overlays/page
    # fallback.  Production callers always provide ``page_sizes_pt``.
    if page_size_pt is None:
        return {
            "paddle_bbox_px": union,
            "paddle_page_size_px": page_size_px,
            "geometry_space": "paddle_pixels_noncanonical",
        }
    target_width, target_height = page_size_pt
    if not all(math.isfinite(value) for value in page_size_pt) or target_width <= 0 or target_height <= 0:
        return {}
    scale_x = target_width / page_size_px[0]
    scale_y = target_height / page_size_px[1]
    return {
        "bbox_pt": [
            union[0] * scale_x,
            union[1] * scale_y,
            union[2] * scale_x,
            union[3] * scale_y,
        ],
        "page_size_pt": [float(target_width), float(target_height)],
        "geometry_precision": "paddle_native_scaled",
    }


def _matches_native_block(draft: SegmentDraft, block: dict[str, Any]) -> bool:
    label = str(block.get("block_label") or "").casefold()
    if draft.kind == SegmentKind.image:
        return label in _IMAGE_LABELS
    if draft.kind == SegmentKind.table:
        return label in _TABLE_LABELS
    if draft.kind == SegmentKind.equation:
        return label in _EQUATION_LABELS
    if label in _IMAGE_LABELS | _TABLE_LABELS | _EQUATION_LABELS:
        return False
    draft_text = _layout_text(draft.source_text)
    block_text = _layout_text(block.get("block_content"))
    return bool(draft_text and block_text and draft_text in block_text)


def _attach_native_geometry(
    out_dir: Path,
    drafts: list[SegmentDraft],
    page_sizes_pt: Mapping[int, tuple[float, float]] | None,
) -> None:
    """Привязать Markdown-сегменты к bbox из native Paddle JSON.

    Paddle превращает один layout-блок в несколько Markdown-абзацев. Поэтому
    несколько последовательных сегментов могут намеренно получить один bbox;
    scan_pdf объединит их перевод и нарисует блок ровно один раз.
    """
    pages: dict[int, dict[str, Any]] = {}
    for path in sorted(out_dir.glob("*_res.json")):
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
            page_idx = page.get("page_index") if isinstance(page, dict) else None
            if isinstance(page_idx, int) and isinstance(page.get("parsing_res_list"), list):
                pages[page_idx] = page
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.warning("paddle-vl: не удалось прочитать layout JSON %s", path.name)

    attached = 0
    for page_idx, page in pages.items():
        ignored_labels = _markdown_ignored_labels(page)
        blocks = [
            block
            for block in page["parsing_res_list"]
            if isinstance(block, dict)
            and str(block.get("block_label") or "").casefold() not in ignored_labels
            and (
                _layout_tokens(block.get("block_content"))
                or str(block.get("block_label") or "").casefold()
                in _IMAGE_LABELS | _TABLE_LABELS | _EQUATION_LABELS
            )
        ]
        cursor = 0
        token_offsets: dict[int, int] = {}
        for draft in (item for item in drafts if item.page_idx == page_idx):
            match_idx: int | None = None
            token_end: int | None = None
            if draft.kind in (SegmentKind.table, SegmentKind.image, SegmentKind.equation):
                for idx in range(cursor, len(blocks)):
                    if _matches_native_block(draft, blocks[idx]):
                        match_idx = idx
                        break
            else:
                draft_tokens = _layout_tokens(draft.source_text)
                if draft_tokens:
                    # If a native block has already supplied a previous
                    # Markdown fragment, continue that same block first.
                    if cursor < len(blocks) and token_offsets.get(cursor, 0) > 0:
                        block_tokens = _layout_tokens(blocks[cursor].get("block_content"))
                        match_start = _token_subsequence(
                            block_tokens,
                            draft_tokens,
                            token_offsets[cursor],
                        )
                        if match_start is not None:
                            match_idx = cursor
                            token_end = match_start + len(draft_tokens)

                    # New fragments prefer an exact whole-block match anywhere
                    # ahead.  Thus Markdown `Total` cannot bind to `Subtotal`
                    # when the actual `Total` block follows it.
                    exact_start = cursor + 1 if match_idx is None and token_offsets.get(cursor, 0) else cursor
                    if match_idx is None:
                        for idx in range(exact_start, len(blocks)):
                            if _layout_tokens(blocks[idx].get("block_content")) == draft_tokens:
                                match_idx = idx
                                token_end = len(draft_tokens)
                                break

                    # Paddle can split one native block into several Markdown
                    # paragraphs.  The fallback is token-boundary aware and
                    # ordered, never an unrestricted character substring.
                    if match_idx is None:
                        for idx in range(cursor, len(blocks)):
                            block_tokens = _layout_tokens(blocks[idx].get("block_content"))
                            start = token_offsets.get(idx, 0)
                            match_start = _token_subsequence(block_tokens, draft_tokens, start)
                            if match_start is not None:
                                match_idx = idx
                                token_end = match_start + len(draft_tokens)
                                break
            if match_idx is None:
                continue
            page_size_pt = page_sizes_pt.get(page_idx) if page_sizes_pt is not None else None
            geometry = _native_geometry(blocks[match_idx], page, page_size_pt)
            if geometry:
                draft.meta.update(geometry)
                attached += 1
            # Один text-layout блок Paddle разворачивает в несколько Markdown-
            # абзацев, поэтому текст может повторно матчиться на текущий блок.
            # Структурный блок даёт ровно один segment: после него обязательно
            # продвигаемся, иначе две таблицы/картинки получат первый bbox.
            if draft.kind in (SegmentKind.table, SegmentKind.image, SegmentKind.equation):
                cursor = match_idx + 1
            elif token_end is not None:
                block_size = len(_layout_tokens(blocks[match_idx].get("block_content")))
                if token_end >= block_size:
                    cursor = match_idx + 1
                    token_offsets.pop(match_idx, None)
                else:
                    cursor = match_idx
                    token_offsets[match_idx] = token_end
    if pages:
        logger.info("paddle-vl: geometry attached to %d/%d segments", attached, len(drafts))


def paddle_to_segments(
    out_dir: Path,
    page_sizes_pt: Mapping[int, tuple[float, float]] | None = None,
) -> list[SegmentDraft]:
    """Постраничный Markdown PaddleOCR-VL → SegmentDraft (idx по порядку).

    ``page_sizes_pt`` обязателен для канонической геометрии: Paddle возвращает
    bbox в пикселях своего растра, а overlay хранит координаты PDF в пунктах.
    """
    files = sorted(
        out_dir.glob("doc_*.md"),
        key=_page_index,
    )
    drafts: list[SegmentDraft] = []
    for f in files:
        pidx = _page_index(f)
        md = f.read_text(encoding="utf-8")
        pos = 0
        for m in _TABLE_RE.finditer(md):
            _text_blocks(md[pos : m.start()], pidx, drafts)
            cells, rows = parse_table(m.group(0))
            if cells:
                preview = "\n".join(" | ".join(c["text"] for c in row) for row in cells)
                drafts.append(
                    SegmentDraft(
                        0,
                        SegmentKind.table,
                        preview,
                        pidx,
                        meta={"table_cells": cells, "table_rows": rows, "caption": ""},
                    )
                )
            pos = m.end()
        _text_blocks(md[pos:], pidx, drafts)
    for i, d in enumerate(drafts):
        d.idx = i
    _attach_native_geometry(out_dir, drafts, page_sizes_pt)
    return drafts
