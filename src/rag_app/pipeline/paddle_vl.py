"""PaddleOCR-VL 1.6 как альтернативный парсер pdf_text.

On-demand: воркер запускает `deploy/parsers/run_paddle_cli.py` из изолированного
paddle-venv (грузит модель на GPU4 на время парса, потом освобождает). Скрипт
кладёт постраничный Markdown `<stem>_<page>.md`; адаптер `paddle_to_segments`
разбирает md → SegmentDraft (заголовки `#`, таблицы `<table>` через parse_table,
остальное — абзацы, с капом длины под лимит перевода).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path

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


def _img_meta(rel: str) -> dict:
    rel = (rel or "").strip().split()[0] if (rel or "").strip() else ""
    return {"img_path": rel} if rel and not rel.startswith(("http://", "https://", "data:")) else {}


async def run_paddle(pdf_path: Path, out_dir: Path) -> Path:
    """Прогон PaddleOCR-VL; возвращает каталог с постраничным Markdown (doc_N.md)."""
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
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.paddle_timeout_s)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"paddle-vl: таймаут {settings.paddle_timeout_s}s") from None
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
                SegmentDraft(0, SegmentKind.heading, hm.group(2).strip(), pidx,
                             heading_level=len(hm.group(1)))
            )
            continue
        for piece in _cap([clean]):
            drafts.append(SegmentDraft(0, SegmentKind.paragraph, piece, pidx))


def paddle_to_segments(out_dir: Path) -> list[SegmentDraft]:
    """Постраничный Markdown PaddleOCR-VL → SegmentDraft (idx по порядку)."""
    files = sorted(
        out_dir.glob("doc_*.md"),
        key=lambda p: int(re.search(r"_(\d+)\.md$", p.name).group(1)),
    )
    drafts: list[SegmentDraft] = []
    for f in files:
        pidx = int(re.search(r"_(\d+)\.md$", f.name).group(1))
        md = f.read_text(encoding="utf-8")
        pos = 0
        for m in _TABLE_RE.finditer(md):
            _text_blocks(md[pos:m.start()], pidx, drafts)
            cells, rows = parse_table(m.group(0))
            if cells:
                preview = "\n".join(" | ".join(c["text"] for c in row) for row in cells)
                drafts.append(SegmentDraft(
                    0, SegmentKind.table, preview, pidx,
                    meta={"table_cells": cells, "table_rows": rows, "caption": ""},
                ))
            pos = m.end()
        _text_blocks(md[pos:], pidx, drafts)
    for i, d in enumerate(drafts):
        d.idx = i
    return drafts
