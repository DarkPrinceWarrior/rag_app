"""Парсинг PDF через MinerU CLI → content_list.json (единый структурный формат).

CLI вместо программного API: внутренний API MinerU нестабилен между версиями,
формат content_list — стабильный контракт.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from rag_app.config import settings

logger = logging.getLogger(__name__)


class NoTextLayerError(Exception):
    """Скан без текстового слоя — OCR-ветка появится на этапе 2."""


# pdfium НЕ потокобезопасен: конкурентные вызовы из asyncio.to_thread дают
# segfault (воркер умирает молча — поймано нагрузочным тестом этапа 5).
# Все обращения к pypdfium2 в процессе — под одним замком.
PDFIUM_LOCK = threading.Lock()


def pdf_info(path: Path, sample_pages: int = 5) -> tuple[int, bool]:
    """(число страниц, есть ли текстовый слой). Проверка — миллисекунды (roadmap § 3.1)."""
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(path))
        try:
            n_pages = len(doc)
            chars = 0
            for i in range(min(sample_pages, n_pages)):
                textpage = doc[i].get_textpage()
                chars += len(textpage.get_text_bounded() or "")
                if chars > 100:
                    return n_pages, True
            return n_pages, chars > 100
        finally:
            doc.close()


async def run_mineru(
    input_pdf: Path,
    out_dir: Path,
    *,
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
) -> Path:
    """Запуск mineru CLI; возвращает путь к *_content_list.json.

    backend/method/lang переопределяют дефолты (settings). Для форс-OCR битого
    cmap используем backend="vlm-engine" (MinerU 3.3 VLM — multilingual, не нужен
    -m/-l); pipeline-бэкенд требует -m/-l.

    Девайс в MinerU 3.x задаётся только через env MINERU_DEVICE_MODE
    (флага -d в CLI больше нет).
    """
    # mineru лежит в том же venv, что и воркер; PATH в tmux/systemd может его не содержать
    mineru_bin = Path(sys.executable).with_name("mineru")
    be = backend or settings.mineru_backend
    cmd = [
        str(mineru_bin) if mineru_bin.exists() else "mineru",
        "-p", str(input_pdf),
        "-o", str(out_dir),
        "-b", be,
    ]
    if be == "pipeline":
        # -m auto: текстовый слой / OCR выбирается постранично (roadmap § 3.1);
        # VLM-бэкенды multilingual и метод/язык не принимают
        cmd += ["-m", method or settings.mineru_method, "-l", lang or settings.mineru_lang]
    env = dict(os.environ, MINERU_DEVICE_MODE=settings.mineru_device)
    logger.info("mineru: %s (device=%s)", " ".join(cmd), settings.mineru_device)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.mineru_timeout_s)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"mineru: таймаут {settings.mineru_timeout_s}s") from None
    if proc.returncode != 0:
        tail = out.decode(errors="replace")[-3000:]
        raise RuntimeError(f"mineru: код {proc.returncode}\n{tail}")

    candidates = sorted(out_dir.rglob("*_content_list.json"))
    if not candidates:
        tail = out.decode(errors="replace")[-3000:]
        raise RuntimeError(f"mineru: content_list.json не найден в {out_dir}\n{tail}")
    return candidates[0]


def load_content_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"неожиданный формат content_list: {type(data)}")
    return data


def _norm_text(text: str) -> str:
    return "".join(text.lower().split())[:40]


@dataclass
class BlockGeometry:
    """Геометрия блоков из *_middle.json — bbox в ПУНКТАХ страницы.

    bbox из content_list — во внутреннем масштабе рендера MinerU (непригоден
    для наложения), а para_blocks из middle.json — в координатах страницы
    (проверено по pdfium): их и используем для оверлея сканов.
    """

    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    text_map: dict[tuple[int, str], list[float]] = field(default_factory=dict)
    typed: dict[tuple[int, str], list[list[float]]] = field(default_factory=dict)

    def match_text(self, page_idx: int | None, text: str) -> list[float] | None:
        if page_idx is None:
            return None
        return self.text_map.get((page_idx, _norm_text(text)))

    def pop_typed(self, page_idx: int | None, block_type: str) -> list[float] | None:
        if page_idx is None:
            return None
        lst = self.typed.get((page_idx, block_type))
        return lst.pop(0) if lst else None


_TYPED_BLOCKS = {"table": "table", "image": "image", "interline_equation": "equation"}


def load_block_geometry(content_list_path: Path) -> BlockGeometry:
    geo = BlockGeometry()
    middle_path = Path(str(content_list_path).replace("_content_list.json", "_middle.json"))
    if not middle_path.exists():
        logger.warning("middle.json не найден: %s — оверлей сканов будет недоступен", middle_path)
        return geo
    data = json.loads(middle_path.read_text(encoding="utf-8"))
    for page in data.get("pdf_info", []):
        p_idx = page.get("page_idx")
        size = page.get("page_size")
        if p_idx is None or not size:
            continue
        geo.page_sizes[p_idx] = (float(size[0]), float(size[1]))
        for blk in page.get("para_blocks", []):
            btype = blk.get("type")
            bbox = blk.get("bbox")
            if not bbox:
                continue
            if btype in _TYPED_BLOCKS:
                geo.typed.setdefault((p_idx, _TYPED_BLOCKS[btype]), []).append(bbox)
                continue
            text = "".join(
                span.get("content", "")
                for line in blk.get("lines", [])
                for span in line.get("spans", [])
            )
            if text.strip():
                geo.text_map[(p_idx, _norm_text(text))] = bbox
    return geo
