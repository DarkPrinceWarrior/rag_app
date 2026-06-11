"""Парсинг PDF через MinerU CLI → content_list.json (единый структурный формат).

CLI вместо программного API: внутренний API MinerU нестабилен между версиями,
формат content_list — стабильный контракт.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from rag_app.config import settings

logger = logging.getLogger(__name__)


class NoTextLayerError(Exception):
    """Скан без текстового слоя — OCR-ветка появится на этапе 2."""


def pdf_info(path: Path, sample_pages: int = 5) -> tuple[int, bool]:
    """(число страниц, есть ли текстовый слой). Проверка — миллисекунды (roadmap § 3.1)."""
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


async def run_mineru(input_pdf: Path, out_dir: Path) -> Path:
    """Запуск mineru CLI; возвращает путь к *_content_list.json.

    Девайс в MinerU 3.x задаётся только через env MINERU_DEVICE_MODE
    (флага -d в CLI больше нет).
    """
    cmd = [
        "mineru",
        "-p", str(input_pdf),
        "-o", str(out_dir),
        "-b", settings.mineru_backend,
    ]
    if settings.mineru_backend == "pipeline":
        cmd += ["-l", settings.mineru_lang]
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
