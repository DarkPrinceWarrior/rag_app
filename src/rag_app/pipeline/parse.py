"""Парсинг PDF через MinerU CLI → content_list.json (единый структурный формат).

CLI вместо программного API: внутренний API MinerU нестабилен между версиями,
формат content_list — стабильный контракт.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    # mineru: явный путь из настроек (изолированный venv для VLM/vllm) или сосед
    # текущего python (общий venv); PATH в tmux/systemd может бинарь не содержать
    mineru_bin = (
        Path(settings.mineru_bin) if settings.mineru_bin else Path(sys.executable).with_name("mineru")
    )
    binpath = str(mineru_bin) if mineru_bin.exists() else "mineru"
    # vllm-движок MinerU не уважает MINERU_DEVICE_MODE и берёт cuda:0 → пиннингуем
    # карту через CUDA_VISIBLE_DEVICES. PATH с .venv/bin: vllm зовёт `ninja` для
    # JIT-компиляции — без него VLM-движок падает на инициализации.
    gpu_idx = settings.mineru_device.rsplit(":", 1)[-1]
    venv_bin = str(Path(sys.executable).parent)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu_idx,
        MINERU_DEVICE_MODE="cuda:0",
        PATH=venv_bin + os.pathsep + os.environ.get("PATH", ""),
    )

    def build(be: str) -> list[str]:
        cmd = [binpath, "-p", str(input_pdf), "-o", str(out_dir), "-b", be]
        if be == "pipeline":
            # -m auto: текстовый слой / OCR постранично (roadmap § 3.1); -l только pipeline
            cmd += ["-m", method or settings.mineru_method, "-l", lang or settings.mineru_lang]
        else:
            # VLM-бэкенды: http-client требует URL сервера; -t выключает ложные таблицы
            if be.endswith("-http-client"):
                cmd += ["-u", settings.mineru_vlm_url]
            cmd += ["-t", str(settings.mineru_table_enable)]
        return cmd

    async def _run(cmd: list[str]) -> Path:
        logger.info("mineru: %s (GPU=%s)", " ".join(cmd), gpu_idx)
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.mineru_timeout_s)
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"mineru: таймаут {settings.mineru_timeout_s}s") from None
        if proc.returncode != 0:
            raise RuntimeError(f"mineru: код {proc.returncode}\n{out.decode(errors='replace')[-3000:]}")
        candidates = sorted(out_dir.rglob("*_content_list.json"))
        if not candidates:
            raise RuntimeError(f"mineru: content_list.json не найден в {out_dir}\n{out.decode(errors='replace')[-2000:]}")
        return candidates[0]

    be = backend or settings.mineru_backend
    try:
        return await _run(build(be))
    except Exception as exc:
        # VLM-сервер недоступен/упал → не блокируем документ, парсим pipeline'ом
        # (sorted() предпочтёт «auto/» над «vlm/», если остался частичный вывод)
        if be != "pipeline" and "pipeline" not in (backend or ""):
            logger.warning("mineru backend=%s упал (%s) — фолбэк на pipeline", be, exc)
            return await _run(build("pipeline"))
        raise


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
    # построчная геометрия блока (x0, текст) — для восстановления переносов/отступов
    block_lines: dict[tuple[int, str], list[tuple[float, str]]] = field(default_factory=dict)

    def match_text(self, page_idx: int | None, text: str) -> list[float] | None:
        if page_idx is None:
            return None
        return self.text_map.get((page_idx, _norm_text(text)))

    def pop_typed(self, page_idx: int | None, block_type: str) -> list[float] | None:
        if page_idx is None:
            return None
        lst = self.typed.get((page_idx, block_type))
        return lst.pop(0) if lst else None

    def reflow(self, page_idx: int | None, text: str) -> str | None:
        """Восстановить переносы и отступы списка/оглавления из строк middle.json.

        content_list схлопывает оглавление (лидер-точки + номера страниц) в один
        абзац-«кашу». Здесь, если блок похож на список (≥4 строк, большинство
        кончается цифрой — номером пункта/страницы), возвращаем многострочный
        текст: одна запись на строку, отступ по x0 (уровень вложенности). Иначе
        None — обычные абзацы оставляем флоу-текстом.
        """
        if page_idx is None:
            return None
        lines = self.block_lines.get((page_idx, _norm_text(text)))
        if not lines or len(lines) < 4:
            return None
        ends_digit = sum(1 for _, t in lines if t.strip() and t.strip()[-1].isdigit())
        if ends_digit < len(lines) * 0.5:
            return None
        # уровни отступа по левому краю (кластеризация x0 округлением до 5 pt)
        xs = sorted({round(x0 / 5) * 5 for x0, _ in lines})
        level = {x: i for i, x in enumerate(xs)}
        out: list[str] = []
        for x0, t in lines:
            lv = min(level[round(x0 / 5) * 5], 4)
            # пробел между названием и слипшимся номером страницы: «…устройства244»
            entry = re.sub(r"(\D)(\d[\d\s]*)$", r"\1 \2", t.strip())
            out.append("    " * lv + entry)
        return "\n".join(out)


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
            lines = blk.get("lines", [])
            text = "".join(
                span.get("content", "") for line in lines for span in line.get("spans", [])
            )
            if text.strip():
                key = (p_idx, _norm_text(text))
                geo.text_map[key] = bbox
                # построчно: (x0, текст строки) — для reflow списков/оглавлений
                rows: list[tuple[float, str]] = []
                for line in lines:
                    lt = "".join(sp.get("content", "") for sp in line.get("spans", []))
                    lb = line.get("bbox")
                    if lt.strip() and lb:
                        rows.append((float(lb[0]), lt))
                if rows:
                    geo.block_lines[key] = rows
    return geo
