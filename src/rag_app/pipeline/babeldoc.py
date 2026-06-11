"""BabelDOC: PDF → переведённый PDF с сохранением вёрстки (roadmap § 3.3.A).

AGPL-3.0 → изоляция (roadmap § 9): отдельный venv /root/services/babeldoc,
вызов ТОЛЬКО через CLI (никаких импортов), конфигурация снаружи,
исходники не модифицируем. Перевод идёт через наш vLLM (OpenAI-совместимый).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from rag_app.config import settings

logger = logging.getLogger(__name__)


class BabelDocUnavailableError(Exception):
    """CLI не установлен/выключен — PDF-экспорт с вёрсткой пропускается."""


async def run_babeldoc(
    input_pdf: Path, out_dir: Path, ocr_workaround: bool = False
) -> tuple[Path | None, Path | None]:
    """Возвращает (mono_pdf, dual_pdf): только перевод / EN+RU постранично.

    ocr_workaround → --auto-enable-ocr-workaround: для searchable-сканов
    (растр + текстовый слой) BabelDOC кладёт перевод на белых плашках;
    image-only сканы он не берёт даже так («no paragraphs» — проверено,
    они идут через наш оверлей).
    """
    if not settings.babeldoc_enabled:
        raise BabelDocUnavailableError("BabelDOC выключен (RAG_BABELDOC_ENABLED=false)")
    babeldoc = Path(settings.babeldoc_bin)
    if not babeldoc.exists():
        raise BabelDocUnavailableError(f"нет бинаря {babeldoc} — deploy/setup_babeldoc.sh")

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(babeldoc),
        "--files", str(input_pdf),
        "--openai",
        "--openai-model", settings.llm_model,
        "--openai-base-url", settings.llm_base_url,
        "--openai-api-key", settings.llm_api_key,
        "--lang-in", "en",
        "--lang-out", "ru",
        "--output", str(out_dir),
        "--qps", str(settings.babeldoc_qps),
        "--watermark-output-mode", "no_watermark",
    ]
    if ocr_workaround:
        cmd.append("--auto-enable-ocr-workaround")
    logger.info("babeldoc: %s", input_pdf.name)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.babeldoc_timeout_s)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"babeldoc: таймаут {settings.babeldoc_timeout_s}s") from None
    if proc.returncode != 0:
        tail = out.decode(errors="replace")[-3000:]
        raise RuntimeError(f"babeldoc: код {proc.returncode}\n{tail}")

    pdfs = list(out_dir.rglob("*.pdf"))
    mono = next((p for p in pdfs if ".mono." in p.name), None)
    dual = next((p for p in pdfs if ".dual." in p.name), None)
    if mono is None and dual is None:
        tail = out.decode(errors="replace")[-2000:]
        raise RuntimeError(f"babeldoc: PDF не найден в {out_dir}\n{tail}")
    return mono, dual
