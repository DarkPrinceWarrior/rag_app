"""Рендер офисных документов в PDF через LibreOffice headless.

Для просмотра «как в Microsoft»: docx/xlsx/pptx (оригинал и перевод) → PDF,
который показывается в том же pdf.js-вьювере, что и обычные PDF. Вёрстка,
таблицы, слайды сохраняются — в отличие от плоского текстового рендера.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from pathlib import Path

SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")


class OfficeRenderError(RuntimeError):
    pass


def _convert_sync(src: Path, out_dir: Path, timeout_s: int) -> Path:
    if not SOFFICE:
        raise OfficeRenderError("LibreOffice (soffice) не установлен")
    # своя UserInstallation на конвертацию — иначе параллельные soffice конфликтуют
    profile = out_dir / f"lo_{uuid.uuid4().hex[:8]}"
    cmd = [
        SOFFICE,
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        f"-env:UserInstallation=file://{profile}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(src),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise OfficeRenderError(f"soffice таймаут ({timeout_s}s)") from exc
    pdf = out_dir / f"{src.stem}.pdf"
    if not pdf.exists():
        raise OfficeRenderError(
            f"soffice не создал PDF (rc={proc.returncode}): {proc.stderr.decode('utf-8','ignore')[:300]}"
        )
    return pdf


async def render_to_pdf(src: Path, out_dir: Path, timeout_s: int = 150) -> bytes:
    """OOXML-файл → PDF-байты (LibreOffice). Бросает OfficeRenderError при сбое.

    Для .pptx предварительно включаем автоподгонку текста (перенос + «ужать до
    фигуры»), иначе LibreOffice выпускает длинный/переведённый текст за границы
    фигур и наезжает на картинки. Затрагивает только PDF-просмотр."""
    render_src = src
    if src.suffix.lower() == ".pptx":
        try:
            from rag_app.pipeline.ooxml import pptx_autofit

            fitted = out_dir / f"{src.stem}__fit.pptx"
            await asyncio.to_thread(pptx_autofit, src, fitted)
            render_src = fitted
        except Exception:
            render_src = src
    pdf = await asyncio.to_thread(_convert_sync, render_src, out_dir, timeout_s)
    return pdf.read_bytes()
