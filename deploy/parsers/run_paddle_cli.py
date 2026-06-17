"""CLI-обёртка PaddleOCR-VL 1.6 для воркера: PDF → постраничный Markdown.

Запускается из изолированного paddle-venv (settings.paddle_venv_python) как
подпроцесс: `python run_paddle_cli.py <input.pdf> <out_dir>`. Модель PaddleOCR-VL
1.6 тянется автоматически в ~/.paddlex и грузится на GPU (CUDA_VISIBLE_DEVICES
задаёт воркер). Кладёт <stem>_<page>.md — их разбирает pipeline.paddle_vl.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    inp, out = sys.argv[1], sys.argv[2]
    Path(out).mkdir(parents=True, exist_ok=True)
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL()
    n = 0
    for res in pipeline.predict(inp):
        res.save_to_markdown(save_path=out)
        n += 1
    print(f"paddle: {n} pages -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
