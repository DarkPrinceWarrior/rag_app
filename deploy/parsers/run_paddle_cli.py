"""CLI-обёртка PaddleOCR-VL для воркера: PDF → постраничный Markdown.

Запускается из изолированного paddle-venv (settings.paddle_venv_python) как
подпроцесс: `python run_paddle_cli.py <input.pdf> <out_dir>`. VLM-распознавание
идёт на ПОСТОЯННЫЙ genai vLLM-сервер (paddlex_genai_server, PaddleOCR-VL-0.9B) —
адрес в env PADDLE_VL_SERVER_URL; layout-детекция выполняется локально на
CUDA_VISIBLE_DEVICES. Без сервера PaddleOCRVL() в 3.7 виснет (нет inference-движка),
поэтому on-demand не используем. save_to_markdown кладёт <stem>_<page>.md и
вырезанные картинки в подпапку рядом — их разбирает pipeline.paddle_vl.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    inp, out = sys.argv[1], sys.argv[2]
    Path(out).mkdir(parents=True, exist_ok=True)
    from paddleocr import PaddleOCRVL

    server = os.environ.get("PADDLE_VL_SERVER_URL")
    kwargs = {"vl_rec_backend": "vllm-server", "vl_rec_server_url": server} if server else {}
    pipeline = PaddleOCRVL(**kwargs)
    n = 0
    for res in pipeline.predict(inp):
        res.save_to_markdown(save_path=out)
        n += 1
    print(f"paddle: {n} pages -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
