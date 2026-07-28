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
    # Layout coordinates must stay in the same orientation and rectangular
    # coordinate space as the source PDF.  Paddle's document preprocessor can
    # otherwise rotate or unwarp the raster before layout detection, making a
    # simple px -> PDF-point transform invalid for the translated overlay.
    for res in pipeline.predict(
        inp,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    ):
        # Markdown is the canonical content consumed by the parser.  Keep the
        # native block JSON alongside it so scan exports can retain Paddle's
        # layout coordinates instead of leaving the original text as a
        # background image.
        res.save_to_json(save_path=out)
        res.save_to_markdown(save_path=out)
        n += 1
    print(f"paddle: {n} pages -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
