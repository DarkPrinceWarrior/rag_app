"""Изолированный direct-Transformers smoke LightOnOCR-2 на сложных PDF-страницах."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import re
import time
from pathlib import Path
from typing import Any

_MODEL = "lightonai/LightOnOCR-2-1B"
_REVISION = "c97bd377f04481830395218fa8951df9deaba756"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_BOX_RE = re.compile(r"image\s*<?\s*(\d+),(\d+),(\d+),(\d+)\s*>?", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def _runtime() -> str:
    packages = ("transformers", "torch", "pillow", "pypdfium2")
    return ";".join(
        f"{package}={importlib.metadata.version(package)}" for package in packages
    )


def _quality_signals(text: str) -> dict[str, Any]:
    characters = len(text)
    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    image_boxes = []
    for match in _IMAGE_BOX_RE.finditer(text):
        values = tuple(int(value) for value in match.groups())
        if 0 <= values[0] < values[2] <= 1000 and 0 <= values[1] < values[3] <= 1000:
            image_boxes.append(values)
    longest_run = 0
    current_run = 0
    previous = ""
    for character in text:
        if character == previous:
            current_run += 1
        else:
            previous = character
            current_run = 1
        longest_run = max(longest_run, current_run)
    return {
        "characters": characters,
        "nonempty_lines": len(nonempty_lines),
        "markdown_table_lines": sum(
            line.count("|") >= 2 for line in nonempty_lines
        ),
        "html_table_count": len(re.findall(r"<table\b", text, flags=re.I)),
        "display_math_count": text.count("$$") // 2,
        "image_bbox_count": len(image_boxes),
        "longest_repeated_character_run": longest_run,
        "acceptable_smoke": bool(
            text.strip()
            and characters <= 4 * 1024 * 1024
            and longest_run <= max(128, characters // 4)
        ),
    }


def _render_pdf(pdf_path: Path, dpi: int, max_image_dimension: int):
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    document = pdfium.PdfDocument(pdf_path.read_bytes())
    try:
        if len(document) != 1:
            raise ValueError(f"ожидалась одна страница: {pdf_path.name}")
        page = document[0]
        try:
            image = page.render(scale=dpi / 72).to_pil().convert("RGB")
            image.thumbnail((max_image_dimension, max_image_dimension))
        finally:
            page.close()
    finally:
        document.close()
    return image


class _LightOnOCR:
    def __init__(self, snapshot: Path) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            LightOnOcrForConditionalGeneration,
            LightOnOcrProcessor,
        )

        self._torch = torch
        self._processor = LightOnOcrProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            snapshot,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        self._model.eval()

    def generate(self, image: Any, *, max_new_tokens: int) -> str:
        import base64

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        conversation = [
            {"role": "user", "content": [{"type": "image", "url": data_url}]}
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(
                device=self._model.device,
                dtype=self._model.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--revision", default=_REVISION)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-image-dimension", type=int, default=1540)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not _COMMIT_RE.fullmatch(args.revision):
        parser.error("--revision должен быть полным lowercase commit SHA")
    if not 72 <= args.dpi <= 300:
        parser.error("--dpi должен быть в диапазоне 72..300")
    if not 512 <= args.max_image_dimension <= 4096:
        parser.error("--max-image-dimension должен быть в диапазоне 512..4096")
    if not 1 <= args.max_new_tokens <= 32768:
        parser.error("--max-new-tokens должен быть в диапазоне 1..32768")
    for path in args.pdf:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".pdf":
            parser.error(f"небезопасный или отсутствующий PDF: {path}")

    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    snapshot = Path(snapshot_download(args.model, revision=args.revision)).resolve()
    if snapshot.name != args.revision:
        raise RuntimeError(f"HF snapshot не закреплен ожидаемой ревизией: {snapshot}")
    inference = _LightOnOCR(snapshot)
    load_latency = round(time.monotonic() - started, 3)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "runtime": _runtime(),
        "dpi": args.dpi,
        "max_image_dimension": args.max_image_dimension,
        "max_new_tokens": args.max_new_tokens,
        "load_latency_s": load_latency,
        "results": {},
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    for pdf_path in args.pdf:
        result: dict[str, Any]
        try:
            render_started = time.monotonic()
            image = _render_pdf(pdf_path, args.dpi, args.max_image_dimension)
            render_latency = round(time.monotonic() - render_started, 3)
            inference_started = time.monotonic()
            text = inference.generate(image, max_new_tokens=args.max_new_tokens)
            inference_latency = round(time.monotonic() - inference_started, 3)
            raw_path = args.output_dir / f"{pdf_path.name}.txt"
            raw_path.write_text(text, encoding="utf-8")
            result = {
                "status": "ok",
                "source_sha256": _sha256(pdf_path),
                "render_latency_s": render_latency,
                "inference_latency_s": inference_latency,
                "signals": _quality_signals(text),
            }
        except Exception as exc:
            result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        summary["results"][pdf_path.name] = result
        _atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps({pdf_path.name: result}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
