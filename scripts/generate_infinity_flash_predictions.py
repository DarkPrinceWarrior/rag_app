"""Сгенерировать канонические Infinity-Parser2-Flash predictions для parser A/B.

Скрипт запускается в отдельном GPU-окружении с официальным ``infinity_parser2``
и не импортируется API/worker. Входом служит corpus manifest ParseBench, выходом
— ``<source.pdf>.json`` для ``benchmark_complex_parsers.py --prediction``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

_MODEL = "infly/Infinity-Parser2-Flash"
_REVISION = "9837b83778196e6107b3767ca62eb5bdfc08f22a"
_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PROMPT_DOC2JSON = """
Extract layout information from the provided PDF image.
For each layout element, output its bbox, category, and the text content within the bbox.
Bbox format: [x1, y1, x2, y2].
Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula',
'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote',
'table_footnote', 'page_footnote', 'footer'].
Text extraction and formatting:
1) For 'figure', the text field must be an empty string.
2) For 'formula', format text as LaTeX.
3) For 'table', format text as HTML.
4) For all other categories, format text as Markdown.
The output text must be exactly the original text from the image, with no translation or rewriting.
Sort all layout elements in human reading order.
Final output must be a single JSON object.
""".strip()
_CATEGORIES = {
    "header",
    "title",
    "text",
    "figure",
    "table",
    "formula",
    "figure_caption",
    "table_caption",
    "formula_caption",
    "figure_footnote",
    "table_footnote",
    "page_footnote",
    "footer",
}
_PARAGRAPH_CATEGORIES = {
    "header",
    "text",
    "figure_caption",
    "table_caption",
    "formula_caption",
    "figure_footnote",
    "table_footnote",
    "page_footnote",
    "footer",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _reject_constant(value: str) -> None:
    raise ValueError(f"недопустимая JSON-константа: {value}")


def _parse_layout(raw_output: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_output, str):
        raise ValueError("Infinity output должен быть строкой")
    stripped = raw_output.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped, parse_constant=_reject_constant)
    if isinstance(parsed, dict):
        containers = [parsed.get(name) for name in ("layout", "elements", "result")]
        parsed = next((value for value in containers if isinstance(value, list)), None)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Infinity output должен содержать непустой массив layout elements")
    if len(parsed) > 100_000 or any(not isinstance(item, dict) for item in parsed):
        raise ValueError("Infinity layout имеет неверный размер или тип элемента")
    return parsed


def _span(attributes: list[tuple[str, str | None]], name: str) -> int:
    for key, value in attributes:
        if key == name:
            try:
                span = int(value or 1)
            except ValueError as exc:
                raise ValueError(f"table {name} должен быть целым числом") from exc
            if span < 1 or span > 1000:
                raise ValueError(f"table {name} вне диапазона 1..1000")
            return span
    return 1


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, Any]]] = []
        self._cell: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"}:
            self._cell = []
            self._colspan = _span(attrs, "colspan")
            self._rowspan = _span(attrs, "rowspan")
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag not in {"td", "th"} or self._cell is None:
            return
        if not self.rows:
            self.rows.append([])
        text = " ".join("".join(self._cell).split())
        self.rows[-1].append({"text": text, "colspan": self._colspan, "rowspan": self._rowspan})
        self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _table_cells(html: str) -> list[list[dict[str, Any]]]:
    parser = _TableParser()
    parser.feed(html)
    rows = [row for row in parser.rows if row]
    if not rows:
        raise ValueError("Infinity table не содержит распознаваемых HTML-ячеек")
    if sum(len(row) for row in rows) > 200_000:
        raise ValueError("Infinity table превышает лимит ячеек")
    return rows


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field}: ожидалось число")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: ожидалось конечное число")
    return result


def _layout_to_segments(
    raw_output: str,
    *,
    image_size_px: tuple[int, int],
    page_size_pt: tuple[float, float],
) -> list[dict[str, Any]]:
    image_width, image_height = image_size_px
    page_width, page_height = page_size_pt
    if image_width < 1 or image_height < 1 or page_width <= 0 or page_height <= 0:
        raise ValueError("размер изображения и страницы должен быть положительным")

    segments: list[dict[str, Any]] = []
    for index, element in enumerate(_parse_layout(raw_output)):
        category = element.get("category")
        if category not in _CATEGORIES:
            raise ValueError(f"layout[{index}].category: неизвестная категория {category!r}")
        text = element.get("text")
        if not isinstance(text, str) or len(text) > 2_000_000:
            raise ValueError(f"layout[{index}].text: ожидалась ограниченная строка")
        bbox = element.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"layout[{index}].bbox: нужны четыре координаты")
        x0, y0, x1, y1 = [_finite_number(value, field=f"layout[{index}].bbox") for value in bbox]
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > 1000 or y1 > 1000:
            raise ValueError(f"layout[{index}].bbox: нормализованный bbox вне 0..1000")
        bbox_pt = [
            round(x0 * page_width / 1000, 4),
            round(y0 * page_height / 1000, 4),
            round(x1 * page_width / 1000, 4),
            round(y1 * page_height / 1000, 4),
        ]
        if category == "title":
            kind = "heading"
            heading_level = 1
        elif category == "table":
            kind = "table"
            heading_level = None
        elif category == "formula":
            kind = "equation"
            heading_level = None
        elif category == "figure":
            kind = "image"
            heading_level = None
        elif category in _PARAGRAPH_CATEGORIES:
            kind = "paragraph"
            heading_level = None
        else:
            raise AssertionError(f"unmapped Infinity category: {category}")
        meta: dict[str, Any] = {
            "bbox_pt": bbox_pt,
            "page_size_pt": [round(page_width, 4), round(page_height, 4)],
            "parser_category": category,
            "source_bbox_1000": [x0, y0, x1, y1],
            "render_size_px": [image_width, image_height],
        }
        if kind == "table":
            try:
                meta["table_cells"] = _table_cells(text)
            except ValueError as exc:
                kind = "paragraph"
                meta["table_parse_error"] = str(exc)
        segments.append(
            {
                "idx": index,
                "kind": kind,
                "source_text": text,
                "page_idx": 0,
                "heading_level": heading_level,
                "meta": meta,
            }
        )
    return segments


def _prediction_complete(path: Path, *, source_sha256: str, revision: str) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == _SCHEMA_VERSION
        and value.get("source") == {"file": path.name.removesuffix(".json"), "sha256": source_sha256}
        and isinstance(value.get("model"), dict)
        and value["model"].get("revision") == revision
        and isinstance(value.get("segments"), list)
        and value["segments"]
    )


def _runtime() -> str:
    packages = ("transformers", "torch", "qwen-vl-utils", "pymupdf", "pillow")
    versions = []
    for package in packages:
        try:
            versions.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{package}=missing")
    return ";".join(versions)


class _DirectInfinityParser:
    def __init__(self, snapshot: Path) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        self._torch = torch
        model_object: Any = AutoModelForImageTextToText.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
        )
        self._model = model_object.to("cuda")
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(snapshot)

    def parse(self, image: Any, *, max_new_tokens: int, max_time_s: float) -> str:
        from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 2048,
                        "max_pixels": 16_777_216,
                    },
                    {"type": "text", "text": _PROMPT_DOC2JSON},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, _ = process_vision_info(messages, image_patch_size=16)
        inputs = self._processor(
            text=text,
            images=image_inputs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._model.device) if isinstance(value, self._torch.Tensor) else value
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_time=max_time_s,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated, strict=True)
        ]
        outputs = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(outputs) != 1 or not isinstance(outputs[0], str):
            raise ValueError("Infinity direct runner вернул неожиданный batch")
        return outputs[0]


def _load_parser(model: str, revision: str) -> tuple[_DirectInfinityParser, str, float]:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    started = time.monotonic()
    snapshot = Path(snapshot_download(repo_id=model, revision=revision)).resolve()
    if snapshot.name != revision:
        raise ValueError(f"HF snapshot не закреплен ожидаемой ревизией: {snapshot}")
    parser = _DirectInfinityParser(snapshot)
    return parser, _runtime(), round(time.monotonic() - started, 3)


def _render_pdf(pdf: Path, dpi: int) -> tuple[Any, tuple[int, int], tuple[float, float], float]:
    import pymupdf  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    started = time.monotonic()
    document = pymupdf.open(pdf)
    try:
        if document.page_count != 1:
            raise ValueError(f"benchmark corpus ожидает одностраничный PDF: {pdf.name}")
        page = document[0]
        page_size = (float(page.rect.width), float(page.rect.height))
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        image.load()
    finally:
        document.close()
    return image, image.size, page_size, round(time.monotonic() - started, 3)


def _validate_pages(manifest: Any, corpus_dir: Path, categories: set[str]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pages"), list):
        raise ValueError("corpus manifest должен содержать pages")
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(manifest["pages"]):
        if not isinstance(page, dict):
            raise ValueError(f"manifest.pages[{index}] должен быть объектом")
        if categories and page.get("category") not in categories:
            continue
        filename = page.get("file")
        expected_sha256 = page.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"manifest.pages[{index}].file небезопасен")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
            raise ValueError(f"manifest.pages[{index}].sha256 неверен")
        pdf = corpus_dir / filename
        if pdf.is_symlink() or not pdf.is_file() or _sha256_file(pdf) != expected_sha256:
            raise ValueError(f"PDF отсутствует или SHA256 не совпадает: {filename}")
        pages.append(page)
    if not pages:
        raise ValueError("после фильтрации не осталось страниц")
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--revision", default=_REVISION)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-new-tokens", type=int, default=32_768)
    parser.add_argument("--max-generation-seconds", type=float, default=600.0)
    parser.add_argument("--categories", nargs="+", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not _COMMIT.fullmatch(args.revision):
        parser.error("--revision должен быть полным lowercase commit SHA")
    if args.dpi < 72 or args.dpi > 300:
        parser.error("--dpi должен быть в диапазоне 72..300")
    if args.max_new_tokens < 1 or args.max_new_tokens > 65_536:
        parser.error("--max-new-tokens должен быть в диапазоне 1..65536")
    if args.max_generation_seconds <= 0 or args.max_generation_seconds > 3600:
        parser.error("--max-generation-seconds должен быть в диапазоне (0, 3600]")
    if args.limit < 0:
        parser.error("--limit должен быть >= 0")

    manifest_path = args.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    pages = _validate_pages(manifest, args.corpus_dir, set(args.categories))
    if args.limit:
        pages = pages[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    summary_path = args.output_dir / "_summary.json"
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source": manifest.get("source"),
        "source_revision": manifest.get("source_revision"),
        "model": args.model,
        "revision": args.revision,
        "dpi": args.dpi,
        "max_new_tokens": args.max_new_tokens,
        "max_generation_seconds": args.max_generation_seconds,
        "results": {},
    }

    pending = []
    for page in pages:
        prediction_path = args.output_dir / f"{page['file']}.json"
        if args.resume and _prediction_complete(
            prediction_path,
            source_sha256=page["sha256"],
            revision=args.revision,
        ):
            summary["results"][page["file"]] = {"status": "skipped", "reason": "resume"}
        else:
            pending.append(page)
    if not pending:
        _atomic_write_json(summary_path, summary)
        return

    inference, runtime, load_latency_s = _load_parser(args.model, args.revision)
    summary["runtime"] = runtime
    summary["load_latency_s"] = load_latency_s
    _atomic_write_json(summary_path, summary)

    for page in pending:
        filename = page["file"]
        pdf = args.corpus_dir / filename
        prediction_path = args.output_dir / f"{filename}.json"
        raw_path = raw_dir / f"{filename}.txt"
        print(f"{filename}: infinity_flash", flush=True)
        try:
            image, image_size, page_size, render_latency_s = _render_pdf(pdf, args.dpi)
            started = time.monotonic()
            raw_output = inference.parse(
                image,
                max_new_tokens=args.max_new_tokens,
                max_time_s=args.max_generation_seconds,
            )
            inference_latency_s = round(time.monotonic() - started, 3)
            segments = _layout_to_segments(
                raw_output,
                image_size_px=image_size,
                page_size_pt=page_size,
            )
            raw_path.write_text(raw_output, encoding="utf-8")
            prediction = {
                "schema_version": _SCHEMA_VERSION,
                "source": {"file": filename, "sha256": page["sha256"]},
                "model": {"id": args.model, "revision": args.revision, "runtime": runtime},
                "latency_s": inference_latency_s,
                "segments": segments,
                "render": {
                    "dpi": args.dpi,
                    "latency_s": render_latency_s,
                    "image_size_px": list(image_size),
                    "page_size_pt": list(page_size),
                },
            }
            _atomic_write_json(prediction_path, prediction)
            result = {
                "status": "ok",
                "segments": len(segments),
                "latency_s": inference_latency_s,
                "render_latency_s": render_latency_s,
            }
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
            (raw_dir / f"{filename}.error.txt").write_text(str(exc), encoding="utf-8")
        summary["results"][filename] = result
        _atomic_write_json(summary_path, summary)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if result["status"] == "error" and args.fail_fast:
            raise RuntimeError(f"Infinity prediction failed for {filename}: {result['error']}")


if __name__ == "__main__":
    main()
