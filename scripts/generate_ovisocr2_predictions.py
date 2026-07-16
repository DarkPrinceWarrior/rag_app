"""Generate canonical parser predictions from an isolated OvisOCR2 vLLM server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rag_app.db.models import SegmentKind  # type: ignore[import-untyped]
from rag_app.pipeline.segments import parse_table  # type: ignore[import-untyped]

_MODEL = "ATH-MaaS/OvisOCR2"
_REVISION = "65c619d374b55d4152e85150fc1b003700bc1f0c"
_SCHEMA_VERSION = 1
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BBOX_IMAGE = re.compile(
    r'<img\s+src=["\']images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg["\']\s*/?>',
    re.IGNORECASE,
)
_STRUCTURED_BLOCK = re.compile(
    r"(<table\b.*?</table>|<img\s+src=[\"']images/bbox_\d+_\d+_\d+_\d+\.jpg[\"']\s*/?>|"
    r"\\\[.*?\\\]|\$\$.*?\$\$)",
    re.IGNORECASE | re.DOTALL,
)
_PROMPT = (
    "Extract all readable content from the image in natural human reading order and output the result "
    "as a single Markdown document. For charts or images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are '
    "bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: "
    "<table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with open(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1/chat/completions"
    ):
        raise ValueError("OvisOCR2 endpoint должен быть локальным http://127.0.0.1:<port>/v1/chat/completions")
    return value


def clean_truncated_repeats(
    text: str,
    *,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Remove a cyclic repeated tail using the official OvisOCR2 algorithm."""

    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(1, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]
    return text


def _render_pdf(pdf: Path, dpi: int) -> tuple[bytes, tuple[int, int], tuple[float, float], float]:
    import pypdfium2 as pdfium  # type: ignore[import-not-found]

    started = time.monotonic()
    document = pdfium.PdfDocument(pdf)
    try:
        if len(document) != 1:
            raise ValueError(f"benchmark corpus ожидает одностраничный PDF: {pdf.name}")
        page = document[0]
        width_pt, height_pt = page.get_size()
        image = page.render(scale=dpi / 72).to_pil().convert("RGB")
        output = io.BytesIO()
        image.save(output, format="PNG")
    finally:
        document.close()
    return (
        output.getvalue(),
        image.size,
        (float(width_pt), float(height_pt)),
        round(time.monotonic() - started, 3),
    )


def _request_ocr(
    endpoint: str,
    image_png: bytes,
    *,
    served_model: str,
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, dict[str, Any], float]:
    image = base64.b64encode(image_png).decode("ascii")
    payload = {
        "model": served_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OvisOCR2 request failed: {type(exc).__name__}") from exc
    latency_s = round(time.monotonic() - started, 3)
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OvisOCR2 вернул неожиданный OpenAI-ответ") from exc
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OvisOCR2 вернул пустой Markdown")
    metadata = {
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }
    return clean_truncated_repeats(text.strip()), metadata, latency_s


def _text_segments(text: str, page_idx: int) -> list[dict[str, Any]]:
    text = re.sub(r"\A```(?:markdown)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\Z", "", text)
    segments: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        heading = re.fullmatch(r"(#{1,6})\s+(.+)", block, flags=re.DOTALL)
        if heading:
            segments.append(
                {
                    "kind": SegmentKind.heading.value,
                    "source_text": heading.group(2).strip(),
                    "page_idx": page_idx,
                    "heading_level": len(heading.group(1)),
                    "meta": {},
                }
            )
        else:
            segments.append(
                {
                    "kind": SegmentKind.paragraph.value,
                    "source_text": block,
                    "page_idx": page_idx,
                    "heading_level": None,
                    "meta": {},
                }
            )
    return segments


def markdown_to_segments(markdown: str, page_size_pt: tuple[float, float]) -> list[dict[str, Any]]:
    width_pt, height_pt = page_size_pt
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in _STRUCTURED_BLOCK.finditer(markdown):
        segments.extend(_text_segments(markdown[cursor : match.start()], 0))
        block = match.group(0)
        lower = block.lstrip().lower()
        if lower.startswith("<table"):
            cells, rows = parse_table(block)
            if cells:
                preview = "\n".join(" | ".join(cell["text"] for cell in row) for row in cells)
                segments.append(
                    {
                        "kind": SegmentKind.table.value,
                        "source_text": preview,
                        "page_idx": 0,
                        "heading_level": None,
                        "meta": {"table_cells": cells, "table_rows": rows, "caption": ""},
                    }
                )
        elif lower.startswith("<img"):
            bbox = _BBOX_IMAGE.fullmatch(block.strip())
            if bbox:
                left, top, right, bottom = (int(value) for value in bbox.groups())
                if 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000:
                    segments.append(
                        {
                            "kind": SegmentKind.image.value,
                            "source_text": "Визуальная область",
                            "page_idx": 0,
                            "heading_level": None,
                            "meta": {
                                "bbox_pt": [
                                    left * width_pt / 1000,
                                    top * height_pt / 1000,
                                    right * width_pt / 1000,
                                    bottom * height_pt / 1000,
                                ],
                                "page_size_pt": [width_pt, height_pt],
                                "ovis_bbox_1000": [left, top, right, bottom],
                            },
                        }
                    )
        else:
            formula = block.strip()
            if formula.startswith("$$"):
                formula = formula[2:-2].strip()
            elif formula.startswith("\\["):
                formula = formula[2:-2].strip()
            if formula:
                segments.append(
                    {
                        "kind": SegmentKind.equation.value,
                        "source_text": formula,
                        "page_idx": 0,
                        "heading_level": None,
                        "meta": {},
                    }
                )
        cursor = match.end()
    segments.extend(_text_segments(markdown[cursor:], 0))
    if not segments and markdown.strip():
        segments = _text_segments(markdown, 0)
    for index, segment in enumerate(segments):
        segment["idx"] = index
    return segments


def _prediction_complete(path: Path, *, source_sha256: str, revision: str) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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


def _validate_pages(manifest: Any, corpus_dir: Path, categories: set[str]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pages"), list):
        raise ValueError("corpus manifest должен содержать pages")
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(manifest["pages"]):
        if not isinstance(page, dict) or (categories and page.get("category") not in categories):
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
    parser.add_argument("--endpoint", default="http://127.0.0.1:18120/v1/chat/completions")
    parser.add_argument("--served-model", default="ovisocr2")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--revision", default=_REVISION)
    parser.add_argument("--runtime", default="vllm-openai")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--categories", nargs="+", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    try:
        args.endpoint = _validate_endpoint(args.endpoint)
    except ValueError as exc:
        parser.error(str(exc))
    if not _COMMIT.fullmatch(args.revision):
        parser.error("--revision должен быть полным lowercase commit SHA")
    if not 72 <= args.dpi <= 300:
        parser.error("--dpi должен быть в диапазоне 72..300")
    if not 1 <= args.max_tokens <= 16_384:
        parser.error("--max-tokens должен быть в диапазоне 1..16384")
    if not 0 < args.timeout <= 3600:
        parser.error("--timeout должен быть в диапазоне (0, 3600]")
    if args.limit < 0:
        parser.error("--limit должен быть >= 0")

    manifest = json.loads((args.corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    pages = _validate_pages(manifest, args.corpus_dir, set(args.categories))
    if args.limit:
        pages = pages[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source": manifest.get("source"),
        "source_revision": manifest.get("source_revision"),
        "model": args.model,
        "revision": args.revision,
        "runtime": args.runtime,
        "endpoint": args.endpoint,
        "dpi": args.dpi,
        "max_tokens": args.max_tokens,
        "results": {},
    }
    summary_path = args.output_dir / "_summary.json"

    for page in pages:
        filename = page["file"]
        prediction_path = args.output_dir / f"{filename}.json"
        if args.resume and _prediction_complete(
            prediction_path,
            source_sha256=page["sha256"],
            revision=args.revision,
        ):
            summary["results"][filename] = {"status": "skipped", "reason": "resume"}
            continue
        print(f"{filename}: ovisocr2", flush=True)
        try:
            image_png, image_size, page_size, render_latency_s = _render_pdf(
                args.corpus_dir / filename,
                args.dpi,
            )
            markdown, inference, latency_s = _request_ocr(
                args.endpoint,
                image_png,
                served_model=args.served_model,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
            )
            segments = markdown_to_segments(markdown, page_size)
            if not segments:
                raise ValueError("OvisOCR2 Markdown не дал ни одного сегмента")
            raw_path = raw_dir / f"{filename}.md"
            raw_path.write_text(markdown, encoding="utf-8")
            raw_path.chmod(0o600)
            prediction = {
                "schema_version": _SCHEMA_VERSION,
                "source": {"file": filename, "sha256": page["sha256"]},
                "model": {"id": args.model, "revision": args.revision, "runtime": args.runtime},
                "latency_s": latency_s,
                "segments": segments,
                "render": {
                    "dpi": args.dpi,
                    "latency_s": render_latency_s,
                    "image_size_px": list(image_size),
                    "page_size_pt": list(page_size),
                },
                "inference": inference,
            }
            _atomic_write_json(prediction_path, prediction)
            summary["results"][filename] = {
                "status": "ok",
                "latency_s": latency_s,
                "segments": len(segments),
                **inference,
            }
        except Exception as exc:
            summary["results"][filename] = {"status": "error", "error": str(exc)}
            if args.fail_fast:
                _atomic_write_json(summary_path, summary)
                raise
        _atomic_write_json(summary_path, summary)
    _atomic_write_json(summary_path, summary)


if __name__ == "__main__":
    main()
