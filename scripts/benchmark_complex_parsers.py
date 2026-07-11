"""Последовательный A/B парсеров на открытом сложном корпусе ParseBench."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from rag_app.db.models import SegmentKind
from rag_app.pipeline.paddle_vl import paddle_to_segments, run_paddle
from rag_app.pipeline.parse import (
    backfill_text_layer,
    load_content_list,
    pdf_info,
    read_pdf_text_by_page,
    run_mineru,
)
from rag_app.pipeline.parse_quality import evaluate_parse, quality_metadata
from rag_app.pipeline.segments import SegmentDraft, content_list_to_segments

_PREDICTION_SCHEMA_VERSION = 1
_MAX_PREDICTION_BYTES = 64 * 1024 * 1024
_MAX_SEGMENTS = 100_000
_MAX_SOURCE_TEXT_CHARS = 2_000_000
_MAX_META_BYTES = 8 * 1024 * 1024
_MAX_TABLE_CELLS = 200_000
_MAX_CELL_TEXT_CHARS = 64_000
_BACKEND_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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


def _parse_prediction_spec(spec: str) -> tuple[str, Path]:
    name, separator, directory = spec.partition("=")
    if not separator or not _BACKEND_NAME.fullmatch(name) or not directory:
        raise ValueError(
            "prediction должен иметь вид backend=/path/to/results; backend: [a-z][a-z0-9_-]{0,63}"
        )
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"каталог prediction не найден: {path}")
    return name, path


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field}: ожидалось конечное число")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: ожидалось конечное число")
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"prediction: недопустимая JSON-константа {value}")


def _validate_table_cells(value: Any, *, segment_idx: int) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"segments[{segment_idx}].meta.table_cells: нужен непустой массив строк")
    total = 0
    for row_idx, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"segments[{segment_idx}].meta.table_cells[{row_idx}]: нужна непустая строка")
        total += len(row)
        if total > _MAX_TABLE_CELLS:
            raise ValueError(f"segments[{segment_idx}].meta.table_cells: слишком много ячеек")
        for cell_idx, cell in enumerate(row):
            field = f"segments[{segment_idx}].meta.table_cells[{row_idx}][{cell_idx}]"
            if not isinstance(cell, dict):
                raise ValueError(f"{field}: ожидался JSON-объект")
            text = cell.get("text")
            if not isinstance(text, str) or len(text) > _MAX_CELL_TEXT_CHARS:
                raise ValueError(f"{field}.text: нужна ограниченная строка")
            for span_name in ("colspan", "rowspan"):
                span = cell.get(span_name)
                if isinstance(span, bool) or not isinstance(span, int) or span < 1 or span > 1000:
                    raise ValueError(f"{field}.{span_name}: нужно целое число 1..1000")


def _validate_meta(meta: Any, *, segment_idx: int, kind: SegmentKind) -> dict[str, Any]:
    if not isinstance(meta, dict) or not all(isinstance(key, str) for key in meta):
        raise ValueError(f"segments[{segment_idx}].meta: ожидался JSON-объект")
    encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_META_BYTES:
        raise ValueError(f"segments[{segment_idx}].meta: превышен лимит {_MAX_META_BYTES} байт")

    bbox = meta.get("bbox_pt")
    page_size = meta.get("page_size_pt")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"segments[{segment_idx}].meta.bbox_pt: нужны четыре координаты")
        coords = [_number(item, field=f"segments[{segment_idx}].meta.bbox_pt") for item in bbox]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise ValueError(f"segments[{segment_idx}].meta.bbox_pt: вырожденный bbox")
        if not isinstance(page_size, list) or len(page_size) != 2:
            raise ValueError(f"segments[{segment_idx}].meta.page_size_pt обязателен вместе с bbox_pt")
        width, height = [
            _number(item, field=f"segments[{segment_idx}].meta.page_size_pt") for item in page_size
        ]
        if width <= 0 or height <= 0:
            raise ValueError(f"segments[{segment_idx}].meta.page_size_pt: размер должен быть > 0")
        if coords[0] < 0 or coords[1] < 0 or coords[2] > width or coords[3] > height:
            raise ValueError(f"segments[{segment_idx}].meta.bbox_pt: bbox вне страницы")
    elif page_size is not None:
        raise ValueError(f"segments[{segment_idx}].meta.page_size_pt задан без bbox_pt")

    table_cells = meta.get("table_cells")
    if kind is SegmentKind.table:
        _validate_table_cells(table_cells, segment_idx=segment_idx)
    elif table_cells is not None:
        raise ValueError(f"segments[{segment_idx}].meta.table_cells допустим только для table")
    if kind is SegmentKind.image and bbox is None:
        raise ValueError(f"segments[{segment_idx}].meta.bbox_pt обязателен для image")
    return meta


def _prediction_to_drafts(
    payload: Any,
    *,
    source_filename: str,
    source_sha256: str,
    n_pages: int,
) -> tuple[list[SegmentDraft], dict[str, str], float]:
    if not isinstance(payload, dict):
        raise ValueError("prediction: ожидался JSON-объект")
    if payload.get("schema_version") != _PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"prediction.schema_version должен быть {_PREDICTION_SCHEMA_VERSION}")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("prediction.source: ожидался JSON-объект")
    if source.get("file") != source_filename:
        raise ValueError("prediction.source.file не совпадает с corpus manifest")
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise ValueError("проверенный source_sha256 имеет неверный формат")
    if source.get("sha256") != source_sha256:
        raise ValueError("prediction.source.sha256 не совпадает с проверенным PDF")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("prediction.model: ожидался JSON-объект")
    provenance: dict[str, str] = {}
    for field in ("id", "revision", "runtime"):
        value = model.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(f"prediction.model.{field}: нужна непустая строка до 512 символов")
        provenance[field] = value

    latency_s = _number(payload.get("latency_s"), field="prediction.latency_s")
    if latency_s < 0:
        raise ValueError("prediction.latency_s должен быть >= 0")

    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("prediction.segments: нужен непустой массив")
    if len(segments) > _MAX_SEGMENTS:
        raise ValueError(f"prediction.segments: превышен лимит {_MAX_SEGMENTS}")

    drafts: list[SegmentDraft] = []
    for position, raw in enumerate(segments):
        if not isinstance(raw, dict):
            raise ValueError(f"segments[{position}]: ожидался JSON-объект")
        idx = raw.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx != position:
            raise ValueError(f"segments[{position}].idx: reading order должен быть 0..N-1")
        raw_kind = raw.get("kind")
        if not isinstance(raw_kind, str):
            raise ValueError(f"segments[{position}].kind: неизвестный тип")
        try:
            kind = SegmentKind(raw_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"segments[{position}].kind: неизвестный тип") from exc
        source_text = raw.get("source_text")
        if not isinstance(source_text, str) or len(source_text) > _MAX_SOURCE_TEXT_CHARS:
            raise ValueError(
                f"segments[{position}].source_text: нужна строка до {_MAX_SOURCE_TEXT_CHARS} символов"
            )
        page_idx = raw.get("page_idx")
        if isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0 or page_idx >= n_pages:
            raise ValueError(f"segments[{position}].page_idx: страница вне диапазона")
        heading_level = raw.get("heading_level")
        if heading_level is not None and (
            isinstance(heading_level, bool)
            or not isinstance(heading_level, int)
            or heading_level < 1
            or heading_level > 6
        ):
            raise ValueError(f"segments[{position}].heading_level: нужен уровень 1..6 или null")
        drafts.append(
            SegmentDraft(
                idx=idx,
                kind=kind,
                source_text=source_text,
                page_idx=page_idx,
                heading_level=heading_level,
                meta=_validate_meta(raw.get("meta", {}), segment_idx=position, kind=kind),
            )
        )
    return drafts, provenance, latency_s


async def _run_prediction(
    pdf: Path,
    backend: str,
    prediction_dir: Path,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    prediction_path = prediction_dir / f"{pdf.name}.json"
    if prediction_path.is_symlink() or not prediction_path.is_file():
        raise ValueError(f"prediction-файл не найден или небезопасен: {prediction_path}")
    if prediction_path.stat().st_size > _MAX_PREDICTION_BYTES:
        raise ValueError(f"prediction-файл превышает {_MAX_PREDICTION_BYTES} байт")
    payload = json.loads(
        prediction_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    n_pages, has_text = await asyncio.to_thread(pdf_info, pdf)
    native_text_by_page = await asyncio.to_thread(read_pdf_text_by_page, pdf) if has_text else None
    drafts, provenance, latency_s = _prediction_to_drafts(
        payload,
        source_filename=pdf.name,
        source_sha256=source_sha256,
        n_pages=n_pages,
    )
    quality = evaluate_parse(
        drafts,
        n_pages=n_pages,
        native_text_by_page=native_text_by_page,
    )
    return {
        "status": "ok",
        "latency_s": round(latency_s, 3),
        "n_pages": n_pages,
        "has_text_layer": has_text,
        "quality": quality_metadata(
            quality,
            backend=backend,
            raw_report=quality,
            backfilled_pages=[],
        ),
        "provenance": {
            **provenance,
            "schema_version": _PREDICTION_SCHEMA_VERSION,
            "prediction_file": prediction_path.name,
            "source_sha256": source_sha256,
        },
        **_stats(drafts),
    }


def _stats(drafts: list[SegmentDraft]) -> dict[str, Any]:
    kinds = {kind.value: 0 for kind in SegmentKind}
    table_cells = 0
    for draft in drafts:
        kinds[draft.kind.value] += 1
        cells = draft.meta.get("table_cells")
        if isinstance(cells, list):
            table_cells += sum(len(row) for row in cells if isinstance(row, list))
    return {
        "segments": len(drafts),
        "source_chars": sum(len(draft.source_text) for draft in drafts),
        "kinds": kinds,
        "table_cells": table_cells,
    }


def _benchmark_proxies(page: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Добавить проверяемые структурные сигналы, которых нет в quality score."""
    if result.get("status") != "ok":
        return {}

    category = page["category"]
    kinds = result["kinds"]
    if category == "table":
        expected_cells = page["selection"].get("cells")
        actual_cells = result["table_cells"]
        output: dict[str, Any] = {
            "table_detected": kinds[SegmentKind.table.value] > 0,
            "expected_table_cells": expected_cells,
        }
        if isinstance(expected_cells, int) and expected_cells > 0:
            output["table_cell_count_ratio"] = round(actual_cells / expected_cells, 4)
        return output
    if category == "chart":
        return {"visual_region_preserved": kinds[SegmentKind.image.value] > 0}
    return {}


def _aggregates(output: dict[str, Any]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    pages = list(output["results"].values())
    for backend in output["backends"]:
        completed = [page[backend] for page in pages if page.get(backend, {}).get("status") == "ok"]
        if not completed:
            continue
        latencies = [result["latency_s"] for result in completed]
        table_results = [
            page[backend]
            for page in pages
            if page["category"] == "table" and page.get(backend, {}).get("status") == "ok"
        ]
        chart_results = [
            page[backend]
            for page in pages
            if page["category"] == "chart" and page.get(backend, {}).get("status") == "ok"
        ]
        aggregates[backend] = {
            "completed_pages": len(completed),
            "latency_mean_s": round(statistics.mean(latencies), 3),
            "latency_median_s": round(statistics.median(latencies), 3),
            "source_chars": sum(result["source_chars"] for result in completed),
            "segments": sum(result["segments"] for result in completed),
            "tables": sum(result["kinds"][SegmentKind.table.value] for result in completed),
            "table_cells": sum(result["table_cells"] for result in completed),
            "images": sum(result["kinds"][SegmentKind.image.value] for result in completed),
            "empty_text_pages": sum(result["source_chars"] == 0 for result in completed),
            "quality_mean": round(
                statistics.mean(result["quality"]["score"] for result in completed),
                4,
            ),
            "table_pages_detected": sum(
                bool(result["benchmark"].get("table_detected")) for result in table_results
            ),
            "chart_pages_with_visual": sum(
                bool(result["benchmark"].get("visual_region_preserved")) for result in chart_results
            ),
        }
    return aggregates


async def _run_backend(
    pdf: Path,
    backend: str,
    output_dir: Path,
) -> dict[str, Any]:
    n_pages, has_text = await asyncio.to_thread(pdf_info, pdf)
    native_text_by_page = await asyncio.to_thread(read_pdf_text_by_page, pdf) if has_text else None
    backend_dir = output_dir / backend / pdf.stem
    shutil.rmtree(backend_dir, ignore_errors=True)
    backend_dir.mkdir(parents=True)
    started = time.monotonic()

    if backend == "mineru":
        content_list = await run_mineru(pdf, backend_dir)
        drafts = content_list_to_segments(load_content_list(content_list))
        raw_drafts = list(drafts)
        backfilled_pages: list[int] = []
        if has_text:
            drafts, backfilled_pages = await asyncio.to_thread(
                backfill_text_layer,
                pdf,
                drafts,
                native_text_by_page=native_text_by_page,
            )
    elif backend == "paddle_vl":
        await run_paddle(pdf, backend_dir)
        drafts = paddle_to_segments(backend_dir)
        raw_drafts = list(drafts)
        backfilled_pages = []
    else:
        raise ValueError(f"неизвестный backend: {backend}")

    latency_s = round(time.monotonic() - started, 3)
    quality = evaluate_parse(
        drafts,
        n_pages=n_pages,
        native_text_by_page=native_text_by_page,
    )
    raw_quality = evaluate_parse(
        raw_drafts,
        n_pages=n_pages,
        native_text_by_page=native_text_by_page,
    )
    return {
        "status": "ok",
        "latency_s": latency_s,
        "n_pages": n_pages,
        "has_text_layer": has_text,
        "quality": quality_metadata(
            quality,
            backend=backend,
            raw_report=raw_quality,
            backfilled_pages=backfilled_pages,
        ),
        **_stats(drafts),
    }


async def _main(args: argparse.Namespace) -> None:
    manifest_path = args.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = [page for page in manifest["pages"] if not args.categories or page["category"] in args.categories]
    if not pages:
        raise ValueError(f"в manifest нет категорий: {args.categories}")
    prediction_dirs: dict[str, Path] = args.predictions
    all_backends = [*args.backends, *prediction_dirs]
    if not all_backends:
        raise ValueError("нужен хотя бы один встроенный backend или external prediction")
    if len(all_backends) != len(set(all_backends)):
        raise ValueError("имена встроенных backend и external prediction должны быть уникальны")

    verified_sha256: dict[str, str] = {}
    for page in pages:
        filename = page["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"небезопасное имя PDF в manifest: {filename!r}")
        pdf = args.corpus_dir / filename
        if pdf.is_symlink() or not pdf.is_file():
            raise ValueError(f"PDF из manifest не найден или небезопасен: {pdf}")
        expected_sha256 = page.get("sha256")
        actual_sha256 = await asyncio.to_thread(_sha256_file, pdf)
        if (
            not isinstance(expected_sha256, str)
            or not _SHA256.fullmatch(expected_sha256)
            or actual_sha256 != expected_sha256
        ):
            raise ValueError(f"SHA256 не совпадает для {filename}")
        verified_sha256[filename] = actual_sha256

    output: dict[str, Any] = {
        "corpus_manifest": str(manifest_path),
        "source": manifest["source"],
        "source_revision": manifest.get("source_revision"),
        "backends": all_backends,
        "external_predictions": {backend: str(directory) for backend, directory in prediction_dirs.items()},
        "categories": args.categories,
        "results": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"

    for page in pages:
        filename = page["file"]
        pdf = args.corpus_dir / filename
        output["results"][filename] = {
            "category": page["category"],
            "selection": page["selection"],
            "source_sha256": verified_sha256[filename],
        }
        for backend in all_backends:
            print(f"{filename}: {backend}", flush=True)
            result: dict[str, Any]
            try:
                if backend in prediction_dirs:
                    result = await _run_prediction(
                        pdf,
                        backend,
                        prediction_dirs[backend],
                        source_sha256=verified_sha256[filename],
                    )
                else:
                    result = await _run_backend(pdf, backend, args.output_dir)
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
            result["benchmark"] = _benchmark_proxies(page, result)
            output["results"][filename][backend] = result
            output["aggregates"] = _aggregates(output)
            _atomic_write_json(summary_path, output)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--backends",
        nargs="*",
        default=["mineru", "paddle_vl"],
        help="встроенные backend; передайте пустой список перед --prediction для external-only",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="BACKEND=DIR",
        help="канонические внешние prediction-файлы <source.pdf>.json",
    )
    parser.add_argument("--categories", nargs="+")
    args = parser.parse_args()
    predictions = dict(_parse_prediction_spec(spec) for spec in args.prediction)
    if len(predictions) != len(args.prediction):
        parser.error("имена --prediction должны быть уникальны")
    args.predictions = predictions
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
