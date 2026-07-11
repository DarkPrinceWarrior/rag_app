"""Последовательный A/B парсеров на открытом сложном корпусе ParseBench."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from rag_app.db.models import SegmentKind
from rag_app.pipeline.paddle_vl import paddle_to_segments, run_paddle
from rag_app.pipeline.parse import backfill_text_layer, load_content_list, pdf_info, run_mineru
from rag_app.pipeline.parse_quality import evaluate_parse, quality_metadata
from rag_app.pipeline.segments import SegmentDraft, content_list_to_segments


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
                bool(result["benchmark"].get("visual_region_preserved"))
                for result in chart_results
            ),
        }
    return aggregates


async def _run_backend(
    pdf: Path,
    backend: str,
    output_dir: Path,
) -> dict[str, Any]:
    n_pages, has_text = await asyncio.to_thread(pdf_info, pdf)
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
            drafts, backfilled_pages = await asyncio.to_thread(backfill_text_layer, pdf, drafts)
    elif backend == "paddle_vl":
        await run_paddle(pdf, backend_dir)
        drafts = paddle_to_segments(backend_dir)
        raw_drafts = list(drafts)
        backfilled_pages = []
    else:
        raise ValueError(f"неизвестный backend: {backend}")

    latency_s = round(time.monotonic() - started, 3)
    quality = evaluate_parse(drafts, n_pages=n_pages)
    raw_quality = evaluate_parse(raw_drafts, n_pages=n_pages)
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
    pages = [
        page
        for page in manifest["pages"]
        if not args.categories or page["category"] in args.categories
    ]
    if not pages:
        raise ValueError(f"в manifest нет категорий: {args.categories}")
    output: dict[str, Any] = {
        "corpus_manifest": str(manifest_path),
        "source": manifest["source"],
        "backends": args.backends,
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
        }
        for backend in args.backends:
            print(f"{filename}: {backend}", flush=True)
            try:
                result = await _run_backend(pdf, backend, args.output_dir)
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
            result["benchmark"] = _benchmark_proxies(page, result)
            output["results"][filename][backend] = result
            output["aggregates"] = _aggregates(output)
            summary_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--backends", nargs="+", default=["mineru", "paddle_vl"])
    parser.add_argument("--categories", nargs="+")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
