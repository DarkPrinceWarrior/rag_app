from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

_BENCHMARK = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "benchmark_complex_parsers.py")
)
_aggregates = _BENCHMARK["_aggregates"]
_benchmark_proxies = _BENCHMARK["_benchmark_proxies"]


def _result(*, tables: int = 0, cells: int = 0, images: int = 0) -> dict[str, Any]:
    return {
        "status": "ok",
        "latency_s": 2.0,
        "source_chars": 100,
        "segments": 3,
        "table_cells": cells,
        "kinds": {
            "heading": 0,
            "paragraph": 2,
            "table": tables,
            "equation": 0,
            "image": images,
        },
        "quality": {"score": 1.0},
    }


def test_table_benchmark_proxy_exposes_missing_structure() -> None:
    page = {"category": "table", "selection": {"cells": 200}}
    result = _result(tables=1, cells=150)

    assert _benchmark_proxies(page, result) == {
        "table_detected": True,
        "expected_table_cells": 200,
        "table_cell_count_ratio": 0.75,
    }


def test_aggregates_count_structural_pages() -> None:
    table = _result(tables=1, cells=150)
    table["benchmark"] = {"table_detected": True}
    chart = _result(images=1)
    chart["benchmark"] = {"visual_region_preserved": True}
    output = {
        "backends": ["parser"],
        "results": {
            "table.pdf": {"category": "table", "parser": table},
            "chart.pdf": {"category": "chart", "parser": chart},
        },
    }

    assert _aggregates(output)["parser"] == {
        "completed_pages": 2,
        "latency_mean_s": 2.0,
        "latency_median_s": 2.0,
        "source_chars": 200,
        "segments": 6,
        "tables": 1,
        "table_cells": 150,
        "images": 1,
        "empty_text_pages": 0,
        "quality_mean": 1.0,
        "table_pages_detected": 1,
        "chart_pages_with_visual": 1,
    }
