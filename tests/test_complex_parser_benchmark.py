from __future__ import annotations

import argparse
import asyncio
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

_BENCHMARK = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "benchmark_complex_parsers.py"))
_aggregates = _BENCHMARK["_aggregates"]
_atomic_write_json = _BENCHMARK["_atomic_write_json"]
_benchmark_proxies = _BENCHMARK["_benchmark_proxies"]
_main = _BENCHMARK["_main"]
_parse_prediction_spec = _BENCHMARK["_parse_prediction_spec"]
_prediction_to_drafts = _BENCHMARK["_prediction_to_drafts"]


def _prediction() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": {"file": "hard.pdf", "sha256": "a" * 64},
        "model": {
            "id": "infly/Infinity-Parser2-Flash",
            "revision": "9837b837",
            "runtime": "transformers-5.4.0",
        },
        "latency_s": 10.8,
        "segments": [
            {
                "idx": 0,
                "kind": "table",
                "source_text": "EPC value 16.5 MPa",
                "page_idx": 0,
                "heading_level": None,
                "meta": {
                    "bbox_pt": [10.0, 20.0, 300.0, 400.0],
                    "page_size_pt": [595.0, 842.0],
                    "table_cells": [[{"text": "16.5 MPa", "colspan": 1, "rowspan": 1}]],
                },
            }
        ],
    }


def _result(*, tables: int = 0, cells: int = 0, images: int = 0) -> dict[str, Any]:
    return {
        "status": "ok",
        "latency_s": 2.0,
        "source_chars": 100,
        "segments": 3,
        "table_cells": cells,
        "table_rows": 10 if tables else 0,
        "table_nonempty_cells": 120 if tables else 0,
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
    page = {"category": "table", "selection": {"cells": 200, "rows": 8}}
    result = _result(tables=1, cells=150)

    assert _benchmark_proxies(page, result) == {
        "table_detected": True,
        "expected_table_cells": 200,
        "expected_table_rows": 8,
        "table_cell_count_ratio": 0.75,
        "table_nonempty_cell_count_ratio": 0.6,
        "table_row_count_ratio": 1.25,
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
        "table_rows": 10,
        "table_nonempty_cells": 120,
        "images": 1,
        "empty_text_pages": 0,
        "quality_mean": 1.0,
        "table_pages_detected": 1,
        "chart_pages_with_visual": 1,
    }


def test_external_prediction_is_converted_with_provenance_and_geometry() -> None:
    drafts, provenance, latency = _prediction_to_drafts(
        _prediction(),
        source_filename="hard.pdf",
        source_sha256="a" * 64,
        n_pages=1,
    )

    assert latency == 10.8
    assert provenance == {
        "id": "infly/Infinity-Parser2-Flash",
        "revision": "9837b837",
        "runtime": "transformers-5.4.0",
    }
    assert len(drafts) == 1
    assert drafts[0].idx == 0
    assert drafts[0].kind.value == "table"
    assert drafts[0].meta["bbox_pt"] == [10.0, 20.0, 300.0, 400.0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["source"].update(sha256="b" * 64), "source.sha256"),
        (lambda payload: payload["segments"][0].update(idx=1), "reading order"),
        (
            lambda payload: payload["segments"][0]["meta"].update(bbox_pt=[10.0, 20.0, 600.0, 400.0]),
            "bbox вне страницы",
        ),
        (lambda payload: payload["segments"][0].update(page_idx=1), "страница вне диапазона"),
        (
            lambda payload: payload["segments"][0]["meta"].update(table_cells=[[{"text": "x"}]]),
            "colspan",
        ),
    ],
)
def test_external_prediction_rejects_invalid_contract(mutate, message: str) -> None:
    payload = _prediction()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        _prediction_to_drafts(
            payload,
            source_filename="hard.pdf",
            source_sha256="a" * 64,
            n_pages=1,
        )


def test_prediction_spec_and_atomic_summary(tmp_path: Path) -> None:
    name, directory = _parse_prediction_spec(f"infinity_flash={tmp_path}")
    assert name == "infinity_flash"
    assert directory == tmp_path.resolve()

    destination = tmp_path / "summary.json"
    _atomic_write_json(destination, {"run": 1})
    _atomic_write_json(destination, {"run": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"run": 2}
    assert list(tmp_path.glob(".summary.json.*")) == []


def test_main_rejects_corpus_sha_before_parser_start(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "hard.pdf").write_bytes(b"not-a-pdf")
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "source": "test",
                "pages": [
                    {
                        "file": "hard.pdf",
                        "sha256": "0" * 64,
                        "category": "table",
                        "selection": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        corpus_dir=corpus,
        output_dir=tmp_path / "out",
        backends=["mineru"],
        predictions={},
        categories=None,
    )

    with pytest.raises(ValueError, match="SHA256 не совпадает"):
        asyncio.run(_main(args))
    assert not args.output_dir.exists()


def test_main_rejects_run_without_any_backend(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(
        json.dumps({"source": "test", "pages": [{"category": "table"}]}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        corpus_dir=corpus,
        output_dir=tmp_path / "out",
        backends=[],
        predictions={},
        categories=None,
    )

    with pytest.raises(ValueError, match="хотя бы один"):
        asyncio.run(_main(args))
