from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "requalify_parser_benchmark.py"
    spec = importlib.util.spec_from_file_location("requalify_parser_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requalify = _load_script()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    pdf_root = tmp_path / "pdf"
    output_root = tmp_path / "output"
    pdf_root.mkdir()
    (output_root / "mineru" / "sample").mkdir(parents=True)
    pdf = pdf_root / "sample.pdf"
    pdf.write_bytes(b"immutable pdf fixture")
    source_sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
    content = [{"type": "text", "text": "private fixture text", "page_idx": 0, "bbox": [0, 0, 10, 10]}]
    (output_root / "mineru" / "sample" / "sample_content_list.json").write_text(
        json.dumps(content),
        encoding="utf-8",
    )
    summary = {
        "benchmark_schema_version": 1,
        "source": "fixture",
        "backends": ["mineru"],
        "runtime_provenance": {"mineru": "3.3.1"},
        "results": {
            "sample.pdf": {
                "category": "text",
                "selection": {},
                "source_sha256": source_sha256,
                "mineru": {"status": "ok", "n_pages": 1, "latency_s": 4.25},
            }
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, output_root, pdf_root, tmp_path / "migrated.json", source_sha256


def test_requalifies_saved_output_without_text_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, output, pdf_root, destination, _ = _fixture(tmp_path)
    monkeypatch.setattr(requalify, "pdf_info", lambda _: (1, False))

    migrated = requalify.requalify_summary(summary, output, pdf_root, destination)

    result = migrated["results"]["sample.pdf"]["mineru"]
    assert migrated["benchmark_schema_version"] == 2
    assert migrated["runtime_provenance"] == {"mineru": "3.3.1"}
    assert result["latency_s"] == 4.25
    assert result["raw_stats"]["source_chars"] == 20
    assert result["source_chars"] == 20
    assert result["bbox_segments"] == 1
    assert len(result["content_list_sha256"]) == 64
    assert migrated["aggregates"]["mineru"]["completed_pages"] == 1
    assert "private fixture text" not in destination.read_text(encoding="utf-8")


def test_rejects_non_exact_outputs_and_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, output, pdf_root, destination, _ = _fixture(tmp_path)
    monkeypatch.setattr(requalify, "pdf_info", lambda _: (1, False))
    (output / "mineru" / "extra").mkdir()
    with pytest.raises(requalify.RequalificationError, match="exact benchmark"):
        requalify.requalify_summary(summary, output, pdf_root, destination)

    (output / "mineru" / "extra").rmdir()
    destination.write_text("do not replace", encoding="utf-8")
    with pytest.raises(requalify.RequalificationError, match="destination already exists"):
        requalify.requalify_summary(summary, output, pdf_root, destination)
    assert destination.read_text(encoding="utf-8") == "do not replace"
