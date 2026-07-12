from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_lightonocr2.py"
    spec = importlib.util.spec_from_file_location("benchmark_lightonocr2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_signals_detects_tables_boxes_and_repetition() -> None:
    module = _module()
    signals = module._quality_signals(
        "| A | B |\n|---|---|\n| 1 | 2 |\nimage<10,20,300,400>\n$$x=1$$"
    )

    assert signals["markdown_table_lines"] == 3
    assert signals["image_bbox_count"] == 1
    assert signals["display_math_count"] == 1
    assert signals["acceptable_smoke"] is True

    repeated = module._quality_signals("!" * 8192)
    assert repeated["acceptable_smoke"] is False


def test_quality_signals_ignores_invalid_boxes() -> None:
    module = _module()
    signals = module._quality_signals("image<500,500,100,100>")
    assert signals["image_bbox_count"] == 0


def test_quality_signals_accepts_bbox_variant_output_format() -> None:
    module = _module()
    signals = module._quality_signals(
        "![image](image_1.png)80,100,914,430\n"
        "![image](image_2.png)80,520,914,820"
    )

    assert signals["image_bbox_count"] == 2
