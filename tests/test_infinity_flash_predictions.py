from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_RUNNER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "generate_infinity_flash_predictions.py")
)
_layout_to_segments = _RUNNER["_layout_to_segments"]
_parse_layout = _RUNNER["_parse_layout"]
_prediction_complete = _RUNNER["_prediction_complete"]


def test_layout_to_segments_scales_bbox_and_preserves_structures() -> None:
    raw = json.dumps(
        [
            {"bbox": [100, 200, 500, 300], "category": "title", "text": "Specification"},
            {
                "bbox": [100, 400, 1000, 800],
                "category": "table",
                "text": "<table><tr><th>A</th><th>B</th></tr><tr><td colspan='2'>16.5 MPa</td></tr></table>",
            },
            {"bbox": [200, 850, 800, 950], "category": "figure", "text": ""},
        ]
    )

    segments = _layout_to_segments(
        raw,
        image_size_px=(1200, 1600),
        page_size_pt=(600.0, 800.0),
    )

    assert [segment["kind"] for segment in segments] == ["heading", "table", "image"]
    assert segments[0]["meta"]["bbox_pt"] == [60.0, 160.0, 300.0, 240.0]
    assert segments[1]["meta"]["table_cells"] == [
        [
            {"text": "A", "colspan": 1, "rowspan": 1},
            {"text": "B", "colspan": 1, "rowspan": 1},
        ],
        [{"text": "16.5 MPa", "colspan": 2, "rowspan": 1}],
    ]
    assert segments[2]["source_text"] == ""
    assert [segment["idx"] for segment in segments] == [0, 1, 2]


def test_parse_layout_accepts_fenced_object_container() -> None:
    raw = (
        "```json\n"
        + json.dumps({"elements": [{"bbox": [1, 2, 3, 4], "category": "text", "text": "x"}]})
        + "\n```"
    )

    assert _parse_layout(raw)[0]["text"] == "x"


@pytest.mark.parametrize(
    ("element", "message"),
    [
        ({"bbox": [0, 0, 10, 10], "category": "unknown", "text": "x"}, "category"),
        ({"bbox": [0, 0, 1300, 10], "category": "text", "text": "x"}, "вне 0..1000"),
        ({"bbox": [0, 0, 10, 10], "category": "text", "text": None}, "text"),
    ],
)
def test_layout_to_segments_rejects_invalid_model_output(element: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _layout_to_segments(
            json.dumps([element]),
            image_size_px=(1200, 1600),
            page_size_pt=(600.0, 800.0),
        )


def test_prediction_complete_requires_matching_revision_and_source(tmp_path: Path) -> None:
    prediction = tmp_path / "hard.pdf.json"
    prediction.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"file": "hard.pdf", "sha256": "a" * 64},
                "model": {"revision": "b" * 40},
                "segments": [{"idx": 0}],
            }
        ),
        encoding="utf-8",
    )

    assert _prediction_complete(prediction, source_sha256="a" * 64, revision="b" * 40)
    assert not _prediction_complete(prediction, source_sha256="c" * 64, revision="b" * 40)


def test_malformed_table_is_preserved_as_visible_paragraph_failure() -> None:
    segments = _layout_to_segments(
        json.dumps([{"bbox": [0, 0, 1000, 1000], "category": "table", "text": "not html"}]),
        image_size_px=(1200, 1600),
        page_size_pt=(600.0, 800.0),
    )

    assert segments[0]["kind"] == "paragraph"
    assert "HTML-ячеек" in segments[0]["meta"]["table_parse_error"]
