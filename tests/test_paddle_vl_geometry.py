from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

from rag_app.db.models import Segment, SegmentKind
from rag_app.pipeline.paddle_vl import paddle_to_segments
from rag_app.pipeline.page_fallback import (
    _has_precise_geometry,
    remap_selected_page_drafts,
)
from rag_app.pipeline.scan_pdf import _overlay_groups, _plain_overlay_text


def _segment(
    idx: int,
    translated_text: str,
    *,
    bbox: list[float] | None,
) -> Segment:
    meta = {"bbox_pt": bbox, "page_size_pt": [200.0, 300.0]} if bbox is not None else {}
    return Segment(
        document_id=uuid.uuid4(),
        idx=idx,
        page_idx=0,
        kind=SegmentKind.paragraph,
        source_text=f"source-{idx}",
        translated_text=translated_text,
        meta=meta,
    )


def test_paddle_markdown_fragments_receive_native_block_geometry(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Alpha\n\nBeta\n\nGamma", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Alpha\nBeta",
                        "block_bbox": [100, 200, 500, 400],
                        "group_id": 0,
                    },
                    {
                        "block_label": "text",
                        "block_content": "",
                        "block_bbox": [600, 200, 1100, 400],
                        "group_id": 0,
                    },
                    {
                        "block_label": "text",
                        "block_content": "Gamma",
                        "block_bbox": [100, 500, 500, 600],
                        "group_id": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    drafts = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})

    assert [draft.source_text for draft in drafts] == ["Alpha", "Beta", "Gamma"]
    assert drafts[0].meta["bbox_pt"] == [50.0, 100.0, 550.0, 200.0]
    assert drafts[1].meta["bbox_pt"] == drafts[0].meta["bbox_pt"]
    assert drafts[2].meta["bbox_pt"] == [50.0, 250.0, 250.0, 300.0]
    assert all(draft.meta["page_size_pt"] == [600.0, 800.0] for draft in drafts)
    assert _has_precise_geometry(drafts, page_size=(600.0, 800.0))


def test_invalid_paddle_layout_json_keeps_markdown_parse_available(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Still parsed", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text("{not-json", encoding="utf-8")

    drafts = paddle_to_segments(tmp_path)

    assert len(drafts) == 1
    assert drafts[0].source_text == "Still parsed"
    assert "bbox_pt" not in drafts[0].meta


def test_consecutive_paddle_tables_use_distinct_native_blocks(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text(
        "<table><tr><td>First</td></tr></table>\n\n<table><tr><td>Second</td></tr></table>",
        encoding="utf-8",
    )
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "table",
                        "block_content": "<table><tr><td>First</td></tr></table>",
                        "block_bbox": [100, 200, 500, 400],
                    },
                    {
                        "block_label": "table",
                        "block_content": "<table><tr><td>Second</td></tr></table>",
                        "block_bbox": [100, 500, 500, 700],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    drafts = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})

    assert [draft.kind for draft in drafts] == [SegmentKind.table, SegmentKind.table]
    assert drafts[0].meta["bbox_pt"] == [50.0, 100.0, 250.0, 200.0]
    assert drafts[1].meta["bbox_pt"] == [50.0, 250.0, 250.0, 350.0]


def test_empty_group_box_is_not_shared_by_two_populated_blocks(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Alpha\n\nBeta", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Alpha",
                        "block_bbox": [100, 200, 500, 400],
                        "group_id": 0,
                    },
                    {
                        "block_label": "text",
                        "block_content": "Beta",
                        "block_bbox": [600, 200, 1000, 400],
                        "group_id": 0,
                    },
                    {
                        "block_label": "text",
                        "block_content": "",
                        "block_bbox": [100, 450, 1000, 500],
                        "group_id": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    drafts = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})

    assert drafts[0].meta["bbox_pt"] == [50.0, 100.0, 250.0, 200.0]
    assert drafts[1].meta["bbox_pt"] == [300.0, 100.0, 500.0, 200.0]


def test_repeated_text_consumes_distinct_native_blocks(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Same\n\nSame", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Same",
                        "block_bbox": [100, 200, 500, 400],
                    },
                    {
                        "block_label": "text",
                        "block_content": "Same",
                        "block_bbox": [600, 500, 1000, 700],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    drafts = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})

    assert drafts[0].meta["bbox_pt"] == [50.0, 100.0, 250.0, 200.0]
    assert drafts[1].meta["bbox_pt"] == [300.0, 250.0, 500.0, 350.0]


def test_exact_total_block_wins_over_subtotal_substring(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Total", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Subtotal",
                        "block_bbox": [100, 200, 500, 400],
                    },
                    {
                        "block_label": "text",
                        "block_content": "Total",
                        "block_bbox": [600, 500, 1000, 700],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    draft = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})[0]

    assert draft.meta["bbox_pt"] == [300.0, 250.0, 500.0, 350.0]


def test_total_does_not_match_inside_subtotal(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Total", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Subtotal",
                        "block_bbox": [100, 200, 500, 400],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    draft = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})[0]

    assert "bbox_pt" not in draft.meta


def test_markdown_ignored_native_blocks_are_not_geometry_candidates(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Body", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "model_settings": {"markdown_ignore_labels": ["header"]},
                "parsing_res_list": [
                    {
                        "block_label": "header",
                        "block_content": "Body",
                        "block_bbox": [100, 20, 500, 100],
                    },
                    {
                        "block_label": "text",
                        "block_content": "Body",
                        "block_bbox": [100, 200, 500, 400],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    draft = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})[0]

    assert draft.meta["bbox_pt"] == [50.0, 100.0, 250.0, 200.0]


def test_paddle_geometry_is_clamped_to_page_bounds(tmp_path: Path) -> None:
    (tmp_path / "doc_0.md").write_text("Bounded", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Bounded",
                        "block_bbox": [-20, -10, 1300, 1700],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    draft = paddle_to_segments(tmp_path, {0: (600.0, 800.0)})[0]

    assert draft.meta["bbox_pt"] == [0.0, 0.0, 600.0, 800.0]


def test_paddle_pixels_are_not_persisted_as_pdf_points_without_page_size(
    tmp_path: Path,
) -> None:
    (tmp_path / "doc_0.md").write_text("Diagnostic", encoding="utf-8")
    (tmp_path / "doc_0_res.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "width": 1200,
                "height": 1600,
                "parsing_res_list": [
                    {
                        "block_label": "text",
                        "block_content": "Diagnostic",
                        "block_bbox": [100, 200, 500, 400],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    draft = paddle_to_segments(tmp_path)[0]

    assert "bbox_pt" not in draft.meta
    assert draft.meta["paddle_bbox_px"] == [100.0, 200.0, 500.0, 400.0]
    assert draft.meta["geometry_space"] == "paddle_pixels_noncanonical"


def test_heterogeneous_selected_pages_keep_canonical_geometry_after_remap(
    tmp_path: Path,
) -> None:
    for page_idx, text in enumerate(("Wide", "Narrow")):
        (tmp_path / f"doc_{page_idx}.md").write_text(text, encoding="utf-8")
        (tmp_path / f"doc_{page_idx}_res.json").write_text(
            json.dumps(
                {
                    "page_index": page_idx,
                    "width": 1200,
                    "height": 1600,
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": text,
                            "block_bbox": [100, 200, 500, 400],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    reduced = paddle_to_segments(
        tmp_path,
        {
            0: (600.0, 800.0),
            1: (300.0, 400.0),
        },
    )
    remapped = remap_selected_page_drafts(reduced, (2, 5))

    wide = [draft for draft in remapped if draft.page_idx == 2]
    narrow = [draft for draft in remapped if draft.page_idx == 5]
    assert wide[0].meta["bbox_pt"] == [50.0, 100.0, 250.0, 200.0]
    assert narrow[0].meta["bbox_pt"] == [25.0, 50.0, 125.0, 100.0]
    assert _has_precise_geometry(wide, page_size=(600.0, 800.0))
    assert _has_precise_geometry(narrow, page_size=(300.0, 400.0))


def test_paddle_runner_disables_orientation_and_unwarping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class _Result:
        def save_to_json(self, *, save_path: str) -> None:
            calls["json_path"] = save_path

        def save_to_markdown(self, *, save_path: str) -> None:
            calls["markdown_path"] = save_path

    class _PaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            calls["init"] = kwargs

        def predict(self, input_path: str, **kwargs):
            calls["input"] = input_path
            calls["predict"] = kwargs
            return [_Result()]

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(PaddleOCRVL=_PaddleOCRVL),
    )
    monkeypatch.setenv("PADDLE_VL_SERVER_URL", "http://paddle.invalid/v1")
    monkeypatch.setattr(sys, "argv", ["run_paddle_cli.py", "source.pdf", str(tmp_path)])
    runner_path = Path(__file__).parents[1] / "deploy" / "parsers" / "run_paddle_cli.py"
    spec = importlib.util.spec_from_file_location("run_paddle_cli_candidate", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert runner.main() == 0
    assert calls["predict"] == {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }


def test_scan_overlay_groups_fragments_sharing_one_layout_block() -> None:
    first = _segment(0, "Первая строка", bbox=[10.0, 20.0, 100.0, 80.0])
    second = _segment(1, "Вторая строка", bbox=[10.0, 20.0, 100.0, 80.0])
    other = _segment(2, "Другой блок", bbox=[10.0, 100.0, 100.0, 160.0])
    missing = _segment(3, "Без геометрии", bbox=None)

    pages, skipped = _overlay_groups([second, missing, other, first])

    assert skipped == 1
    assert [[segment.idx for segment in group] for group in pages[0]] == [[0, 1], [2]]


def test_scan_overlay_removes_paddle_latex_wrappers() -> None:
    assert (
        _plain_overlay_text(
            "Акт № $ \\underline{\\text{CKCEC-A-0003}} $ от 《___》\n"
            r"\underline{28.02.2023}"
        )
        == "Акт № CKCEC-A-0003 от «___»\n28.02.2023"
    )
    assert _plain_overlay_text("Стоимость: $100") == "Стоимость: $100"
