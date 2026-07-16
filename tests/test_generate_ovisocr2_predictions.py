from __future__ import annotations

import runpy
from pathlib import Path

_RUNNER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "generate_ovisocr2_predictions.py")
)
_prediction_complete = _RUNNER["_prediction_complete"]
_validate_endpoint = _RUNNER["_validate_endpoint"]
clean_truncated_repeats = _RUNNER["clean_truncated_repeats"]
markdown_to_segments = _RUNNER["markdown_to_segments"]


def test_validate_endpoint_restricts_server_to_loopback() -> None:
    endpoint = "http://127.0.0.1:18120/v1/chat/completions"
    assert _validate_endpoint(endpoint) == endpoint

    for unsafe in (
        "https://127.0.0.1:18120/v1/chat/completions",
        "http://192.168.1.2:18120/v1/chat/completions",
        "http://localhost:18120/other",
    ):
        try:
            _validate_endpoint(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe endpoint accepted: {unsafe}")


def test_markdown_to_segments_preserves_structures_and_bbox() -> None:
    markdown = """# Title

Intro text.

<table><tr><th rowspan="2">A</th><th>B</th></tr><tr><td>42</td></tr></table>

\\[x^2 + y^2\\]

<img src="images/bbox_100_200_600_800.jpg" />
"""
    segments = markdown_to_segments(markdown, (600.0, 800.0))

    assert [segment["idx"] for segment in segments] == list(range(len(segments)))
    assert [segment["kind"] for segment in segments] == [
        "heading",
        "paragraph",
        "table",
        "equation",
        "image",
    ]
    assert segments[0]["heading_level"] == 1
    assert segments[2]["meta"]["table_cells"][0][0] == {
        "text": "A",
        "colspan": 1,
        "rowspan": 2,
    }
    assert segments[4]["meta"]["bbox_pt"] == [60.0, 160.0, 360.0, 640.0]


def test_clean_truncated_repeats_removes_only_long_cyclic_tail() -> None:
    prefix = "A" * 8000
    unit = "broken-tail-"
    repeated = prefix + unit * 12
    cleaned = clean_truncated_repeats(repeated)
    assert len(cleaned) < len(repeated)
    assert cleaned.startswith(prefix)
    assert clean_truncated_repeats("short " + unit * 12) == "short " + unit * 12


def test_prediction_complete_binds_revision_and_source(tmp_path) -> None:
    source_hash = "a" * 64
    revision = "b" * 40
    path = tmp_path / "page.pdf.json"
    path.write_text(
        '{"schema_version":1,"source":{"file":"page.pdf","sha256":"'
        + source_hash
        + '"},"model":{"revision":"'
        + revision
        + '"},"segments":[{}]}',
        encoding="utf-8",
    )
    assert _prediction_complete(path, source_sha256=source_hash, revision=revision)
    assert not _prediction_complete(path, source_sha256="c" * 64, revision=revision)
