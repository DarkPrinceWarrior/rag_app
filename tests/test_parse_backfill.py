from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.db.models import SegmentKind
from rag_app.pipeline.parse import backfill_text_layer
from rag_app.pipeline.segments import SegmentDraft


def test_backfill_uses_supplied_native_text_without_opening_pdf() -> None:
    native = "This native text layer contains the authoritative paragraph. " * 5

    drafts, filled = backfill_text_layer(
        Path("does-not-exist.pdf"),
        [],
        native_text_by_page={0: native},
    )

    assert filled == [0]
    assert len(drafts) == 1
    assert drafts[0].idx == 0
    assert drafts[0].page_idx == 0
    assert drafts[0].kind == SegmentKind.paragraph
    assert drafts[0].meta == {"backfill": True}
    assert "authoritative paragraph" in drafts[0].source_text


def test_backfill_rejects_sparse_page_mapping() -> None:
    with pytest.raises(ValueError, match="dense zero-based"):
        backfill_text_layer(
            Path("does-not-exist.pdf"),
            [SegmentDraft(0, SegmentKind.paragraph, "text", 1)],
            native_text_by_page={1: "text" * 100},
        )
