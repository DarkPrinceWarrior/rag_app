from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

from rag_app.db.models import SegmentKind
from rag_app.pipeline.page_routing import PageRouteDecision, RouteRole
from rag_app.pipeline.page_routing_shadow import PageRoutingPlan
from rag_app.pipeline.segments import SegmentDraft
from rag_app.workers import tasks


class _Storage:
    async def download_to(self, bucket: str, key: str, local: Path) -> None:
        del bucket, key
        local.write_bytes(b"pdf")


def _draft(page: int, text: str, kind: SegmentKind = SegmentKind.paragraph) -> SegmentDraft:
    meta = {"table_cells": [[{"text": text}]]} if kind == SegmentKind.table else {}
    return SegmentDraft(0, kind, text, page, meta=meta)


def test_worker_helper_merges_only_accepted_page(monkeypatch) -> None:
    uploaded: list[SegmentDraft] = []

    async def fake_vlm(*args, **kwargs) -> list[SegmentDraft]:
        del args, kwargs
        return [_draft(1, "A", SegmentKind.table)]

    async def fake_upload(storage, doc_id, base_dir, drafts) -> None:
        del storage, doc_id, base_dir
        uploaded.extend(drafts)

    monkeypatch.setattr(tasks, "_vlm_segments", fake_vlm)
    monkeypatch.setattr(tasks, "_pdf_page_sizes", lambda _: {0: (600.0, 800.0), 1: (600.0, 800.0)})
    monkeypatch.setattr(tasks, "_upload_segment_images", fake_upload)
    routing = PageRoutingPlan(
        (),
        (PageRouteDecision(1, RouteRole.parser_fallback, "table_requires_structure_check"),),
        (PageRouteDecision(1, RouteRole.parser_fallback, "table_requires_structure_check"),),
    )
    doc = SimpleNamespace(id=uuid.uuid4(), s3_key_original="doc/source.pdf")

    merged, metadata, accepted_pages = asyncio.run(
        tasks._apply_paddle_page_fallback(
            _Storage(),
            doc,
            [_draft(0, "page zero"), _draft(1, "plain")],
            [_draft(0, "page zero"), _draft(1, "plain")],
            routing,
            n_pages=2,
            parser_revision=4,
        )
    )

    assert [draft.source_text for draft in merged] == ["page zero", "A"]
    assert merged[1].meta["parser_backend"] == "paddle_vl"
    assert merged[1].meta["parser_revision"] == 4
    assert len(uploaded) == 1
    assert metadata["accepted_page_count"] == 1
    assert accepted_pages == frozenset({1})
