from __future__ import annotations

import pytest

from rag_app.rag.retrieve import (
    _DENSE_SQL,
    _IMG_CHUNKS_SQL,
    _SPARSE_SQL,
    _VISUAL_PAGES_SQL,
)


@pytest.mark.parametrize(
    "query",
    (_DENSE_SQL, _SPARSE_SQL, _VISUAL_PAGES_SQL, _IMG_CHUNKS_SQL),
)
def test_retrieval_only_uses_completed_documents(query: str) -> None:
    assert "d.status = 'done'" in query
