from __future__ import annotations

import uuid

from rag_app.storage.cleanup import orphan_document_ids


def test_orphan_document_ids_only_accepts_uuid_prefixes() -> None:
    live = uuid.uuid4()
    orphan = uuid.uuid4()

    assert orphan_document_ids(
        [
            f"{live}/view.pdf",
            f"{orphan}/translations/zh.docx",
            "not-a-document/system.json",
            str(uuid.uuid4()),
            "",
        ],
        {live},
    ) == {orphan}
