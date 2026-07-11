from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import pytest

from rag_app.storage.s3 import Storage


@dataclass
class _Object:
    object_name: str


class _Client:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.removed: list[tuple[str, str]] = []

    def list_objects(self, bucket: str, *, prefix: str, recursive: bool):
        assert bucket == "artifacts"
        assert recursive is True
        return (_Object(name) for name in self.names if name.startswith(prefix))

    def remove_object(self, bucket: str, key: str) -> None:
        self.removed.append((bucket, key))


def test_remove_document_objects_uses_exact_uuid_prefix() -> None:
    document_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    prefix = f"{document_id}/"
    client = _Client(
        [
            prefix + "content.json",
            prefix + "sidecars/r1/p000000/kie/a.json",
            "11111111-1111-1111-1111-111111111112/keep.json",
        ]
    )
    storage = Storage.__new__(Storage)
    storage.client = client  # type: ignore[assignment]

    removed = asyncio.run(storage.remove_document_objects("artifacts", document_id))

    assert removed == 2
    assert client.removed == [
        ("artifacts", prefix + "content.json"),
        ("artifacts", prefix + "sidecars/r1/p000000/kie/a.json"),
    ]


class _EscapingClient(_Client):
    def list_objects(self, bucket: str, *, prefix: str, recursive: bool):
        return iter([_Object("other-document/leak.json")])


def test_remove_document_objects_rejects_outside_prefix() -> None:
    storage = Storage.__new__(Storage)
    storage.client = _EscapingClient([])  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(
            storage.remove_document_objects(
                "artifacts",
                uuid.UUID("11111111-1111-1111-1111-111111111111"),
            )
        )
