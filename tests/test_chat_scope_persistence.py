from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from rag_app.api.routes.chat import ChatIn, _apply_requested_scope, _scope_values
from rag_app.db.models import ChatSession


def _id(value: int) -> uuid.UUID:
    return uuid.UUID(int=value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (ChatIn(message="q", scope_kind="all"), (None, None, None)),
        (ChatIn(message="q", scope_kind="folder", folder_id=_id(10)), (None, _id(10), None)),
        (ChatIn(message="q", scope_kind="docs", document_id=_id(20)), (_id(20), None, None)),
        (
            ChatIn(message="q", scope_kind="docs", document_ids=[_id(20), _id(21)]),
            (None, None, [_id(20), _id(21)]),
        ),
    ],
)
def test_scope_values_are_canonical_and_persistable(
    body: ChatIn,
    expected: tuple[uuid.UUID | None, uuid.UUID | None, list[uuid.UUID] | None],
) -> None:
    assert _scope_values(body) == expected


def test_explicit_all_scope_clears_a_persisted_multi_document_scope() -> None:
    session = ChatSession(
        id=_id(1),
        title="scope",
        owner_sub="synthetic-owner",
        document_ids=[_id(20), _id(21)],
    )

    _apply_requested_scope(session, ChatIn(message="q", scope_kind="all"))

    assert session.document_id is None
    assert session.folder_id is None
    assert session.document_ids is None


def test_legacy_request_without_scope_keeps_the_persisted_scope() -> None:
    session = ChatSession(
        id=_id(1),
        title="scope",
        owner_sub="synthetic-owner",
        document_ids=[_id(20), _id(21)],
    )

    _apply_requested_scope(session, ChatIn(message="q"))

    assert session.document_ids == [_id(20), _id(21)]


@pytest.mark.parametrize(
    "payload",
    [
        {"scope_kind": "all", "document_id": _id(1)},
        {"scope_kind": "folder"},
        {"scope_kind": "docs"},
        {"document_id": _id(1), "folder_id": _id(2)},
        {"scope_kind": "docs", "document_ids": [_id(1), _id(1)]},
    ],
)
def test_invalid_or_ambiguous_scope_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatIn(message="q", **payload)
