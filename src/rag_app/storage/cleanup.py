"""Pure helpers for safe object-storage cleanup."""

from __future__ import annotations

import uuid
from collections.abc import Iterable


def orphan_document_ids(object_names: Iterable[str], live_ids: set[uuid.UUID]) -> set[uuid.UUID]:
    """Return UUID top-level prefixes absent from the database; ignore other keys."""
    candidates: set[uuid.UUID] = set()
    for object_name in object_names:
        prefix, separator, _rest = object_name.partition("/")
        if not separator:
            continue
        try:
            candidates.add(uuid.UUID(prefix))
        except ValueError:
            continue
    return candidates - live_ids
