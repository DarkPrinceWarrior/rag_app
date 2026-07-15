#!/usr/bin/env python3
"""Fail-closed in-place backfill of deterministic chunk hierarchy metadata."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections.abc import Sequence

from sqlalchemy import select

from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Chunk, Segment
from rag_app.db.rls import reset_principal, set_principal
from rag_app.rag.chunking import segments_to_chunks
from rag_app.rag.hierarchy_backfill import ChunkHierarchyUpdate, build_hierarchy_updates


async def run(*, apply: bool, document_ids: Sequence[uuid.UUID]) -> dict[str, int | bool]:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    principal_token = set_principal("hierarchy-backfill", True)
    try:
        async with sessionmaker() as session:
            if document_ids:
                ids = tuple(dict.fromkeys(document_ids))
            else:
                ids = tuple(
                    (
                        await session.execute(
                            select(Chunk.document_id).distinct().order_by(Chunk.document_id)
                        )
                    )
                    .scalars()
                    .all()
                )
            all_updates: list[tuple[Chunk, ChunkHierarchyUpdate]] = []
            for document_id in ids:
                segments = list(
                    (
                        await session.execute(
                            select(Segment)
                            .where(Segment.document_id == document_id)
                            .order_by(Segment.idx)
                        )
                    )
                    .scalars()
                    .all()
                )
                chunks = list(
                    (
                        await session.execute(
                            select(Chunk)
                            .where(Chunk.document_id == document_id)
                            .order_by(Chunk.idx)
                        )
                    )
                    .scalars()
                    .all()
                )
                updates = build_hierarchy_updates(segments_to_chunks(segments), chunks)
                all_updates.extend(zip(chunks, updates, strict=True))
            changed = 0
            for chunk, update in all_updates:
                if not update.changed:
                    continue
                changed += 1
                if apply:
                    chunk.meta = update.meta
            if apply:
                await session.commit()
            else:
                await session.rollback()
            return {
                "applied": apply,
                "document_count": len(ids),
                "chunk_count": len(all_updates),
                "changed_chunk_count": changed,
            }
    finally:
        reset_principal(principal_token)
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="commit validated metadata updates")
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        type=uuid.UUID,
        help="limit to a document UUID; repeatable",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = asyncio.run(run(apply=args.apply, document_ids=args.document_id))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
