"""Inventory and optionally remove export prefixes without a live document row."""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Document
from rag_app.storage.cleanup import orphan_document_ids
from rag_app.storage.s3 import Storage


async def main(*, apply: bool) -> int:
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    storage = Storage()
    try:
        async with sessionmaker() as session:
            live_ids = set((await session.execute(select(Document.id))).scalars().all())

        def _names() -> list[str]:
            return [
                item.object_name
                for item in storage.client.list_objects(
                    settings.bucket_exports,
                    recursive=True,
                )
                if item.object_name
            ]

        orphans = orphan_document_ids(await asyncio.to_thread(_names), live_ids)
        print(f"orphan export prefixes: {len(orphans)}; mode={'apply' if apply else 'dry-run'}")
        for document_id in sorted(orphans, key=str):
            print(document_id)
            if apply:
                await storage.remove_document_objects(settings.bucket_exports, document_id)
        return len(orphans)
    finally:
        await engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete orphan UUID prefixes; without this flag the command is read-only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(apply=args.apply))
