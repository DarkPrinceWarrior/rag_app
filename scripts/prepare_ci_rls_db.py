"""Prepare the disposable PostgreSQL database used by the real RLS CI test."""

from __future__ import annotations

import asyncio
import os
import time

import asyncpg  # type: ignore[import-untyped]
from sqlalchemy.engine import make_url

_ENV = "RAG_RLS_TEST_ADMIN_URL"


async def _connect_with_retry(dsn: str) -> asyncpg.Connection:
    deadline = time.monotonic() + 60
    while True:
        try:
            return await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresConnectionError):
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(1)


async def main() -> None:
    raw_url = os.environ.get(_ENV)
    if not raw_url:
        raise RuntimeError(f"{_ENV} is required")
    url = make_url(raw_url)
    database = url.database or ""
    if "rls_test" not in database.lower():
        raise RuntimeError("refusing to prepare a database whose name does not contain rls_test")

    maintenance_url = url.set(database="postgres").render_as_string(hide_password=False)
    connection = await _connect_with_retry(maintenance_url)
    try:
        for role, bypass_rls in (("rag", False), ("rag_api", False), ("rag_worker", True)):
            exists = await connection.fetchval("SELECT 1 FROM pg_roles WHERE rolname = $1", role)
            if not exists:
                suffix = " BYPASSRLS" if bypass_rls else " NOBYPASSRLS"
                await connection.execute(f'CREATE ROLE "{role}" NOLOGIN{suffix}')
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
        if not exists:
            await connection.execute(f'CREATE DATABASE "{database}"')
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
