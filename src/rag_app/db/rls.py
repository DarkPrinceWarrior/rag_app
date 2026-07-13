"""Fail-closed RLS context and production role checks.

Every API transaction receives ``app.user_id`` and ``app.is_admin`` from a
request-scoped ContextVar.  Missing context is anonymous and therefore sees
no protected rows.  Workers and migrations do not impersonate an API
principal: they use dedicated PostgreSQL roles with explicit RLS bypass.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import event, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session


@dataclass(frozen=True, slots=True)
class Principal:
    user_sub: str | None = None
    is_admin: bool = False


_principal: contextvars.ContextVar[Principal | None] = contextvars.ContextVar(
    "db_principal", default=None
)


def current_principal() -> Principal:
    """Return the active principal; missing context is always anonymous."""
    return _principal.get() or Principal()


def set_principal(user_sub: str, is_admin: bool) -> contextvars.Token[Principal | None]:
    """Set an authenticated API principal and return its restoration token."""
    if not user_sub or not user_sub.strip():
        raise ValueError("RLS principal requires a non-empty user_sub")
    return _principal.set(Principal(user_sub=user_sub, is_admin=bool(is_admin)))


def reset_principal(token: contextvars.Token[Principal | None]) -> None:
    """Restore the context that preceded ``set_principal``."""
    _principal.reset(token)


_SET_GUC = text(
    "SELECT set_config('app.user_id', :u, true), set_config('app.is_admin', :a, true)"
)


@event.listens_for(Session, "after_begin")
def _apply_rls_guc(session, transaction, connection) -> None:  # noqa: ANN001
    """Apply transaction-local GUCs; any PostgreSQL failure aborts the query."""
    if connection.dialect.name != "postgresql":
        return
    principal = current_principal()
    connection.execute(
        _SET_GUC,
        {
            "u": principal.user_sub or "",
            "a": "on" if principal.is_admin else "off",
        },
    )


PROTECTED_TABLES: tuple[str, ...] = (
    "documents",
    "folders",
    "chunks",
    "segments",
    "document_translations",
    "segment_versions",
    "page_embeddings",
    "document_structured_artifacts",
    "chat_sessions",
    "chat_messages",
    "audit_log",
    "memory_events",
    "memory_items",
    "memory_candidates",
    "memory_audit_log",
    "memory_item_sources",
)

_ROLE_STATE = text(
    """
    SELECT r.rolname,
           r.rolsuper,
           r.rolbypassrls,
           EXISTS (
               SELECT 1
               FROM pg_roles privileged
               WHERE (privileged.rolsuper OR privileged.rolbypassrls)
                 AND privileged.oid <> r.oid
                 AND pg_has_role(r.oid, privileged.oid, 'MEMBER')
           ) AS privileged_membership
    FROM pg_roles r
    WHERE r.rolname = current_user
    """
)

_TABLE_STATE = text(
    """
    SELECT c.relname,
           c.relowner = (SELECT oid FROM pg_roles WHERE rolname = current_user) AS is_owner,
           c.relrowsecurity,
           c.relforcerowsecurity,
           row_security_active(c.oid) AS rls_active,
           EXISTS (
               SELECT 1 FROM pg_policy p WHERE p.polrelid = c.oid
           ) AS has_policy
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = current_schema()
      AND c.relname = ANY(CAST(:tables AS text[]))
    """
)


async def _read_role_state(engine: AsyncEngine) -> tuple[RowMapping, Sequence[RowMapping]]:
    if engine.dialect.name != "postgresql":
        raise RuntimeError("RLS role checks require PostgreSQL")
    async with engine.connect() as connection:
        role = (await connection.execute(_ROLE_STATE)).mappings().one()
        tables = (
            await connection.execute(_TABLE_STATE, {"tables": list(PROTECTED_TABLES)})
        ).mappings().all()
    return role, tables


async def assert_api_rls_role(engine: AsyncEngine, *, required: bool) -> None:
    """Refuse startup when the API role could bypass the isolation boundary."""
    if not required:
        return
    role, table_rows = await _read_role_state(engine)
    if role["rolsuper"] or role["rolbypassrls"] or role["privileged_membership"]:
        raise RuntimeError(f"unsafe API database role: {role['rolname']}")

    by_name = {row["relname"]: row for row in table_rows}
    missing = sorted(set(PROTECTED_TABLES) - by_name.keys())
    unsafe = sorted(
        name
        for name, row in by_name.items()
        if row["is_owner"]
        or not row["relrowsecurity"]
        or not row["relforcerowsecurity"]
        or not row["rls_active"]
        or not row["has_policy"]
    )
    if missing or unsafe:
        raise RuntimeError(
            "unsafe API RLS schema: "
            f"missing={','.join(missing) or '-'}; unsafe={','.join(unsafe) or '-'}"
        )


async def assert_worker_rls_role(engine: AsyncEngine) -> None:
    """Workers must run under a dedicated non-superuser BYPASSRLS role."""
    role, _ = await _read_role_state(engine)
    if role["rolsuper"] or not role["rolbypassrls"]:
        raise RuntimeError(f"unsafe worker database role: {role['rolname']}")
