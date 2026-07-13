from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute, Dependant
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session

from rag_app.api import auth as auth_module
from rag_app.api.auth import get_current_user
from rag_app.config import settings
from rag_app.db import rls
from rag_app.db.rls import (
    PROTECTED_TABLES,
    Principal,
    _apply_rls_guc,
    current_principal,
    reset_principal,
    set_principal,
)


@dataclass
class _Dialect:
    name: str


class _RecordingConnection:
    def __init__(self, dialect: str = "postgresql") -> None:
        self.dialect = _Dialect(dialect)
        self.calls: list[tuple[Any, dict[str, str]]] = []

    def execute(self, statement: Any, parameters: dict[str, str]) -> None:
        self.calls.append((statement, parameters))


class _FailingConnection(_RecordingConnection):
    def execute(self, statement: Any, parameters: dict[str, str]) -> None:
        raise RuntimeError("set_config failed")


def _guc_parameters(connection: _RecordingConnection) -> dict[str, str]:
    _apply_rls_guc(None, None, connection)
    assert len(connection.calls) == 1
    return connection.calls[0][1]


def _depends_on(dependant: Dependant, dependency: object) -> bool:
    return dependant.call is dependency or any(
        _depends_on(child, dependency) for child in dependant.dependencies
    )


def test_missing_principal_is_default_deny() -> None:
    assert current_principal() == Principal(user_sub=None, is_admin=False)
    assert _guc_parameters(_RecordingConnection()) == {"u": "", "a": "off"}


def test_principal_tokens_restore_nested_context() -> None:
    outer = set_principal("user-a", False)
    try:
        assert current_principal() == Principal(user_sub="user-a", is_admin=False)
        assert _guc_parameters(_RecordingConnection()) == {"u": "user-a", "a": "off"}

        inner = set_principal("admin-b", True)
        try:
            assert current_principal() == Principal(user_sub="admin-b", is_admin=True)
            assert _guc_parameters(_RecordingConnection()) == {"u": "admin-b", "a": "on"}
        finally:
            reset_principal(inner)

        assert current_principal() == Principal(user_sub="user-a", is_admin=False)
    finally:
        reset_principal(outer)

    assert current_principal() == Principal(user_sub=None, is_admin=False)


@pytest.mark.parametrize("user_sub", ["", " ", "\t"])
def test_empty_principal_is_rejected(user_sub: str) -> None:
    with pytest.raises(ValueError, match="non-empty user_sub"):
        set_principal(user_sub, False)


def test_postgresql_guc_failure_is_not_suppressed() -> None:
    with pytest.raises(RuntimeError, match="set_config failed"):
        _apply_rls_guc(None, None, _FailingConnection())


def test_sqlite_transactions_skip_postgresql_guc() -> None:
    engine = create_engine("sqlite://")
    try:
        with Session(engine) as session:
            assert session.scalar(text("SELECT 1")) == 1
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_auth_yield_dependency_restores_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    dependency = get_current_user(request)

    user = await anext(dependency)
    assert user.sub == "local-dev"
    assert current_principal() == Principal(user_sub="local-dev", is_admin=True)

    await dependency.aclose()
    assert current_principal() == Principal(user_sub=None, is_admin=False)


@pytest.mark.asyncio
async def test_missing_bearer_token_is_401_and_does_not_set_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        await auth_module._authenticate_user(request)

    assert exc_info.value.status_code == 401
    assert current_principal() == Principal()


@pytest.mark.asyncio
async def test_invalid_azp_is_401_and_does_not_set_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(auth_module.jwt, "get_unverified_header", lambda token: {"kid": "k"})

    async def get_key(kid: str) -> object:
        return object()

    monkeypatch.setattr(auth_module._jwks, "get_key", get_key)
    monkeypatch.setattr(
        auth_module.jwt,
        "decode",
        lambda *args, **kwargs: {
            "sub": "user-a",
            "azp": "foreign-client",
            "realm_access": {"roles": ["user"]},
        },
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", b"Bearer token")],
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth_module._authenticate_user(request)

    assert exc_info.value.status_code == 401
    assert current_principal() == Principal()


@pytest.mark.asyncio
async def test_auth_principal_lives_through_stream_and_cleans_afterwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    observed: list[Principal] = []
    app = FastAPI()

    @app.get("/stream", dependencies=[Depends(get_current_user)])
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            observed.append(current_principal())
            yield b"first\n"
            await asyncio.sleep(0)
            observed.append(current_principal())
            yield b"second\n"

        return StreamingResponse(body(), media_type="text/plain")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/stream")

    assert response.status_code == 200
    assert response.text == "first\nsecond\n"
    assert observed == [
        Principal(user_sub="local-dev", is_admin=True),
        Principal(user_sub="local-dev", is_admin=True),
    ]
    assert current_principal() == Principal(user_sub=None, is_admin=False)


def test_every_private_api_route_has_auth_dependency() -> None:
    from rag_app.api.main import app

    unauthenticated: list[tuple[str, tuple[str, ...]]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith("/api") or route.path == "/api/config":
            continue
        if not _depends_on(route.dependant, get_current_user):
            unauthenticated.append((route.path, tuple(sorted(route.methods or ()))))

    assert unauthenticated == []


def _safe_table_rows() -> list[dict[str, object]]:
    return [
        {
            "relname": table,
            "is_owner": False,
            "relrowsecurity": True,
            "relforcerowsecurity": True,
            "rls_active": True,
            "has_policy": True,
        }
        for table in PROTECTED_TABLES
    ]


@pytest.mark.asyncio
async def test_api_role_gate_accepts_only_enforced_non_bypass_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def read_state(engine: AsyncEngine):
        return {
            "rolname": "rag_api",
            "rolsuper": False,
            "rolbypassrls": False,
            "privileged_membership": False,
        }, _safe_table_rows()

    monkeypatch.setattr(rls, "_read_role_state", read_state)
    await rls.assert_api_rls_role(cast(AsyncEngine, object()), required=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_field", ["rolsuper", "rolbypassrls", "privileged_membership"])
async def test_api_role_gate_rejects_privileged_role(
    monkeypatch: pytest.MonkeyPatch, unsafe_field: str
) -> None:
    role = {
        "rolname": "unsafe",
        "rolsuper": False,
        "rolbypassrls": False,
        "privileged_membership": False,
    }
    role[unsafe_field] = True

    async def read_state(engine: AsyncEngine):
        return role, _safe_table_rows()

    monkeypatch.setattr(rls, "_read_role_state", read_state)
    with pytest.raises(RuntimeError, match="unsafe API database role"):
        await rls.assert_api_rls_role(cast(AsyncEngine, object()), required=True)


@pytest.mark.asyncio
async def test_api_role_gate_rejects_missing_or_unforced_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _safe_table_rows()[1:]
    rows[0]["relforcerowsecurity"] = False

    async def read_state(engine: AsyncEngine):
        return {
            "rolname": "rag_api",
            "rolsuper": False,
            "rolbypassrls": False,
            "privileged_membership": False,
        }, rows

    monkeypatch.setattr(rls, "_read_role_state", read_state)
    with pytest.raises(RuntimeError, match="unsafe API RLS schema"):
        await rls.assert_api_rls_role(cast(AsyncEngine, object()), required=True)
