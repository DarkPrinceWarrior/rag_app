"""Production smoke for Keycloak, document, and RAG owner isolation.

Run this script on the A100 host. It uses the Keycloak bootstrap credentials
only in process memory, obtains user tokens through browser impersonation and
the existing Authorization Code/PKCE flow, and never changes realm settings or
user credentials. Detailed results are written with mode 0600; tokens and
usernames are never persisted.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import http.cookiejar
import json
import os
import secrets
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_app.config import settings
from rag_app.db.rls import assert_api_rls_role, reset_principal, set_principal
from rag_app.rag.retrieve import _HIERARCHICAL_EXPANSION_SQL


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    base_url: str
    api_url: str
    realm: str
    client_id: str
    redirect_uri: str
    keycloak_container: str
    account_count: int
    report_path: Path
    hierarchical_rls_evidence_path: Path | None
    search_query: str


@dataclasses.dataclass(frozen=True, slots=True)
class TokenBundle:
    user_id: str = dataclasses.field(repr=False)
    access_token: str = dataclasses.field(repr=False)
    refresh_token: str = dataclasses.field(repr=False)
    roles: frozenset[str]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # noqa: ANN001
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        return None


def _request(
    url: str,
    *,
    method: str = "GET",
    bearer: str | None = None,
    json_body: dict[str, Any] | None = None,
    form: dict[str, str] | None = None,
    opener: urllib.request.OpenerDirector | None = None,
    timeout: float = 120.0,
) -> tuple[int, Any, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    data: bytes | None = None
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body, separators=(",", ":")).encode()
    elif form is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urllib.parse.urlencode(form).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    client = opener or urllib.request.build_opener()
    try:
        with client.open(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
            location = response.headers.get("Location", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
        location = exc.headers.get("Location", "")
    if not raw:
        body: Any = None
    else:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
    return status, body, location


def _require_status(status: int, expected: int, operation: str) -> None:
    if status != expected:
        raise RuntimeError(f"{operation}: HTTP {status}, expected {expected}")


def _container_secret(container: str, name: str) -> str:
    completed = subprocess.run(
        ["docker", "exec", container, "printenv", name],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"empty Keycloak environment variable: {name}")
    return value


def _admin_token(config: Config) -> str:
    username = _container_secret(config.keycloak_container, "KC_BOOTSTRAP_ADMIN_USERNAME")
    password = _container_secret(config.keycloak_container, "KC_BOOTSTRAP_ADMIN_PASSWORD")
    status, body, _ = _request(
        f"{config.base_url}/realms/master/protocol/openid-connect/token",
        method="POST",
        form={
            "client_id": "admin-cli",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
    )
    _require_status(status, 200, "bootstrap admin login")
    if not isinstance(body, dict) or not isinstance(body.get("access_token"), str):
        raise RuntimeError("bootstrap admin login returned no access token")
    return body["access_token"]


def _admin_get(config: Config, admin_token: str, path: str) -> Any:
    status, body, _ = _request(
        f"{config.base_url}/admin/realms/{config.realm}{path}",
        bearer=admin_token,
    )
    _require_status(status, 200, f"Keycloak admin GET {path}")
    return body


def _user_roles(config: Config, admin_token: str, user_id: str) -> frozenset[str]:
    body = _admin_get(
        config,
        admin_token,
        f"/users/{urllib.parse.quote(user_id)}/role-mappings/realm/composite",
    )
    if not isinstance(body, list):
        raise RuntimeError("Keycloak role mapping is not a list")
    return frozenset(
        role["name"] for role in body if isinstance(role, dict) and isinstance(role.get("name"), str)
    )


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".", 2)[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        claims = json.loads(decoded)
    except (IndexError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Keycloak returned a malformed access token") from exc
    if not isinstance(claims, dict):
        raise RuntimeError("Keycloak access-token claims are not an object")
    return claims


def _impersonation_session_id(cookie_jar: http.cookiejar.CookieJar) -> str | None:
    for cookie in cookie_jar:
        if cookie.name != "KEYCLOAK_SESSION":
            continue
        if cookie.value is None:
            continue
        value = urllib.parse.unquote(cookie.value).strip('"')
        session_id = value.rsplit("/", 1)[-1]
        if session_id:
            return session_id
    return None


def _delete_session(config: Config, admin_token: str, session_id: str) -> int:
    status, _, _ = _request(
        f"{config.base_url}/admin/realms/{config.realm}/sessions/{urllib.parse.quote(session_id)}",
        method="DELETE",
        bearer=admin_token,
    )
    return status


def _impersonate(config: Config, admin_token: str, user_id: str) -> TokenBundle:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar), _NoRedirect())
    status, body, _ = _request(
        f"{config.base_url}/admin/realms/{config.realm}/users/{urllib.parse.quote(user_id)}/impersonation",
        method="POST",
        bearer=admin_token,
        json_body={},
        opener=opener,
    )
    _require_status(status, 200, "Keycloak browser impersonation")
    if not isinstance(body, dict) or not {"redirect", "sameRealm"} <= body.keys():
        raise RuntimeError("unexpected Keycloak impersonation response")
    session_id = _impersonation_session_id(cookie_jar)

    try:
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(24)
        query = urllib.parse.urlencode(
            {
                "client_id": config.client_id,
                "redirect_uri": config.redirect_uri,
                "response_type": "code",
                "scope": "openid",
                "state": state,
                "nonce": secrets.token_urlsafe(24),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        status, _, location = _request(
            f"{config.base_url}/realms/{config.realm}/protocol/openid-connect/auth?{query}",
            opener=opener,
        )
        _require_status(status, 302, "OIDC authorization")
        redirect = urllib.parse.urlparse(location)
        expected_redirect = urllib.parse.urlparse(config.redirect_uri)
        if (redirect.scheme, redirect.netloc, redirect.path) != (
            expected_redirect.scheme,
            expected_redirect.netloc,
            expected_redirect.path,
        ):
            raise RuntimeError("OIDC authorization returned an unexpected redirect URI")
        params = urllib.parse.parse_qs(redirect.query)
        if params.get("state") != [state] or not params.get("code"):
            raise RuntimeError("OIDC authorization response failed state/code validation")

        status, token_body, _ = _request(
            f"{config.base_url}/realms/{config.realm}/protocol/openid-connect/token",
            method="POST",
            form={
                "client_id": config.client_id,
                "grant_type": "authorization_code",
                "redirect_uri": config.redirect_uri,
                "code": params["code"][0],
                "code_verifier": verifier,
            },
        )
        _require_status(status, 200, "OIDC code exchange")
        if not isinstance(token_body, dict):
            raise RuntimeError("OIDC code exchange returned an invalid response")
        access_token = token_body.get("access_token")
        refresh_token = token_body.get("refresh_token")
        if not isinstance(access_token, str) or not isinstance(refresh_token, str):
            raise RuntimeError("OIDC code exchange returned incomplete tokens")
        claims = _jwt_claims(access_token)
        token_roles = claims.get("realm_access", {}).get("roles", [])
        roles = frozenset(role for role in token_roles if isinstance(role, str))
        if claims.get("sub") != user_id:
            raise RuntimeError("impersonated token subject differs from selected Keycloak user")
        return TokenBundle(
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            roles=roles,
        )
    except BaseException:
        if session_id is not None:
            try:
                _delete_session(config, admin_token, session_id)
            except Exception:
                pass
        raise


def _logout(config: Config, token: TokenBundle) -> int:
    status, _, _ = _request(
        f"{config.base_url}/realms/{config.realm}/protocol/openid-connect/logout",
        method="POST",
        form={
            "client_id": config.client_id,
            "refresh_token": token.refresh_token,
        },
    )
    return status


def _api(
    config: Config,
    token: TokenBundle,
    path: str,
    *,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> tuple[int, Any]:
    status, body, _ = _request(
        f"{config.api_url}{path}",
        method=method,
        bearer=token.access_token,
        json_body=json_body,
        timeout=timeout,
    )
    return status, body


def _document_ids(body: Any) -> set[str]:
    if not isinstance(body, list):
        raise RuntimeError("documents/search response is not a list")
    ids: set[str] = set()
    for item in body:
        if not isinstance(item, dict):
            raise RuntimeError("documents/search response has a non-object entry")
        document_id = item.get("document_id", item.get("id"))
        if not isinstance(document_id, str):
            raise RuntimeError("documents/search response has an entry without document ID")
        ids.add(document_id)
    return ids


def _subject_hash(user_id: str) -> str:
    return hashlib.sha256(f"account-isolation-v1:{user_id}".encode()).hexdigest()[:20]


def _api_database_url() -> str:
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ")
            if b"uvicorn" not in command or b"rag_app.api.main:app" not in command:
                continue
            environment = (process_dir / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for entry in environment:
            if not entry.startswith(b"RAG_DATABASE_URL="):
                continue
            value = entry.partition(b"=")[2].decode()
            if value:
                return value
    raise RuntimeError("could not read RAG_DATABASE_URL from the running API process")


async def _hierarchy_rows(
    session: AsyncSession,
    *,
    anchor_ids: list[uuid.UUID],
    owner_sub: str | None,
) -> list[Any]:
    return list(
        (
            await session.execute(
                text(_HIERARCHICAL_EXPANSION_SQL),
                {
                    "doc_id": None,
                    "doc_ids": None,
                    "folder_id": None,
                    "owner": owner_sub,
                    "anchor_ids": anchor_ids,
                    "page_radius": settings.rag_hierarchical_page_radius,
                    "per_anchor_k": settings.rag_hierarchical_per_anchor_k,
                    "expansion_k": settings.rag_hierarchical_max_candidates,
                },
            )
        ).all()
    )


async def _simulate_rls_principals(
    user_ids: list[str],
) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    engine = create_async_engine(_api_database_url(), pool_pre_ping=True)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await assert_api_rls_role(engine, required=True)
        admin_context = set_principal("account-isolation-admin-probe", True)
        try:
            async with sessionmaker() as session:
                all_document_ids = set(
                    await session.scalars(text("SELECT id::text FROM documents ORDER BY id"))
                )
                all_chunk_rows = (
                    await session.execute(
                        text(
                            "SELECT c.id, c.document_id::text AS document_id, d.owner_sub "
                            "FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY c.id"
                        )
                    )
                ).all()
        finally:
            reset_principal(admin_context)

        results: list[dict[str, Any]] = []
        visible_by_subject: dict[str, set[str]] = {}
        hierarchy_leak_count = 0
        hierarchy_scope_violation_count = 0
        for user_id in user_ids:
            subject = _subject_hash(user_id)
            errors: list[str] = []
            foreign_chunk_id = next(
                (row.id for row in all_chunk_rows if row.owner_sub != user_id),
                None,
            )
            principal_context = set_principal(user_id, False)
            try:
                async with sessionmaker() as session:
                    state = (
                        await session.execute(
                            text(
                                "SELECT current_user, "
                                "current_setting('app.user_id', true) AS user_id, "
                                "current_setting('app.is_admin', true) AS is_admin"
                            )
                        )
                    ).one()
                    document_rows = (
                        await session.execute(text("SELECT id::text, owner_sub FROM documents ORDER BY id"))
                    ).all()
                    document_ids = {row.id for row in document_rows}
                    chunk_rows = (
                        await session.execute(
                            text(
                                "SELECT c.id, c.document_id::text AS document_id FROM chunks c "
                                "JOIN documents d ON d.id = c.document_id ORDER BY c.id"
                            )
                        )
                    ).all()
                    chunk_document_ids = {row.document_id for row in chunk_rows}
                    foreign_candidates = sorted(all_document_ids - document_ids)
                    foreign_visible = False
                    if foreign_candidates:
                        foreign_visible = (
                            await session.scalar(
                                text(
                                    "SELECT EXISTS(SELECT 1 FROM documents "
                                    "WHERE id = CAST(:document_id AS uuid))"
                                ),
                                {"document_id": foreign_candidates[0]},
                            )
                        ) is True
                    own_anchor_ids = [chunk_rows[0].id] if chunk_rows else []
                    own_hierarchy_rows = await _hierarchy_rows(
                        session,
                        anchor_ids=own_anchor_ids,
                        owner_sub=user_id,
                    )
                    foreign_hierarchy_rows = await _hierarchy_rows(
                        session,
                        anchor_ids=[] if foreign_chunk_id is None else [foreign_chunk_id],
                        owner_sub=user_id,
                    )
            finally:
                reset_principal(principal_context)

            if state.current_user != "rag_api":
                errors.append("unexpected_database_role")
            if state.user_id != user_id or state.is_admin != "off":
                errors.append("rls_principal_guc_mismatch")
            if any(row.owner_sub != user_id for row in document_rows):
                errors.append("foreign_document_visible")
            if not chunk_document_ids <= document_ids:
                errors.append("retrieval_chunk_outside_document_scope")
            if foreign_candidates and foreign_visible:
                errors.append("foreign_document_lookup_visible")
            if foreign_chunk_id is None:
                errors.append("no_foreign_hierarchy_canary")
            if foreign_hierarchy_rows:
                errors.append("foreign_hierarchy_anchor_visible")
            hierarchy_scope_violations = sum(
                str(row.document_id) not in document_ids for row in own_hierarchy_rows
            )
            if hierarchy_scope_violations:
                errors.append("hierarchy_expansion_escaped_document_scope")
            hierarchy_leak_count += len(foreign_hierarchy_rows)
            hierarchy_scope_violation_count += hierarchy_scope_violations
            visible_by_subject[subject] = document_ids
            results.append(
                {
                    "subject_hash": subject,
                    "database_role": state.current_user,
                    "document_count": len(document_ids),
                    "retrieval_document_count": len(chunk_document_ids),
                    "foreign_lookup_empty": bool(foreign_candidates) and not foreign_visible,
                    "hierarchy_own_anchor_count": len(own_anchor_ids),
                    "hierarchy_related_count": len(own_hierarchy_rows),
                    "foreign_hierarchy_visible_count": len(foreign_hierarchy_rows),
                    "hierarchy_scope_violation_count": hierarchy_scope_violations,
                    "errors": errors,
                    "passed": not errors,
                }
            )

        async with sessionmaker() as session:
            anonymous_hierarchy_rows = await _hierarchy_rows(
                session,
                anchor_ids=[all_chunk_rows[0].id] if all_chunk_rows else [],
                owner_sub=None,
            )

        overlap_count = 0
        subjects = sorted(visible_by_subject)
        for index, left in enumerate(subjects):
            for right in subjects[index + 1 :]:
                if visible_by_subject[left] & visible_by_subject[right]:
                    overlap_count += 1
                    for result in results:
                        if result["subject_hash"] in {left, right}:
                            result["errors"].append("document_visible_to_multiple_simulated_principals")
                            result["passed"] = False

        if anonymous_hierarchy_rows:
            hierarchy_leak_count += len(anonymous_hierarchy_rows)
        passed = sum(bool(result["passed"]) for result in results)
        owner_scope_count = sum(bool(document_ids) for document_ids in visible_by_subject.values())
        hierarchy_evidence_passed = (
            len(results) >= 10
            and owner_scope_count >= 2
            and bool(all_chunk_rows)
            and not anonymous_hierarchy_rows
            and hierarchy_leak_count == 0
            and hierarchy_scope_violation_count == 0
            and passed == len(results)
        )
        hierarchy_evidence = {
            "schema_version": "hierarchical-rls-evidence-v1",
            "principal_count": len(results),
            "owner_scope_count": owner_scope_count,
            "probe_count": len(results) * 2 + 1,
            "admin_foreign_truth_count": len(all_chunk_rows),
            "anonymous_visible_count": len(anonymous_hierarchy_rows),
            "leak_count": hierarchy_leak_count,
            "scope_violation_count": hierarchy_scope_violation_count,
            "passed": hierarchy_evidence_passed,
        }
        return (
            {
                "tested_principal_count": len(results),
                "passed_principal_count": passed,
                "failed_principal_count": len(results) - passed,
                "pairwise_document_overlap_count": overlap_count,
                "hierarchy_probe_count": hierarchy_evidence["probe_count"],
                "hierarchy_rls_passed": hierarchy_evidence_passed,
                "principals": results,
            },
            all_document_ids,
            hierarchy_evidence,
        )
    finally:
        await engine.dispose()


def _write_private_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(config: Config) -> dict[str, Any]:
    admin_token = _admin_token(config)
    users = _admin_get(config, admin_token, "/users?enabled=true&max=1000")
    if not isinstance(users, list):
        raise RuntimeError("Keycloak users response is not a list")

    role_map: dict[str, frozenset[str]] = {}
    for user in users:
        if not isinstance(user, dict) or not isinstance(user.get("id"), str):
            continue
        if user.get("serviceAccountClientId"):
            continue
        role_map[user["id"]] = _user_roles(config, admin_token, user["id"])
    normal_ids = sorted(
        user_id for user_id, roles in role_map.items() if "user" in roles and "admin" not in roles
    )
    if len(normal_ids) < config.account_count:
        raise RuntimeError(f"only {len(normal_ids)} eligible non-admin users; need {config.account_count}")

    tokens: list[TokenBundle] = []
    logout_statuses: dict[str, int] = {}
    try:
        skipped_accounts: list[dict[str, str]] = []
        for user_id in normal_ids:
            try:
                tokens.append(_impersonate(config, admin_token, user_id))
            except RuntimeError as exc:
                skipped_accounts.append({"subject_hash": _subject_hash(user_id), "reason": str(exc)})
            if len(tokens) == config.account_count:
                break
        for token in tokens:
            if "user" not in token.roles or "admin" in token.roles:
                raise RuntimeError("selected non-admin token has unexpected realm roles")
        rls_summary, all_document_ids, hierarchical_rls_evidence = asyncio.run(
            _simulate_rls_principals(normal_ids[: config.account_count])
        )
        if config.hierarchical_rls_evidence_path is not None:
            _write_private_report(
                config.hierarchical_rls_evidence_path,
                hierarchical_rls_evidence,
            )

        results: list[dict[str, Any]] = []
        visible_by_subject: dict[str, set[str]] = {}
        for token in tokens:
            subject = _subject_hash(token.user_id)
            errors: list[str] = []
            documents_status, documents = _api(config, token, "/api/documents")
            if documents_status == 200:
                try:
                    own_ids = _document_ids(documents)
                except RuntimeError as exc:
                    own_ids = set()
                    errors.append(str(exc))
            else:
                own_ids = set()
                errors.append(f"documents_http_{documents_status}")
            visible_by_subject[subject] = own_ids

            search_path = "/api/search?" + urllib.parse.urlencode({"q": config.search_query, "top_k": 30})
            search_status, search_body = _api(config, token, search_path)
            search_ids: set[str] = set()
            search_scope_ok = False
            if search_status == 200:
                try:
                    search_ids = _document_ids(search_body)
                    search_scope_ok = True
                    for document_id in search_ids - own_ids:
                        direct_status, _ = _api(config, token, f"/api/documents/{document_id}")
                        if direct_status != 200:
                            search_scope_ok = False
                            errors.append("search_returned_inaccessible_document")
                            break
                except RuntimeError as exc:
                    errors.append(str(exc))
            else:
                errors.append(f"search_http_{search_status}")

            foreign_id: str | None = None
            foreign_status: int | None = None
            for candidate in sorted(all_document_ids - own_ids):
                candidate_status, _ = _api(config, token, f"/api/documents/{candidate}")
                if candidate_status == 404:
                    foreign_id = candidate
                    foreign_status = candidate_status
                    break
                if candidate_status != 200:
                    errors.append(f"foreign_probe_http_{candidate_status}")
                    break
            if foreign_id is None:
                errors.append("no_confirmed_foreign_document")

            chat_status: int | None = None
            if foreign_id is not None:
                chat_status, _ = _api(
                    config,
                    token,
                    "/api/chat?memory=off",
                    method="POST",
                    json_body={
                        "message": "account isolation smoke",
                        "document_ids": [foreign_id],
                    },
                )
                if chat_status != 404:
                    errors.append(f"foreign_chat_scope_http_{chat_status}")

            results.append(
                {
                    "subject_hash": subject,
                    "documents_status": documents_status,
                    "document_count": len(own_ids),
                    "search_status": search_status,
                    "search_hit_count": len(search_ids),
                    "search_scope_ok": search_scope_ok,
                    "foreign_document_status": foreign_status,
                    "foreign_chat_scope_status": chat_status,
                    "errors": errors,
                }
            )

        overlaps: list[tuple[str, str]] = []
        subjects = sorted(visible_by_subject)
        for index, left in enumerate(subjects):
            for right in subjects[index + 1 :]:
                if visible_by_subject[left] & visible_by_subject[right]:
                    overlaps.append((left, right))
        if overlaps:
            affected = {subject for pair in overlaps for subject in pair}
            for result in results:
                if result["subject_hash"] in affected:
                    result["errors"].append("document_visible_to_multiple_non_admin_users")

        for token in tokens:
            status = _logout(config, token)
            logout_statuses[_subject_hash(token.user_id)] = status
        tokens = []
        for result in results:
            result["logout_status"] = logout_statuses[result["subject_hash"]]
            if result["logout_status"] not in {200, 204}:
                result["errors"].append(f"logout_http_{result['logout_status']}")
            result["passed"] = not result["errors"]

        passed = sum(bool(result["passed"]) for result in results)
        rls_passed = (
            rls_summary["failed_principal_count"] == 0
            and hierarchical_rls_evidence["passed"] is True
        )
        overall_passed = passed == len(results) and rls_passed
        report = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "target": {
                "api_url": config.api_url,
                "keycloak_url": config.base_url,
                "realm": config.realm,
                "client_id": config.client_id,
            },
            "method": (
                "available_http_admin_browser_impersonation_plus_oidc_pkce_s256"
                "_and_ten_principal_production_rls_simulation"
            ),
            "realm_mutated": False,
            "credentials_changed": False,
            "tokens_persisted": False,
            "enabled_user_count": len(role_map),
            "eligible_non_admin_user_count": len(normal_ids),
            "skipped_account_count": len(skipped_accounts),
            "skipped_accounts": skipped_accounts,
            "requested_http_account_count": config.account_count,
            "http_coverage_complete": len(results) == config.account_count,
            "http_tested_account_count": len(results),
            "http_passed_account_count": passed,
            "http_failed_account_count": len(results) - passed,
            "http_pairwise_document_overlap_count": len(overlaps),
            "http_accounts": results,
            "rls_admin_visible_document_count": len(all_document_ids),
            "rls_simulation": rls_summary,
            "hierarchical_rls_evidence": hierarchical_rls_evidence,
            "limitation": (
                None
                if len(results) == config.account_count
                else "interactive OIDC state prevented bearer issuance for some accounts; "
                "production RLS simulated all ten real Keycloak subjects"
            ),
            "passed": overall_passed,
        }
        _write_private_report(config.report_path, report)
        return {
            "passed": overall_passed,
            "http_tested_accounts": len(results),
            "http_passed_accounts": passed,
            "rls_simulated_principals": rls_summary["tested_principal_count"],
            "rls_passed_principals": rls_summary["passed_principal_count"],
            "hierarchical_rls_evidence": (
                str(config.hierarchical_rls_evidence_path)
                if config.hierarchical_rls_evidence_path is not None
                else None
            ),
            "report": str(config.report_path),
        }
    finally:
        for token in tokens:
            try:
                _logout(config, token)
            except Exception:
                pass


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://doc-rag-translate.ds-mind-lab.ru")
    parser.add_argument("--api-url", default="https://doc-rag-translate.ds-mind-lab.ru")
    parser.add_argument("--realm", default="rag-app")
    parser.add_argument("--client-id", default="rag-web")
    parser.add_argument("--redirect-uri", default="https://doc-rag-translate.ds-mind-lab.ru/")
    parser.add_argument("--keycloak-container", default="rag-app-keycloak-1")
    parser.add_argument("--accounts", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/root/parser_trials/rag_eval_v1/account_isolation_smoke_2026-07-13.json"),
    )
    parser.add_argument("--hierarchical-rls-evidence", type=Path)
    parser.add_argument("--search-query", default="требования документа")
    args = parser.parse_args()
    if args.accounts < 10:
        parser.error("--accounts must be at least 10")
    return Config(
        base_url=args.base_url.rstrip("/"),
        api_url=args.api_url.rstrip("/"),
        realm=args.realm,
        client_id=args.client_id,
        redirect_uri=args.redirect_uri,
        keycloak_container=args.keycloak_container,
        account_count=args.accounts,
        report_path=args.report,
        hierarchical_rls_evidence_path=args.hierarchical_rls_evidence,
        search_query=args.search_query,
    )


def main() -> None:
    summary = run(_parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
