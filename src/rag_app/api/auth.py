"""OIDC-аутентификация (Keycloak) и RBAC (roadmap § 9, этап 5).

RAG_AUTH_ENABLED=false (dev) — все запросы идут от встроенного пользователя
local-dev с ролью admin; true — bearer-токен обязателен, подпись проверяется
по JWKS realm'а (кэш ключей 1 час). Роли — realm_access.roles.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx
import jwt
from fastapi import Depends, HTTPException, Request

from rag_app.config import settings
from rag_app.db.rls import set_principal

logger = logging.getLogger(__name__)


@dataclass
class User:
    sub: str
    username: str
    roles: set[str] = field(default_factory=set)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


_DEV_USER = User(sub="local-dev", username="local-dev", roles={"user", "admin"})


class _JwksCache:
    def __init__(self) -> None:
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at = 0.0

    async def get_key(self, kid: str) -> jwt.PyJWK:
        if kid not in self._keys or time.monotonic() - self._fetched_at > 3600:
            url = settings.oidc_jwks_url or f"{settings.oidc_issuer}/protocol/openid-connect/certs"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            # в JWKS есть и enc-ключи (RSA-OAEP) — берём только подписи
            self._keys = {
                k["kid"]: jwt.PyJWK(k)
                for k in resp.json()["keys"]
                if k.get("use") == "sig"
            }
            self._fetched_at = time.monotonic()
        if kid not in self._keys:
            raise HTTPException(401, "неизвестный ключ подписи токена")
        return self._keys[kid]


_jwks = _JwksCache()


async def get_current_user(request: Request) -> User:
    """FastAPI-зависимость: пользователь запроса (или 401)."""
    if not settings.auth_enabled:
        request.state.user = _DEV_USER
        set_principal(_DEV_USER.sub, _DEV_USER.is_admin)  # RLS-контекст (§4.7.1)
        return _DEV_USER

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "нужен bearer-токен", headers={"WWW-Authenticate": "Bearer"})
    token = auth_header.removeprefix("Bearer ").strip()

    try:
        kid = jwt.get_unverified_header(token).get("kid", "")
        key = await _jwks.get_key(kid)
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=settings.oidc_issuer,
            options={"verify_aud": False},  # Keycloak ставит aud=account; сверяем azp
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("невалидный токен: %s", exc)
        raise HTTPException(401, "невалидный токен", headers={"WWW-Authenticate": "Bearer"}) from None

    azp = claims.get("azp")
    if azp not in (settings.oidc_client_id, "rag-extension"):
        raise HTTPException(401, f"токен выписан не нашему клиенту ({azp})")

    user = User(
        sub=claims["sub"],
        username=claims.get("preferred_username", claims["sub"]),
        roles=set(claims.get("realm_access", {}).get("roles", [])) & {"user", "admin"},
    )
    if not user.roles:
        raise HTTPException(403, "нет ролей rag-app (user/admin)")
    request.state.user = user
    set_principal(user.sub, user.is_admin)  # RLS-контекст (§4.7.1)
    return user


require_user = Depends(get_current_user)
