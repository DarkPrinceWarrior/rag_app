"""Запись в append-only аудит (roadmap § 9)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

from rag_app.api.auth import User
from rag_app.db.models import AuditLog

logger = logging.getLogger(__name__)


async def audit(
    request: Request,
    action: str,
    object_type: str | None = None,
    object_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Аудит не должен ронять бизнес-операцию — ошибки только в лог."""
    user: User | None = getattr(request.state, "user", None)
    try:
        async with request.app.state.sessionmaker() as session:
            session.add(
                AuditLog(
                    user_sub=user.sub if user else "anonymous",
                    username=user.username if user else None,
                    action=action,
                    object_type=object_type,
                    object_id=str(object_id) if object_id else None,
                    detail=detail,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("audit: не записалось (action=%s)", action)
