"""Восстановление документов, застрявших в промежуточных статусах."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import update

from rag_app.config import settings
from rag_app.db.models import Document, DocumentStatus, DocumentTranslation

# Сторож срабатывает только после штатного ARQ timeout и дополнительного окна,
# поэтому не конкурирует с задачей, которая ещё имеет право выполняться.
STATUS_RECOVERY_GRACE_S = 300

RECOVERABLE_DOCUMENT_STATUSES = (
    DocumentStatus.uploaded,
    DocumentStatus.parsing,
    DocumentStatus.parsed,
    DocumentStatus.translating,
    DocumentStatus.translated,
    DocumentStatus.exporting,
)
RECOVERABLE_TRANSLATION_STATUSES = ("translating", "exporting")


def stale_status_cutoff(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    return current - timedelta(seconds=settings.job_timeout_s + STATUS_RECOVERY_GRACE_S)


async def recover_stale_documents(ctx: dict) -> dict[str, int]:
    """Перевести просроченные промежуточные статусы в явную ошибку.

    Порог заведомо больше тайм-аута ARQ. Это не lease/requeue: сторож не создаёт
    дублей, а открывает безопасный пользовательский retry с новой parse revision.
    """

    now = datetime.now(UTC)
    cutoff = stale_status_cutoff(now)
    async with ctx["sessionmaker"]() as session:
        documents = await session.execute(
            update(Document)
            .where(
                Document.status.in_(RECOVERABLE_DOCUMENT_STATUSES),
                Document.updated_at < cutoff,
            )
            .values(
                status=DocumentStatus.error,
                error="обработка прервана: промежуточный статус просрочен; повторите операцию",
                updated_at=now,
            )
        )
        translations = await session.execute(
            update(DocumentTranslation)
            .where(
                DocumentTranslation.status.in_(RECOVERABLE_TRANSLATION_STATUSES),
                DocumentTranslation.updated_at < cutoff,
            )
            .values(
                status="error",
                error="обработка прервана: промежуточный статус просрочен; запустите перевод повторно",
                updated_at=now,
            )
        )
        await session.commit()
    return {
        "documents": int(documents.rowcount or 0),
        "translations": int(translations.rowcount or 0),
    }
