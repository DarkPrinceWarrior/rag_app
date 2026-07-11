"""Обёртка над MinIO. Клиент minio синхронный — вызовы уводим в поток."""

from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path

from minio import Minio

from rag_app.config import settings


class Storage:
    def __init__(self) -> None:
        self.client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
        )

    async def ensure_buckets(self) -> None:
        def _ensure() -> None:
            for bucket in (
                settings.bucket_originals,
                settings.bucket_artifacts,
                settings.bucket_translated,
                settings.bucket_exports,
            ):
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)

        await asyncio.to_thread(_ensure)

    async def put_bytes(
        self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> None:
        await asyncio.to_thread(
            self.client.put_object,
            bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        def _get() -> bytes:
            resp = self.client.get_object(bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return await asyncio.to_thread(_get)

    async def download_to(self, bucket: str, key: str, path: Path) -> None:
        await asyncio.to_thread(self.client.fget_object, bucket, key, str(path))

    async def remove_object(self, bucket: str, key: str) -> None:
        """Best-effort удаление объекта (для удаления документа). Отсутствие
        объекта — не ошибка: чистка идёт по ключам, которых могло и не быть."""
        try:
            await asyncio.to_thread(self.client.remove_object, bucket, key)
        except Exception:
            pass

    async def remove_document_objects(self, bucket: str, document_id: uuid.UUID) -> int:
        """Удалить только объекты под точным UUID-префиксом документа."""

        prefix = f"{document_id}/"

        def _remove() -> int:
            removed = 0
            for item in self.client.list_objects(bucket, prefix=prefix, recursive=True):
                key = item.object_name
                if not key or not key.startswith(prefix):
                    raise RuntimeError("MinIO returned an object outside the requested prefix")
                self.client.remove_object(bucket, key)
                removed += 1
            return removed

        return await asyncio.to_thread(_remove)
