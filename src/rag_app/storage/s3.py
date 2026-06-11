"""Обёртка над MinIO. Клиент minio синхронный — вызовы уводим в поток."""

from __future__ import annotations

import asyncio
import io
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
