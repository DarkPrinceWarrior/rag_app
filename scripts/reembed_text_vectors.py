"""Безопасная in-place пересборка всех текстовых векторов.

По умолчанию выполняется только preflight и подсчёт строк. Запись разрешается
исключительно с ``--apply`` и checkpoint-файлом:

    uv run python scripts/reembed_text_vectors.py
    uv run python scripts/reembed_text_vectors.py \
      --apply --model-revision 8ca3ff382cf1de715e05acac8b553e0a084680d0 \
      --checkpoint /root/model_trials/reembed-nemotron3.json

Перед ``--apply`` остановите API и воркеры, которые создают/изменяют индекс,
память и translation memory. Скрипт проверяет неизменность набора и текста
целевых строк, сохраняет ID и обновляет только четыре vector-поля.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import re
import stat
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_CHECKPOINT_SCHEMA = "docragenslate-reembed-text-vectors-v3"
_ENV_MAX_BYTES = 128 * 1024
_CHECKPOINT_MAX_BYTES = 1024 * 1024
_RESPONSE_MAX_BYTES = 64 * 1024 * 1024
_BODY_MAX_CHARS = 8000
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ReembeddingError(RuntimeError):
    """Fail-closed operational error."""


@dataclass(frozen=True, slots=True)
class TargetSpec:
    key: str
    manifest_sql: str
    fetch_sql: str
    update_sql: str
    verify_sql: str
    resume_sql: str = ""
    vector_manifest_sql: str = ""
    prefix_vector_manifest_sql: str = ""


@dataclass(frozen=True, slots=True)
class SourceManifest:
    count: int
    md5: str


@dataclass(frozen=True, slots=True)
class SourceRow:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Verification:
    eligible: int
    non_null: int
    wrong_dimension: int


@dataclass(frozen=True, slots=True)
class ResumeState:
    count_through_id: int
    max_eligible_id: str | None
    cursor_exists: bool


@dataclass(frozen=True, slots=True)
class RunIdentity:
    model: str
    model_revision: str
    profile: str
    dimension: int
    native_dimension: int
    endpoint_sha256: str
    database_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "model_revision": self.model_revision,
            "profile": self.profile,
            "dimension": self.dimension,
            "native_dimension": self.native_dimension,
            "endpoint_sha256": self.endpoint_sha256,
            "database_sha256": self.database_sha256,
        }


class Repository(Protocol):
    async def source_manifest(self, target: TargetSpec) -> SourceManifest: ...

    async def fetch_batch(
        self, target: TargetSpec, *, after_id: str | None, limit: int
    ) -> list[SourceRow]: ...

    async def commit_batch(
        self,
        target: TargetSpec,
        rows: Sequence[SourceRow],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def verify(self, target: TargetSpec, dimension: int) -> Verification: ...

    async def resume_state(
        self, target: TargetSpec, last_id: str | None
    ) -> ResumeState: ...

    async def vector_manifest(self, target: TargetSpec) -> str: ...

    async def prefix_vector_manifest(
        self, target: TargetSpec, last_id: str
    ) -> str: ...


_CHUNK_MANIFEST = """
SELECT count(*)::bigint AS count,
       md5(coalesce(string_agg(
           c.id::text || ':' || md5(coalesce({source}, '')),
           ',' ORDER BY c.id
       ), '')) AS manifest_md5
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.status = 'done'
"""

_CHUNK_FETCH = """
SELECT c.id::text AS id, coalesce({source}, '') AS source_text
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.status = 'done'
  AND (:after_id IS NULL OR c.id > CAST(:after_id AS uuid))
ORDER BY c.id
LIMIT :limit
"""

_CHUNK_UPDATE = """
UPDATE chunks AS c
SET {vector} = CAST(:embedding AS vector)
FROM documents AS d
WHERE c.id = CAST(:id AS uuid)
  AND c.document_id = d.id
  AND d.status = 'done'
  AND coalesce({source}, '') = :source_text
"""

_CHUNK_VERIFY = """
SELECT count(*)::bigint AS eligible,
       count({vector})::bigint AS non_null,
       count(*) FILTER (
           WHERE {vector} IS NOT NULL AND vector_dims({vector}) <> :dimension
       )::bigint AS wrong_dimension
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.status = 'done'
"""

_CHUNK_RESUME = """
SELECT count(*) FILTER (
           WHERE :last_id IS NOT NULL AND c.id <= CAST(:last_id AS uuid)
       )::bigint AS count_through_id,
       max(c.id)::text AS max_eligible_id,
       coalesce(bool_or(
           :last_id IS NOT NULL AND c.id = CAST(:last_id AS uuid)
       ), false) AS cursor_exists
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.status = 'done'
"""

_CHUNK_VECTOR_MANIFEST = """
SELECT md5(coalesce(string_agg(
           c.id::text || ':' || md5(coalesce({vector}::text, '')),
           ',' ORDER BY c.id
       ), '')) AS vector_manifest_md5
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.status = 'done'
"""

_CHUNK_PREFIX_VECTOR_MANIFEST = """
SELECT md5(coalesce(string_agg(
           c.id::text || ':' || md5(coalesce({vector}::text, '')),
           ',' ORDER BY c.id
       ), '')) AS vector_manifest_md5
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE d.status = 'done'
  AND c.id <= CAST(:last_id AS uuid)
"""


def _simple_target(
    *,
    key: str,
    table: str,
    source: str,
    vector: str,
    eligibility: str,
) -> TargetSpec:
    return TargetSpec(
        key=key,
        manifest_sql=f"""
SELECT count(*)::bigint AS count,
       md5(coalesce(string_agg(
           id::text || ':' || md5(coalesce({source}, '')),
           ',' ORDER BY id
       ), '')) AS manifest_md5
FROM {table}
WHERE {eligibility}
""",
        fetch_sql=f"""
SELECT id::text AS id, coalesce({source}, '') AS source_text
FROM {table}
WHERE {eligibility}
  AND (:after_id IS NULL OR id > CAST(:after_id AS uuid))
ORDER BY id
LIMIT :limit
""",
        update_sql=f"""
UPDATE {table}
SET {vector} = CAST(:embedding AS vector)
WHERE id = CAST(:id AS uuid)
  AND {eligibility}
  AND coalesce({source}, '') = :source_text
""",
        verify_sql=f"""
SELECT count(*)::bigint AS eligible,
       count({vector})::bigint AS non_null,
       count(*) FILTER (
           WHERE {vector} IS NOT NULL AND vector_dims({vector}) <> :dimension
       )::bigint AS wrong_dimension
FROM {table}
WHERE {eligibility}
""",
        resume_sql=f"""
SELECT count(*) FILTER (
           WHERE :last_id IS NOT NULL AND id <= CAST(:last_id AS uuid)
       )::bigint AS count_through_id,
       max(id)::text AS max_eligible_id,
       coalesce(bool_or(
           :last_id IS NOT NULL AND id = CAST(:last_id AS uuid)
       ), false) AS cursor_exists
FROM {table}
WHERE {eligibility}
""",
        vector_manifest_sql=f"""
SELECT md5(coalesce(string_agg(
           id::text || ':' || md5(coalesce({vector}::text, '')),
           ',' ORDER BY id
       ), '')) AS vector_manifest_md5
FROM {table}
WHERE {eligibility}
""",
        prefix_vector_manifest_sql=f"""
SELECT md5(coalesce(string_agg(
           id::text || ':' || md5(coalesce({vector}::text, '')),
           ',' ORDER BY id
       ), '')) AS vector_manifest_md5
FROM {table}
WHERE {eligibility}
  AND id <= CAST(:last_id AS uuid)
""",
    )


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        key="chunks.emb_en",
        manifest_sql=_CHUNK_MANIFEST.format(source="c.text_en"),
        fetch_sql=_CHUNK_FETCH.format(source="c.text_en"),
        update_sql=_CHUNK_UPDATE.format(source="c.text_en", vector="emb_en"),
        verify_sql=_CHUNK_VERIFY.format(vector="c.emb_en"),
        resume_sql=_CHUNK_RESUME,
        vector_manifest_sql=_CHUNK_VECTOR_MANIFEST.format(vector="c.emb_en"),
        prefix_vector_manifest_sql=_CHUNK_PREFIX_VECTOR_MANIFEST.format(
            vector="c.emb_en"
        ),
    ),
    TargetSpec(
        key="chunks.emb_ru",
        manifest_sql=_CHUNK_MANIFEST.format(source="c.text_ru"),
        fetch_sql=_CHUNK_FETCH.format(source="c.text_ru"),
        update_sql=_CHUNK_UPDATE.format(source="c.text_ru", vector="emb_ru"),
        verify_sql=_CHUNK_VERIFY.format(vector="c.emb_ru"),
        resume_sql=_CHUNK_RESUME,
        vector_manifest_sql=_CHUNK_VECTOR_MANIFEST.format(vector="c.emb_ru"),
        prefix_vector_manifest_sql=_CHUNK_PREFIX_VECTOR_MANIFEST.format(
            vector="c.emb_ru"
        ),
    ),
    _simple_target(
        key="memory_items.embedding",
        table="memory_items",
        source="content",
        vector="embedding",
        eligibility="status = 'active' AND deleted_at IS NULL",
    ),
    _simple_target(
        key="translation_memory.source_embedding",
        table="translation_memory",
        source="source_text",
        vector="source_embedding",
        eligibility="status = 'approved' AND revoked_at IS NULL",
    ),
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def database_identity_sha256(database_url: str) -> str:
    """Fingerprint a database location without retaining credentials."""

    try:
        parsed = make_url(database_url)
        payload = {
            "backend": parsed.get_backend_name(),
            "host": parsed.host or "",
            "port": parsed.port,
            "database": parsed.database or "",
        }
    except Exception as exc:
        raise ReembeddingError("database URL is invalid") from exc
    return sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def validate_model_revision(value: str | None, *, required: bool) -> str:
    revision = (value or "").strip()
    if not revision:
        if required:
            raise ReembeddingError("--model-revision is required with --apply")
        return "<unpinned-dry-run>"
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ReembeddingError("model revision must be exactly 40 hexadecimal characters")
    return revision.lower()


def normalize_embedding_base_url(value: str) -> str:
    """Allow only an explicit loopback HTTP(S) endpoint rooted exactly at /v1."""

    raw = value.strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ReembeddingError("embedding URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ReembeddingError("embedding URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ReembeddingError("embedding URL must not contain query or fragment")
    if parsed.path.rstrip("/") != "/v1":
        raise ReembeddingError("embedding URL path must be exactly /v1")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ReembeddingError("embedding URL has an invalid port") from exc
    if port is None:
        raise ReembeddingError("embedding URL must contain an explicit port")
    hostname = parsed.hostname
    if hostname is None:
        raise ReembeddingError("embedding URL has no hostname")
    if hostname.casefold() != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise ReembeddingError("embedding URL must use a loopback address")
        except ValueError as exc:
            raise ReembeddingError("embedding URL must use a loopback address") from exc
    host = f"[{hostname}]" if ":" in hostname else hostname
    return urlunsplit((parsed.scheme, f"{host}:{port}", "/v1", "", ""))


def _decode_env_value(raw: str, *, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise ReembeddingError(f"env file line {line_number} has an unterminated quote")
        if value[0] == "'":
            return value[1:-1]
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ReembeddingError(f"env file line {line_number} has invalid quoting") from exc
        if not isinstance(decoded, str):
            raise ReembeddingError(f"env file line {line_number} is not a string")
        return decoded
    return value


def load_env_file(path: Path) -> tuple[str, ...]:
    """Load plain KEY=VALUE lines without shell evaluation or overriding env."""

    expanded = path.expanduser()
    try:
        info = expanded.lstat()
    except FileNotFoundError as exc:
        raise ReembeddingError("env file does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReembeddingError("env file must be a regular non-symlink file")
    if info.st_size > _ENV_MAX_BYTES:
        raise ReembeddingError("env file is too large")
    try:
        raw = expanded.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReembeddingError("env file cannot be read as UTF-8") from exc
    loaded: list[str] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            raise ReembeddingError(f"env file line {line_number} uses forbidden shell syntax")
        key, separator, value = stripped.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY.fullmatch(key):
            raise ReembeddingError(f"env file line {line_number} is not KEY=VALUE")
        if "\x00" in value:
            raise ReembeddingError(f"env file line {line_number} contains NUL")
        if key not in os.environ:
            os.environ[key] = _decode_env_value(value, line_number=line_number)
            loaded.append(key)
    return tuple(loaded)


def format_document_input(text_value: str, profile: str) -> str:
    raw = text_value.strip()
    if profile == "qwen3":
        return raw[:_BODY_MAX_CHARS] or "."
    if profile == "nemotron3":
        if raw.casefold().startswith("passage:"):
            raw = raw[len("passage:") :].lstrip() or "."
        body = raw[:_BODY_MAX_CHARS] or "."
        return f"passage: {body}"
    raise ReembeddingError("unsupported embedding input profile")


def normalize_vector(vector: Sequence[object], dimension: int) -> list[float]:
    if dimension <= 0 or len(vector) < dimension:
        raise ReembeddingError("embedding response has an invalid dimension")
    values: list[float] = []
    for item in vector[:dimension]:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReembeddingError("embedding response contains a non-number")
        value = float(item)
        if not math.isfinite(value):
            raise ReembeddingError("embedding response contains a non-finite value")
        values.append(value)
    squared_norm = math.fsum(value * value for value in values)
    if not math.isfinite(squared_norm) or squared_norm <= 0.0:
        raise ReembeddingError("embedding response has zero norm")
    norm = math.sqrt(squared_norm)
    return [value / norm for value in values]


class SafeEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        profile: str,
        dimension: int,
        timeout_s: float,
    ) -> None:
        self.base_url = normalize_embedding_base_url(base_url)
        self.model = model.strip()
        self.profile = profile
        self.dimension = dimension
        self.native_dimension: int | None = None
        if not self.model:
            raise ReembeddingError("embedding model must not be empty")
        if profile not in {"qwen3", "nemotron3"}:
            raise ReembeddingError("unsupported embedding input profile")
        if dimension <= 0:
            raise ReembeddingError("embedding dimension must be positive")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            trust_env=False,
            follow_redirects=False,
            headers={"Authorization": "Bearer local"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self, method: str, path: str, *, payload: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        request = self._client.build_request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            json=payload,
        )
        response = await self._client.send(request, stream=True)
        try:
            if response.is_redirect:
                raise ReembeddingError("embedding endpoint redirect refused")
            if response.status_code < 200 or response.status_code >= 300:
                raise ReembeddingError("embedding endpoint returned a non-success status")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > _RESPONSE_MAX_BYTES:
                    raise ReembeddingError("embedding endpoint response is too large")
        finally:
            await response.aclose()
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReembeddingError("embedding endpoint returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ReembeddingError("embedding endpoint returned a non-object")
        return decoded

    async def preflight(self) -> None:
        models = await self._request_json("GET", "models")
        cards = models.get("data")
        if not isinstance(cards, list) or self.model not in {
            card.get("id") for card in cards if isinstance(card, dict)
        }:
            raise ReembeddingError("requested embedding model is not served")
        vectors = await self.embed(["DocRAGenslate embedding preflight"])
        if len(vectors) != 1 or len(vectors[0]) != self.dimension:
            raise ReembeddingError("embedding preflight dimension mismatch")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            raise ReembeddingError("embedding batch must not be empty")
        inputs = [format_document_input(value, self.profile) for value in texts]
        response = await self._request_json(
            "POST",
            "embeddings",
            payload={"model": self.model, "input": inputs},
        )
        if response.get("model") != self.model:
            raise ReembeddingError("embedding endpoint returned a different model")
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            raise ReembeddingError("embedding endpoint returned an incomplete batch")
        indexed: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ReembeddingError("embedding endpoint returned an invalid item")
            index = item.get("index")
            vector = item.get("embedding")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ReembeddingError("embedding response has an invalid index")
            if not isinstance(vector, list) or index in indexed:
                raise ReembeddingError("embedding response has invalid vectors")
            native_dimension = len(vector)
            if self.native_dimension is None:
                self.native_dimension = native_dimension
            elif native_dimension != self.native_dimension:
                raise ReembeddingError(
                    "embedding endpoint returned inconsistent native dimensions"
                )
            indexed[index] = normalize_vector(vector, self.dimension)
        expected = list(range(len(inputs)))
        if sorted(indexed) != expected:
            raise ReembeddingError("embedding response indices do not match the request")
        return [indexed[index] for index in expected]


class SqlRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def source_manifest(self, target: TargetSpec) -> SourceManifest:
        async with self._sessionmaker() as session:
            row = (await session.execute(text(target.manifest_sql))).mappings().one()
        return SourceManifest(count=int(row["count"]), md5=str(row["manifest_md5"]))

    async def fetch_batch(
        self, target: TargetSpec, *, after_id: str | None, limit: int
    ) -> list[SourceRow]:
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(
                    text(target.fetch_sql),
                    {"after_id": after_id, "limit": limit},
                )
            ).mappings().all()
        return [SourceRow(id=str(row["id"]), text=str(row["source_text"])) for row in rows]

    async def commit_batch(
        self,
        target: TargetSpec,
        rows: Sequence[SourceRow],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(rows) != len(vectors):
            raise ReembeddingError("row/vector batch lengths differ")
        async with self._sessionmaker() as session, session.begin():
            for row, vector in zip(rows, vectors, strict=True):
                result = await session.execute(
                    text(target.update_sql),
                    {
                        "id": row.id,
                        "source_text": row.text,
                        "embedding": json.dumps(vector, separators=(",", ":")),
                    },
                )
                if result.rowcount != 1:  # type: ignore[attr-defined]
                    raise ReembeddingError(
                        f"{target.key}: source row changed during re-embedding"
                    )

    async def verify(self, target: TargetSpec, dimension: int) -> Verification:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(text(target.verify_sql), {"dimension": dimension})
            ).mappings().one()
        return Verification(
            eligible=int(row["eligible"]),
            non_null=int(row["non_null"]),
            wrong_dimension=int(row["wrong_dimension"]),
        )

    async def resume_state(
        self, target: TargetSpec, last_id: str | None
    ) -> ResumeState:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(text(target.resume_sql), {"last_id": last_id})
            ).mappings().one()
        return ResumeState(
            count_through_id=int(row["count_through_id"]),
            max_eligible_id=(
                str(row["max_eligible_id"])
                if row["max_eligible_id"] is not None
                else None
            ),
            cursor_exists=bool(row["cursor_exists"]),
        )

    async def vector_manifest(self, target: TargetSpec) -> str:
        async with self._sessionmaker() as session:
            value = (
                await session.execute(text(target.vector_manifest_sql))
            ).scalar_one()
        return self._validated_vector_manifest(target, value)

    async def prefix_vector_manifest(
        self, target: TargetSpec, last_id: str
    ) -> str:
        async with self._sessionmaker() as session:
            value = (
                await session.execute(
                    text(target.prefix_vector_manifest_sql),
                    {"last_id": last_id},
                )
            ).scalar_one()
        return self._validated_vector_manifest(target, value)

    @staticmethod
    def _validated_vector_manifest(target: TargetSpec, value: object) -> str:
        manifest = str(value)
        if not re.fullmatch(r"[0-9a-f]{32}", manifest):
            raise ReembeddingError(f"{target.key}: vector manifest is invalid")
        return manifest


def _checkpoint_identity(data: Mapping[str, object]) -> RunIdentity:
    identity = data.get("identity")
    if not isinstance(identity, dict):
        raise ReembeddingError("checkpoint has no identity")
    try:
        model = identity["model"]
        model_revision = identity["model_revision"]
        profile = identity["profile"]
        dimension = identity["dimension"]
        native_dimension = identity["native_dimension"]
        endpoint_sha256 = identity["endpoint_sha256"]
        database_sha256 = identity["database_sha256"]
    except KeyError as exc:
        raise ReembeddingError("checkpoint identity is incomplete") from exc
    if (
        not isinstance(model, str)
        or not isinstance(model_revision, str)
        or not isinstance(profile, str)
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or isinstance(native_dimension, bool)
        or not isinstance(native_dimension, int)
        or not isinstance(endpoint_sha256, str)
        or not isinstance(database_sha256, str)
    ):
        raise ReembeddingError("checkpoint identity is invalid")
    if (
        not model
        or not re.fullmatch(r"[0-9a-f]{40}", model_revision)
        or profile not in {"qwen3", "nemotron3"}
        or dimension <= 0
        or native_dimension < dimension
        or not re.fullmatch(r"[0-9a-f]{64}", endpoint_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", database_sha256)
    ):
        raise ReembeddingError("checkpoint identity values are invalid")
    return RunIdentity(
        model,
        model_revision,
        profile,
        dimension,
        native_dimension,
        endpoint_sha256,
        database_sha256,
    )


def checkpoint_payload_sha256(data: Mapping[str, object]) -> str:
    unsigned = dict(data)
    unsigned.pop("payload_sha256", None)
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    try:
        info = expanded.lstat()
    except FileNotFoundError as exc:
        raise ReembeddingError("checkpoint does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ReembeddingError("checkpoint must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ReembeddingError("checkpoint permissions must not allow group/other access")
    if info.st_size > _CHECKPOINT_MAX_BYTES:
        raise ReembeddingError("checkpoint is too large")
    try:
        decoded = json.loads(expanded.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReembeddingError("checkpoint cannot be read") from exc
    if not isinstance(decoded, dict):
        raise ReembeddingError("checkpoint payload is not an object")
    recorded_hash = decoded.get("payload_sha256")
    if (
        not isinstance(recorded_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_hash)
        or recorded_hash != checkpoint_payload_sha256(decoded)
    ):
        raise ReembeddingError("checkpoint payload hash mismatch")
    if decoded.get("schema") != _CHECKPOINT_SCHEMA:
        raise ReembeddingError("checkpoint schema is invalid")
    _checkpoint_identity(decoded)
    if not isinstance(decoded.get("targets"), dict):
        raise ReembeddingError("checkpoint target state is invalid")
    return decoded


def save_checkpoint(path: Path, data: Mapping[str, object]) -> None:
    expanded = path.expanduser()
    parent = expanded.parent
    if not parent.is_dir():
        raise ReembeddingError("checkpoint parent directory does not exist")
    if expanded.exists() or expanded.is_symlink():
        info = expanded.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ReembeddingError("checkpoint must be a regular non-symlink file")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ReembeddingError("checkpoint permissions must be private")
    signed_data = dict(data)
    signed_data["payload_sha256"] = checkpoint_payload_sha256(signed_data)
    payload = (
        json.dumps(
            signed_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _CHECKPOINT_MAX_BYTES:
        raise ReembeddingError("checkpoint payload is too large")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{expanded.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, expanded)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _new_checkpoint(
    identity: RunIdentity,
    manifests: Mapping[str, SourceManifest],
) -> dict[str, Any]:
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "identity": identity.as_dict(),
        "targets": {
            key: {
                "source_count": manifest.count,
                "source_manifest_md5": manifest.md5,
                "last_id": None,
                "processed": 0,
                "complete": False,
                "prefix_vector_manifest_md5": None,
                "vector_manifest_md5": None,
            }
            for key, manifest in manifests.items()
        },
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, object],
    identity: RunIdentity,
    manifests: Mapping[str, SourceManifest],
) -> None:
    if _checkpoint_identity(checkpoint) != identity:
        raise ReembeddingError("checkpoint belongs to another model/profile/endpoint")
    targets = checkpoint.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(manifests):
        raise ReembeddingError("checkpoint target set does not match this run")
    for key, manifest in manifests.items():
        progress = targets.get(key)
        if not isinstance(progress, dict):
            raise ReembeddingError(f"{key}: checkpoint progress is invalid")
        if (
            progress.get("source_count") != manifest.count
            or progress.get("source_manifest_md5") != manifest.md5
        ):
            raise ReembeddingError(f"{key}: eligible source rows changed")
        last_id = progress.get("last_id")
        if last_id is not None:
            try:
                uuid.UUID(str(last_id))
            except ValueError as exc:
                raise ReembeddingError(f"{key}: checkpoint cursor is invalid") from exc
        processed = progress.get("processed")
        complete = progress.get("complete")
        prefix_vector_manifest = progress.get("prefix_vector_manifest_md5")
        vector_manifest = progress.get("vector_manifest_md5")
        if (
            isinstance(processed, bool)
            or not isinstance(processed, int)
            or processed < 0
            or processed > manifest.count
            or not isinstance(complete, bool)
        ):
            raise ReembeddingError(f"{key}: checkpoint counters are invalid")
        if (processed == 0) != (last_id is None):
            raise ReembeddingError(f"{key}: checkpoint cursor/count are inconsistent")
        if processed == 0:
            if prefix_vector_manifest is not None:
                raise ReembeddingError(
                    f"{key}: empty checkpoint prefix has a vector attestation"
                )
        elif not isinstance(prefix_vector_manifest, str) or not re.fullmatch(
            r"[0-9a-f]{32}", prefix_vector_manifest
        ):
            raise ReembeddingError(
                f"{key}: processed checkpoint prefix has no vector attestation"
            )
        if complete and processed != manifest.count:
            raise ReembeddingError(f"{key}: completed checkpoint has the wrong count")
        if complete:
            if not isinstance(vector_manifest, str) or not re.fullmatch(
                r"[0-9a-f]{32}", vector_manifest
            ):
                raise ReembeddingError(
                    f"{key}: completed checkpoint has no vector attestation"
                )
        elif vector_manifest is not None:
            raise ReembeddingError(
                f"{key}: incomplete checkpoint has a vector attestation"
            )


async def _validate_resume_against_database(
    repository: Repository,
    checkpoint: Mapping[str, object],
    manifests: Mapping[str, SourceManifest],
    targets: Sequence[TargetSpec],
) -> None:
    target_state = checkpoint.get("targets")
    if not isinstance(target_state, dict):
        raise ReembeddingError("checkpoint target state is invalid")
    for target in targets:
        progress = target_state.get(target.key)
        if not isinstance(progress, dict):
            raise ReembeddingError(f"{target.key}: checkpoint progress is invalid")
        last_id = progress["last_id"]
        processed = progress["processed"]
        complete = progress["complete"]
        state = await repository.resume_state(target, last_id)
        if state.count_through_id != processed:
            raise ReembeddingError(
                f"{target.key}: checkpoint cursor does not match database count"
            )
        if last_id is not None and not state.cursor_exists:
            raise ReembeddingError(
                f"{target.key}: checkpoint cursor is not an eligible row"
            )
        if last_id is not None:
            current_prefix = await repository.prefix_vector_manifest(
                target, last_id
            )
            if current_prefix != progress["prefix_vector_manifest_md5"]:
                raise ReembeddingError(
                    f"{target.key}: processed prefix vector attestation changed"
                )
        if complete:
            expected_last_id = state.max_eligible_id
            if manifests[target.key].count == 0:
                if last_id is not None or expected_last_id is not None:
                    raise ReembeddingError(
                        f"{target.key}: empty completed checkpoint has a cursor"
                    )
            elif last_id != expected_last_id:
                raise ReembeddingError(
                    f"{target.key}: completed checkpoint does not reach the final row"
                )
            current_vectors = await repository.vector_manifest(target)
            if current_vectors != progress["vector_manifest_md5"]:
                raise ReembeddingError(
                    f"{target.key}: completed checkpoint vector attestation changed"
                )


async def collect_manifests(
    repository: Repository, targets: Sequence[TargetSpec]
) -> dict[str, SourceManifest]:
    return {
        target.key: await repository.source_manifest(target)
        for target in targets
    }


async def apply_reembedding(
    *,
    repository: Repository,
    embed_batch: Callable[[Sequence[str]], Awaitable[list[list[float]]]],
    identity: RunIdentity,
    targets: Sequence[TargetSpec],
    batch_size: int,
    checkpoint_path: Path,
) -> dict[str, Verification]:
    if batch_size <= 0:
        raise ReembeddingError("batch size must be positive")
    manifests = await collect_manifests(repository, targets)
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        checkpoint = load_checkpoint(checkpoint_path)
        _validate_checkpoint(checkpoint, identity, manifests)
        await _validate_resume_against_database(
            repository, checkpoint, manifests, targets
        )
    else:
        checkpoint = _new_checkpoint(identity, manifests)
        save_checkpoint(checkpoint_path, checkpoint)

    target_state = checkpoint["targets"]
    if not isinstance(target_state, dict):
        raise ReembeddingError("checkpoint target state is invalid")
    for target in targets:
        progress = target_state[target.key]
        if not isinstance(progress, dict):
            raise ReembeddingError(f"{target.key}: checkpoint progress is invalid")
        if progress["complete"]:
            continue
        while True:
            latest_manifest = await repository.source_manifest(target)
            if latest_manifest != manifests[target.key]:
                raise ReembeddingError(f"{target.key}: source rows changed during run")
            rows = await repository.fetch_batch(
                target,
                after_id=progress["last_id"],
                limit=batch_size,
            )
            if not rows:
                break
            vectors = await embed_batch([row.text for row in rows])
            if len(vectors) != len(rows):
                raise ReembeddingError(f"{target.key}: endpoint returned wrong batch size")
            if any(len(vector) != identity.dimension for vector in vectors):
                raise ReembeddingError(f"{target.key}: endpoint returned wrong vector dimension")
            await repository.commit_batch(target, rows, vectors)
            progress["last_id"] = rows[-1].id
            progress["processed"] += len(rows)
            if progress["processed"] > manifests[target.key].count:
                raise ReembeddingError(f"{target.key}: processed count exceeded source count")
            progress[
                "prefix_vector_manifest_md5"
            ] = await repository.prefix_vector_manifest(
                target, progress["last_id"]
            )
            save_checkpoint(checkpoint_path, checkpoint)
        if progress["processed"] != manifests[target.key].count:
            raise ReembeddingError(f"{target.key}: not every eligible row was processed")
        final_state = await repository.resume_state(target, progress["last_id"])
        if (
            final_state.count_through_id != progress["processed"]
            or (
                manifests[target.key].count > 0
                and (
                    not final_state.cursor_exists
                    or progress["last_id"] != final_state.max_eligible_id
                )
            )
            or (
                manifests[target.key].count == 0
                and (
                    progress["last_id"] is not None
                    or final_state.max_eligible_id is not None
                )
            )
        ):
            raise ReembeddingError(f"{target.key}: final cursor verification failed")
        progress["vector_manifest_md5"] = await repository.vector_manifest(target)
        if manifests[target.key].count > 0 and (
            progress["prefix_vector_manifest_md5"]
            != progress["vector_manifest_md5"]
        ):
            raise ReembeddingError(
                f"{target.key}: final prefix/vector attestations differ"
            )
        progress["complete"] = True
        save_checkpoint(checkpoint_path, checkpoint)

    final_manifests = await collect_manifests(repository, targets)
    if final_manifests != manifests:
        raise ReembeddingError("source rows changed before final verification")
    verification = {
        target.key: await repository.verify(target, identity.dimension)
        for target in targets
    }
    for key, result in verification.items():
        expected = manifests[key].count
        if (
            result.eligible != expected
            or result.non_null != expected
            or result.wrong_dimension != 0
        ):
            raise ReembeddingError(f"{key}: final vector verification failed")
    return verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="In-place пересборка текстовых vector(1024); по умолчанию dry-run.",
        allow_abbrev=False,
    )
    parser.add_argument("--apply", action="store_true", help="разрешить запись в БД")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="приватный checkpoint; обязателен с --apply, существующий автоматически resume",
    )
    parser.add_argument("--env-file", type=Path, help="дополнительный KEY=VALUE файл")
    parser.add_argument(
        "--model-revision",
        help="неизменяемая ревизия весов; обязательна с --apply",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


async def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size < 1 or args.batch_size > 128:
        raise ReembeddingError("batch size must be between 1 and 128")
    if args.timeout <= 0 or args.timeout > 600:
        raise ReembeddingError("timeout must be between 0 and 600 seconds")
    if args.apply and args.checkpoint is None:
        raise ReembeddingError("--checkpoint is required with --apply")
    if not args.apply and args.checkpoint is not None:
        raise ReembeddingError("--checkpoint is only used with --apply")
    model_revision = validate_model_revision(
        args.model_revision,
        required=bool(args.apply),
    )
    if args.env_file is not None:
        load_env_file(args.env_file)

    from rag_app.config import Settings
    from rag_app.db.engine import create_engine, create_sessionmaker
    from rag_app.db.models import EMBEDDING_DIM, MEMORY_DIM
    from rag_app.db.rls import assert_worker_rls_role

    runtime = Settings()
    base_url = normalize_embedding_base_url(runtime.embed_base_url)
    if runtime.embed_dim != EMBEDDING_DIM or runtime.embed_dim != MEMORY_DIM:
        raise ReembeddingError("configured dimension does not match all vector columns")
    embedding_client = SafeEmbeddingClient(
        base_url=base_url,
        model=runtime.embed_model,
        profile=runtime.embed_input_profile,
        dimension=runtime.embed_dim,
        timeout_s=args.timeout,
    )
    engine = create_engine()
    try:
        await assert_worker_rls_role(engine)
        await embedding_client.preflight()
        native_dimension = embedding_client.native_dimension
        if native_dimension is None:
            raise ReembeddingError("embedding preflight returned no native dimension")
        identity = RunIdentity(
            model=runtime.embed_model,
            model_revision=model_revision,
            profile=runtime.embed_input_profile,
            dimension=runtime.embed_dim,
            native_dimension=native_dimension,
            endpoint_sha256=sha256_text(base_url),
            database_sha256=database_identity_sha256(runtime.database_url),
        )
        repository = SqlRepository(create_sessionmaker(engine))
        manifests = await collect_manifests(repository, TARGETS)
        if not args.apply:
            verification = {
                target.key: await repository.verify(target, identity.dimension)
                for target in TARGETS
            }
            return {
                "mode": "dry-run",
                "identity": identity.as_dict(),
                "targets": {
                    key: {
                        "eligible": manifest.count,
                        "source_manifest_md5": manifest.md5,
                        "current_non_null": verification[key].non_null,
                        "current_wrong_dimension": verification[key].wrong_dimension,
                    }
                    for key, manifest in manifests.items()
                },
            }
        assert args.checkpoint is not None
        async with engine.connect() as lock_connection:
            locked = bool(
                (
                    await lock_connection.execute(
                        text(
                            "SELECT pg_try_advisory_lock("
                            "hashtextextended('docragenslate-reembed-text-vectors', 0))"
                        )
                    )
                ).scalar_one()
            )
            if not locked:
                raise ReembeddingError("another text re-embedding run holds the database lock")
            try:
                applied = await apply_reembedding(
                    repository=repository,
                    embed_batch=embedding_client.embed,
                    identity=identity,
                    targets=TARGETS,
                    batch_size=args.batch_size,
                    checkpoint_path=args.checkpoint,
                )
            finally:
                await lock_connection.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtextextended('docragenslate-reembed-text-vectors', 0))"
                    )
                )
        return {
            "mode": "apply",
            "identity": identity.as_dict(),
            "targets": {
                key: {
                    "eligible": result.eligible,
                    "non_null": result.non_null,
                    "wrong_dimension": result.wrong_dimension,
                }
                for key, result in applied.items()
            },
        }
    finally:
        await embedding_client.close()
        await engine.dispose()


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 — CLI boundary must not leak DSN/secrets
        print(
            f"re-embedding failed ({type(exc).__name__}); details suppressed",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
