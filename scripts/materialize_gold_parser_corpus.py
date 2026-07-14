#!/usr/bin/env python3
"""Materialize the private Gold documents as a byte-pinned parser corpus."""

from __future__ import annotations

import argparse
import asyncio
import enum
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from rag_app.config import settings
from rag_app.db.engine import create_engine
from rag_app.db.models import Chunk, Document, DocumentStatus
from rag_app.eval.gold_set import (
    DocumentSnapshot,
    GoldRecord,
    load_gold_set,
    make_scope_id,
    parsed_chunks_sha256,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarRecord,
    SidecarDocument,
    bind_gold_sidecar,
    load_private_sidecar,
)
from rag_app.storage.s3 import Storage

_MAX_PRIVATE_INPUT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_DOCUMENT_BYTES = 512 * 1024 * 1024
_MAX_DOCUMENT_BYTES_LIMIT = 2 * 1024 * 1024 * 1024
_MAX_CONTROL_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_CONTROL_CHUNKS = 100_000
_MAX_OOXML_ENTRIES = 100_000
_MAX_CONTENT_TYPES_BYTES = 2 * 1024 * 1024
_PDF_HEADER = re.compile(rb"%PDF-(?:1\.[0-7]|2\.0)(?:\r\n|\r|\n)")
_ELIGIBLE_STATUS = DocumentStatus.done.value
_OOXML_MAIN_PARTS = {
    "word/document.xml",
    "xl/workbook.xml",
    "ppt/presentation.xml",
}


class CorpusMaterializationError(ValueError):
    """Fail-closed error whose message excludes private document metadata."""


@dataclass(frozen=True, slots=True)
class DocumentBinding:
    document_id: uuid.UUID
    snapshot: DocumentSnapshot


@dataclass(frozen=True, slots=True)
class ScopePlan:
    scope_id: str
    snapshots: tuple[DocumentSnapshot, ...]
    anchors: tuple[DocumentBinding, ...]


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    scopes: tuple[ScopePlan, ...]

    @property
    def snapshots(self) -> tuple[DocumentSnapshot, ...]:
        return tuple(snapshot for scope in self.scopes for snapshot in scope.snapshots)


@dataclass(frozen=True, slots=True)
class ScopeRequest:
    scope_id: str
    document_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True, slots=True)
class ResolvedScope:
    scope_id: str
    rows: tuple[DocumentRow, ...]


@dataclass(frozen=True, slots=True)
class DocumentRow:
    document_id: uuid.UUID
    s3_key_original: str
    page_count: int | None
    status: str


@dataclass(frozen=True, slots=True)
class ControlChunkRow:
    document_id: uuid.UUID
    idx: int
    kind: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    text_en: str


@dataclass(frozen=True, slots=True)
class _InputGuard:
    path: Path
    identity: tuple[int, int, int, int, int, int]

    def assert_unchanged(self) -> None:
        try:
            info = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise CorpusMaterializationError("private input disappeared while being loaded") from exc
        if _file_identity(info) != self.identity:
            raise CorpusMaterializationError("private input changed while being loaded")


class DocumentResolver(Protocol):
    async def resolve(self, scopes: tuple[ScopeRequest, ...]) -> Sequence[ResolvedScope]: ...

    async def resolve_control_chunks(
        self, document_ids: tuple[uuid.UUID, ...]
    ) -> Sequence[ControlChunkRow]: ...


class ObjectStorage(Protocol):
    async def get_bytes(self, bucket: str, key: str) -> bytes: ...


class DatabaseDocumentResolver:
    """Resolve the one-owner ``done`` corpus selected by private sidecar anchors."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def resolve(self, scopes: tuple[ScopeRequest, ...]) -> Sequence[ResolvedScope]:
        if not scopes:
            return ()
        try:
            resolved: list[ResolvedScope] = []
            owner_by_scope: dict[str, str] = {}
            async with self._engine.connect() as connection:
                async with connection.begin():
                    await connection.execute(
                        text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                    )
                    for scope in scopes:
                        anchor_result = await connection.execute(
                            select(Document.id, Document.owner_sub).where(
                                Document.id.in_(scope.document_ids)
                            )
                        )
                        owner_sub = _single_anchor_owner(
                            scope.document_ids,
                            tuple((row.id, row.owner_sub) for row in anchor_result.all()),
                        )
                        _register_scope_owner(owner_by_scope, scope.scope_id, owner_sub)
                        result = await connection.execute(
                            select(
                                Document.id,
                                Document.s3_key_original,
                                Document.page_count,
                                Document.status,
                            ).where(
                                Document.owner_sub == owner_sub,
                                Document.status == DocumentStatus.done,
                            )
                        )
                        resolved.append(
                            ResolvedScope(
                                scope.scope_id,
                                tuple(
                                    DocumentRow(
                                        document_id=row.id,
                                        s3_key_original=row.s3_key_original,
                                        page_count=row.page_count,
                                        status=_status_value(row.status),
                                    )
                                    for row in result.all()
                                ),
                            )
                        )
        except CorpusMaterializationError:
            raise
        except Exception as exc:
            raise CorpusMaterializationError(
                f"read-only document lookup failed ({type(exc).__name__})"
            ) from None
        return tuple(resolved)

    async def resolve_control_chunks(
        self, document_ids: tuple[uuid.UUID, ...]
    ) -> Sequence[ControlChunkRow]:
        if not document_ids:
            return ()
        try:
            async with self._engine.connect() as connection:
                async with connection.begin():
                    await connection.execute(
                        text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                    )
                    result = await connection.execute(
                        select(
                            Chunk.document_id,
                            Chunk.idx,
                            Chunk.kind,
                            Chunk.heading_path,
                            Chunk.page_start,
                            Chunk.page_end,
                            Chunk.text_en,
                        )
                        .where(Chunk.document_id.in_(document_ids))
                        .order_by(Chunk.document_id, Chunk.idx)
                    )
                    rows = result.all()
        except Exception as exc:
            raise CorpusMaterializationError(
                f"read-only control chunk lookup failed ({type(exc).__name__})"
            ) from None
        return tuple(
            ControlChunkRow(
                document_id=row.document_id,
                idx=row.idx,
                kind=row.kind or "section",
                heading_path=row.heading_path or "",
                page_start=row.page_start,
                page_end=row.page_end,
                text_en=(row.text_en or "").strip(),
            )
            for row in rows
        )


def _single_anchor_owner(
    document_ids: Sequence[uuid.UUID],
    owner_rows: Sequence[tuple[uuid.UUID, str]],
) -> str:
    expected_ids = set(document_ids)
    rows_by_id: dict[uuid.UUID, str] = {}
    for document_id, owner_sub in owner_rows:
        if document_id in rows_by_id:
            raise CorpusMaterializationError("anchor lookup returned duplicate document rows")
        rows_by_id[document_id] = owner_sub
    if set(rows_by_id) != expected_ids:
        raise CorpusMaterializationError("anchor lookup did not return the exact sidecar document set")
    owners = set(rows_by_id.values())
    if len(owners) != 1:
        raise CorpusMaterializationError("sidecar anchor documents do not have one owner")
    owner_sub = next(iter(owners))
    if not owner_sub:
        raise CorpusMaterializationError("sidecar anchor documents do not have one owner")
    return owner_sub


def _register_scope_owner(owner_by_scope: dict[str, str], scope_id: str, owner_sub: str) -> None:
    if scope_id in owner_by_scope:
        raise CorpusMaterializationError("resolver received a duplicate Gold scope")
    if owner_sub in owner_by_scope.values():
        raise CorpusMaterializationError("distinct Gold scopes resolve to the same owner")
    if make_scope_id(owner_sub) != scope_id:
        raise CorpusMaterializationError("database owner does not match the Gold scope")
    owner_by_scope[scope_id] = owner_sub


def _status_value(value: object) -> str:
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        stat.S_IMODE(info.st_mode),
    )


def _require_private_input(path: Path, *, label: str) -> _InputGuard:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        resolved = absolute.resolve(strict=True)
        info = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise CorpusMaterializationError(f"{label} is unavailable ({type(exc).__name__})") from None
    if resolved != absolute or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CorpusMaterializationError(f"{label} must be a real regular file without symlinks")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise CorpusMaterializationError(f"{label} permissions must be exactly 0600")
    if info.st_size < 1 or info.st_size > _MAX_PRIVATE_INPUT_BYTES:
        raise CorpusMaterializationError(f"{label} size is outside the accepted range")
    return _InputGuard(absolute, _file_identity(info))


def _index_snapshots(records: Sequence[GoldRecord]) -> dict[str, DocumentSnapshot]:
    snapshots: dict[str, DocumentSnapshot] = {}
    for record in records:
        for snapshot in record.document_scope:
            previous = snapshots.get(snapshot.document_ref)
            if previous is not None and previous != snapshot:
                raise CorpusMaterializationError("Gold contains conflicting document snapshots")
            snapshots[snapshot.document_ref] = snapshot
    return snapshots


def _add_mapping(
    by_id: dict[uuid.UUID, DocumentBinding],
    id_by_ref: dict[str, uuid.UUID],
    snapshots: Mapping[str, DocumentSnapshot],
    document: SidecarDocument,
) -> None:
    snapshot = snapshots.get(document.document_ref)
    if snapshot is None:
        raise CorpusMaterializationError("sidecar document has no matching Gold snapshot")
    previous = by_id.get(document.document_id)
    if previous is not None and previous.snapshot.document_ref != document.document_ref:
        raise CorpusMaterializationError("one document ID maps to conflicting document references")
    previous_id = id_by_ref.get(document.document_ref)
    if previous_id is not None and previous_id != document.document_id:
        raise CorpusMaterializationError("one document reference maps to conflicting document IDs")
    by_id[document.document_id] = DocumentBinding(document.document_id, snapshot)
    id_by_ref[document.document_ref] = document.document_id


def collect_corpus_plan(
    records: Sequence[GoldRecord],
    bound_sidecars: Mapping[str, PrivateSidecarRecord],
) -> CorpusPlan:
    record_by_case = {record.case_id: record for record in records}
    if len(record_by_case) != len(records) or set(bound_sidecars) != set(record_by_case):
        raise CorpusMaterializationError("bound sidecar cases do not exactly match the Gold release")
    records_by_scope: dict[str, list[GoldRecord]] = {}
    for record in records:
        records_by_scope.setdefault(record.scope_id, []).append(record)

    scopes: list[ScopePlan] = []
    global_refs: set[str] = set()
    global_anchor_ids: set[uuid.UUID] = set()
    for scope_id in sorted(records_by_scope):
        scope_records = records_by_scope[scope_id]
        snapshots = _index_snapshots(scope_records)
        by_id: dict[uuid.UUID, DocumentBinding] = {}
        id_by_ref: dict[str, uuid.UUID] = {}
        for record in sorted(scope_records, key=lambda item: item.case_id):
            for document in bound_sidecars[record.case_id].source_documents:
                _add_mapping(by_id, id_by_ref, snapshots, document)
        if not by_id:
            raise CorpusMaterializationError("Gold scope has no private sidecar anchors")
        if global_refs.intersection(snapshots):
            raise CorpusMaterializationError("a document snapshot is shared across Gold scopes")
        if global_anchor_ids.intersection(by_id):
            raise CorpusMaterializationError("a sidecar anchor ID is shared across Gold scopes")
        global_refs.update(snapshots)
        global_anchor_ids.update(by_id)
        scopes.append(
            ScopePlan(
                scope_id=scope_id,
                snapshots=tuple(sorted(snapshots.values(), key=lambda item: item.source_sha256)),
                anchors=tuple(sorted(by_id.values(), key=lambda item: item.snapshot.source_sha256)),
            )
        )
    if not scopes:
        raise CorpusMaterializationError("Gold release contains no scopes")
    return CorpusPlan(scopes=tuple(scopes))


def load_bound_corpus_plan(
    gold_path: Path,
    sidecar_path: Path,
    *,
    repository_root: Path,
) -> CorpusPlan:
    gold_guard = _require_private_input(gold_path, label="Gold release")
    sidecar_guard = _require_private_input(sidecar_path, label="generator sidecar")
    records, _ = load_gold_set(gold_guard.path, mode="release", repository_root=repository_root)
    sidecars = load_private_sidecar(sidecar_guard.path, repository_root=repository_root)
    bound = bind_gold_sidecar(records, sidecars)
    gold_guard.assert_unchanged()
    sidecar_guard.assert_unchanged()
    return collect_corpus_plan(records, bound)


def _require_output_path(output_dir: Path) -> Path:
    output = Path(os.path.abspath(output_dir.expanduser()))
    parent = output.parent
    try:
        resolved_parent = parent.resolve(strict=True)
        parent_info = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise CorpusMaterializationError(
            f"output parent is unavailable ({type(exc).__name__})"
        ) from None
    if (
        resolved_parent != parent
        or stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
    ):
        raise CorpusMaterializationError("output parent must be a real directory without symlinks")
    if os.path.lexists(output):
        raise CorpusMaterializationError("output corpus path must not already exist")
    return output


def _atomic_private_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o600:
            raise CorpusMaterializationError("private output file permissions are not 0600")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_pdf(payload: bytes, *, expected_page_count: int) -> None:
    if _PDF_HEADER.match(payload) is None or b"%%EOF" not in payload[-1024:]:
        raise CorpusMaterializationError("original object is not a structurally recognizable PDF")
    try:
        document = pdfium.PdfDocument(payload)
        try:
            actual_page_count = len(document)
        finally:
            document.close()
    except Exception:
        raise CorpusMaterializationError("original object cannot be decoded as PDF") from None
    if actual_page_count != expected_page_count:
        raise CorpusMaterializationError("PDF page count does not match the Gold snapshot")


def _validate_ooxml(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > _MAX_OOXML_ENTRIES:
                raise CorpusMaterializationError("OOXML entry count is outside the accepted range")
            names: set[str] = set()
            for entry in entries:
                name = entry.filename
                normalized = name[:-1] if entry.is_dir() and name.endswith("/") else name
                parts = normalized.replace("\\", "/").split("/")
                if (
                    name in names
                    or not normalized
                    or name.startswith(("/", "\\"))
                    or "\\" in name
                    or any(part in {"", ".", ".."} for part in parts)
                    or entry.flag_bits & 0x1
                ):
                    raise CorpusMaterializationError("OOXML archive structure is unsafe")
                names.add(name)
            if "[Content_Types].xml" not in names or not names.intersection(_OOXML_MAIN_PARTS):
                raise CorpusMaterializationError("ZIP object is not a supported OOXML document")
            content_types_info = archive.getinfo("[Content_Types].xml")
            if content_types_info.file_size > _MAX_CONTENT_TYPES_BYTES:
                raise CorpusMaterializationError("OOXML content types part is too large")
            content_types = archive.read(content_types_info)
            ET.fromstring(content_types)
    except CorpusMaterializationError:
        raise
    except (ET.ParseError, OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        raise CorpusMaterializationError("original object cannot be decoded as OOXML") from None


def _source_format(payload: bytes) -> str:
    if _PDF_HEADER.match(payload) is not None:
        return "pdf"
    if payload.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        _validate_ooxml(payload)
        return "ooxml"
    raise CorpusMaterializationError("original object format is neither PDF nor supported OOXML")


def _build_control_artifact(
    controls: Sequence[tuple[DocumentRow, DocumentSnapshot]],
    chunk_rows: Sequence[ControlChunkRow],
) -> bytes:
    expected_ids = {row.document_id for row, _snapshot in controls}
    rows_by_document: dict[uuid.UUID, list[ControlChunkRow]] = {
        document_id: [] for document_id in expected_ids
    }
    if len(expected_ids) != len(controls):
        raise CorpusMaterializationError("control documents must be unique")
    if len(chunk_rows) > _MAX_CONTROL_CHUNKS:
        raise CorpusMaterializationError("control chunk count exceeds the accepted limit")
    for chunk in chunk_rows:
        target = rows_by_document.get(chunk.document_id)
        if target is None:
            raise CorpusMaterializationError("chunk lookup returned a document outside controls")
        target.append(chunk)

    artifact_controls: list[dict[str, Any]] = []
    for row, snapshot in sorted(controls, key=lambda item: item[1].source_sha256):
        chunks = sorted(rows_by_document[row.document_id], key=lambda item: item.idx)
        if not chunks or len({item.idx for item in chunks}) != len(chunks):
            raise CorpusMaterializationError("control chunks must be non-empty with unique indices")
        serialized_chunks = [
            {
                "idx": chunk.idx,
                "kind": chunk.kind,
                "heading_path": chunk.heading_path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text_en,
            }
            for chunk in chunks
        ]
        if parsed_chunks_sha256(serialized_chunks) != snapshot.parsed_content_sha256:
            raise CorpusMaterializationError("production chunks do not match the Gold parsed snapshot")
        artifact_controls.append(
            {
                "source_sha256": snapshot.source_sha256,
                "page_count": snapshot.page_count,
                "parsed_content_sha256": snapshot.parsed_content_sha256,
                "chunks": serialized_chunks,
            }
        )
    payload = _canonical_json_bytes(
        {
            "schema_version": 1,
            "source": "private-rag-gold-ooxml-controls",
            "controls": artifact_controls,
        }
    )
    if len(payload) > _MAX_CONTROL_ARTIFACT_BYTES:
        raise CorpusMaterializationError("control artifact exceeds the accepted size")
    return payload


def _validate_scope_rows(
    scope: ScopePlan, rows: Sequence[DocumentRow]
) -> dict[uuid.UUID, DocumentRow]:
    rows_by_id: dict[uuid.UUID, DocumentRow] = {}
    for row in rows:
        if row.document_id in rows_by_id:
            raise CorpusMaterializationError("read-only lookup returned duplicate document rows")
        rows_by_id[row.document_id] = row
    if len(rows_by_id) != len(scope.snapshots):
        raise CorpusMaterializationError("owner document count does not match its exact Gold scope")
    for row in rows_by_id.values():
        if not row.s3_key_original:
            raise CorpusMaterializationError("Gold document has no original object key")
        if row.status != _ELIGIBLE_STATUS:
            raise CorpusMaterializationError("owner document is not in done status")
    if not {anchor.document_id for anchor in scope.anchors} <= set(rows_by_id):
        raise CorpusMaterializationError("owner document set is missing a sidecar anchor")
    return rows_by_id


def _validate_resolved_scopes(
    plan: CorpusPlan, resolved: Sequence[ResolvedScope]
) -> dict[str, dict[uuid.UUID, DocumentRow]]:
    plan_by_id = {scope.scope_id: scope for scope in plan.scopes}
    resolved_by_id: dict[str, ResolvedScope] = {}
    for scope in resolved:
        if scope.scope_id in resolved_by_id:
            raise CorpusMaterializationError("resolver returned a duplicate Gold scope")
        resolved_by_id[scope.scope_id] = scope
    if set(resolved_by_id) != set(plan_by_id):
        raise CorpusMaterializationError("resolver did not return the exact Gold scope set")
    return {
        scope_id: _validate_scope_rows(plan_by_id[scope_id], resolved_by_id[scope_id].rows)
        for scope_id in plan_by_id
    }


async def materialize_gold_parser_corpus(
    plan: CorpusPlan,
    output_dir: Path,
    *,
    resolver: DocumentResolver,
    storage: ObjectStorage,
    bucket_originals: str,
    expected_documents: int = 10,
    expected_pdfs: int = 7,
    expected_controls: int = 3,
    max_document_bytes: int = _DEFAULT_MAX_DOCUMENT_BYTES,
) -> Path:
    if min(expected_documents, expected_pdfs, expected_controls) < 0 or expected_documents < 1:
        raise CorpusMaterializationError("expected corpus counts are outside the accepted range")
    if expected_pdfs + expected_controls != expected_documents:
        raise CorpusMaterializationError("expected PDF and control counts do not sum to total documents")
    if len(plan.snapshots) != expected_documents:
        raise CorpusMaterializationError("bound Gold document count does not match the expected count")
    if not plan.scopes or len({scope.scope_id for scope in plan.scopes}) != len(plan.scopes):
        raise CorpusMaterializationError("Gold scopes must be non-empty and unique")
    if not plan.snapshots or len({item.document_ref for item in plan.snapshots}) != len(plan.snapshots):
        raise CorpusMaterializationError("Gold snapshots must be non-empty and unique")
    if len({item.source_sha256 for item in plan.snapshots}) != len(plan.snapshots):
        raise CorpusMaterializationError("Gold source hashes must be unique")
    all_anchor_ids: list[uuid.UUID] = []
    for scope in plan.scopes:
        if not scope.snapshots or not scope.anchors:
            raise CorpusMaterializationError("each Gold scope needs snapshots and sidecar anchors")
        scope_refs = {item.document_ref for item in scope.snapshots}
        if len(scope_refs) != len(scope.snapshots):
            raise CorpusMaterializationError("Gold scope snapshots must be unique")
        if any(item.snapshot.document_ref not in scope_refs for item in scope.anchors):
            raise CorpusMaterializationError("sidecar anchor is outside its exact Gold scope")
        all_anchor_ids.extend(item.document_id for item in scope.anchors)
    if len(set(all_anchor_ids)) != len(all_anchor_ids):
        raise CorpusMaterializationError("sidecar anchor IDs must be globally unique")
    if not 1 <= max_document_bytes <= _MAX_DOCUMENT_BYTES_LIMIT:
        raise CorpusMaterializationError("maximum document size is outside the accepted range")

    output = _require_output_path(output_dir)
    requests = tuple(
        ScopeRequest(
            scope.scope_id,
            tuple(sorted((item.document_id for item in scope.anchors), key=lambda item: item.hex)),
        )
        for scope in sorted(plan.scopes, key=lambda item: item.scope_id)
    )
    try:
        resolved = await resolver.resolve(requests)
    except CorpusMaterializationError:
        raise
    except Exception as exc:
        raise CorpusMaterializationError(
            f"read-only document lookup failed ({type(exc).__name__})"
        ) from None
    rows_by_scope = _validate_resolved_scopes(plan, resolved)

    output_created = False
    try:
        try:
            output.mkdir(mode=0o700)
            output_created = True
        except OSError as exc:
            raise CorpusMaterializationError(
                f"unable to create fresh output corpus ({type(exc).__name__})"
            ) from None
        os.chmod(output, 0o700)
        pages_by_sha256: dict[str, dict[str, Any]] = {}
        matched_sha256: set[str] = set()
        control_documents: list[tuple[DocumentRow, DocumentSnapshot]] = []
        snapshot_by_sha256 = {item.source_sha256: item for item in plan.snapshots}
        for scope in plan.scopes:
            scope_snapshot_by_sha256 = {item.source_sha256: item for item in scope.snapshots}
            anchor_sha256_by_id = {
                item.document_id: item.snapshot.source_sha256 for item in scope.anchors
            }
            for row in rows_by_scope[scope.scope_id].values():
                try:
                    payload = await storage.get_bytes(bucket_originals, row.s3_key_original)
                except Exception as exc:
                    raise CorpusMaterializationError(
                        f"original object read failed ({type(exc).__name__})"
                    ) from None
                if len(payload) < 1 or len(payload) > max_document_bytes:
                    raise CorpusMaterializationError("original object size is outside the accepted range")
                actual_sha256 = hashlib.sha256(payload).hexdigest()
                snapshot = scope_snapshot_by_sha256.get(actual_sha256)
                if snapshot is None:
                    raise CorpusMaterializationError("owner document is outside its exact Gold scope")
                if actual_sha256 in matched_sha256:
                    raise CorpusMaterializationError("multiple owner documents match one Gold snapshot")
                matched_sha256.add(actual_sha256)
                anchor_sha256 = anchor_sha256_by_id.get(row.document_id)
                if anchor_sha256 is not None and anchor_sha256 != actual_sha256:
                    raise CorpusMaterializationError(
                        "sidecar anchor ID does not match its Gold document reference"
                    )
                if row.page_count is not None and row.page_count != snapshot.page_count:
                    raise CorpusMaterializationError(
                        "database page count does not match the Gold snapshot "
                        f"(database={row.page_count!r}, Gold={snapshot.page_count})"
                    )
                source_format = await asyncio.to_thread(_source_format, payload)
                if source_format == "pdf":
                    await asyncio.to_thread(
                        _validate_pdf,
                        payload,
                        expected_page_count=snapshot.page_count,
                    )
                    filename = f"{actual_sha256}.pdf"
                    _atomic_private_write(output / filename, payload)
                    pages_by_sha256[actual_sha256] = {
                        "file": filename,
                        "sha256": actual_sha256,
                        "category": "layout",
                        "selection": {
                            "document_ref": snapshot.document_ref,
                            "page_count": snapshot.page_count,
                        },
                    }
                elif source_format == "ooxml":
                    control_documents.append((row, snapshot))
                else:
                    raise CorpusMaterializationError("unsupported source format classification")

        if matched_sha256 != set(snapshot_by_sha256):
            raise CorpusMaterializationError("owner documents do not match every Gold snapshot exactly once")
        if len(pages_by_sha256) != expected_pdfs or len(control_documents) != expected_controls:
            raise CorpusMaterializationError("PDF and OOXML counts do not match the expected corpus split")

        control_ids = tuple(
            sorted((row.document_id for row, _snapshot in control_documents), key=lambda item: item.hex)
        )
        try:
            control_chunks = await resolver.resolve_control_chunks(control_ids)
        except CorpusMaterializationError:
            raise
        except Exception as exc:
            raise CorpusMaterializationError(
                f"read-only control chunk lookup failed ({type(exc).__name__})"
            ) from None
        controls_payload = _build_control_artifact(control_documents, control_chunks)
        controls_sha256 = hashlib.sha256(controls_payload).hexdigest()
        controls_path = output / "controls.json"
        _atomic_private_write(controls_path, controls_payload)

        pages = [
            pages_by_sha256[item.source_sha256]
            for item in plan.snapshots
            if item.source_sha256 in pages_by_sha256
        ]

        revision_payload = {
            "pdfs": [
                {
                    "document_ref": item["selection"]["document_ref"],
                    "page_count": item["selection"]["page_count"],
                    "sha256": item["sha256"],
                }
                for item in pages
            ],
            "controls": {
                "sha256": controls_sha256,
                "count": len(control_documents),
            },
        }
        source_revision = hashlib.sha256(_canonical_json_bytes(revision_payload)).hexdigest()
        manifest = {
            "manifest_version": 1,
            "source": "private-rag-gold-release",
            "source_revision": source_revision,
            "pages": pages,
            "controls": {
                "file": "controls.json",
                "sha256": controls_sha256,
                "count": len(control_documents),
            },
        }
        manifest_path = output / "manifest.json"
        _atomic_private_write(manifest_path, _canonical_json_bytes(manifest))
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest_path
    except BaseException:
        if not output_created:
            raise
        try:
            shutil.rmtree(output)
        except OSError:
            raise CorpusMaterializationError("failed to remove an incomplete private corpus") from None
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Private reviewed Gold release JSONL (0600)")
    parser.add_argument(
        "--sidecar",
        type=Path,
        required=True,
        help="Private generator sidecar JSONL paired with the release (0600)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh private corpus directory")
    parser.add_argument("--expected-documents", type=int, default=10)
    parser.add_argument("--expected-pdfs", type=int, default=7)
    parser.add_argument("--expected-controls", type=int, default=3)
    parser.add_argument(
        "--max-document-bytes",
        type=int,
        default=_DEFAULT_MAX_DOCUMENT_BYTES,
        help="Per-document byte ceiling (default: 512 MiB)",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> Path:
    plan = load_bound_corpus_plan(
        args.gold,
        args.sidecar,
        repository_root=Path.cwd(),
    )
    engine = create_engine()
    try:
        return await materialize_gold_parser_corpus(
            plan,
            args.output_dir,
            resolver=DatabaseDocumentResolver(engine),
            storage=Storage(),
            bucket_originals=settings.bucket_originals,
            expected_documents=args.expected_documents,
            expected_pdfs=args.expected_pdfs,
            expected_controls=args.expected_controls,
            max_document_bytes=args.max_document_bytes,
        )
    finally:
        await engine.dispose()


def main() -> int:
    args = _parse_args()
    try:
        manifest_path = asyncio.run(_main(args))
    except (CorpusMaterializationError, ValueError) as exc:
        print(f"materialization rejected: {exc}", file=sys.stderr)
        return 2
    print(f"private parser corpus ready: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
