"""Fail-closed paired retrieval evaluation over two parser outputs.

The evaluator intentionally keeps private Gold text in memory only.  Its JSON
artifact contains hashes, numeric aggregates, and public model/runtime facts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from rag_app.config import settings
from rag_app.eval.gold_set import (
    GoldRecord,
    gold_record_case_sha256,
    make_document_ref,
    parsed_chunks_sha256,
)
from rag_app.eval.parser_rag_linkage import (
    ParserRagLinkageError,
    gold_document_snapshot,
    parser_corpus_snapshot,
)
from rag_app.eval.private_artifacts import (
    PrivateArtifactError,
    read_private_json,
    write_private_json_fresh,
)
from rag_app.eval.private_sidecar import PrivateSidecarRecord
from rag_app.pipeline.parse import (
    backfill_text_layer,
    load_content_list,
    pdf_info,
    read_pdf_text_by_page,
)
from rag_app.pipeline.segments import SegmentDraft, content_list_to_segments

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b")
_MAX_SUMMARY_BYTES = 64 * 1024 * 1024
_MAX_CONTENT_LIST_BYTES = 512 * 1024 * 1024
_MAX_CONTROLS_BYTES = 512 * 1024 * 1024
_MAX_CONTROL_CHUNKS = 100_000
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_EXPECTED_PDF_DOCUMENTS = 7
_EXPECTED_PDF_PAGES = 61
_EXPECTED_CONTROL_DOCUMENTS = 3
_MIN_BASELINE_COVERAGE = 0.50
_METRIC_NAMES = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "evidence_coverage",
    "page_coverage",
    "page_coverage_at_10",
)


class ShadowRetrievalError(ValueError):
    """A sanitized, fail-closed evaluation error."""


class EmbedderLike(Protocol):
    async def embed(self, texts: list[str], batch: int | None = None) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


class RerankerLike(Protocol):
    async def rerank(self, query: str, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class ShadowChunk:
    key: str
    document_ref: str
    source_sha256: str
    index: int
    kind: str
    text: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True, slots=True)
class EvidenceLocator:
    key: str
    document_ref: str
    page_start: int
    page_end: int
    grade: int
    source_text: str
    source_text_sha256: str
    source_anchors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_sha256: str
    query: str
    answerable: bool
    language: str
    hop_type: str
    content_types: tuple[str, ...]
    document_refs: frozenset[str]
    locators: tuple[EvidenceLocator, ...]


@dataclass(frozen=True, slots=True)
class RankedCase:
    case: RetrievalCase
    baseline: Mapping[str, float]
    candidate: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CorpusLoad:
    chunks: tuple[ShadowChunk, ...]
    manifest_sha256: str
    content_lists_sha256: str
    document_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class ControlCorpus:
    chunks: tuple[ShadowChunk, ...]
    documents: Mapping[str, int]
    pdf_documents: Mapping[str, int]
    artifact_sha256: str
    manifest_sha256: str
    chunks_manifest_sha256: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class PairEvaluation:
    ranked_cases: tuple[RankedCase, ...]
    baseline_embedding_latency_ms: float
    candidate_embedding_latency_ms: float


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_local_retrieval_endpoints() -> None:
    """Private Gold text may only be sent to loopback model services."""

    for value in (settings.embed_base_url, settings.rerank_base_url):
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ShadowRetrievalError("retrieval model endpoints must be loopback-only")


def load_benchmark_summary(path: Path) -> tuple[dict[str, Any], str]:
    """Load one immutable schema-v2 summary without reflecting its contents in errors."""

    source = path.expanduser()
    try:
        info = source.lstat()
    except OSError as error:
        raise ShadowRetrievalError(f"unable to stat benchmark summary ({type(error).__name__})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ShadowRetrievalError("benchmark summary must be a regular non-symlink file")
    if info.st_size > _MAX_SUMMARY_BYTES:
        raise ShadowRetrievalError("benchmark summary exceeds size limit")
    try:
        raw = source.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
        raise ShadowRetrievalError(f"invalid benchmark summary ({type(error).__name__})") from None
    if not isinstance(value, dict):
        raise ShadowRetrievalError("benchmark summary must be a JSON object")
    return cast(dict[str, Any], value), _sha256_bytes(raw)


def _normalize_text(value: str) -> str:
    return "\n".join(" ".join(line.split()) for line in value.splitlines() if line.strip()).strip()


def _normalize_match_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _source_anchors(value: str) -> tuple[str, ...]:
    """Build overlapping private anchors that survive deterministic chunk boundaries."""

    normalized = _normalize_match_text(value)
    if not normalized:
        return ()
    if len(normalized) <= 96:
        return (normalized,)
    words = normalized.split()
    anchors: list[str]
    if len(words) >= 8:
        window = 8
        starts = [*range(0, len(words) - window + 1, 4), len(words) - window]
        anchors = [" ".join(words[start : start + window]) for start in starts]
    else:
        window = 48
        starts = [*range(0, len(normalized) - window + 1, 24), len(normalized) - window]
        anchors = [normalized[start : start + window] for start in starts]
    return tuple(dict.fromkeys(anchors))


def source_evidence_manifest_sha256(source_text_by_chunk_id: Mapping[uuid.UUID, str]) -> str:
    """Hash private source evidence identities/content without serializing either."""

    rows: list[dict[str, str]] = []
    for chunk_id, source_text in sorted(source_text_by_chunk_id.items(), key=lambda item: str(item[0])):
        normalized = _normalize_match_text(source_text)
        if not normalized:
            raise ShadowRetrievalError("source evidence text must be non-empty")
        rows.append(
            {
                "chunk_identity_sha256": _sha256_bytes(str(chunk_id).encode("ascii")),
                "source_text_sha256": _sha256_bytes(normalized.encode("utf-8")),
            }
        )
    if not rows:
        raise ShadowRetrievalError("source evidence manifest must be non-empty")
    return _sha256_bytes(_canonical_bytes(rows))


def _split_bounded(text: str, limit: int) -> list[str]:
    if limit < 1:
        raise ShadowRetrievalError("chunk size must be positive")
    remaining = text.strip()
    parts: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < max(1, limit // 3):
            split_at = limit
        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _chunk_key(source_sha256: str, index: int, text: str, pages: tuple[int | None, int | None]) -> str:
    payload = {
        "index": index,
        "pages": pages,
        "source_sha256": source_sha256,
        "text_sha256": _sha256_bytes(text.encode()),
    }
    return _sha256_bytes(_canonical_bytes(payload))


def stable_text_chunks(
    drafts: Sequence[SegmentDraft],
    *,
    source_sha256: str,
    page_count: int,
    max_chars: int,
) -> list[ShadowChunk]:
    """Create deterministic, hard-bounded chunks without ORM or database state."""

    if _SHA256.fullmatch(source_sha256) is None or page_count < 1 or max_chars < 32:
        raise ShadowRetrievalError("invalid chunking inputs")
    document_ref = make_document_ref(source_sha256)
    chunks: list[ShadowChunk] = []
    headings: list[str] = []
    buffer: list[str] = []
    pages: list[int] = []

    def emit(text: str, kind: str, chunk_pages: Sequence[int]) -> None:
        normalized = text.strip()
        if not normalized:
            return
        for piece in _split_bounded(normalized, max_chars):
            start = min(chunk_pages) if chunk_pages else None
            end = max(chunk_pages) if chunk_pages else None
            index = len(chunks)
            chunks.append(
                ShadowChunk(
                    key=_chunk_key(source_sha256, index, piece, (start, end)),
                    document_ref=document_ref,
                    source_sha256=source_sha256,
                    index=index,
                    kind=kind,
                    text=piece,
                    page_start=start,
                    page_end=end,
                )
            )

    def flush() -> None:
        if buffer:
            emit("\n".join(buffer), "section", pages)
            buffer.clear()
            pages.clear()

    for draft in drafts:
        page = draft.page_idx
        if page is not None and (page < 0 or page >= page_count):
            raise ShadowRetrievalError("content_list page index exceeds benchmark page count")
        text = _normalize_text(draft.source_text or "")
        kind = str(draft.kind.value if hasattr(draft.kind, "value") else draft.kind)
        if kind == "heading":
            flush()
            level = max(draft.heading_level or 1, 1)
            del headings[level - 1 :]
            if text:
                headings.append(text)
                buffer.extend(_split_bounded(text, max_chars))
                if page is not None:
                    pages.append(page)
            continue
        if not text:
            continue
        if kind in {"table", "image"}:
            flush()
            prefix = " > ".join(headings)
            standalone = f"{prefix}\n{text}" if prefix else text
            emit(standalone, kind, [page] if page is not None else [])
            continue
        for piece in _split_bounded(text, max_chars):
            separator = 1 if buffer else 0
            current_size = sum(len(item) for item in buffer) + max(0, len(buffer) - 1)
            if buffer and current_size + separator + len(piece) > max_chars:
                flush()
            buffer.append(piece)
            if page is not None:
                pages.append(page)
    flush()
    if not chunks:
        raise ShadowRetrievalError("parser output produced no text chunks")
    if any(len(chunk.text) > max_chars for chunk in chunks):
        raise AssertionError("bounded chunk invariant failed")
    return chunks


def _secure_content_list(document_dir: Path) -> Path:
    try:
        info = document_dir.lstat()
    except OSError as error:
        raise ShadowRetrievalError(
            f"unable to stat parser output directory ({type(error).__name__})"
        ) from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ShadowRetrievalError("parser document output must be a regular directory")
    candidates = sorted(document_dir.rglob("*_content_list.json"))
    if len(candidates) != 1:
        raise ShadowRetrievalError("parser document output must contain exactly one content_list")
    candidate = candidates[0]
    parent = candidate.parent
    while parent != document_dir:
        try:
            parent_info = parent.lstat()
        except OSError as error:
            raise ShadowRetrievalError(
                f"unable to stat content_list parent ({type(error).__name__})"
            ) from None
        if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
            raise ShadowRetrievalError("content_list parents must be regular directories")
        if document_dir not in parent.parents:
            raise ShadowRetrievalError("content_list escapes its parser document directory")
        parent = parent.parent
    try:
        info = candidate.lstat()
    except OSError as error:
        raise ShadowRetrievalError(f"unable to stat content_list ({type(error).__name__})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ShadowRetrievalError("content_list must be a regular non-symlink file")
    if info.st_size > _MAX_CONTENT_LIST_BYTES:
        raise ShadowRetrievalError("content_list exceeds size limit")
    return candidate


def _benchmark_text_sha256(drafts: Sequence[SegmentDraft]) -> str:
    """Reproduce schema-v2 benchmark text hashing without importing its CLI."""

    manifest: list[dict[str, Any]] = []
    for draft in drafts:
        cells = draft.meta.get("table_cells")
        bbox_pt = draft.meta.get("bbox_pt")
        native_bbox = draft.meta.get("bbox")
        manifest.append(
            {
                "bbox": native_bbox if isinstance(native_bbox, list) else None,
                "bbox_pt": bbox_pt if isinstance(bbox_pt, list) else None,
                "heading_level": draft.heading_level,
                "idx": draft.idx,
                "kind": draft.kind.value,
                "page_idx": draft.page_idx,
                "source_text": draft.source_text,
                "table_cells": cells if isinstance(cells, list) else None,
            }
        )
    return _sha256_bytes(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _summary_result(
    summary: Mapping[str, Any],
    source_sha256: str,
) -> Mapping[str, Any]:
    results = summary.get("results")
    if not isinstance(results, Mapping):
        raise ShadowRetrievalError("benchmark summary results are invalid")
    matches = [
        (name, row)
        for name, row in results.items()
        if isinstance(row, Mapping) and row.get("source_sha256") == source_sha256
    ]
    if len(matches) != 1:
        raise ShadowRetrievalError("benchmark summary/output document binding is not exact")
    name, row = matches[0]
    if name != f"{source_sha256}.pdf":
        raise ShadowRetrievalError("benchmark summary PDF identity is invalid")
    backend = row.get("mineru")
    if not isinstance(backend, Mapping) or backend.get("status") != "ok":
        raise ShadowRetrievalError("benchmark summary MinerU result is invalid")
    return backend


def _secure_pdf(pdf_root: Path, source_sha256: str) -> Path:
    pdf_path = pdf_root / f"{source_sha256}.pdf"
    try:
        info = pdf_path.lstat()
    except OSError as error:
        raise ShadowRetrievalError(f"unable to stat private source PDF ({type(error).__name__})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ShadowRetrievalError("private source PDF must be a regular non-symlink file")
    return pdf_path


def load_parser_corpus(
    output_root: Path,
    documents: Mapping[str, int],
    *,
    summary: Mapping[str, Any],
    pdf_root: Path,
    max_chars: int,
) -> CorpusLoad:
    """Load and bind each output to its schema-v2 result and immutable source PDF."""

    root = output_root.expanduser()
    try:
        root_info = root.lstat()
    except OSError as error:
        raise ShadowRetrievalError(f"unable to stat parser output root ({type(error).__name__})") from None
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ShadowRetrievalError("parser output root must be a regular directory")
    mineru_root = root / "mineru"
    try:
        mineru_info = mineru_root.lstat()
    except OSError as error:
        raise ShadowRetrievalError(f"unable to stat MinerU output root ({type(error).__name__})") from None
    if stat.S_ISLNK(mineru_info.st_mode) or not stat.S_ISDIR(mineru_info.st_mode):
        raise ShadowRetrievalError("MinerU output root must be a regular directory")
    source_root = pdf_root.expanduser()
    try:
        source_root_info = source_root.lstat()
        output_entries = list(mineru_root.iterdir())
    except OSError as error:
        raise ShadowRetrievalError(
            f"unable to inspect parser corpus roots ({type(error).__name__})"
        ) from None
    if stat.S_ISLNK(source_root_info.st_mode) or not stat.S_ISDIR(source_root_info.st_mode):
        raise ShadowRetrievalError("private PDF root must be a regular directory")
    if {entry.name for entry in output_entries} != set(documents):
        raise ShadowRetrievalError("parser output root does not contain the exact benchmark documents")
    for entry in output_entries:
        entry_info = entry.lstat()
        if stat.S_ISLNK(entry_info.st_mode) or not stat.S_ISDIR(entry_info.st_mode):
            raise ShadowRetrievalError("parser output root contains an invalid document entry")

    chunks: list[ShadowChunk] = []
    content_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source_sha256, page_count in sorted(documents.items()):
        if _SHA256.fullmatch(source_sha256) is None:
            raise ShadowRetrievalError("parser corpus contains an invalid source digest")
        backend_result = _summary_result(summary, source_sha256)
        if backend_result.get("n_pages") != page_count:
            raise ShadowRetrievalError("benchmark summary page count does not match its corpus")
        raw_stats = backend_result.get("raw_stats")
        raw_text_sha256 = raw_stats.get("text_sha256") if isinstance(raw_stats, Mapping) else None
        final_text_sha256 = backend_result.get("text_sha256")
        if (
            not isinstance(raw_text_sha256, str)
            or _SHA256.fullmatch(raw_text_sha256) is None
            or not isinstance(final_text_sha256, str)
            or _SHA256.fullmatch(final_text_sha256) is None
        ):
            raise ShadowRetrievalError("benchmark summary text hashes are invalid")
        pdf_path = _secure_pdf(source_root, source_sha256)
        source_file_sha256 = _sha256_file(pdf_path)
        if source_file_sha256 != source_sha256:
            raise ShadowRetrievalError("private source PDF SHA does not match the benchmark")
        content_path = _secure_content_list(root / "mineru" / source_sha256)
        content_sha256 = _sha256_file(content_path)
        try:
            n_pages, has_text = pdf_info(pdf_path)
            if n_pages != page_count:
                raise ShadowRetrievalError("private source PDF page count does not match the benchmark")
            native_text_by_page = read_pdf_text_by_page(pdf_path) if has_text else None
            drafts = content_list_to_segments(load_content_list(content_path))
            raw_hash_before_backfill = _benchmark_text_sha256(drafts)
            raw_drafts = list(drafts)
            final_drafts = drafts
            if has_text:
                final_drafts, _ = backfill_text_layer(
                    pdf_path,
                    drafts,
                    native_text_by_page=native_text_by_page,
                )
            raw_hash_after_backfill = _benchmark_text_sha256(raw_drafts)
        except Exception as error:
            if isinstance(error, ShadowRetrievalError):
                raise
            raise ShadowRetrievalError(
                f"unable to bind parser content_list ({type(error).__name__})"
            ) from None
        if raw_text_sha256 not in {raw_hash_before_backfill, raw_hash_after_backfill}:
            raise ShadowRetrievalError("parser content_list does not match benchmark raw text hash")
        if _benchmark_text_sha256(final_drafts) != final_text_sha256:
            raise ShadowRetrievalError("parser content_list does not match benchmark final text hash")
        recorded_content_sha256 = backend_result.get("content_list_sha256")
        if recorded_content_sha256 is not None and recorded_content_sha256 != content_sha256:
            raise ShadowRetrievalError("parser content_list file SHA does not match benchmark")
        if _sha256_file(content_path) != content_sha256 or _sha256_file(pdf_path) != source_file_sha256:
            raise ShadowRetrievalError("parser content or source PDF changed during evaluation")
        document_chunks = stable_text_chunks(
            final_drafts,
            source_sha256=source_sha256,
            page_count=page_count,
            max_chars=max_chars,
        )
        chunks.extend(document_chunks)
        content_rows.append({"source_sha256": source_sha256, "content_list_sha256": content_sha256})
        manifest_rows.append(
            {
                "source_sha256": source_sha256,
                "page_count": page_count,
                "content_list_sha256": content_sha256,
                "chunks": [
                    {
                        "key": chunk.key,
                        "kind": chunk.kind,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    }
                    for chunk in document_chunks
                ],
            }
        )
    return CorpusLoad(
        chunks=tuple(chunks),
        manifest_sha256=_sha256_bytes(_canonical_bytes(manifest_rows)),
        content_lists_sha256=_sha256_bytes(_canonical_bytes(content_rows)),
        document_count=len(documents),
        chunk_count=len(chunks),
    )


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ShadowRetrievalError(f"{label} schema is invalid")
    return cast(Mapping[str, Any], value)


def _gold_snapshots(records: Sequence[GoldRecord]) -> dict[str, tuple[int, str]]:
    snapshots: dict[str, tuple[int, str]] = {}
    for record in records:
        for document in record.document_scope:
            value = (document.page_count, document.parsed_content_sha256)
            existing = snapshots.setdefault(document.source_sha256, value)
            if existing != value:
                raise ShadowRetrievalError("Gold document snapshots are inconsistent")
    if not snapshots:
        raise ShadowRetrievalError("Gold set contains no document snapshots")
    return snapshots


def load_control_corpus(path: Path, records: Sequence[GoldRecord]) -> ControlCorpus:
    """Load pinned OOXML controls and verify their production chunk snapshots."""

    try:
        artifact = read_private_json(path.expanduser(), max_bytes=_MAX_CONTROLS_BYTES)
        manifest_artifact = read_private_json(
            path.expanduser().parent / "manifest.json",
            max_bytes=_MAX_SUMMARY_BYTES,
        )
    except PrivateArtifactError as error:
        raise ShadowRetrievalError(f"unable to read private controls ({type(error).__name__})") from None

    root = _exact_keys(
        artifact.value,
        {"schema_version", "source", "controls"},
        "controls artifact",
    )
    if root["schema_version"] != 1 or root["source"] != "private-rag-gold-ooxml-controls":
        raise ShadowRetrievalError("controls artifact identity is invalid")
    raw_controls = root["controls"]
    if not isinstance(raw_controls, list) or len(raw_controls) != _EXPECTED_CONTROL_DOCUMENTS:
        raise ShadowRetrievalError("controls artifact must contain exactly three documents")

    manifest = _exact_keys(
        manifest_artifact.value,
        {"manifest_version", "source", "source_revision", "pages", "controls"},
        "private corpus manifest",
    )
    if manifest["manifest_version"] != 1 or manifest["source"] != "private-rag-gold-release":
        raise ShadowRetrievalError("private corpus manifest identity is invalid")
    source_revision = manifest["source_revision"]
    if not isinstance(source_revision, str) or _SHA256.fullmatch(source_revision) is None:
        raise ShadowRetrievalError("private corpus source revision is invalid")
    controls_meta = _exact_keys(
        manifest["controls"],
        {"file", "sha256", "count"},
        "private corpus controls metadata",
    )
    if (
        controls_meta["file"] != path.name
        or controls_meta["sha256"] != artifact.sha256
        or controls_meta["count"] != _EXPECTED_CONTROL_DOCUMENTS
    ):
        raise ShadowRetrievalError("controls artifact SHA or identity does not match its manifest")

    raw_pages = manifest["pages"]
    if not isinstance(raw_pages, list) or len(raw_pages) != _EXPECTED_PDF_DOCUMENTS:
        raise ShadowRetrievalError("private corpus manifest must contain exactly seven PDFs")
    pdf_documents: dict[str, int] = {}
    revision_pdfs: list[dict[str, Any]] = []
    for raw_page in raw_pages:
        page = _exact_keys(
            raw_page,
            {"file", "sha256", "category", "selection"},
            "private corpus PDF",
        )
        source_sha256 = page["sha256"]
        selection = _exact_keys(
            page["selection"],
            {"document_ref", "page_count"},
            "private corpus PDF selection",
        )
        page_count = selection["page_count"]
        if (
            not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
            or page["file"] != f"{source_sha256}.pdf"
            or page["category"] != "layout"
            or selection["document_ref"] != make_document_ref(source_sha256)
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 1
            or source_sha256 in pdf_documents
        ):
            raise ShadowRetrievalError("private corpus PDF metadata is invalid")
        pdf_documents[source_sha256] = page_count
        revision_pdfs.append(
            {
                "document_ref": selection["document_ref"],
                "page_count": page_count,
                "sha256": source_sha256,
            }
        )
    if sum(pdf_documents.values()) != _EXPECTED_PDF_PAGES:
        raise ShadowRetrievalError("private corpus PDF page total must equal 61")
    expected_revision = _sha256_bytes(
        _canonical_bytes(
            {
                "pdfs": revision_pdfs,
                "controls": {
                    "sha256": artifact.sha256,
                    "count": _EXPECTED_CONTROL_DOCUMENTS,
                },
            }
        )
    )
    if source_revision != expected_revision:
        raise ShadowRetrievalError("private corpus source revision does not match its contents")

    gold = _gold_snapshots(records)
    documents: dict[str, int] = {}
    chunks: list[ShadowChunk] = []
    chunks_manifest: list[dict[str, Any]] = []
    total_chunks = 0
    for raw_control in raw_controls:
        control = _exact_keys(
            raw_control,
            {"source_sha256", "page_count", "parsed_content_sha256", "chunks"},
            "OOXML control",
        )
        source_sha256 = control["source_sha256"]
        page_count = control["page_count"]
        parsed_sha256 = control["parsed_content_sha256"]
        if (
            not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 1
            or not isinstance(parsed_sha256, str)
            or _SHA256.fullmatch(parsed_sha256) is None
            or source_sha256 in documents
        ):
            raise ShadowRetrievalError("OOXML control identity is invalid")
        raw_chunks = control["chunks"]
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ShadowRetrievalError("OOXML control chunks must be non-empty")
        total_chunks += len(raw_chunks)
        if total_chunks > _MAX_CONTROL_CHUNKS:
            raise ShadowRetrievalError("OOXML control chunk count exceeds the accepted limit")
        serialized: list[dict[str, Any]] = []
        document_chunks: list[ShadowChunk] = []
        for raw_chunk in raw_chunks:
            chunk = _exact_keys(
                raw_chunk,
                {"idx", "kind", "heading_path", "page_start", "page_end", "text"},
                "OOXML control chunk",
            )
            serialized_chunk = dict(chunk)
            try:
                parsed_chunks_sha256([serialized_chunk])
            except ValueError as error:
                raise ShadowRetrievalError(
                    f"OOXML control chunk is invalid ({type(error).__name__})"
                ) from None
            idx = cast(int, chunk["idx"])
            kind = cast(str, chunk["kind"])
            text = cast(str, chunk["text"])
            page_start = cast(int | None, chunk["page_start"])
            page_end = cast(int | None, chunk["page_end"])
            if (page_start is not None and page_start >= page_count) or (
                page_end is not None and page_end >= page_count
            ):
                raise ShadowRetrievalError("OOXML control chunk exceeds its Gold page count")
            serialized.append(serialized_chunk)
            document_chunks.append(
                ShadowChunk(
                    key=_chunk_key(source_sha256, idx, text, (page_start, page_end)),
                    document_ref=make_document_ref(source_sha256),
                    source_sha256=source_sha256,
                    index=idx,
                    kind=kind,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
        try:
            actual_parsed_sha256 = parsed_chunks_sha256(serialized)
        except ValueError as error:
            raise ShadowRetrievalError(f"OOXML control chunks are invalid ({type(error).__name__})") from None
        if actual_parsed_sha256 != parsed_sha256:
            raise ShadowRetrievalError("OOXML control parsed_chunks_sha256 does not match")
        if gold.get(source_sha256) != (page_count, parsed_sha256):
            raise ShadowRetrievalError("OOXML control does not match its Gold parsed snapshot")
        documents[source_sha256] = page_count
        chunks.extend(sorted(document_chunks, key=lambda item: item.index))
        chunks_manifest.append(
            {
                "source_sha256": source_sha256,
                "page_count": page_count,
                "parsed_content_sha256": parsed_sha256,
                "chunk_keys": sorted(chunk.key for chunk in document_chunks),
            }
        )
    return ControlCorpus(
        chunks=tuple(chunks),
        documents=documents,
        pdf_documents=pdf_documents,
        artifact_sha256=artifact.sha256,
        manifest_sha256=manifest_artifact.sha256,
        chunks_manifest_sha256=_sha256_bytes(_canonical_bytes(chunks_manifest)),
        source_revision=source_revision,
    )


def _locator_range(page: int, page_start: int | None, page_end: int | None) -> tuple[int, int]:
    start = page_start if page_start is not None else page - 1
    end = page_end if page_end is not None else page - 1
    if start < 0 or end < start:
        raise ShadowRetrievalError("sidecar contains an invalid page locator")
    return start, end


def build_retrieval_cases(
    records: Sequence[GoldRecord],
    sidecars_by_id: Mapping[str, PrivateSidecarRecord],
    source_text_by_chunk_id: Mapping[uuid.UUID, str],
) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    used_source_chunks: set[uuid.UUID] = set()
    for record in records:
        sidecar = sidecars_by_id.get(record.case_id)
        if sidecar is None:
            raise ShadowRetrievalError("Gold/sidecar binding is incomplete")
        page_counts = {item.document_ref: item.page_count for item in record.document_scope}
        grades = {item.evidence_id: item.relevance_grade for item in record.evidence}
        private_locators: Sequence[Any] = (
            sidecar.exact_evidence if record.answerable else sidecar.retrieval_probe
        )
        locators: list[EvidenceLocator] = []
        for item in private_locators:
            if item.document_ref not in page_counts:
                raise ShadowRetrievalError("sidecar locator is outside Gold document scope")
            start, end = _locator_range(item.page, item.page_start, item.page_end)
            if end >= page_counts[item.document_ref]:
                raise ShadowRetrievalError("sidecar locator exceeds Gold page count")
            locator_id = item.evidence_id if record.answerable else str(item.chunk_id)
            grade = grades.get(locator_id, 1)
            source_text = source_text_by_chunk_id.get(item.chunk_id)
            if source_text is None:
                raise ShadowRetrievalError("source evidence lookup is incomplete")
            normalized_source_text = _normalize_match_text(source_text)
            if not normalized_source_text:
                raise ShadowRetrievalError("source evidence text must be non-empty")
            source_anchors = _source_anchors(normalized_source_text)
            if not source_anchors:
                raise AssertionError("source evidence anchor invariant failed")
            used_source_chunks.add(item.chunk_id)
            locators.append(
                EvidenceLocator(
                    key=_sha256_bytes(
                        _canonical_bytes(
                            {
                                "document_ref": item.document_ref,
                                "locator": locator_id,
                                "page_start": start,
                                "page_end": end,
                            }
                        )
                    ),
                    document_ref=item.document_ref,
                    page_start=start,
                    page_end=end,
                    grade=grade,
                    source_text=normalized_source_text,
                    source_text_sha256=_sha256_bytes(normalized_source_text.encode("utf-8")),
                    source_anchors=source_anchors,
                )
            )
        if not locators:
            if record.answerable:
                raise ShadowRetrievalError("every answerable Gold case needs exact evidence")
            continue
        case_sha256 = gold_record_case_sha256(record)
        if _SHA256.fullmatch(case_sha256) is None:
            raise AssertionError("Gold case hash invariant failed")
        cases.append(
            RetrievalCase(
                case_sha256=case_sha256,
                query=record.question,
                answerable=record.answerable,
                language=record.language,
                hop_type=record.hop_type,
                content_types=record.content_types,
                document_refs=frozenset(page_counts),
                locators=tuple(locators),
            )
        )
    if len({case.case_sha256 for case in cases}) != len(cases):
        raise ShadowRetrievalError("Gold case hashes must be unique")
    if set(source_text_by_chunk_id) != used_source_chunks:
        raise ShadowRetrievalError("source evidence lookup does not match sidecar locators exactly")
    return sorted(cases, key=lambda item: item.case_sha256)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ShadowRetrievalError("embedding dimensions are inconsistent")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not math.isfinite(dot) or left_norm == 0.0 or right_norm == 0.0:
        raise ShadowRetrievalError("embedding contains invalid values")
    return dot / (left_norm * right_norm)


def _validated_vectors(vectors: Sequence[Sequence[float]], expected: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise ShadowRetrievalError("embedder returned an unexpected vector count")
    result = [[float(value) for value in vector] for vector in vectors]
    dimensions = {len(vector) for vector in result}
    if dimensions == {0} or len(dimensions) != 1:
        raise ShadowRetrievalError("embedder returned inconsistent dimensions")
    if any(not math.isfinite(value) for vector in result for value in vector):
        raise ShadowRetrievalError("embedder returned non-finite values")
    return result


def _chunk_matches(chunk: ShadowChunk, locator: EvidenceLocator) -> bool:
    normalized_chunk = _normalize_match_text(chunk.text)
    content_matches = bool(
        locator.source_text
        and _sha256_bytes(locator.source_text.encode("utf-8")) == locator.source_text_sha256
        and (
            normalized_chunk == locator.source_text
            or locator.source_text in normalized_chunk
            or any(anchor in normalized_chunk for anchor in locator.source_anchors)
        )
    )
    return bool(
        chunk.document_ref == locator.document_ref
        and chunk.page_start is not None
        and chunk.page_end is not None
        and chunk.page_start <= locator.page_end
        and chunk.page_end >= locator.page_start
        and content_matches
    )


def retrieval_metrics(
    ranked: Sequence[ShadowChunk],
    all_chunks: Sequence[ShadowChunk],
    locators: Sequence[EvidenceLocator],
    *,
    latency_ms: float,
) -> dict[str, float]:
    if not locators:
        raise ShadowRetrievalError("retrieval metrics require at least one locator")

    def covered(chunks: Sequence[ShadowChunk], limit: int | None = None) -> set[str]:
        subset = chunks if limit is None else chunks[:limit]
        return {
            locator.key for locator in locators if any(_chunk_matches(chunk, locator) for chunk in subset)
        }

    recalls = {cutoff: len(covered(ranked, cutoff)) / len(locators) for cutoff in (1, 5, 10)}
    first_rank = next(
        (
            rank
            for rank, chunk in enumerate(ranked[:10], start=1)
            if any(_chunk_matches(chunk, locator) for locator in locators)
        ),
        None,
    )
    seen: set[str] = set()
    dcg = 0.0
    for rank, chunk in enumerate(ranked[:10], start=1):
        matches = [
            locator for locator in locators if locator.key not in seen and _chunk_matches(chunk, locator)
        ]
        if matches:
            best = max(matches, key=lambda item: item.grade)
            seen.add(best.key)
            dcg += (2**best.grade - 1) / math.log2(rank + 1)
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(sorted((item.grade for item in locators), reverse=True)[:10], start=1)
    )
    expected_pages = {
        (locator.document_ref, page)
        for locator in locators
        for page in range(locator.page_start, locator.page_end + 1)
    }

    def covered_pages(chunks: Sequence[ShadowChunk]) -> set[tuple[str, int]]:
        return {
            page_key
            for page_key in expected_pages
            if any(
                chunk.document_ref == page_key[0]
                and chunk.page_start is not None
                and chunk.page_end is not None
                and chunk.page_start <= page_key[1] <= chunk.page_end
                for chunk in chunks
            )
        }

    return {
        "recall_at_1": recalls[1],
        "recall_at_5": recalls[5],
        "recall_at_10": recalls[10],
        "mrr_at_10": 1.0 / first_rank if first_rank is not None else 0.0,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "evidence_coverage": len(covered(all_chunks)) / len(locators),
        "page_coverage": len(covered_pages(all_chunks)) / len(expected_pages),
        "page_coverage_at_10": len(covered_pages(ranked[:10])) / len(expected_pages),
        "latency_ms": latency_ms,
    }


async def _rank_case(
    case: RetrievalCase,
    query_vector: Sequence[float],
    chunks: Sequence[ShadowChunk],
    vectors: Mapping[str, Sequence[float]],
    reranker: RerankerLike,
    *,
    dense_top_k: int,
    rerank_top_k: int,
    common_query_latency_ms: float,
) -> tuple[list[ShadowChunk], float]:
    started = time.perf_counter()
    scoped = [chunk for chunk in chunks if chunk.document_ref in case.document_refs]
    if not scoped:
        raise ShadowRetrievalError("case document scope has no parser chunks")
    dense = sorted(
        ((_cosine(query_vector, vectors[chunk.key]), chunk) for chunk in scoped),
        key=lambda item: (-item[0], item[1].key),
    )[:dense_top_k]
    rerank_input = dense[:rerank_top_k]
    scores = await reranker.rerank(case.query, [chunk.text for _, chunk in rerank_input])
    if len(scores) != len(rerank_input) or any(not math.isfinite(float(score)) for score in scores):
        raise ShadowRetrievalError("reranker returned invalid scores")
    ranked = [
        item[2]
        for item in sorted(
            (
                (float(score), dense_score, chunk)
                for score, (dense_score, chunk) in zip(scores, rerank_input, strict=True)
            ),
            key=lambda item: (-item[0], -item[1], item[2].key),
        )
    ]
    latency_ms = common_query_latency_ms + (time.perf_counter() - started) * 1000.0
    return ranked, latency_ms


async def evaluate_pair(
    cases: Sequence[RetrievalCase],
    baseline_chunks: Sequence[ShadowChunk],
    candidate_chunks: Sequence[ShadowChunk],
    embedder: EmbedderLike,
    reranker: RerankerLike,
    *,
    dense_top_k: int,
    rerank_top_k: int,
) -> PairEvaluation:
    """Run the same embedder/reranker instance over both parser variants."""

    if dense_top_k < 10 or rerank_top_k < 10 or dense_top_k < rerank_top_k:
        raise ShadowRetrievalError("retrieval cutoffs must retain at least top 10")
    baseline_started = time.perf_counter()
    baseline_vectors = _validated_vectors(
        await embedder.embed([chunk.text for chunk in baseline_chunks]),
        len(baseline_chunks),
    )
    baseline_embedding_latency_ms = (time.perf_counter() - baseline_started) * 1000.0
    candidate_started = time.perf_counter()
    candidate_vectors = _validated_vectors(
        await embedder.embed([chunk.text for chunk in candidate_chunks]),
        len(candidate_chunks),
    )
    candidate_embedding_latency_ms = (time.perf_counter() - candidate_started) * 1000.0
    baseline_by_key = dict(zip((chunk.key for chunk in baseline_chunks), baseline_vectors, strict=True))
    candidate_by_key = dict(zip((chunk.key for chunk in candidate_chunks), candidate_vectors, strict=True))

    ranked_cases: list[RankedCase] = []
    for case in cases:
        query_started = time.perf_counter()
        query_vector = _validated_vectors([await embedder.embed_query(case.query)], 1)[0]
        query_latency_ms = (time.perf_counter() - query_started) * 1000.0
        # Alternate order deterministically to avoid a systematic warm-cache advantage.
        baseline_first = int(case.case_sha256[0], 16) % 2 == 0
        if baseline_first:
            baseline_ranked, baseline_latency = await _rank_case(
                case,
                query_vector,
                baseline_chunks,
                baseline_by_key,
                reranker,
                dense_top_k=dense_top_k,
                rerank_top_k=rerank_top_k,
                common_query_latency_ms=query_latency_ms,
            )
            candidate_ranked, candidate_latency = await _rank_case(
                case,
                query_vector,
                candidate_chunks,
                candidate_by_key,
                reranker,
                dense_top_k=dense_top_k,
                rerank_top_k=rerank_top_k,
                common_query_latency_ms=query_latency_ms,
            )
        else:
            candidate_ranked, candidate_latency = await _rank_case(
                case,
                query_vector,
                candidate_chunks,
                candidate_by_key,
                reranker,
                dense_top_k=dense_top_k,
                rerank_top_k=rerank_top_k,
                common_query_latency_ms=query_latency_ms,
            )
            baseline_ranked, baseline_latency = await _rank_case(
                case,
                query_vector,
                baseline_chunks,
                baseline_by_key,
                reranker,
                dense_top_k=dense_top_k,
                rerank_top_k=rerank_top_k,
                common_query_latency_ms=query_latency_ms,
            )
        ranked_cases.append(
            RankedCase(
                case=case,
                baseline=retrieval_metrics(
                    baseline_ranked,
                    [chunk for chunk in baseline_chunks if chunk.document_ref in case.document_refs],
                    case.locators,
                    latency_ms=baseline_latency,
                ),
                candidate=retrieval_metrics(
                    candidate_ranked,
                    [chunk for chunk in candidate_chunks if chunk.document_ref in case.document_refs],
                    case.locators,
                    latency_ms=candidate_latency,
                ),
            )
        )
    return PairEvaluation(
        ranked_cases=tuple(ranked_cases),
        baseline_embedding_latency_ms=baseline_embedding_latency_ms,
        candidate_embedding_latency_ms=candidate_embedding_latency_ms,
    )


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _aggregate(rows: Sequence[RankedCase]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "baseline": None, "candidate": None, "delta": None}
    baseline = {
        metric: sum(float(row.baseline[metric]) for row in rows) / len(rows) for metric in _METRIC_NAMES
    }
    candidate = {
        metric: sum(float(row.candidate[metric]) for row in rows) / len(rows) for metric in _METRIC_NAMES
    }
    for label, source in (("baseline", baseline), ("candidate", candidate)):
        latencies = [float(getattr(row, label)["latency_ms"]) for row in rows]
        source["latency_mean_ms"] = sum(latencies) / len(latencies)
        source["latency_p50_ms"] = _nearest_rank(latencies, 0.50)
        source["latency_p95_ms"] = _nearest_rank(latencies, 0.95)
    delta = {metric: candidate[metric] - baseline[metric] for metric in _METRIC_NAMES}
    delta["latency_mean_ms"] = candidate["latency_mean_ms"] - baseline["latency_mean_ms"]
    wins = {
        metric: {
            "candidate_better": sum(
                float(row.candidate[metric]) > float(row.baseline[metric]) for row in rows
            ),
            "equal": sum(float(row.candidate[metric]) == float(row.baseline[metric]) for row in rows),
            "baseline_better": sum(
                float(row.candidate[metric]) < float(row.baseline[metric]) for row in rows
            ),
        }
        for metric in ("recall_at_5", "ndcg_at_10")
    }
    return {
        "count": len(rows),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "paired_outcomes": wins,
    }


def build_aggregates(evaluation: PairEvaluation) -> dict[str, Any]:
    rows = list(evaluation.ranked_cases)
    answerable = [row for row in rows if row.case.answerable]
    no_answer = [row for row in rows if not row.case.answerable]
    slices: dict[str, Any] = {}
    dimensions: dict[str, Sequence[str]] = {
        "language": ("ru", "en", "zh"),
        "hop_type": ("single", "multi", "cross_document"),
        "content_type": tuple(sorted({value for row in answerable for value in row.case.content_types})),
    }
    for dimension, values in dimensions.items():
        slices[dimension] = {}
        for value in values:
            subset = [
                row
                for row in answerable
                if (
                    row.case.language == value
                    if dimension == "language"
                    else row.case.hop_type == value
                    if dimension == "hop_type"
                    else value in row.case.content_types
                )
            ]
            slices[dimension][value] = _aggregate(subset)
    return {
        "answerable": _aggregate(answerable),
        "no_answer_probe": _aggregate(no_answer),
        "slices": slices,
        "embedding_latency_ms": {
            "baseline": evaluation.baseline_embedding_latency_ms,
            "candidate": evaluation.candidate_embedding_latency_ms,
            "delta": evaluation.candidate_embedding_latency_ms - evaluation.baseline_embedding_latency_ms,
        },
    }


def decide_candidate(
    aggregates: Mapping[str, Any],
    *,
    max_regression: float = 0.01,
    min_slice_cases: int = 1,
) -> dict[str, Any]:
    """Accept non-inferiority; any >1pp overall or slice regression rejects."""

    if not 0.0 <= max_regression <= 1.0 or min_slice_cases < 1:
        raise ShadowRetrievalError("invalid decision thresholds")
    answerable = cast(Mapping[str, Any], aggregates["answerable"])
    if int(answerable["count"]) < 1 or answerable["delta"] is None:
        raise ShadowRetrievalError("decision requires answerable cases")
    no_answer_probe = cast(Mapping[str, Any], aggregates["no_answer_probe"])
    if int(no_answer_probe["count"]) < 1 or no_answer_probe["delta"] is None:
        raise ShadowRetrievalError("decision requires no-answer probe cases")
    failures: list[str] = []
    gate_metrics = ("recall_at_5", "ndcg_at_10", "evidence_coverage", "page_coverage")
    absolute_metrics = ("recall_at_10", "evidence_coverage")
    for label, cohort in (("answerable", answerable), ("no_answer_probe", no_answer_probe)):
        baseline = cohort.get("baseline")
        if not isinstance(baseline, Mapping):
            raise ShadowRetrievalError("decision requires absolute baseline metrics")
        for metric in absolute_metrics:
            value = baseline.get(metric)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ShadowRetrievalError("absolute baseline metric is invalid")
            if float(value) < _MIN_BASELINE_COVERAGE:
                failures.append(f"baseline_below_floor:{label}:{metric}")
    overall_delta = cast(Mapping[str, float], answerable["delta"])
    for metric in gate_metrics:
        if float(overall_delta[metric]) < -max_regression - 1e-12:
            failures.append(f"overall_regression:{metric}")
    no_answer_delta = cast(Mapping[str, float], no_answer_probe["delta"])
    for metric in gate_metrics:
        if float(no_answer_delta[metric]) < -max_regression - 1e-12:
            failures.append(f"no_answer_probe_regression:{metric}")

    slices = cast(Mapping[str, Mapping[str, Mapping[str, Any]]], aggregates["slices"])
    answerable_count = int(answerable["count"])
    rare_cutoff = max(5, math.ceil(answerable_count * 0.10))
    checked_slices = 0
    rare_slices = 0
    for dimension, values in slices.items():
        for value, aggregate in values.items():
            count = int(aggregate["count"])
            if count < min_slice_cases:
                failures.append(f"insufficient_slice:{dimension}:{value}")
                continue
            checked_slices += 1
            if count <= rare_cutoff:
                rare_slices += 1
            delta = cast(Mapping[str, float], aggregate["delta"])
            for metric in ("recall_at_5", "ndcg_at_10"):
                if float(delta[metric]) < -max_regression - 1e-12:
                    failures.append(f"slice_regression:{dimension}:{value}:{metric}")
    improved = any(float(overall_delta[metric]) > 1e-12 for metric in ("recall_at_5", "ndcg_at_10"))
    return {
        "accepted": not failures,
        "improved": improved,
        "failure_codes": sorted(failures),
        "thresholds": {
            "max_regression_pp": max_regression * 100.0,
            "min_slice_cases": min_slice_cases,
            "min_baseline_recall_or_coverage": _MIN_BASELINE_COVERAGE,
        },
        "checked_slices": checked_slices,
        "rare_slices": rare_slices,
    }


def _hash_string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ShadowRetrievalError("required provenance string is missing")
    return _sha256_bytes(value.encode())


def _runtime_provenance(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = summary.get("runtime_provenance")
    if not isinstance(value, dict):
        raise ShadowRetrievalError("benchmark runtime provenance is missing")
    client = value.get("client")
    server = value.get("server")
    model = value.get("model")
    controlled = value.get("controlled")
    if (
        not isinstance(client, dict)
        or not isinstance(server, dict)
        or not isinstance(model, dict)
        or not isinstance(controlled, dict)
    ):
        raise ShadowRetrievalError("benchmark runtime provenance is incomplete")
    client_version = client.get("version")
    server_version = server.get("mineru_version")
    vllm_version = server.get("vllm_version")
    model_snapshot = model.get("snapshot_sha")
    model_manifest = model.get("manifest_sha256")
    for item in (client_version, server_version, vllm_version):
        if not isinstance(item, str) or not item:
            raise ShadowRetrievalError("benchmark runtime version is missing")
    if client_version != server_version:
        raise ShadowRetrievalError("benchmark client/server MinerU versions differ")
    if not isinstance(model_snapshot, str) or _REVISION_SHA.fullmatch(model_snapshot) is None:
        raise ShadowRetrievalError("benchmark model snapshot is invalid")
    if not isinstance(model_manifest, str) or _SHA256.fullmatch(model_manifest) is None:
        raise ShadowRetrievalError("benchmark model manifest digest is invalid")
    parser_backend = controlled.get("parser_backend")
    table_enable = controlled.get("table_enable")
    server_inference_args = controlled.get("server_inference_args")
    repetition_penalty = controlled.get("repetition_penalty")
    sampling_patch_sha256 = controlled.get("sampling_patch_sha256")
    if not isinstance(parser_backend, str) or not parser_backend:
        raise ShadowRetrievalError("benchmark controlled parser backend is missing")
    if not isinstance(table_enable, bool):
        raise ShadowRetrievalError("benchmark controlled table setting is invalid")
    if not isinstance(server_inference_args, list) or not server_inference_args or not all(
        isinstance(item, str) and item for item in server_inference_args
    ):
        raise ShadowRetrievalError("benchmark controlled server arguments are invalid")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, int | float)
        or not math.isfinite(float(repetition_penalty))
        or float(repetition_penalty) <= 0
    ):
        raise ShadowRetrievalError("benchmark controlled sampling setting is invalid")
    if (
        not isinstance(sampling_patch_sha256, str)
        or _SHA256.fullmatch(sampling_patch_sha256) is None
    ):
        raise ShadowRetrievalError("benchmark controlled sampling patch digest is invalid")
    return {
        "runtime_provenance_sha256": _sha256_bytes(_canonical_bytes(value)),
        "mineru_version": client_version,
        "vllm_version": vllm_version,
        "model_snapshot_sha": model_snapshot,
        "model_manifest_sha256": model_manifest,
        "controlled_sha256": _sha256_bytes(_canonical_bytes(controlled)),
    }


def validate_pair_linkage(
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    records: Sequence[GoldRecord],
    controls: ControlCorpus,
    *,
    backend: str = "mineru",
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], dict[str, Any]]:
    try:
        baseline = parser_corpus_snapshot(baseline_summary, backend=backend)
        candidate = parser_corpus_snapshot(candidate_summary, backend=backend)
        gold_documents = gold_document_snapshot(records)
    except ParserRagLinkageError as error:
        raise ShadowRetrievalError(str(error)) from None
    baseline_documents = dict(baseline.documents)
    candidate_documents = dict(candidate.documents)
    control_documents = dict(controls.documents)
    if baseline.source_revision != candidate.source_revision:
        raise ShadowRetrievalError("parser summaries use different source revisions")
    if baseline_documents != candidate_documents:
        raise ShadowRetrievalError("parser summaries must cover the same seven PDFs")
    if baseline.source_revision != controls.source_revision:
        raise ShadowRetrievalError("parser summaries do not match the controls source revision")
    if baseline_documents != dict(controls.pdf_documents):
        raise ShadowRetrievalError("a PDF or OOXML control moved between corpus partitions")
    if set(baseline_documents).intersection(control_documents):
        raise ShadowRetrievalError("parser PDFs and OOXML controls must be disjoint")
    combined = {**baseline_documents, **control_documents}
    extra = set(combined).difference(gold_documents)
    missing = set(gold_documents).difference(combined)
    if extra:
        raise ShadowRetrievalError("controls/parser corpus contains a document outside Gold")
    if missing:
        raise ShadowRetrievalError("controls/parser corpus is missing a Gold document")
    if combined != gold_documents:
        raise ShadowRetrievalError("controls/parser page counts do not match Gold")
    if (
        len(baseline_documents) != _EXPECTED_PDF_DOCUMENTS
        or sum(baseline_documents.values()) != _EXPECTED_PDF_PAGES
        or len(control_documents) != _EXPECTED_CONTROL_DOCUMENTS
    ):
        raise ShadowRetrievalError("confirmed Gold 7-PDF/3-control composition does not match")
    baseline_runtime = _runtime_provenance(baseline_summary)
    candidate_runtime = _runtime_provenance(candidate_summary)
    if baseline_runtime["mineru_version"] == candidate_runtime["mineru_version"]:
        raise ShadowRetrievalError("parser variants must use different MinerU versions")
    for key in (
        "vllm_version",
        "model_snapshot_sha",
        "model_manifest_sha256",
        "controlled_sha256",
    ):
        if baseline_runtime[key] != candidate_runtime[key]:
            raise ShadowRetrievalError(f"parser variants differ in controlled runtime field {key}")
    linkage = {
        "schema_version": "parser-rag-linkage-v2",
        "eligible": True,
        "reason_codes": [],
        "counts": {
            "parser_pdf_documents": len(baseline_documents),
            "parser_pdf_pages": sum(baseline_documents.values()),
            "ooxml_control_documents": len(control_documents),
            "ooxml_control_pages": sum(control_documents.values()),
            "gold_documents": len(gold_documents),
            "combined_documents": len(combined),
        },
    }
    return linkage, baseline_documents, baseline_runtime, candidate_runtime


def _assert_public_report(value: Any, *, parent_key: str = "") -> None:
    forbidden_keys = {
        "question",
        "query",
        "text",
        "exact_quote",
        "reference_answer",
        "case_id",
        "document_id",
        "chunk_id",
        "path",
        "url",
        "s3",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            categorical_text = parent_key == "content_type" and key.lower() == "text"
            if (
                key.lower() in forbidden_keys
                and not categorical_text
                or key.lower().endswith(("_path", "_url"))
            ):
                raise ShadowRetrievalError(f"report contains a forbidden private field ({key})")
            _assert_public_report(item, parent_key=key)
    elif isinstance(value, list):
        for item in value:
            _assert_public_report(item, parent_key=parent_key)
    elif isinstance(value, str):
        if _UUID.search(value) or "s3://" in value.lower():
            raise ShadowRetrievalError("report contains a forbidden private identifier")
        if parent_key == "case_hashes" and _SHA256.fullmatch(value) is None:
            raise ShadowRetrievalError("report case hash is invalid")


def build_report(
    *,
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    baseline_summary_sha256: str,
    candidate_summary_sha256: str,
    gold_sha256: str,
    sidecar_sha256: str,
    source_evidence_manifest_sha256: str,
    linkage: Mapping[str, Any],
    baseline_runtime: Mapping[str, Any],
    candidate_runtime: Mapping[str, Any],
    baseline_corpus: CorpusLoad,
    candidate_corpus: CorpusLoad,
    control_corpus: ControlCorpus,
    evaluation: PairEvaluation,
    dense_top_k: int,
    rerank_top_k: int,
    max_regression: float,
    min_slice_cases: int,
) -> dict[str, Any]:
    aggregates = build_aggregates(evaluation)
    decision = decide_candidate(
        aggregates,
        max_regression=max_regression,
        min_slice_cases=min_slice_cases,
    )
    case_hashes = sorted(row.case.case_sha256 for row in evaluation.ranked_cases)
    source_revision = baseline_summary.get("source_revision")
    if source_revision != candidate_summary.get("source_revision"):
        raise ShadowRetrievalError("parser source revisions differ")
    if source_revision != control_corpus.source_revision:
        raise ShadowRetrievalError("parser summaries and controls use different source revisions")
    if _SHA256.fullmatch(source_evidence_manifest_sha256) is None:
        raise ShadowRetrievalError("source evidence manifest SHA is invalid")
    payload: dict[str, Any] = {
        "schema_version": "parser-shadow-retrieval-v2",
        "case_hashes": case_hashes,
        "counts": {
            "cases": len(case_hashes),
            "answerable": sum(row.case.answerable for row in evaluation.ranked_cases),
            "no_answer": sum(not row.case.answerable for row in evaluation.ranked_cases),
            "documents": baseline_corpus.document_count + len(control_corpus.documents),
            "parser_pdf_documents": baseline_corpus.document_count,
            "ooxml_control_documents": len(control_corpus.documents),
            "ooxml_control_chunks": len(control_corpus.chunks),
            "baseline_chunks": baseline_corpus.chunk_count + len(control_corpus.chunks),
            "candidate_chunks": candidate_corpus.chunk_count + len(control_corpus.chunks),
        },
        "linkage": linkage,
        "provenance": {
            "gold_sha256": gold_sha256,
            "sidecar_sha256": sidecar_sha256,
            "source_evidence_manifest_sha256": source_evidence_manifest_sha256,
            "source_revision_sha256": _hash_string(source_revision),
            "controls": {
                "artifact_sha256": control_corpus.artifact_sha256,
                "manifest_sha256": control_corpus.manifest_sha256,
                "chunks_manifest_sha256": control_corpus.chunks_manifest_sha256,
            },
            "baseline": {
                "summary_sha256": baseline_summary_sha256,
                "run_label_sha256": _hash_string(baseline_summary.get("run_label")),
                "corpus_manifest_sha256": baseline_corpus.manifest_sha256,
                "content_lists_sha256": baseline_corpus.content_lists_sha256,
                **baseline_runtime,
            },
            "candidate": {
                "summary_sha256": candidate_summary_sha256,
                "run_label_sha256": _hash_string(candidate_summary.get("run_label")),
                "corpus_manifest_sha256": candidate_corpus.manifest_sha256,
                "content_lists_sha256": candidate_corpus.content_lists_sha256,
                **candidate_runtime,
            },
            "retrieval": {
                "embed_model_sha256": _hash_string(settings.embed_model),
                "rerank_model_sha256": _hash_string(settings.rerank_model),
                "embed_dim": settings.embed_dim,
                "chunk_max_chars": settings.chunk_max_chars,
                "dense_top_k": dense_top_k,
                "rerank_top_k": rerank_top_k,
            },
        },
        "aggregates": aggregates,
        "decision": decision,
    }
    payload["payload_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    _assert_public_report(payload)
    return payload


def write_report(path: Path, report: Mapping[str, Any]) -> str:
    """Atomically create a non-replacing, owner-only report and return file SHA."""

    _assert_public_report(report)
    content = _canonical_bytes(report)
    if len(content) > _MAX_REPORT_BYTES:
        raise ShadowRetrievalError("shadow retrieval report exceeds size limit")
    try:
        artifact = write_private_json_fresh(path.expanduser(), content, max_bytes=_MAX_REPORT_BYTES)
    except Exception as error:
        raise ShadowRetrievalError(f"unable to publish private report ({type(error).__name__})") from None
    return artifact.sha256


__all__ = [
    "CorpusLoad",
    "ControlCorpus",
    "EmbedderLike",
    "EvidenceLocator",
    "PairEvaluation",
    "RankedCase",
    "RerankerLike",
    "RetrievalCase",
    "ShadowChunk",
    "ShadowRetrievalError",
    "build_aggregates",
    "build_report",
    "build_retrieval_cases",
    "decide_candidate",
    "evaluate_pair",
    "load_benchmark_summary",
    "load_control_corpus",
    "load_parser_corpus",
    "retrieval_metrics",
    "source_evidence_manifest_sha256",
    "stable_text_chunks",
    "validate_pair_linkage",
    "validate_local_retrieval_endpoints",
    "write_report",
]
