from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_app.eval.gold_set import GoldRecord

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ParserRagLinkageError(ValueError):
    """A sanitized failure while validating parser/RAG corpus linkage."""


@dataclass(frozen=True)
class ParserCorpusSnapshot:
    source_revision: str
    documents: Mapping[str, int]


def parser_corpus_snapshot(
    payload: Mapping[str, Any],
    *,
    backend: str,
) -> ParserCorpusSnapshot:
    """Extract a content-free document snapshot from a parser benchmark report."""

    if payload.get("benchmark_schema_version") != 2:
        raise ParserRagLinkageError("parser report schema must be version 2")
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ParserRagLinkageError("parser report source revision is missing")
    backends = payload.get("backends")
    if not isinstance(backends, list) or backend not in backends:
        raise ParserRagLinkageError("requested parser backend is absent")
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ParserRagLinkageError("parser report has no results")

    documents: dict[str, int] = {}
    for row in results.values():
        if not isinstance(row, dict):
            raise ParserRagLinkageError("parser report result is invalid")
        source_sha256 = row.get("source_sha256")
        if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
            raise ParserRagLinkageError("parser report source digest is invalid")
        backend_result = row.get(backend)
        if not isinstance(backend_result, dict) or backend_result.get("status") != "ok":
            raise ParserRagLinkageError("parser report contains an unsuccessful page")
        page_count = backend_result.get("n_pages")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
            raise ParserRagLinkageError("parser report page count is invalid")
        if source_sha256 in documents:
            raise ParserRagLinkageError("parser report contains a duplicate source digest")
        documents[source_sha256] = page_count
    return ParserCorpusSnapshot(source_revision=revision, documents=documents)


def gold_document_snapshot(records: Sequence[GoldRecord]) -> dict[str, int]:
    """Return stable source-byte digests and page counts without private case data."""

    documents: dict[str, int] = {}
    for record in records:
        for document in record.document_scope:
            existing = documents.setdefault(document.source_sha256, document.page_count)
            if existing != document.page_count:
                raise ParserRagLinkageError("gold document snapshots are inconsistent")
    if not documents:
        raise ParserRagLinkageError("gold set has no document snapshots")
    return documents


def build_linkage_report(
    baseline: ParserCorpusSnapshot,
    candidate: ParserCorpusSnapshot,
    gold_documents: Mapping[str, int],
) -> dict[str, Any]:
    """Decide whether downstream RAG evaluation is methodologically eligible."""

    reasons: list[str] = []
    reports_match = (
        baseline.source_revision == candidate.source_revision
        and dict(baseline.documents) == dict(candidate.documents)
    )
    if not reports_match:
        reasons.append("parser_reports_use_different_corpora")

    matched = set(gold_documents).intersection(baseline.documents)
    missing = set(gold_documents).difference(baseline.documents)
    extra = set(baseline.documents).difference(gold_documents)
    page_count_mismatches = sum(
        baseline.documents[source_sha256] != page_count
        for source_sha256, page_count in gold_documents.items()
        if source_sha256 in baseline.documents
    )
    if missing:
        reasons.append("gold_documents_missing_from_parser_corpus")
    if extra:
        reasons.append("parser_corpus_contains_documents_outside_gold")
    if page_count_mismatches:
        reasons.append("gold_document_page_count_mismatch")

    return {
        "schema_version": "parser-rag-linkage-v1",
        "eligible": not reasons,
        "reason_codes": reasons,
        "counts": {
            "parser_documents": len(baseline.documents),
            "gold_documents": len(gold_documents),
            "matched_gold_documents": len(matched),
            "missing_gold_documents": len(missing),
            "extra_parser_documents": len(extra),
            "page_count_mismatches": page_count_mismatches,
        },
    }


__all__ = [
    "ParserCorpusSnapshot",
    "ParserRagLinkageError",
    "build_linkage_report",
    "gold_document_snapshot",
    "parser_corpus_snapshot",
]
