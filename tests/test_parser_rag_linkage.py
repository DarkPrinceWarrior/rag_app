from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_app.eval.parser_rag_linkage import (
    ParserCorpusSnapshot,
    ParserRagLinkageError,
    build_linkage_report,
    gold_document_snapshot,
    parser_corpus_snapshot,
)


def _report(*, revision: str = "corpus-v1", digest: str = "a" * 64, pages: int = 1):
    return {
        "benchmark_schema_version": 2,
        "source_revision": revision,
        "backends": ["mineru"],
        "results": {
            "page.pdf": {
                "source_sha256": digest,
                "mineru": {"status": "ok", "n_pages": pages},
            }
        },
    }


def test_parser_corpus_snapshot_rejects_unsuccessful_page() -> None:
    payload = _report()
    payload["results"]["page.pdf"]["mineru"]["status"] = "error"

    with pytest.raises(ParserRagLinkageError, match="unsuccessful"):
        parser_corpus_snapshot(payload, backend="mineru")


def test_gold_document_snapshot_rejects_inconsistent_versions() -> None:
    digest = "a" * 64
    records = [
        SimpleNamespace(document_scope=[SimpleNamespace(source_sha256=digest, page_count=1)]),
        SimpleNamespace(document_scope=[SimpleNamespace(source_sha256=digest, page_count=2)]),
    ]

    with pytest.raises(ParserRagLinkageError, match="inconsistent"):
        gold_document_snapshot(records)  # type: ignore[arg-type]


def test_linkage_report_accepts_exact_document_and_page_binding() -> None:
    digest = "a" * 64
    snapshot = ParserCorpusSnapshot(source_revision="corpus-v1", documents={digest: 3})

    report = build_linkage_report(snapshot, snapshot, {digest: 3})

    assert report["eligible"] is True
    assert report["reason_codes"] == []
    assert report["counts"]["matched_gold_documents"] == 1
    assert report["counts"]["extra_parser_documents"] == 0


def test_linkage_report_rejects_disjoint_gold_without_disclosing_hashes() -> None:
    baseline = ParserCorpusSnapshot(source_revision="corpus-v1", documents={"a" * 64: 1})
    candidate = ParserCorpusSnapshot(source_revision="corpus-v1", documents={"a" * 64: 1})

    report = build_linkage_report(baseline, candidate, {"b" * 64: 7})

    assert report == {
        "schema_version": "parser-rag-linkage-v1",
        "eligible": False,
        "reason_codes": [
            "gold_documents_missing_from_parser_corpus",
            "parser_corpus_contains_documents_outside_gold",
        ],
        "counts": {
            "parser_documents": 1,
            "gold_documents": 1,
            "matched_gold_documents": 0,
            "missing_gold_documents": 1,
            "extra_parser_documents": 1,
            "page_count_mismatches": 0,
        },
    }
    assert "b" * 64 not in str(report)


def test_linkage_report_rejects_corpus_and_page_count_mismatch() -> None:
    digest = "a" * 64
    baseline = ParserCorpusSnapshot(source_revision="corpus-v1", documents={digest: 1})
    candidate = ParserCorpusSnapshot(source_revision="corpus-v2", documents={digest: 2})

    report = build_linkage_report(baseline, candidate, {digest: 2})

    assert report["eligible"] is False
    assert report["reason_codes"] == [
        "parser_reports_use_different_corpora",
        "gold_document_page_count_mismatch",
    ]


def test_linkage_report_rejects_parser_document_outside_gold() -> None:
    gold_digest = "a" * 64
    snapshot = ParserCorpusSnapshot(
        source_revision="corpus-v1",
        documents={gold_digest: 1, "b" * 64: 2},
    )

    report = build_linkage_report(snapshot, snapshot, {gold_digest: 1})

    assert report["eligible"] is False
    assert report["reason_codes"] == ["parser_corpus_contains_documents_outside_gold"]
    assert report["counts"]["extra_parser_documents"] == 1
    assert "b" * 64 not in str(report)
