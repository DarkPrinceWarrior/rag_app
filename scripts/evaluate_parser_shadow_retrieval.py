#!/usr/bin/env python3
"""Run a paired, content-free shadow retrieval gate for two MinerU outputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select, text

from rag_app.config import settings
from rag_app.db.engine import create_engine
from rag_app.db.models import Chunk
from rag_app.eval.gold_set import GoldSetValidationError, load_gold_set
from rag_app.eval.parser_shadow_retrieval import (
    ShadowRetrievalError,
    build_report,
    build_retrieval_cases,
    evaluate_pair,
    load_benchmark_summary,
    load_control_corpus,
    load_parser_corpus,
    source_evidence_manifest_sha256,
    validate_local_retrieval_endpoints,
    validate_pair_linkage,
    write_report,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarError,
    PrivateSidecarRecord,
    bind_gold_sidecar,
    load_private_sidecar,
)
from rag_app.llm.embeddings import Embedder, Reranker

_EXPECTED_CASES = 236
_MAX_REGRESSION = 0.01
_MIN_SLICE_CASES = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_source_chunks(
    sidecars_by_id: Mapping[str, PrivateSidecarRecord],
) -> dict[uuid.UUID, tuple[uuid.UUID, int | None, int | None]]:
    expected: dict[uuid.UUID, tuple[uuid.UUID, int | None, int | None]] = {}
    for sidecar in sidecars_by_id.values():
        private_locators = (
            sidecar.retrieval_probe if sidecar.stratum == "no_answer" else sidecar.exact_evidence
        )
        for raw_locator in private_locators:
            locator = cast(Any, raw_locator)
            identity = (locator.document_id, locator.page_start, locator.page_end)
            previous = expected.setdefault(locator.chunk_id, identity)
            if previous != identity:
                raise ShadowRetrievalError("sidecar source chunk bindings are inconsistent")
    if not expected:
        raise ShadowRetrievalError("sidecar has no source chunks")
    return expected


async def _load_source_text_by_chunk_id(
    sidecars_by_id: Mapping[str, PrivateSidecarRecord],
) -> dict[uuid.UUID, str]:
    """Resolve exact source-language evidence in one read-only repeatable snapshot."""

    expected = _expected_source_chunks(sidecars_by_id)
    engine = create_engine()
    try:
        async with engine.connect() as connection:
            async with connection.begin():
                await connection.execute(
                    text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                )
                result = await connection.execute(
                    select(
                        Chunk.id,
                        Chunk.document_id,
                        Chunk.page_start,
                        Chunk.page_end,
                        Chunk.text_en,
                    ).where(Chunk.id.in_(tuple(expected)))
                )
                rows = result.all()
    except ShadowRetrievalError:
        raise
    except Exception as error:
        raise ShadowRetrievalError(
            f"read-only source evidence lookup failed ({type(error).__name__})"
        ) from None
    finally:
        await engine.dispose()

    source_text: dict[uuid.UUID, str] = {}
    for row in rows:
        identity = expected.get(row.id)
        if identity is None or row.id in source_text:
            raise ShadowRetrievalError("source evidence lookup returned an unexpected or duplicate chunk")
        document_id, page_start, page_end = identity
        if (
            row.document_id != document_id
            or row.page_start != page_start
            or row.page_end != page_end
        ):
            raise ShadowRetrievalError("source evidence document/page binding changed")
        value = (row.text_en or "").strip()
        if not value:
            raise ShadowRetrievalError("source evidence text must be non-empty")
        source_text[row.id] = value
    if set(source_text) != set(expected):
        raise ShadowRetrievalError("source evidence lookup did not return the exact sidecar chunks")
    return source_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed shadow retrieval A/B over private Gold originals",
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-mode", choices=("candidate", "release"), default="release")
    parser.add_argument("--dense-top-k", type=int, default=max(settings.rag_dense_top_k, 20))
    parser.add_argument("--rerank-top-k", type=int, default=max(settings.rag_rerank_top_k, 10))
    args = parser.parse_args()
    if args.dense_top_k < args.rerank_top_k or args.rerank_top_k < 10:
        parser.error("dense/rerank cutoffs must preserve at least top 10")
    return args


async def _run(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    validate_local_retrieval_endpoints()
    baseline_summary, baseline_summary_sha256 = load_benchmark_summary(args.baseline_report)
    candidate_summary, candidate_summary_sha256 = load_benchmark_summary(args.candidate_report)
    if baseline_summary_sha256 == candidate_summary_sha256:
        raise ShadowRetrievalError("baseline and candidate summaries must differ")
    if args.baseline_output.resolve() == args.candidate_output.resolve():
        raise ShadowRetrievalError("baseline and candidate output roots must differ")

    gold_sha256 = _sha256_file(args.gold)
    sidecar_sha256 = _sha256_file(args.sidecar)
    records, _ = load_gold_set(args.gold, mode=args.gold_mode, repository_root=Path.cwd())
    if len(records) != _EXPECTED_CASES:
        raise ShadowRetrievalError("Gold case count does not match the expected paired gate size")
    sidecars = load_private_sidecar(args.sidecar, repository_root=Path.cwd())
    sidecars_by_id = bind_gold_sidecar(records, sidecars)
    source_text_by_chunk_id = await _load_source_text_by_chunk_id(sidecars_by_id)
    source_evidence_sha256 = source_evidence_manifest_sha256(source_text_by_chunk_id)
    if _sha256_file(args.gold) != gold_sha256 or _sha256_file(args.sidecar) != sidecar_sha256:
        raise ShadowRetrievalError("Gold or sidecar changed while it was being evaluated")
    control_corpus = load_control_corpus(args.controls, records)
    linkage, parser_documents, baseline_runtime, candidate_runtime = validate_pair_linkage(
        baseline_summary,
        candidate_summary,
        records,
        control_corpus,
    )
    baseline_corpus = load_parser_corpus(
        args.baseline_output,
        parser_documents,
        summary=baseline_summary,
        pdf_root=args.controls.expanduser().parent,
        max_chars=settings.chunk_max_chars,
    )
    candidate_corpus = load_parser_corpus(
        args.candidate_output,
        parser_documents,
        summary=candidate_summary,
        pdf_root=args.controls.expanduser().parent,
        max_chars=settings.chunk_max_chars,
    )
    cases = build_retrieval_cases(records, sidecars_by_id, source_text_by_chunk_id)
    baseline_chunks = (*baseline_corpus.chunks, *control_corpus.chunks)
    candidate_chunks = (*candidate_corpus.chunks, *control_corpus.chunks)
    embedder = Embedder()
    try:
        evaluation = await evaluate_pair(
            cases,
            baseline_chunks,
            candidate_chunks,
            embedder,
            Reranker(),
            dense_top_k=args.dense_top_k,
            rerank_top_k=args.rerank_top_k,
        )
    finally:
        await embedder.client.close()
    report = build_report(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        baseline_summary_sha256=baseline_summary_sha256,
        candidate_summary_sha256=candidate_summary_sha256,
        gold_sha256=gold_sha256,
        sidecar_sha256=sidecar_sha256,
        source_evidence_manifest_sha256=source_evidence_sha256,
        linkage=linkage,
        baseline_runtime=baseline_runtime,
        candidate_runtime=candidate_runtime,
        baseline_corpus=baseline_corpus,
        candidate_corpus=candidate_corpus,
        control_corpus=control_corpus,
        evaluation=evaluation,
        dense_top_k=args.dense_top_k,
        rerank_top_k=args.rerank_top_k,
        max_regression=_MAX_REGRESSION,
        min_slice_cases=_MIN_SLICE_CASES,
    )
    artifact_sha256 = write_report(args.output, report)
    return report, artifact_sha256


def main() -> int:
    args = _parse_args()
    try:
        report, artifact_sha256 = asyncio.run(_run(args))
    except (
        GoldSetValidationError,
        PrivateSidecarError,
        ShadowRetrievalError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_version": "parser-shadow-retrieval-cli-v2",
                    "status": "error",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 3
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": "parser-shadow-retrieval-cli-v2",
                    "status": "error",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 3
    decision = report["decision"]
    assert isinstance(decision, dict)
    accepted = decision.get("accepted") is True
    print(
        json.dumps(
            {
                "schema_version": "parser-shadow-retrieval-cli-v2",
                "status": "accepted" if accepted else "rejected",
                "artifact_sha256": artifact_sha256,
                "payload_sha256": report["payload_sha256"],
                "case_count": report["counts"]["cases"],
                "failure_codes": decision.get("failure_codes", []),
            },
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
