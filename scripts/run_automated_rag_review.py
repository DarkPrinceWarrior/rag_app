"""Review GoldRecord quality and emit paired release/sidecar JSONL files."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, text

from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.db.models import Chunk, Document
from rag_app.eval.automated_review import (
    LocalQwenJudge,
    RuntimeCaseData,
    RuntimeChunk,
    atomic_review_artifacts,
    require_fresh_output_paths,
    require_loopback_database_url,
    require_private_input_0600,
    run_gold_review_gate,
)
from rag_app.eval.gold_set import ensure_private_gold_path, load_gold_set
from rag_app.eval.private_sidecar import (
    PrivateSidecarRecord,
    bind_gold_sidecar,
    load_private_sidecar,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


async def load_runtime_chunks(
    sidecars: dict[str, PrivateSidecarRecord],
) -> dict[str, RuntimeCaseData]:
    """Resolve source text and owner scope from PostgreSQL without writing."""

    requested: set[uuid.UUID] = set()
    requested_documents: set[uuid.UUID] = set()
    for sidecar in sidecars.values():
        for document in sidecar.source_documents:
            requested_documents.add(document.document_id)
        for evidence in sidecar.exact_evidence:
            requested.add(evidence.chunk_id)
            requested_documents.add(evidence.document_id)
        for probe in sidecar.retrieval_probe:
            requested.add(probe.chunk_id)
            requested_documents.add(probe.document_id)
    require_loopback_database_url(settings.database_url)
    engine = create_engine()
    sessionmaker = create_sessionmaker(engine)
    try:
        async with sessionmaker() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            rows = (
                await session.execute(
                    select(
                        Chunk.id,
                        Chunk.document_id,
                        Chunk.text_ru,
                        Chunk.text_en,
                    ).where(Chunk.id.in_(requested))
                )
            ).all()
            document_rows = (
                await session.execute(
                    select(Document.id, Document.owner_sub).where(Document.id.in_(requested_documents))
                )
            ).all()
    finally:
        await engine.dispose()
    owners = {row.id: row.owner_sub for row in document_rows}
    if set(owners) != requested_documents:
        raise RuntimeError("runtime document owner resolution failed closed")
    resolved = {
        row.id: RuntimeChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            owner_sub=owners[row.document_id],
            text=(row.text_ru or row.text_en or "").strip(),
        )
        for row in rows
    }
    if set(resolved) != requested:
        raise RuntimeError("runtime chunk resolution failed closed")
    output: dict[str, RuntimeCaseData] = {}
    for case_id, sidecar in sidecars.items():
        case_chunks = {evidence.chunk_id: resolved[evidence.chunk_id] for evidence in sidecar.exact_evidence}
        case_chunks.update({probe.chunk_id: resolved[probe.chunk_id] for probe in sidecar.retrieval_probe})
        case_document_ids = {item.document_id for item in sidecar.source_documents}
        case_document_ids.update(item.document_id for item in sidecar.exact_evidence)
        case_document_ids.update(item.document_id for item in sidecar.retrieval_probe)
        output[case_id] = RuntimeCaseData(
            chunks=case_chunks,
            owner_subs=tuple(sorted({owners[document_id] for document_id in case_document_ids})),
        )
    return output


async def async_main(args: argparse.Namespace) -> int:
    gold_path = ensure_private_gold_path(args.gold_set, REPOSITORY_ROOT)
    sidecar_path = ensure_private_gold_path(args.private_sidecar, REPOSITORY_ROOT)
    report_path = ensure_private_gold_path(args.report_output, REPOSITORY_ROOT)
    release_path = ensure_private_gold_path(args.release_output, REPOSITORY_ROOT)
    release_sidecar_path = ensure_private_gold_path(args.release_sidecar_output, REPOSITORY_ROOT)
    gold_path = require_private_input_0600(gold_path, name="gold set")
    sidecar_path = require_private_input_0600(sidecar_path, name="private sidecar")
    report_path, release_path, release_sidecar_path = require_fresh_output_paths(
        (report_path, release_path, release_sidecar_path)
    )
    records, _ = load_gold_set(gold_path, mode="candidate", repository_root=REPOSITORY_ROOT)
    sidecars = bind_gold_sidecar(records, load_private_sidecar(sidecar_path))
    runtime_chunks = await load_runtime_chunks(sidecars)
    judge = LocalQwenJudge(base_url=args.judge_url, model=args.model, timeout=args.timeout)
    try:
        report, release = await run_gold_review_gate(
            records,
            sidecars,
            runtime_chunks,
            judge,
            model=args.model,
            reviewed_at=datetime.now(UTC),
            seed_a=args.seed_a,
            seed_b=args.seed_b,
            seed_adjudicator=args.seed_adjudicator,
            concurrency=args.concurrency,
        )
    finally:
        await judge.close()
    atomic_review_artifacts(
        report_path,
        report,
        release_path,
        release_sidecar_path,
        release,
        sidecars,
    )
    print(
        f"automated GoldRecord review: accepted={report.accepted_count} "
        f"rejected={report.rejected_count} total={report.case_count}"
    )
    print(f"report={report_path}")
    if report.release_accepted:
        print(f"release={release_path}")
        print(f"release_sidecar={release_sidecar_path}")
        return 0
    print("release=not-created")
    print("release_sidecar=not-created")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", type=Path, required=True)
    parser.add_argument("--private-sidecar", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--release-output", type=Path, required=True)
    parser.add_argument("--release-sidecar-output", type=Path, required=True)
    parser.add_argument("--judge-url", default=settings.llm_base_url)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed-a", type=int, default=2026071301)
    parser.add_argument("--seed-b", type=int, default=2026071302)
    parser.add_argument("--seed-adjudicator", type=int, default=2026071303)
    parser.add_argument("--concurrency", type=int, default=2, choices=range(1, 17))
    return parser


def main() -> int:
    return asyncio.run(async_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
