from __future__ import annotations

import asyncio
import importlib.util
import json
import stat
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_app.rag.retrieve import RetrievedChunk


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_script("evaluate_retrieval_bm25")
runner = _load_script("evaluate_hierarchical_retrieval")


def _sha(value: str) -> str:
    return runner._text_sha256(value)


def _chunk(
    index: int,
    *,
    document_id: uuid.UUID | None = None,
    text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.UUID(int=index + 1),
        document_id=document_id or uuid.UUID(int=100 + index),
        filename=f"doc-{index}.pdf",
        heading_path=f"Section {index}",
        kind="section",
        page_start=index,
        page_end=index,
        text_en=text or f"evidence {index}",
        text_ru="",
        meta={"segment_ids": [str(uuid.UUID(int=1000 + index))]},
    )


def _snapshot(
    chunks: tuple[RetrievedChunk, ...],
    *,
    question_sha256: str,
    snapshot_index: int = 0,
    scores: tuple[float, ...] | None = None,
) -> runner.FrozenScoreSnapshot:
    values = scores or tuple(1.0 - index / 100 for index in range(len(chunks)))
    by_hash = {_sha(runner._chunk_text(chunk)): score for chunk, score in zip(chunks, values, strict=True)}
    entries = tuple(
        runner.FrozenScoreEntry(
            question_sha256=question_sha256,
            text_sha256=text_sha256,
            score_hex=score.hex(),
        )
        for text_sha256, score in sorted(by_hash.items())
    )
    return runner.FrozenScoreSnapshot(
        snapshot_index=snapshot_index,
        pair_count=len(entries),
        entries=entries,
        pair_manifest_sha256=runner.canonical_sha256(entries),
        scoring_seconds=0.1,
    )


def test_frozen_score_snapshot_rejects_manifest_or_order_errors() -> None:
    left = runner.FrozenScoreEntry(
        question_sha256=_sha("question"),
        text_sha256=_sha("b"),
        score_hex=(0.5).hex(),
    )
    right = left.model_copy(update={"text_sha256": _sha("a")})
    reversed_entries = tuple(
        sorted((left, right), key=lambda item: item.text_sha256, reverse=True)
    )
    with pytest.raises(ValidationError, match="canonically ordered"):
        runner.FrozenScoreSnapshot(
            snapshot_index=0,
            pair_count=2,
            entries=reversed_entries,
            pair_manifest_sha256=runner.canonical_sha256(reversed_entries),
            scoring_seconds=0.0,
        )
    with pytest.raises(ValidationError, match="manifest hash mismatch"):
        runner.FrozenScoreSnapshot(
            snapshot_index=0,
            pair_count=1,
            entries=(left,),
            pair_manifest_sha256="0" * 64,
            scoring_seconds=0.0,
        )


def test_both_variants_use_the_same_frozen_score_map() -> None:
    chunks = (_chunk(0), _chunk(1), _chunk(2))
    question_sha256 = _sha("question")
    snapshot = _snapshot(
        chunks,
        question_sha256=question_sha256,
        scores=(0.2, 0.9, 0.8),
    )
    scores = runner._score_map(snapshot)
    baseline = runner._rank_with_frozen_scores(
        chunks[:2],
        question_sha256=question_sha256,
        scores=scores,
        top_k=10,
        minimum_score=0.1,
    )
    candidate = runner._rank_with_frozen_scores(
        chunks,
        question_sha256=question_sha256,
        scores=scores,
        top_k=10,
        minimum_score=0.1,
    )
    assert [item.id for item in baseline] == [chunks[1].id, chunks[0].id]
    assert [item.id for item in candidate] == [chunks[1].id, chunks[2].id, chunks[0].id]
    with pytest.raises(runner.HierarchicalEvaluationError, match="missed a candidate pair"):
        runner._rank_with_frozen_scores(
            chunks,
            question_sha256=_sha("other question"),
            scores=scores,
            top_k=10,
            minimum_score=0.1,
        )


def test_route_recall_is_document_level_not_exact_chunk_level() -> None:
    relevant_document = uuid.UUID(int=900)
    anchor = _chunk(0, document_id=relevant_document)
    unrelated_exact_chunk = _chunk(1, document_id=relevant_document)
    assert runner._route_document_recall((anchor,), {relevant_document}, 5) == 1.0
    assert runner._metric_recall((anchor,), {unrelated_exact_chunk.id: 3}, 5) == 0.0


def test_private_artifact_is_mode_0600_and_hmac_bound(tmp_path: Path) -> None:
    work = runner._private_dir(tmp_path / "private")
    path = work / "artifact.json"
    key = b"k" * 32
    raw, digest = runner._write_signed_private_json(
        path,
        artifact_type="test-artifact-v1",
        payload={"value": 1},
        key=key,
    )
    envelope = runner.ArtifactHmac.model_validate_json(
        (work / "artifact.hmac.json").read_bytes(),
        strict=True,
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((work / "artifact.hmac.json").stat().st_mode) == 0o600
    assert envelope.artifact_sha256 == digest
    assert envelope.signature == runner._artifact_signature("test-artifact-v1", raw, key)
    assert envelope.signature != runner._artifact_signature("test-artifact-v1", raw + b" ", key)
    with pytest.raises(FileExistsError):
        runner._write_signed_private_json(
            path,
            artifact_type="test-artifact-v1",
            payload={"value": 1},
            key=key,
        )


class _FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        del query
        self.calls += 1
        return [0.1 * self.calls + index / 100 for index in range(len(texts))]


def test_freeze_scores_keeps_three_independent_common_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeReranker()
    monkeypatch.setattr(runner, "Reranker", lambda: fake)
    chunks = (_chunk(0), _chunk(1))
    record = SimpleNamespace(question="private question", question_sha256=_sha("private question"))
    collected = (
        runner._CollectedCase(
            binding=SimpleNamespace(record=record),
            scope=SimpleNamespace(),
            baseline=chunks[:1],
            candidate=chunks,
            hierarchy_manifest_sha256=_sha("hierarchy"),
            hierarchy_added=1,
        ),
    )
    snapshots = asyncio.run(runner._freeze_scores(collected, snapshot_count=3))
    assert [item.snapshot_index for item in snapshots] == [0, 1, 2]
    assert fake.calls == 3
    assert len({item.pair_manifest_sha256 for item in snapshots}) == 3
    assert all(item.pair_count == 2 for item in snapshots)


def _variant(
    value: float,
    *,
    answerable: bool,
) -> runner.VariantMetrics:
    return runner.VariantMetrics(
        recall_at_5=value if answerable else 0.0,
        recall_at_10=value if answerable else 0.0,
        ndcg_at_10=value if answerable else 0.0,
        full_evidence_at_10=value if answerable else 0.0,
        returned_count=10 if answerable else 0,
        abstained=not answerable,
    )


def _case(index: int, *, improved: bool = True) -> runner.HierarchicalCaseResult:
    answerable = index < 169
    baseline_value = 0.50
    candidate_value = 0.80 if improved else 0.50
    snapshots = tuple(
        runner.SnapshotCaseMetrics(
            snapshot_index=snapshot_index,
            baseline=_variant(baseline_value, answerable=answerable),
            candidate=_variant(candidate_value, answerable=answerable),
            baseline_order_sha256=_sha(f"baseline-{index}-{snapshot_index}"),
            candidate_order_sha256=_sha(f"candidate-{index}-{snapshot_index}"),
        )
        for snapshot_index in range(3)
    )
    languages = ("ru", "en", "zh")
    hops = ("single", "multi", "cross_document")
    contents = ("text", "table", "formula", "figure", "scan")
    scope_index = index % 4
    cluster_index = (index // 4) % 8
    return runner.HierarchicalCaseResult(
        case_id=f"ragq-hier-{index:04d}",
        gold_case_sha256=_sha(f"gold-{index}"),
        question_sha256=_sha(f"question-{index}"),
        query_embedding_sha256=_sha(f"embedding-{index}"),
        scope_id=f"scope-sha256:{_sha(f'scope-{scope_index}')}",
        cluster_id=f"cluster-sha256:{_sha(f'cluster-{cluster_index}')}",
        split="locked",
        language=languages[index % len(languages)],
        hop_type=hops[index % len(hops)],
        content_types=(contents[index % len(contents)],),
        challenge_tags=(() if answerable else ("leakage",)),
        answerable=answerable,
        relevant_count=1 if answerable else 0,
        route_recall_at_5=0.98 if answerable else None,
        evidence_recall_at_20=0.99 if answerable else None,
        baseline_pool_count=20,
        candidate_pool_count=21,
        hierarchical_added=1,
        hierarchy_manifest_sha256=_sha(f"hierarchy-{index}"),
        scope_violation_count=0,
        hierarchical_fallback=False,
        snapshots=snapshots,
    )


def test_quality_gate_accepts_material_improvement_and_rejects_no_gain() -> None:
    accepted = runner.evaluate_quality(tuple(_case(index) for index in range(236)))
    assert accepted.accepted is True
    assert not accepted.failure_codes
    rejected = runner.evaluate_quality(tuple(_case(index, improved=False) for index in range(236)))
    assert rejected.accepted is False
    assert "metric:global:ndcg_at_10_target" in rejected.failure_codes


def test_rls_and_load_models_fail_closed() -> None:
    with pytest.raises(ValidationError):
        runner.RlsEvidence(
            schema_version="hierarchical-rls-evidence-v1",
            principal_count=9,
            owner_scope_count=2,
            probe_count=10,
            admin_foreign_truth_count=1,
            anonymous_visible_count=0,
            leak_count=0,
            scope_violation_count=0,
            passed=True,
        )
    baseline = runner.LoadAggregate(
        request_count=200,
        completed_count=200,
        error_count=0,
        p95_latency_ms=100.0,
        throughput_rps=20.0,
    )
    candidate = runner.LoadAggregate(
        request_count=200,
        completed_count=200,
        error_count=0,
        p95_latency_ms=120.0,
        throughput_rps=19.0,
    )
    with pytest.raises(ValidationError, match="load decision is inconsistent"):
        runner.LoadEvidence(
            concurrency=10,
            observed_peak_concurrency=10,
            requests_per_variant=200,
            warmups_per_variant=20,
            raw_observations_sha256=_sha("raw"),
            baseline=baseline,
            candidate=candidate,
            maximum_candidate_p95_ms=110.0,
            throughput_ratio=0.95,
            passed=True,
        )


def test_cli_defaults_are_release_shaped() -> None:
    parser = runner._parser()
    args = parser.parse_args(
        [
            "--gold",
            "/tmp/gold.jsonl",
            "--sidecar",
            "/tmp/sidecar.jsonl",
            "--work-dir",
            "/tmp/work",
        ]
    )
    assert args.snapshots == 3
    assert args.load_concurrency == 10
    assert args.load_requests_per_variant == 200
    assert "skip_load" not in vars(args)


def test_debug_report_can_never_claim_release_acceptance() -> None:
    report = runner.HierarchicalReport.model_construct(
        mode="debug",
        evaluated_at=datetime.now(UTC),
        cases=(),
        score_snapshots=(),
        quality=SimpleNamespace(accepted=True),
        rls=SimpleNamespace(passed=True),
        load=SimpleNamespace(passed=True),
        release_accepted=True,
    )

    with pytest.raises(ValueError, match="release decision is inconsistent"):
        report.validate_report()


def test_serialized_case_contains_no_question_or_document_text() -> None:
    payload = json.dumps(_case(0).model_dump(mode="json"), sort_keys=True)
    assert "question-0" not in payload
    assert "evidence 0" not in payload
    assert "question_sha256" in payload


def test_load_evidence_uses_the_signed_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_top_k: list[int] = []
    started_base = datetime(2026, 7, 15, tzinfo=UTC)

    async def fake_load_request(**kwargs: object) -> runner.LoadRequestObservation:
        top_k = int(kwargs["top_k"])
        observed_top_k.append(top_k)
        started_at = started_base + timedelta(milliseconds=2 * len(observed_top_k))
        variant = kwargs["variant"]
        pair_index = int(kwargs["pair_index"])
        order_in_pair = int(kwargs["order_in_pair"])
        return runner.LoadRequestObservation(
            request_id=_sha(f"load-{len(observed_top_k)}"),
            pair_index=pair_index,
            order_in_pair=order_in_pair,
            case_id=_case(0).case_id,
            variant=variant,
            started_at=started_at,
            completed_at=started_at + timedelta(milliseconds=1),
            latency_ms=1.0,
            returned_count=top_k,
            order_sha256=_sha(f"order-{len(observed_top_k)}"),
            success=True,
            error_code=None,
        )

    monkeypatch.setattr(runner, "_load_request", fake_load_request)
    asyncio.run(
        runner.generate_load_evidence(
            retriever=object(),
            sessionmaker=object(),
            collected=[object()],
            concurrency=1,
            requests_per_variant=2,
            warmups_per_variant=1,
            top_k=37,
        )
    )

    assert observed_top_k == [37] * 6
