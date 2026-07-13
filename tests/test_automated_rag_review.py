from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_app.eval.automated_review import (
    AutomatedCaseResult,
    AutomatedGateReport,
    CandidateOutput,
    DeterministicChecks,
    JudgeDecision,
    JudgeOutputError,
    JudgeRun,
    RetrievedChunkRef,
    RuntimeCaseData,
    RuntimeChunk,
    _numbers_supported_by_evidence,
    atomic_filtered_release_sidecar_jsonl,
    atomic_release_jsonl,
    atomic_review_artifacts,
    build_release_records,
    deterministic_checks,
    evaluate_case,
    require_fresh_output_paths,
    require_loopback_database_url,
    require_private_input_0600,
    synthesize_gold_review_candidates,
)
from rag_app.eval.gold_set import (
    DocumentSnapshot,
    EvidenceRef,
    GoldRecord,
    gold_record_case_sha256,
    make_document_ref,
    make_evidence_id,
    make_scope_id,
    text_sha256,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarRecord,
    QuantitySpec,
    RetrievalProbe,
    SidecarClassification,
    SidecarDocument,
    SidecarEvidence,
    SidecarGeneration,
    SidecarQuantities,
    load_private_sidecar,
)


class FakeJudge:
    def __init__(self, decisions: dict[str, JudgeDecision | Exception]) -> None:
        self.decisions = decisions
        self.calls: list[tuple[str, int]] = []
        self.payloads: list[dict] = []

    async def __call__(self, variant: str, seed: int, payload: dict) -> JudgeDecision:
        self.calls.append((variant, seed))
        self.payloads.append(payload)
        outcome = self.decisions[variant]
        if isinstance(outcome, Exception):
            raise outcome
        assert "question" in payload
        return outcome


def _accept() -> JudgeDecision:
    return JudgeDecision(
        verdict="accept",
        answer_supported=True,
        evidence_supported=True,
        answerability_correct=True,
        scope_compliant=True,
        reason_codes=(),
    )


def _reject() -> JudgeDecision:
    return JudgeDecision(
        verdict="reject",
        answer_supported=False,
        evidence_supported=True,
        answerability_correct=True,
        scope_compliant=True,
        reason_codes=("answer_mismatch",),
    )


def _positive_case() -> tuple[GoldRecord, PrivateSidecarRecord, RuntimeCaseData]:
    owner_sub = "synthetic-owner"
    scope_id = make_scope_id(owner_sub)
    document_id = uuid.UUID(int=1)
    chunk_id = uuid.UUID(int=2)
    document_sha = "1" * 64
    document_ref = make_document_ref(document_sha)
    source_text = "Required pressure is exactly 16.5 MPa."
    exact_quote = "pressure is exactly 16.5 MPa"
    content_sha = text_sha256(exact_quote)
    evidence_id = make_evidence_id(document_sha, 1, "text", content_sha)
    snapshot = DocumentSnapshot(
        document_ref=document_ref,
        source_sha256=document_sha,
        parsed_content_sha256="2" * 64,
        page_count=1,
    )
    evidence = EvidenceRef(
        evidence_id=evidence_id,
        document_ref=document_ref,
        page=1,
        content_type="text",
        content_sha256=content_sha,
        relevance_grade=3,
        bbox=None,
    )
    question = "What pressure is required?"
    answer = "The required pressure is 16.5 MPa."
    record = GoldRecord(
        schema_version="rag-gold-v1",
        scope_id=scope_id,
        case_id="ragq-case-0001",
        status="candidate",
        language="en",
        question=question,
        question_sha256=text_sha256(question),
        answerable=True,
        reference_answer=answer,
        reference_answer_sha256=text_sha256(answer),
        hop_type="single",
        content_types=("text",),
        challenge_tags=("numbers", "units"),
        document_scope=(snapshot,),
        evidence=(evidence,),
        review=None,
    )
    sidecar = PrivateSidecarRecord(
        schema_version="private-rag-generator-v1",
        case_id=record.case_id,
        gold_case_sha256=gold_record_case_sha256(record),
        scope_id=scope_id,
        stratum="single_hop",
        language="en",
        source_documents=(
            SidecarDocument(
                document_id=document_id,
                document_ref=document_ref,
                source_lang="en",
            ),
        ),
        classification=SidecarClassification(
            content_types=("text",),
            challenge_tags=("numbers", "units"),
            has_numbers=True,
            has_units=True,
            has_standards=False,
        ),
        generation=SidecarGeneration(model="synthetic", seed=7),
        exact_evidence=(
            SidecarEvidence(
                evidence_id=evidence_id,
                document_id=document_id,
                document_ref=document_ref,
                chunk_id=chunk_id,
                chunk_index=0,
                kind="text",
                heading_path="",
                page=1,
                page_start=0,
                page_end=0,
                text_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                content_sha256=content_sha,
                exact_quote=exact_quote,
                retrieval_score=None,
            ),
        ),
        retrieval_probe=(),
        quantities=SidecarQuantities(
            expected=(QuantitySpec(value="16.5", unit="MPa"),),
            supported=(QuantitySpec(value="16.5", unit="MPa"),),
        ),
        validation={
            "answer_supported": True,
            "question_unambiguous": True,
            "uses_all_evidence": True,
        },
    )
    runtime = RuntimeCaseData(
        chunks={
            chunk_id: RuntimeChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                owner_sub=owner_sub,
                text=source_text,
            )
        },
        owner_subs=(owner_sub,),
    )
    return record, sidecar, runtime


def _no_answer_case(language: str) -> tuple[GoldRecord, PrivateSidecarRecord, RuntimeCaseData]:
    owner_sub = "synthetic-owner"
    scope_id = make_scope_id(owner_sub)
    document_id = uuid.UUID(int=3)
    chunk_id = uuid.UUID(int=4)
    document_sha = "3" * 64
    document_ref = make_document_ref(document_sha)
    source_text = "This context does not contain the requested value."
    snapshot = DocumentSnapshot(
        document_ref=document_ref,
        source_sha256=document_sha,
        parsed_content_sha256="4" * 64,
        page_count=1,
    )
    question_by_language = {
        "ru": "Каково отсутствующее значение параметра?",
        "en": "What is the missing parameter value?",
        "zh": "缺失参数的数值是多少？",
    }
    question = question_by_language[language]
    record = GoldRecord(
        schema_version="rag-gold-v1",
        scope_id=scope_id,
        case_id=f"ragq-noanswer-{language}01",
        status="candidate",
        language=language,
        question=question,
        question_sha256=text_sha256(question),
        answerable=False,
        reference_answer=None,
        reference_answer_sha256=None,
        hop_type="single",
        content_types=("text",),
        challenge_tags=(),
        document_scope=(snapshot,),
        evidence=(),
        review=None,
    )
    sidecar = PrivateSidecarRecord(
        schema_version="private-rag-generator-v1",
        case_id=record.case_id,
        gold_case_sha256=gold_record_case_sha256(record),
        scope_id=scope_id,
        stratum="no_answer",
        language=language,
        source_documents=(
            SidecarDocument(
                document_id=document_id,
                document_ref=document_ref,
                source_lang=language,
            ),
        ),
        classification=SidecarClassification(
            content_types=("text",),
            challenge_tags=(),
            has_numbers=False,
            has_units=False,
            has_standards=False,
        ),
        generation=SidecarGeneration(model="synthetic", seed=8),
        exact_evidence=(),
        retrieval_probe=(
            RetrievalProbe(
                document_id=document_id,
                document_ref=document_ref,
                chunk_id=chunk_id,
                page=1,
                page_start=0,
                page_end=0,
                content_sha256=text_sha256(source_text),
                retrieval_score=0.5,
            ),
        ),
        quantities=SidecarQuantities(expected=(), supported=()),
        validation={"answerable_from_top8": False},
    )
    runtime = RuntimeCaseData(
        chunks={
            chunk_id: RuntimeChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                owner_sub=owner_sub,
                text=source_text,
            )
        },
        owner_subs=(owner_sub,),
    )
    return record, sidecar, runtime


def test_gold_review_runs_two_prompts_and_machine_adjudication() -> None:
    record, sidecar, runtime = _positive_case()
    candidate = synthesize_gold_review_candidates([record], {record.case_id: sidecar})[record.case_id]
    judge = FakeJudge(
        {
            "evidence_first": _accept(),
            "claim_by_claim": _reject(),
            "adjudicator": _accept(),
        }
    )

    result = asyncio.run(
        evaluate_case(
            record,
            sidecar,
            candidate,
            runtime,
            judge,
            mode="gold_review",
            seed_a=11,
            seed_b=12,
            seed_adjudicator=13,
        )
    )

    assert result.final_verdict == "accept"
    assert result.adjudicated is True
    assert judge.calls == [
        ("evidence_first", 11),
        ("claim_by_claim", 12),
        ("adjudicator", 13),
    ]
    assert all(payload["content_types"] == ["text"] for payload in judge.payloads)
    assert all(payload["challenge_tags"] == ["numbers", "units"] for payload in judge.payloads)


def test_judge_output_error_is_reported_without_private_payload() -> None:
    record, sidecar, runtime = _positive_case()
    candidate = synthesize_gold_review_candidates([record], {record.case_id: sidecar})[
        record.case_id
    ]
    judge = FakeJudge(
        {
            "evidence_first": JudgeOutputError("invalid_schema"),
            "claim_by_claim": JudgeOutputError("invalid_json"),
            "adjudicator": _accept(),
        }
    )

    result = asyncio.run(
        evaluate_case(
            record,
            sidecar,
            candidate,
            runtime,
            judge,
            mode="gold_review",
            seed_a=11,
            seed_b=12,
            seed_adjudicator=13,
        )
    )

    assert result.final_verdict == "reject"
    assert [run.error_code for run in result.judge_runs] == [
        "invalid_schema",
        "invalid_json",
    ]
    assert all(run.decision is None and run.status == "error" for run in result.judge_runs)


def test_judge_run_status_and_payload_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="successful judge run"):
        JudgeRun(
            variant="evidence_first",
            seed=1,
            status="ok",
            decision=None,
            error_code=None,
        )
    with pytest.raises(ValueError, match="failed judge run"):
        JudgeRun(
            variant="evidence_first",
            seed=1,
            status="error",
            decision=None,
            error_code=None,
        )


def test_number_support_uses_values_not_following_prose_as_units() -> None:
    evidence = "Page 2 requires 16.5 MPa in 2 consecutive stages."

    assert _numbers_supported_by_evidence(
        "The value is 16.5 MPa and it applies on page 2 in 2 stages.",
        [evidence],
    )
    assert not _numbers_supported_by_evidence(
        "The value is 16.5 MPa and it applies on page 3.",
        [evidence],
    )


def test_deterministic_checks_reject_changed_recognized_unit() -> None:
    record, sidecar, runtime = _positive_case()
    candidate = synthesize_gold_review_candidates([record], {record.case_id: sidecar})[
        record.case_id
    ].model_copy(update={"candidate_answer": "The required pressure is 16.5 bar."})

    checks = deterministic_checks(record, sidecar, candidate, runtime, mode="gold_review")

    assert not checks.number_unit_consistent
    assert "number_unit" in checks.failure_codes


def test_deterministic_source_failure_rejects_before_judges() -> None:
    record, sidecar, runtime = _positive_case()
    candidate = synthesize_gold_review_candidates([record], {record.case_id: sidecar})[record.case_id]
    bad_runtime = RuntimeCaseData(
        chunks={
            chunk_id: chunk.model_copy(update={"text": "tampered source"})
            for chunk_id, chunk in runtime.chunks.items()
        },
        owner_subs=runtime.owner_subs,
    )
    judge = FakeJudge(
        {
            "evidence_first": _accept(),
            "claim_by_claim": _accept(),
            "adjudicator": _accept(),
        }
    )

    result = asyncio.run(
        evaluate_case(
            record,
            sidecar,
            candidate,
            bad_runtime,
            judge,
            mode="gold_review",
            seed_a=1,
            seed_b=2,
            seed_adjudicator=3,
        )
    )

    assert result.final_verdict == "reject"
    assert "exact_quote" in result.deterministic.failure_codes
    assert judge.calls == []


@pytest.mark.parametrize("language", ["ru", "en", "zh"])
def test_no_answer_gold_review_is_language_safe(language: str) -> None:
    record, sidecar, runtime = _no_answer_case(language)
    candidate = synthesize_gold_review_candidates([record], {record.case_id: sidecar})[record.case_id]

    checks = deterministic_checks(record, sidecar, candidate, runtime, mode="gold_review")

    assert checks.passed
    assert checks.answerability_consistent


def _release_candidates(count: int = 200) -> tuple[list[GoldRecord], list[AutomatedCaseResult]]:
    records: list[GoldRecord] = []
    results: list[AutomatedCaseResult] = []
    languages = ("ru", "en", "zh")
    content_types = ("text", "table", "formula", "figure", "scan")
    hop_types = ("single", "multi", "cross_document")
    ordinary_tags = ("numbers", "units", "standards", "prompt_injection")
    passed = DeterministicChecks(
        passed=True,
        gold_binding=True,
        exact_quote_source=True,
        retrieval_snapshot=True,
        scope_compliant=True,
        evidence_coverage=True,
        number_unit_consistent=True,
        answerability_consistent=True,
        failure_codes=(),
    )
    for index in range(count):
        no_answer = index < 45
        question = f"Synthetic release question {index}"
        hop_type = hop_types[index % len(hop_types)]
        content_type = content_types[index % len(content_types)]
        challenge = ("leakage",) if index < 5 else (ordinary_tags[(index - 5) % len(ordinary_tags)],)
        first_source_hash = hashlib.sha256(f"source-{index}-1".encode()).hexdigest()
        first_document_ref = make_document_ref(first_source_hash)
        documents = [
            DocumentSnapshot(
                document_ref=first_document_ref,
                source_sha256=first_source_hash,
                parsed_content_sha256=hashlib.sha256(f"parsed-{index}-1".encode()).hexdigest(),
                page_count=2,
            )
        ]
        evidence: list[EvidenceRef] = []
        if not no_answer:
            quote = f"Synthetic supporting evidence {index} part 1"
            quote_hash = text_sha256(quote)
            evidence.append(
                EvidenceRef(
                    evidence_id=make_evidence_id(first_source_hash, 1, content_type, quote_hash),
                    document_ref=first_document_ref,
                    page=1,
                    content_type=content_type,
                    content_sha256=quote_hash,
                    relevance_grade=3,
                    bbox=None,
                )
            )
            if hop_type in {"multi", "cross_document"}:
                second_source_hash = (
                    hashlib.sha256(f"source-{index}-2".encode()).hexdigest()
                    if hop_type == "cross_document"
                    else first_source_hash
                )
                second_document_ref = make_document_ref(second_source_hash)
                if hop_type == "cross_document":
                    documents.append(
                        DocumentSnapshot(
                            document_ref=second_document_ref,
                            source_sha256=second_source_hash,
                            parsed_content_sha256=hashlib.sha256(f"parsed-{index}-2".encode()).hexdigest(),
                            page_count=2,
                        )
                    )
                second_quote = f"Synthetic supporting evidence {index} part 2"
                second_quote_hash = text_sha256(second_quote)
                evidence.append(
                    EvidenceRef(
                        evidence_id=make_evidence_id(second_source_hash, 2, content_type, second_quote_hash),
                        document_ref=second_document_ref,
                        page=2,
                        content_type=content_type,
                        content_sha256=second_quote_hash,
                        relevance_grade=3,
                        bbox=None,
                    )
                )
        answer = None if no_answer else f"Synthetic answer {index}"
        record = GoldRecord(
            schema_version="rag-gold-v1",
            scope_id="scope-sha256:" + "5" * 64,
            case_id=f"ragq-release-{index:04d}",
            status="candidate",
            language=languages[index % len(languages)],
            question=question,
            question_sha256=text_sha256(question),
            answerable=not no_answer,
            reference_answer=answer,
            reference_answer_sha256=text_sha256(answer) if answer is not None else None,
            hop_type=hop_type,
            content_types=(content_type,),
            challenge_tags=challenge,
            document_scope=tuple(documents),
            evidence=tuple(evidence),
            review=None,
        )
        records.append(record)
        results.append(
            AutomatedCaseResult(
                case_id=record.case_id,
                scope_id=record.scope_id,
                gold_record_case_sha256=gold_record_case_sha256(record),
                deterministic=passed,
                judge_runs=(),
                final_verdict="accept",
                adjudicated=index % 2 == 0,
            )
        )
    return records, results


def test_release_requires_200_balanced_accepted_records_and_is_private(tmp_path: Path) -> None:
    records, results = _release_candidates()
    reviewed_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    release, failure = build_release_records(records, results, reviewed_at=reviewed_at)

    assert failure is None
    assert len(release) == 200
    assert all(record.status == "reviewed" and record.review is not None for record in release)
    path = tmp_path / "private" / "release.jsonl"
    atomic_release_jsonl(path, release)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(path.read_text(encoding="utf-8").splitlines()) == 200

    rejected_one = [*results]
    rejected_one[0] = rejected_one[0].model_copy(update={"final_verdict": "reject"})
    failed_release, failure = build_release_records(records, rejected_one, reviewed_at=reviewed_at)
    assert failed_release == []
    assert failure == "accepted subset failed release size or coverage gates"


def test_filtered_release_sidecar_is_atomic_private_and_one_to_one(tmp_path: Path) -> None:
    accepted, accepted_sidecar, _ = _positive_case()
    excluded, excluded_sidecar, _ = _no_answer_case("ru")
    full_sidecar = {
        accepted.case_id: accepted_sidecar,
        excluded.case_id: excluded_sidecar,
    }
    path = tmp_path / "private" / "release-sidecar.jsonl"

    atomic_filtered_release_sidecar_jsonl(path, [accepted], full_sidecar)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    filtered = load_private_sidecar(path)
    assert [record.case_id for record in filtered] == [accepted.case_id]
    assert set(full_sidecar) == {accepted.case_id, excluded.case_id}

    path.write_text("sentinel\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="missing accepted release cases"):
        atomic_filtered_release_sidecar_jsonl(
            path,
            [accepted],
            {excluded.case_id: excluded_sidecar},
        )
    assert path.read_text(encoding="utf-8") == "sentinel\n"


def test_review_preflight_requires_0600_inputs_and_fresh_distinct_outputs(tmp_path: Path) -> None:
    private_input = tmp_path / "candidate.jsonl"
    private_input.write_text("{}\n", encoding="utf-8")
    private_input.chmod(0o600)
    assert require_private_input_0600(private_input, name="gold") == private_input.resolve()

    private_input.chmod(0o640)
    with pytest.raises(ValueError, match="0600"):
        require_private_input_0600(private_input, name="gold")

    outputs = [tmp_path / name for name in ("report.json", "release.jsonl", "sidecar.jsonl")]
    assert require_fresh_output_paths(outputs) == tuple(path.resolve() for path in outputs)
    outputs[1].symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError, match="already exists"):
        require_fresh_output_paths(outputs)


def test_review_artifacts_rollback_release_pair_on_publish_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, sidecar, _ = _positive_case()
    report = AutomatedGateReport(
        model="synthetic-qwen",
        mode="gold_review",
        case_count=1,
        accepted_count=1,
        rejected_count=0,
        release_record_count=1,
        release_accepted=True,
        release_failure=None,
        results=(),
    )
    report_path = tmp_path / "private" / "report.json"
    release_path = tmp_path / "private" / "release.jsonl"
    sidecar_path = tmp_path / "private" / "release.sidecar.jsonl"
    real_link = os.link
    link_calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic publish failure")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", fail_second_link)
    with pytest.raises(OSError, match="synthetic publish failure"):
        atomic_review_artifacts(
            report_path,
            report,
            release_path,
            sidecar_path,
            [record],
            {record.case_id: sidecar},
        )
    assert not report_path.exists()
    assert not release_path.exists()
    assert not sidecar_path.exists()

    monkeypatch.setattr(os, "link", real_link)
    atomic_review_artifacts(
        report_path,
        report,
        release_path,
        sidecar_path,
        [record],
        {record.case_id: sidecar},
    )
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (report_path, release_path, sidecar_path))
    assert [item.case_id for item in load_private_sidecar(sidecar_path)] == [record.case_id]


def test_candidate_output_rejects_citations_outside_retrieval() -> None:
    chunk_id = uuid.UUID(int=10)
    with pytest.raises(ValueError, match="subset"):
        CandidateOutput(
            schema_version="rag-candidate-v1",
            case_id="ragq-case-0001",
            scope_id="scope-sha256:" + "6" * 64,
            gold_record_case_sha256="7" * 64,
            candidate_answer="Synthetic",
            cited_chunk_ids=(chunk_id,),
            retrieved=(
                RetrievedChunkRef(
                    chunk_id=uuid.UUID(int=11),
                    document_id=uuid.UUID(int=12),
                    document_ref="doc-sha256:" + "8" * 64,
                ),
            ),
        )


def test_automated_review_database_must_be_loopback_asyncpg() -> None:
    value = "postgresql+asyncpg://rag:secret@localhost:5433/rag_app"
    assert require_loopback_database_url(value) == value
    with pytest.raises(ValueError, match="loopback"):
        require_loopback_database_url("postgresql+asyncpg://rag:secret@db.internal:5432/rag_app")
    with pytest.raises(ValueError, match="asyncpg"):
        require_loopback_database_url("postgresql://rag:secret@localhost:5433/rag_app")
