from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_app.eval.baseline import (
    BaselineCaseMetrics,
    BaselineConfiguration,
    BaselineModelIdentifiers,
    BaselineModelRevisions,
    BaselineProvenance,
    RankedScores,
    RuntimeModelRevision,
    aggregate_metrics,
)
from rag_app.eval.gold_set import (
    GoldRecord,
    ReviewMetadata,
    gold_record_case_sha256,
    make_document_ref,
    make_evidence_id,
    make_scope_id,
    text_sha256,
)
from rag_app.eval.qualification_evidence import (
    JudgeCaseObservation,
    LoadAttemptObservation,
    LoadRunObservations,
    LocalLicenseEvidence,
    LongContextObservation,
    PairedLoadRequestObservation,
    PairedSemanticSafetyObservation,
    QualificationProvenance,
    RestoredModelWeightManifest,
    RollbackProbeObservation,
    RollbackRawEvidence,
    RollbackSmokeObservation,
    RollbackTraceEvent,
    build_raw_qualification_evidence,
)
from rag_app.eval.release_gate import (
    GateRuntimeProvenance,
    ReleaseGateError,
    ReleaseGatePolicy,
    evaluate_release_gate,
    load_baseline_report,
    load_policy,
)

CONTENT_TYPES = ("text", "table", "formula", "figure", "scan")
HOP_TYPES = ("single", "multi", "cross_document")
LANGUAGES = ("ru", "en", "zh")
BASELINE_SHA = "0" * 64
CANDIDATE_SHA = "1" * 64
GOLD_SHA = "2" * 64
SIDECAR_SHA = "3" * 64
CORPUS_SHA = "4" * 64
RUNTIME_SHA = "5" * 64
GIT_SHA = "7" * 40
GATE_GIT_SHA = "6" * 40
QUALIFICATION_SHA = "8" * 64
POLICY_SHA = "9" * 64
BASELINE_ATTESTATION_SHA = "a" * 64
CANDIDATE_ATTESTATION_SHA = "b" * 64
QUALIFICATION_ATTESTATION_SHA = "c" * 64


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "compare_rag_baselines.py"
    spec = importlib.util.spec_from_file_location("compare_rag_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT = _script_module()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _gate_runtime() -> GateRuntimeProvenance:
    return GateRuntimeProvenance(
        git_sha=GATE_GIT_SHA,
        git_dirty=False,
        comparator_sha256="d" * 64,
        release_gate_sha256="e" * 64,
        private_artifacts_sha256="f" * 64,
        report_attestation_sha256="0" * 64,
        qualification_evidence_sha256="1" * 64,
    )


def _raw_record(index: int) -> dict:
    no_answer = index < 45
    leakage = 40 <= index < 45
    answerable = not no_answer
    hop_type = HOP_TYPES[index % len(HOP_TYPES)] if answerable else "single"
    content_type = CONTENT_TYPES[index % len(CONTENT_TYPES)]
    document_count = 2 if hop_type == "cross_document" else 1
    documents = []
    for document_index in range(document_count):
        source_hash = _sha(f"document-{index}-{document_index}")
        documents.append(
            {
                "document_ref": make_document_ref(source_hash),
                "source_sha256": source_hash,
                "parsed_content_sha256": _sha(f"parsed-{index}-{document_index}"),
                "page_count": 10,
            }
        )
    evidence = []
    if answerable:
        evidence_count = 2 if hop_type in {"multi", "cross_document"} else 1
        for evidence_index in range(evidence_count):
            document = documents[evidence_index % len(documents)]
            content_hash = _sha(f"evidence-{index}-{evidence_index}")
            evidence.append(
                {
                    "evidence_id": make_evidence_id(
                        document["source_sha256"],
                        evidence_index + 1,
                        content_type,
                        content_hash,
                    ),
                    "document_ref": document["document_ref"],
                    "page": evidence_index + 1,
                    "content_type": content_type,
                    "content_sha256": content_hash,
                    "relevance_grade": 3,
                    "bbox": None,
                }
            )
    tags: list[str] = []
    if leakage:
        tags = ["leakage"]
    elif 45 <= index < 50:
        tags = ["prompt_injection"]
    elif 50 <= index < 55:
        tags = ["standards"]
    elif 55 <= index < 60:
        tags = ["numbers"]
    elif 60 <= index < 65:
        tags = ["units"]
    question = f"Synthetic technical question {index}?"
    answer = f"Synthetic supported answer {index}." if answerable else None
    return {
        "schema_version": "rag-gold-v1",
        "case_id": f"ragq-release-gate-{index:04d}",
        "status": "candidate",
        "scope_id": make_scope_id("synthetic-owner"),
        "language": LANGUAGES[index % len(LANGUAGES)],
        "question": question,
        "question_sha256": text_sha256(question),
        "answerable": answerable,
        "reference_answer": answer,
        "reference_answer_sha256": text_sha256(answer) if answer else None,
        "hop_type": hop_type,
        "content_types": [content_type],
        "challenge_tags": tags,
        "document_scope": documents,
        "evidence": evidence,
        "review": None,
    }


def _records() -> list[GoldRecord]:
    records = []
    reviewed_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    for index in range(200):
        candidate = GoldRecord.model_validate_json(json.dumps(_raw_record(index)), strict=True)
        raw = candidate.model_dump(mode="json")
        raw["status"] = "reviewed"
        raw["review"] = ReviewMetadata(
            reviewer_id="auto-test-reviewer-v1",
            reviewed_at=reviewed_at,
            case_sha256=gold_record_case_sha256(candidate),
        ).model_dump(mode="json")
        records.append(GoldRecord.model_validate_json(json.dumps(raw), strict=True))
    return records


def _revision(weight: str) -> RuntimeModelRevision:
    return RuntimeModelRevision(
        endpoint_metadata_sha256="a" * 64,
        runtime_version_sha256="b" * 64,
        runtime_process_sha256="c" * 64,
        local_config_manifest_sha256="d" * 64,
        weight_manifest_sha256=weight,
        weight_file_count=2,
        weight_bytes=1024,
        declared_revision="e" * 40,
    )


def _configuration() -> BaselineConfiguration:
    return BaselineConfiguration(
        top_k=10,
        dense_top_k=50,
        sparse_top_k=50,
        rerank_top_k=20,
        rerank_min_score=0.02,
        embedding_dim=1024,
        visual_enabled=False,
        context_max_chars=28000,
        context_window_tokens=16384,
        output_tokens=2048,
        answer_route="doc_only",
        prompt_sha256="f" * 64,
        temperature=0.2,
        top_p=0.8,
        seed_namespace=2026071300,
        seed_strategy="case-id-sha256-v1",
        enable_thinking=False,
    )


CONFIG_SHA = hashlib.sha256(
    json.dumps(
        _configuration().model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def _provenance(*, candidate: bool, git_sha: str = GIT_SHA) -> BaselineProvenance:
    common_revision = _revision("b" * 64)
    llm_revision = _revision("c" * 64) if candidate else common_revision
    return BaselineProvenance(
        runner="retrieval_direct_answer",
        evaluation_mode="release",
        evaluated_at=datetime(2026, 7, 13, 12, 30 if candidate else 0, tzinfo=UTC),
        git_sha=git_sha,
        git_dirty=False,
        gold_artifact_sha256=GOLD_SHA,
        sidecar_artifact_sha256=SIDECAR_SHA,
        corpus_fingerprint_sha256=CORPUS_SHA,
        runtime_corpus_snapshot_sha256=RUNTIME_SHA,
        scope_count=1,
        document_snapshot_count=200,
        models=BaselineModelIdentifiers(
            llm="candidate-llm" if candidate else "baseline-llm",
            embedding="embedding",
            reranker="reranker",
        ),
        model_revisions=BaselineModelRevisions(
            llm=llm_revision,
            embedding=common_revision,
            reranker=common_revision,
        ),
        configuration=_configuration(),
        configuration_sha256=CONFIG_SHA,
    )


def _case_metrics(record: GoldRecord, index: int, *, improved: bool) -> BaselineCaseMetrics:
    eligible = record.answerable
    score = 1.0 if eligible else None
    quantity_accuracy = None
    if eligible:
        quantity_accuracy = float(index % 4 != 0 or improved)
    ranked = {
        key: RankedScores(
            recall={"value": score},
            mrr={"value": score},
            ndcg={"value": score},
        )
        for key in ("1", "5", "10")
    }
    return BaselineCaseMetrics(
        case_id=record.case_id,
        answerable=record.answerable,
        answerability_correct=True,
        abstained=not record.answerable,
        ranked=ranked,
        citation={
            "citation_precision": score,
            "citation_recall": score,
            "citation_count": 1 if eligible else 0,
        },
        quantities={
            "quantity_unit_accuracy": quantity_accuracy,
            "quantity_unit_recall": score,
            "mentioned_number_count": 1 if eligible else 0,
            "unsupported_number_count": 0,
        },
        retrieval_ms=100.0,
        generation_ms=200.0,
        total_ms=300.0,
    )


def _reports(*, improved: bool = True, candidate_git_sha: str = GIT_SHA):
    records = _records()
    baseline_cases = tuple(
        _case_metrics(record, index, improved=False) for index, record in enumerate(records)
    )
    candidate_cases = tuple(
        _case_metrics(record, index, improved=improved) for index, record in enumerate(records)
    )
    baseline = aggregate_metrics(baseline_cases, provenance=_provenance(candidate=False))
    candidate = aggregate_metrics(
        candidate_cases,
        provenance=_provenance(candidate=True, git_sha=candidate_git_sha),
    )
    return records, baseline, candidate


def _policy() -> ReleaseGatePolicy:
    metric_args = {
        "answerability_accuracy": ("higher", 0.02, 0.0),
        "recall_at_1": ("higher", 0.02, 0.0),
        "recall_at_5": ("higher", 0.01, 0.0),
        "recall_at_10": ("higher", 0.01, 0.0),
        "mrr_at_10": ("higher", 0.01, 0.0),
        "ndcg_at_10": ("higher", 0.01, 0.01),
        "citation_precision": ("higher", 0.02, 0.0),
        "citation_recall": ("higher", 0.02, 0.0),
        "quantity_unit_accuracy": ("higher", 0.02, 0.02),
        "quantity_unit_recall": ("higher", 0.02, 0.0),
        "unsupported_number_rate": ("lower", 0.01, 0.0),
        "latency_p95_ms": ("lower", 250.0, 0.0),
    }
    metrics = []
    for name, (direction, margin, practical) in metric_args.items():
        metrics.append(
            {
                "name": name,
                "direction": direction,
                "absolute_noninferiority_margin": margin,
                "relative_noninferiority_margin": 0.1 if name == "latency_p95_ms" else None,
                "practical_improvement": practical,
                "minimum": None,
                "maximum": 0.4 if name == "unsupported_number_rate" else None,
            }
        )
    return ReleaseGatePolicy.model_validate_json(
        json.dumps(
            {
                "schema_version": "rag-release-policy-v1",
                "policy_id": "synthetic-policy",
                "reference_report_sha256": BASELINE_SHA,
                "reference_git_sha": GIT_SHA,
                "gold_artifact_sha256": GOLD_SHA,
                "sidecar_artifact_sha256": SIDECAR_SHA,
                "corpus_fingerprint_sha256": CORPUS_SHA,
                "runtime_corpus_snapshot_sha256": RUNTIME_SHA,
                "configuration_sha256": CONFIG_SHA,
                "allowed_model_roles": ["llm", "reranker"],
                "target_metric_by_role": {
                    "llm": "quantity_unit_accuracy",
                    "reranker": "ndcg_at_10",
                },
                "allowed_spdx_licenses": ["Apache-2.0"],
                "attestation_key_id": "a" * 64,
                "approved_model_licenses": [
                    {
                        "role": "llm",
                        "model": "candidate-llm",
                        "weight_manifest_sha256": "c" * 64,
                        "spdx_license": "Apache-2.0",
                        "license_text_sha256": _sha("Apache License 2.0"),
                    }
                ],
                "trusted_judge": {
                    "model": "fixed-judge",
                    "declared_revision": "judge-revision",
                    "weight_manifest_sha256": "b" * 64,
                    "config_sha256": "e" * 64,
                    "prompt_sha256": "c" * 64,
                },
                "qualification_max_age_hours": 24,
                "min_case_count": 200,
                "bootstrap_samples": 1000,
                "bootstrap_seed": 1234,
                "familywise_alpha": 0.05,
                "target_alpha": 0.05,
                "metrics": metrics,
                "slice_margin_language_hop": 0.03,
                "slice_margin_content": 0.05,
                "slice_min_statistical_count": 20,
                "long_context_min_cases": 30,
                "long_context_min_window_utilization": 0.85,
                "long_context_max_window_utilization": 0.95,
                "load_min_concurrency": 10,
                "load_min_requests": 200,
                "load_max_p95_regression": 0.1,
                "load_min_throughput_ratio": 0.9,
                "semantic_noninferiority_margin": 0.01,
                "safety_min_cases": 10,
                "rollback_max_seconds": 600.0,
                "rollback_min_smoke_cases": 10,
            }
        )
    )


def _qualification(candidate_report, records: list[GoldRecord] | None = None):
    release_records = records or _records()
    license_bytes = b"Apache License 2.0"
    provenance = QualificationProvenance(
        generated_at=datetime(2026, 7, 13, 13, 0, tzinfo=UTC),
        producer_git_sha=GIT_SHA,
        git_dirty=False,
        candidate_role="llm",
        candidate_model="candidate-llm",
        candidate_declared_revision="e" * 40,
        candidate_weight_manifest_sha256="c" * 64,
        candidate_config_sha256="d" * 64,
        baseline_model="baseline-llm",
        baseline_weight_manifest_sha256="b" * 64,
        baseline_config_sha256="d" * 64,
        rag_configuration_sha256=CONFIG_SHA,
        baseline_report_sha256=BASELINE_SHA,
        candidate_report_sha256=CANDIDATE_SHA,
        gold_artifact_sha256=GOLD_SHA,
        sidecar_artifact_sha256=SIDECAR_SHA,
        corpus_fingerprint_sha256=CORPUS_SHA,
        runtime_corpus_snapshot_sha256=RUNTIME_SHA,
        judge_model="fixed-judge",
        judge_declared_revision="judge-revision",
        judge_weight_manifest_sha256="b" * 64,
        judge_config_sha256="e" * 64,
        judge_prompt_sha256="c" * 64,
        reference_git_sha=GIT_SHA,
    )
    license_evidence = LocalLicenseEvidence(
        role="llm",
        model="candidate-llm",
        weight_manifest_sha256="c" * 64,
        spdx_license="Apache-2.0",
        source_url="https://huggingface.co/example/candidate-llm",
        local_relative_path="LICENSE",
        license_bytes_base64=base64.b64encode(license_bytes).decode(),
        license_byte_count=len(license_bytes),
        license_text_sha256=hashlib.sha256(license_bytes).hexdigest(),
        commercial_on_prem_allowed=True,
    )
    long_context = tuple(
        LongContextObservation(
            case_id=f"long-{index}",
            language=LANGUAGES[index % 3],
            input_tokens=14_000,
            model_context_tokens=16_384,
            outcome="completed",
            duration_ms=100,
            output_sha256=_sha(f"long-{index}"),
        )
        for index in range(30)
    )
    load_requests = tuple(
        PairedLoadRequestObservation(
            request_id=f"request-{index}",
            case_id=release_records[index % len(release_records)].case_id,
            baseline=LoadAttemptObservation(
                started_offset_ms=index * 10,
                finished_offset_ms=index * 10 + 100,
                outcome="completed",
                response_sha256=_sha(f"baseline-load-{index}"),
            ),
            candidate=LoadAttemptObservation(
                started_offset_ms=index * 10,
                finished_offset_ms=index * 10 + 105,
                outcome="completed",
                response_sha256=_sha(f"candidate-load-{index}"),
            ),
        )
        for index in range(200)
    )
    load = LoadRunObservations(
        concurrency=10,
        baseline_duration_ms=20_000,
        candidate_duration_ms=20_000,
        requests=load_requests,
    )

    def judgment(name: str) -> JudgeCaseObservation:
        return JudgeCaseObservation(verdict="pass", response_sha256=_sha(name))

    semantic = tuple(
        PairedSemanticSafetyObservation(
            case_id=record.case_id,
            gold_case_sha256=gold_record_case_sha256(record),
            categories=tuple(
                ["semantic"]
                + (["safety"] if {"leakage", "prompt_injection"} & set(record.challenge_tags) else [])
                + (["standards"] if "standards" in record.challenge_tags else [])
            ),
            baseline_output_sha256=_sha(f"baseline-output-{record.case_id}"),
            candidate_output_sha256=_sha(f"candidate-output-{record.case_id}"),
            baseline=judgment(f"baseline-judge-{record.case_id}"),
            candidate=judgment(f"candidate-judge-{record.case_id}"),
        )
        for record in release_records
    )
    started = datetime(2026, 7, 13, 13, 10, tzinfo=UTC)
    rollback = RollbackRawEvidence(
        reference_report_sha256=BASELINE_SHA,
        restored_git_sha=GIT_SHA,
        restored_model_weight_manifests=tuple(
            RestoredModelWeightManifest(role=role, weight_manifest_sha256="b" * 64)
            for role in ("llm", "embedding", "reranker")
        ),
        restored_configuration_sha256="d" * 64,
        restored_rag_configuration_sha256=CONFIG_SHA,
        restored_runtime_corpus_snapshot_sha256=RUNTIME_SHA,
        trace=(
            RollbackTraceEvent(
                sequence=0,
                kind="rollback_started",
                observed_at=started,
                success=True,
                evidence_sha256=_sha("rollback-started"),
            ),
            RollbackTraceEvent(
                sequence=1,
                kind="config_restored",
                observed_at=started + timedelta(seconds=20),
                success=True,
                evidence_sha256=_sha("config-restored"),
            ),
            RollbackTraceEvent(
                sequence=2,
                kind="code_restored",
                observed_at=started + timedelta(seconds=40),
                success=True,
                evidence_sha256=_sha("code-restored"),
            ),
            RollbackTraceEvent(
                sequence=3,
                kind="services_restarted",
                observed_at=started + timedelta(seconds=60),
                success=True,
                evidence_sha256=_sha("services-restarted"),
            ),
            RollbackTraceEvent(
                sequence=4,
                kind="verification_started",
                observed_at=started + timedelta(seconds=90),
                success=True,
                evidence_sha256=_sha("verification-started"),
            ),
            RollbackTraceEvent(
                sequence=5,
                kind="rollback_completed",
                observed_at=started + timedelta(seconds=120),
                success=True,
                evidence_sha256=_sha("rollback-completed"),
            ),
        ),
        probes=(
            RollbackProbeObservation(
                kind="health", target="/healthz", passed=True, status_code=200, response_sha256=_sha("health")
            ),
            RollbackProbeObservation(
                kind="root", target="/", passed=True, status_code=200, response_sha256=_sha("root")
            ),
            RollbackProbeObservation(
                kind="auth_enabled", target="config", passed=True, response_sha256=_sha("auth")
            ),
            RollbackProbeObservation(
                kind="anonymous_protected",
                target="/api/documents",
                passed=True,
                status_code=401,
                response_sha256=_sha("anonymous"),
            ),
            *tuple(
                RollbackProbeObservation(
                    kind="model_endpoint",
                    target=role,
                    passed=True,
                    status_code=200,
                    response_sha256=_sha(role),
                )
                for role in ("llm", "embedding", "reranker")
            ),
        ),
        smoke=tuple(
            RollbackSmokeObservation(
                case_id=f"smoke-{index}", passed=True, result_sha256=_sha(f"smoke-{index}")
            )
            for index in range(10)
        ),
    )
    return build_raw_qualification_evidence(
        provenance=provenance,
        license=license_evidence,
        long_context_observations=long_context,
        load_observations=load,
        semantic_safety_observations=semantic,
        rollback_trace=rollback,
    )


def _evaluate(*, improved: bool = True, qualification=None):
    records, baseline, candidate = _reports(improved=improved)
    evidence = qualification or _qualification(candidate)
    decision = evaluate_release_gate(
        baseline,
        candidate,
        records,
        evidence,
        _policy(),
        evaluated_at=datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
        baseline_sha256=BASELINE_SHA,
        baseline_attestation_sha256=BASELINE_ATTESTATION_SHA,
        candidate_sha256=CANDIDATE_SHA,
        candidate_attestation_sha256=CANDIDATE_ATTESTATION_SHA,
        gold_sha256=GOLD_SHA,
        sidecar_sha256=SIDECAR_SHA,
        qualification_sha256=QUALIFICATION_SHA,
        qualification_attestation_sha256=QUALIFICATION_ATTESTATION_SHA,
        policy_sha256=POLICY_SHA,
        gate_runtime=_gate_runtime(),
    )
    return decision


def test_accepts_paired_candidate_with_practical_statistical_improvement() -> None:
    decision = _evaluate()
    assert decision.accepted
    assert decision.failure_codes == ()
    target = next(metric for metric in decision.metrics if metric.target_metric)
    assert target.name == "quantity_unit_accuracy"
    assert target.improvement >= 0.02
    assert target.target_ci_low > 0
    assert all(item.passed for item in decision.metrics)
    assert all(item.passed for item in decision.slices)


def test_rejects_same_quality_candidate_without_target_improvement() -> None:
    decision = _evaluate(improved=False)
    assert not decision.accepted
    assert "metric_failed:quantity_unit_accuracy" in decision.failure_codes


def test_rejects_incomplete_operational_qualification() -> None:
    _, _, candidate = _reports()
    evidence = _qualification(candidate)
    first = evidence.load_observations.requests[0]
    failed_attempt = LoadAttemptObservation(
        started_offset_ms=first.candidate.started_offset_ms,
        finished_offset_ms=first.candidate.finished_offset_ms,
        outcome="error",
        error_code="runtime_error",
    )
    broken_load = evidence.load_observations.model_copy(
        update={
            "requests": (
                first.model_copy(update={"candidate": failed_attempt}),
                *evidence.load_observations.requests[1:],
            )
        }
    )
    broken = build_raw_qualification_evidence(
        provenance=evidence.provenance,
        license=evidence.license,
        long_context_observations=evidence.long_context_observations,
        load_observations=broken_load,
        semantic_safety_observations=evidence.semantic_safety_observations,
        rollback_trace=evidence.rollback_trace,
    )
    decision = _evaluate(qualification=broken)
    assert not decision.accepted
    assert "load_gate_failed" in decision.failure_codes


def test_rejects_candidate_from_different_git_even_with_same_evaluator_sources() -> None:
    records, baseline, candidate = _reports(candidate_git_sha="8" * 40)
    with pytest.raises(ReleaseGateError, match=r"incomparable \(git_sha\)"):
        evaluate_release_gate(
            baseline,
            candidate,
            records,
            _qualification(candidate),
            _policy(),
            evaluated_at=datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
            baseline_sha256=BASELINE_SHA,
            baseline_attestation_sha256=BASELINE_ATTESTATION_SHA,
            candidate_sha256=CANDIDATE_SHA,
            candidate_attestation_sha256=CANDIDATE_ATTESTATION_SHA,
            gold_sha256=GOLD_SHA,
            sidecar_sha256=SIDECAR_SHA,
            qualification_sha256=QUALIFICATION_SHA,
            qualification_attestation_sha256=QUALIFICATION_ATTESTATION_SHA,
            policy_sha256=POLICY_SHA,
            gate_runtime=_gate_runtime(),
        )


def test_rejects_reports_from_same_non_reference_git() -> None:
    records, baseline, candidate = _reports(candidate_git_sha="8" * 40)
    baseline = baseline.model_copy(
        update={
            "provenance": baseline.provenance.model_copy(update={"git_sha": "8" * 40})
        }
    )

    with pytest.raises(ReleaseGateError, match=r"pinned policy \(git_sha\)"):
        evaluate_release_gate(
            baseline,
            candidate,
            records,
            _qualification(candidate),
            _policy(),
            evaluated_at=datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
            baseline_sha256=BASELINE_SHA,
            baseline_attestation_sha256=BASELINE_ATTESTATION_SHA,
            candidate_sha256=CANDIDATE_SHA,
            candidate_attestation_sha256=CANDIDATE_ATTESTATION_SHA,
            gold_sha256=GOLD_SHA,
            sidecar_sha256=SIDECAR_SHA,
            qualification_sha256=QUALIFICATION_SHA,
            qualification_attestation_sha256=QUALIFICATION_ATTESTATION_SHA,
            policy_sha256=POLICY_SHA,
            gate_runtime=_gate_runtime(),
        )


@pytest.mark.parametrize("bad_side", ["baseline", "candidate"])
def test_rejects_report_attestation_from_non_reference_git(bad_side: str) -> None:
    policy = _policy()
    baseline_git_sha = "8" * 40 if bad_side == "baseline" else policy.reference_git_sha
    candidate_git_sha = "8" * 40 if bad_side == "candidate" else policy.reference_git_sha
    baseline_attestation = SimpleNamespace(repository_git_sha=baseline_git_sha)
    candidate_attestation = SimpleNamespace(repository_git_sha=candidate_git_sha)

    with pytest.raises(ReleaseGateError, match="attestation Git binding"):
        _SCRIPT._validate_report_attestation_git_bindings(
            baseline_attestation,
            candidate_attestation,
            policy,
        )


def test_rejects_model_alias_without_changed_weight_manifest() -> None:
    records, baseline, candidate = _reports()
    revisions = candidate.provenance.model_revisions.model_copy(
        update={"llm": baseline.provenance.model_revisions.llm}
    )
    provenance = candidate.provenance.model_copy(update={"model_revisions": revisions})
    aliased = candidate.model_copy(update={"provenance": provenance})
    with pytest.raises(ReleaseGateError, match="weight manifest"):
        evaluate_release_gate(
            baseline,
            aliased,
            records,
            _qualification(aliased),
            _policy(),
            evaluated_at=datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
            baseline_sha256=BASELINE_SHA,
            baseline_attestation_sha256=BASELINE_ATTESTATION_SHA,
            candidate_sha256=CANDIDATE_SHA,
            candidate_attestation_sha256=CANDIDATE_ATTESTATION_SHA,
            gold_sha256=GOLD_SHA,
            sidecar_sha256=SIDECAR_SHA,
            qualification_sha256=QUALIFICATION_SHA,
            qualification_attestation_sha256=QUALIFICATION_ATTESTATION_SHA,
            policy_sha256=POLICY_SHA,
            gate_runtime=_gate_runtime(),
        )


def test_qualification_case_counts_are_bound_to_gold_tags() -> None:
    _, _, candidate = _reports()
    evidence = _qualification(candidate)
    observations = list(evidence.semantic_safety_observations)
    target = next(
        index for index, observation in enumerate(observations) if observation.categories == ("semantic",)
    )
    observations[target] = observations[target].model_copy(update={"categories": ("semantic", "safety")})
    broken = build_raw_qualification_evidence(
        provenance=evidence.provenance,
        license=evidence.license,
        long_context_observations=evidence.long_context_observations,
        load_observations=evidence.load_observations,
        semantic_safety_observations=observations,
        rollback_trace=evidence.rollback_trace,
    )
    with pytest.raises(ReleaseGateError, match="judgment binding"):
        _evaluate(qualification=broken)


def test_bootstrap_decision_is_deterministic_and_contains_no_gold_text() -> None:
    first = _evaluate()
    second = _evaluate()
    assert first == second
    serialized = first.model_dump_json()
    assert "Synthetic technical question" not in serialized
    assert "Synthetic supported answer" not in serialized


def test_paired_semantic_noninferiority_rejects_hidden_aggregate_decline() -> None:
    _, _, candidate = _reports()
    evidence = _qualification(candidate)
    observations = list(evidence.semantic_safety_observations)
    index = next(
        offset for offset, observation in enumerate(observations) if observation.categories == ("semantic",)
    )
    failed = JudgeCaseObservation(
        verdict="fail",
        response_sha256=_sha("semantic-failure"),
        reason_codes=("unsupported_claim",),
    )
    observations[index] = observations[index].model_copy(update={"candidate": failed})
    degraded = build_raw_qualification_evidence(
        provenance=evidence.provenance,
        license=evidence.license,
        long_context_observations=evidence.long_context_observations,
        load_observations=evidence.load_observations,
        semantic_safety_observations=observations,
        rollback_trace=evidence.rollback_trace,
    )

    decision = _evaluate(qualification=degraded)

    assert not decision.accepted
    assert "semantic_gate_failed" in decision.failure_codes
    assert decision.qualification.semantic_candidate == pytest.approx(0.995)


def test_rejects_stale_or_future_qualification() -> None:
    _, _, candidate = _reports()
    evidence = _qualification(candidate)
    stale_provenance = evidence.provenance.model_copy(
        update={"generated_at": datetime(2026, 7, 10, 13, 0, tzinfo=UTC)}
    )
    stale = evidence.model_copy(update={"provenance": stale_provenance})

    with pytest.raises(ReleaseGateError, match="binding mismatch"):
        _evaluate(qualification=stale)


def test_slice_regression_is_reported_fail_closed() -> None:
    records, baseline, candidate = _reports()
    cases = list(candidate.cases)
    changed = 0
    for index, record in enumerate(records):
        if record.language != "zh" or not record.answerable:
            continue
        case = cases[index]
        ranked = dict(case.ranked)
        ranked["10"] = RankedScores(
            recall={"value": 0.0},
            mrr={"value": 0.0},
            ndcg={"value": 0.0},
        )
        cases[index] = case.model_copy(update={"ranked": ranked})
        changed += 1
        if changed == 4:
            break
    degraded = aggregate_metrics(cases, provenance=candidate.provenance)
    decision = evaluate_release_gate(
        baseline,
        degraded,
        records,
        _qualification(degraded),
        _policy(),
        evaluated_at=datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
        baseline_sha256=BASELINE_SHA,
        baseline_attestation_sha256=BASELINE_ATTESTATION_SHA,
        candidate_sha256=CANDIDATE_SHA,
        candidate_attestation_sha256=CANDIDATE_ATTESTATION_SHA,
        gold_sha256=GOLD_SHA,
        sidecar_sha256=SIDECAR_SHA,
        qualification_sha256=QUALIFICATION_SHA,
        qualification_attestation_sha256=QUALIFICATION_ATTESTATION_SHA,
        policy_sha256=POLICY_SHA,
        gate_runtime=_gate_runtime(),
    )
    assert not decision.accepted
    assert "slice_failed:language:zh" in decision.failure_codes


def test_loader_recomputes_aggregates_and_requires_0600(tmp_path: Path) -> None:
    _, baseline, _ = _reports()
    path = tmp_path / ".private" / "baseline.json"
    path.parent.mkdir()
    os.chmod(path.parent, 0o700)
    path.write_text(baseline.model_dump_json(), encoding="utf-8")
    os.chmod(path, 0o600)
    assert load_baseline_report(path, repository_root=tmp_path).case_count == 200

    raw = baseline.model_dump(mode="json")
    raw["answerability_accuracy"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReleaseGateError, match="aggregates"):
        load_baseline_report(path, repository_root=tmp_path)

    path.write_text(baseline.model_dump_json(), encoding="utf-8")
    os.chmod(path, 0o640)
    with pytest.raises(ReleaseGateError, match="0600"):
        load_baseline_report(path, repository_root=tmp_path)


def test_loader_binds_configuration_hash_to_payload(tmp_path: Path) -> None:
    _, baseline, _ = _reports()
    raw = baseline.model_dump(mode="json")
    raw["provenance"]["configuration"]["top_k"] = 64
    path = tmp_path / ".private" / "baseline.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(json.dumps(raw), encoding="utf-8")
    os.chmod(path, 0o600)

    with pytest.raises(ReleaseGateError, match="invalid"):
        load_baseline_report(path, repository_root=tmp_path)


def test_loader_rejects_latency_below_component_sum(tmp_path: Path) -> None:
    _, baseline, _ = _reports()
    raw = baseline.model_dump(mode="json")
    raw["cases"][0]["total_ms"] = 1.0
    rebuilt = aggregate_metrics(
        tuple(BaselineCaseMetrics.model_validate(case) for case in raw["cases"]),
        provenance=baseline.provenance,
    )
    path = tmp_path / ".private" / "baseline.json"
    path.parent.mkdir(mode=0o700)
    path.write_text(rebuilt.model_dump_json(), encoding="utf-8")
    os.chmod(path, 0o600)

    with pytest.raises(ReleaseGateError, match="component latency"):
        load_baseline_report(path, repository_root=tmp_path)


def test_rare_content_slice_allows_no_per_case_regression() -> None:
    records, baseline, candidate = _reports()
    rare_ids = {record.case_id for record in records if record.answerable}
    rare_ids = set(sorted(rare_ids)[:10])
    sliced_records = [
        record.model_copy(update={"content_types": ("scan",) if record.case_id in rare_ids else ("text",)})
        for record in records
    ]
    degraded_cases = []
    for case in candidate.cases:
        if case.case_id not in rare_ids:
            degraded_cases.append(case)
            continue
        ranked = dict(case.ranked)
        ranked["10"] = RankedScores(
            recall={"value": 0.01},
            mrr={"value": 0.01},
            ndcg={"value": 0.01},
        )
        degraded_cases.append(case.model_copy(update={"ranked": ranked}))
    degraded = aggregate_metrics(degraded_cases, provenance=candidate.provenance)

    decision = evaluate_release_gate(
        baseline,
        degraded,
        sliced_records,
        _qualification(degraded, sliced_records),
        _policy(),
        evaluated_at=datetime(2026, 7, 13, 14, 0, tzinfo=UTC),
        baseline_sha256=BASELINE_SHA,
        baseline_attestation_sha256=BASELINE_ATTESTATION_SHA,
        candidate_sha256=CANDIDATE_SHA,
        candidate_attestation_sha256=CANDIDATE_ATTESTATION_SHA,
        gold_sha256=GOLD_SHA,
        sidecar_sha256=SIDECAR_SHA,
        qualification_sha256=QUALIFICATION_SHA,
        qualification_attestation_sha256=QUALIFICATION_ATTESTATION_SHA,
        policy_sha256=POLICY_SHA,
        gate_runtime=_gate_runtime(),
    )

    assert not decision.accepted
    assert "slice_failed:content_type:scan" in decision.failure_codes


def test_release_decision_writer_is_fresh_atomic_and_private(tmp_path: Path) -> None:
    output = tmp_path / ".private" / "decision.json"
    output.parent.mkdir(mode=0o700)
    payload = {"accepted": False, "failure_codes": ["PRIVATE-MARKER-NOT-A-CASE"]}
    _SCRIPT._atomic_write_fresh(output, payload, repository_root=tmp_path)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert os.stat(output).st_mode & 0o777 == 0o600
    with pytest.raises(ReleaseGateError, match="already exists"):
        _SCRIPT._atomic_write_fresh(output, payload, repository_root=tmp_path)


def test_production_policy_is_strict_and_pinned() -> None:
    policy_path = Path(__file__).parents[1] / "deploy" / "rag-eval" / "release-policy-v1.json"
    policy = load_policy(policy_path)
    assert policy.bootstrap_samples == 20_000
    assert policy.reference_report_sha256 == (
        "ef79566abdb340d4d7a1504cfea6f7839f08c68d8046320a78ecc0ce374bf336"
    )
    assert policy.allowed_model_roles == ("llm",)
    assert policy.allowed_spdx_licenses == ("Apache-2.0",)
    assert policy.approved_model_licenses == ()
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == _SCRIPT.EXPECTED_POLICY_SHA256


def test_release_cli_has_fixed_policy_and_distinct_exit_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _SCRIPT.build_parser()
    with pytest.raises(SystemExit) as missing:
        parser.parse_args([])
    assert missing.value.code == 64
    required = [
        "baseline.json",
        "candidate.json",
        "gold.jsonl",
        "sidecar.jsonl",
        "qualification.json",
        "--baseline-attestation",
        "baseline.attestation.json",
        "--candidate-attestation",
        "candidate.attestation.json",
        "--qualification-attestation",
        "qualification.attestation.json",
        "--attestation-key",
        "/private/key",
        "--output",
        "decision.json",
    ]
    with pytest.raises(SystemExit) as override:
        parser.parse_args([*required, "--policy", "permissive.json"])
    assert override.value.code == 64

    parsed = parser.parse_args(required)
    monkeypatch.setattr(
        _SCRIPT,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda: parsed),
    )
    for result in (0, 2):
        monkeypatch.setattr(_SCRIPT, "run", lambda _args, code=result: code)
        with pytest.raises(SystemExit) as outcome:
            _SCRIPT.main()
        assert outcome.value.code == result

    monkeypatch.setattr(
        _SCRIPT,
        "run",
        lambda _args: (_ for _ in ()).throw(ReleaseGateError("private detail")),
    )
    with pytest.raises(SystemExit) as invalid:
        _SCRIPT.main()
    assert invalid.value.code == 3

    monkeypatch.setattr(
        _SCRIPT,
        "run",
        lambda _args: (_ for _ in ()).throw(RuntimeError("private detail")),
    )
    with pytest.raises(SystemExit) as operational:
        _SCRIPT.main()
    assert operational.value.code == 4
