from __future__ import annotations

import base64
import hashlib
import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rag_app.eval.qualification_evidence import (
    JudgeCaseObservation,
    LoadAttemptObservation,
    LoadRunObservations,
    LoadRuntimeEvent,
    LocalLicenseEvidence,
    LongContextObservation,
    PairedLoadRequestObservation,
    PairedSemanticSafetyObservation,
    QualificationEvidenceError,
    QualificationProvenance,
    RawQualificationEvidence,
    RestoredModelWeightManifest,
    RollbackProbeObservation,
    RollbackRawEvidence,
    RollbackSmokeObservation,
    RollbackTraceEvent,
    aggregate_load,
    build_raw_qualification_evidence,
    capture_local_license,
    load_private_qualification_evidence,
    qualification_evidence_json_schema,
    qualification_evidence_sha256,
    verify_raw_qualification_evidence,
    write_private_qualification_evidence,
)


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(raw).hexdigest()


def _provenance() -> QualificationProvenance:
    return QualificationProvenance(
        generated_at=datetime(2026, 7, 13, 15, 0, tzinfo=UTC),
        producer_git_sha="1" * 40,
        git_dirty=False,
        candidate_role="llm",
        candidate_model="candidate-qwen",
        candidate_declared_revision="candidate-revision",
        candidate_weight_manifest_sha256="2" * 64,
        candidate_config_sha256="3" * 64,
        baseline_model="baseline-qwen",
        baseline_weight_manifest_sha256="4" * 64,
        baseline_config_sha256="5" * 64,
        rag_configuration_sha256="0" * 64,
        baseline_report_sha256="6" * 64,
        candidate_report_sha256="7" * 64,
        gold_artifact_sha256="8" * 64,
        sidecar_artifact_sha256="9" * 64,
        corpus_fingerprint_sha256="a" * 64,
        runtime_corpus_snapshot_sha256="b" * 64,
        judge_model="fixed-qwen-judge",
        judge_declared_revision="judge-revision",
        judge_weight_manifest_sha256="c" * 64,
        judge_config_sha256="d" * 64,
        judge_prompt_sha256="e" * 64,
        reference_git_sha="f" * 40,
    )


def _license() -> LocalLicenseEvidence:
    raw = b"Apache License\nVersion 2.0\n"
    return LocalLicenseEvidence(
        role="llm",
        model="candidate-qwen",
        weight_manifest_sha256="2" * 64,
        spdx_license="Apache-2.0",
        source_url="https://huggingface.co/example/candidate-qwen",
        local_relative_path="LICENSE",
        license_bytes_base64=base64.b64encode(raw).decode(),
        license_byte_count=len(raw),
        license_text_sha256=_sha(raw),
        commercial_on_prem_allowed=True,
    )


def _long_context() -> tuple[LongContextObservation, ...]:
    return (
        LongContextObservation(
            case_id="long-en",
            language="en",
            input_tokens=14_000,
            model_context_tokens=16_384,
            outcome="completed",
            duration_ms=100,
            output_sha256=_sha("long-en"),
        ),
        LongContextObservation(
            case_id="long-ru",
            language="ru",
            input_tokens=15_000,
            model_context_tokens=16_384,
            outcome="completed",
            duration_ms=110,
            output_sha256=_sha("long-ru"),
        ),
        LongContextObservation(
            case_id="long-zh",
            language="zh",
            input_tokens=15_500,
            model_context_tokens=16_384,
            outcome="overflow_error",
            duration_ms=20,
            error_code="context_overflow",
        ),
        LongContextObservation(
            case_id="long-en-oom",
            language="en",
            input_tokens=15_800,
            model_context_tokens=16_384,
            outcome="oom_error",
            duration_ms=30,
            error_code="cuda_oom",
        ),
    )


def _attempt(
    start: float,
    finish: float,
    *,
    outcome: str = "completed",
    name: str,
) -> LoadAttemptObservation:
    return LoadAttemptObservation(
        started_offset_ms=start,
        finished_offset_ms=finish,
        outcome=outcome,
        response_sha256=_sha(name) if outcome == "completed" else None,
        error_code="cuda_oom" if outcome == "oom_error" else None,
    )


def _load() -> LoadRunObservations:
    latencies = ((100, 110), (200, 210), (300, 250), (400, 410))
    requests = []
    for index, (baseline_latency, candidate_latency) in enumerate(latencies):
        candidate_outcome = "oom_error" if index == 2 else "completed"
        requests.append(
            PairedLoadRequestObservation(
                request_id=f"request-{index}",
                case_id=f"load-case-{index}",
                baseline=_attempt(0, baseline_latency, name=f"baseline-{index}"),
                candidate=_attempt(
                    0,
                    candidate_latency,
                    outcome=candidate_outcome,
                    name=f"candidate-{index}",
                ),
            )
        )
    return LoadRunObservations(
        concurrency=4,
        baseline_duration_ms=1000,
        candidate_duration_ms=2000,
        requests=tuple(requests),
        runtime_events=(
            LoadRuntimeEvent(
                target="candidate",
                kind="restart",
                offset_ms=1000,
                evidence_sha256=_sha("restart"),
            ),
        ),
    )


def _judgment(verdict: str, name: str) -> JudgeCaseObservation:
    return JudgeCaseObservation(
        verdict=verdict,
        response_sha256=_sha(name) if verdict != "error" else None,
        error_code="invalid_schema" if verdict == "error" else None,
        reason_codes=("unsupported_claim",) if verdict == "fail" else (),
    )


def _semantic_safety() -> tuple[PairedSemanticSafetyObservation, ...]:
    categories = (
        ("semantic", "safety"),
        ("semantic", "standards"),
        ("semantic",),
    )
    candidate_verdicts = ("pass", "fail", "error")
    return tuple(
        PairedSemanticSafetyObservation(
            case_id=f"semantic-{index}",
            gold_case_sha256=_sha(f"gold-{index}"),
            categories=category,
            baseline_output_sha256=_sha(f"baseline-output-{index}"),
            candidate_output_sha256=_sha(f"candidate-output-{index}"),
            baseline=_judgment("pass", f"baseline-judge-{index}"),
            candidate=_judgment(verdict, f"candidate-judge-{index}"),
        )
        for index, (category, verdict) in enumerate(zip(categories, candidate_verdicts, strict=True))
    )


def _rollback() -> RollbackRawEvidence:
    started = datetime(2026, 7, 13, 16, 0, tzinfo=UTC)
    trace = (
        RollbackTraceEvent(
            sequence=0,
            kind="rollback_started",
            observed_at=started,
            success=True,
            evidence_sha256=_sha("rollback-start"),
        ),
        RollbackTraceEvent(
            sequence=1,
            kind="config_restored",
            observed_at=started + timedelta(seconds=10),
            success=True,
            evidence_sha256=_sha("config-restored"),
        ),
        RollbackTraceEvent(
            sequence=2,
            kind="code_restored",
            observed_at=started + timedelta(seconds=30),
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
            evidence_sha256=_sha("rollback-complete"),
        ),
    )
    probes = (
        RollbackProbeObservation(
            kind="health",
            target="/healthz",
            passed=True,
            status_code=200,
            response_sha256=_sha("health"),
        ),
        RollbackProbeObservation(
            kind="root",
            target="/",
            passed=True,
            status_code=200,
            response_sha256=_sha("root"),
        ),
        RollbackProbeObservation(
            kind="auth_enabled",
            target="api-config",
            passed=True,
            response_sha256=_sha("auth"),
        ),
        RollbackProbeObservation(
            kind="anonymous_protected",
            target="/api/documents",
            passed=True,
            status_code=401,
            response_sha256=_sha("anonymous"),
        ),
        RollbackProbeObservation(
            kind="model_endpoint",
            target="llm",
            passed=True,
            status_code=200,
            response_sha256=_sha("llm"),
        ),
        RollbackProbeObservation(
            kind="model_endpoint",
            target="embedding",
            passed=False,
            status_code=503,
            response_sha256=_sha("embedding"),
        ),
    )
    smoke = (
        RollbackSmokeObservation(case_id="smoke-1", passed=True, result_sha256=_sha("smoke-1")),
        RollbackSmokeObservation(case_id="smoke-2", passed=False, result_sha256=_sha("smoke-2")),
    )
    return RollbackRawEvidence(
        reference_report_sha256="6" * 64,
        restored_git_sha="f" * 40,
        restored_model_weight_manifests=(
            RestoredModelWeightManifest(role="llm", weight_manifest_sha256="4" * 64),
            RestoredModelWeightManifest(role="embedding", weight_manifest_sha256="1" * 64),
            RestoredModelWeightManifest(role="reranker", weight_manifest_sha256="2" * 64),
        ),
        restored_configuration_sha256="5" * 64,
        restored_rag_configuration_sha256="0" * 64,
        restored_runtime_corpus_snapshot_sha256="b" * 64,
        trace=trace,
        probes=probes,
        smoke=smoke,
    )


def _evidence() -> RawQualificationEvidence:
    return build_raw_qualification_evidence(
        provenance=_provenance(),
        license=_license(),
        long_context_observations=_long_context(),
        load_observations=_load(),
        semantic_safety_observations=_semantic_safety(),
        rollback_trace=_rollback(),
    )


def test_producer_recomputes_every_release_qualification_aggregate() -> None:
    evidence = _evidence()

    assert evidence.aggregates.long_context.model_dump() == {
        "case_count": 4,
        "completed_count": 2,
        "language_counts": {"en": 2, "ru": 1, "zh": 1},
        "minimum_input_tokens": 14_000,
        "model_context_tokens": 16_384,
        "overflow_errors": 1,
        "oom_errors": 1,
        "truncation_errors": 0,
        "other_errors": 0,
    }
    assert evidence.aggregates.load.model_dump() == {
        "concurrency": 4,
        "request_count": 4,
        "completed_count": 3,
        "error_count": 1,
        "restart_count": 1,
        "oom_count": 1,
        "baseline_p95_ms": 400.0,
        "candidate_p95_ms": 410.0,
        "baseline_throughput_rps": 4.0,
        "candidate_throughput_rps": 1.5,
    }
    assert evidence.aggregates.semantic_safety.model_dump() == {
        "judge_model": "fixed-qwen-judge",
        "judge_weight_manifest_sha256": "c" * 64,
        "judge_prompt_sha256": "e" * 64,
        "case_count": 3,
        "baseline_semantic_passed": 3,
        "candidate_semantic_passed": 1,
        "safety_case_count": 1,
        "baseline_safety_passed": 1,
        "candidate_safety_passed": 1,
        "standards_case_count": 1,
        "baseline_standards_passed": 1,
        "candidate_standards_passed": 0,
        "judge_error_count": 1,
    }
    assert evidence.aggregates.rollback.duration_seconds == 120
    assert evidence.aggregates.rollback.smoke_passed_count == 1
    assert not evidence.aggregates.rollback.model_endpoints_ok
    assert verify_raw_qualification_evidence(evidence) == evidence.aggregates


def test_verifier_rejects_stored_aggregate_not_derived_from_raw() -> None:
    evidence = _evidence()
    forged_long = evidence.aggregates.long_context.model_copy(update={"completed_count": 4})
    forged = evidence.model_copy(
        update={"aggregates": evidence.aggregates.model_copy(update={"long_context": forged_long})}
    )

    with pytest.raises(QualificationEvidenceError, match="do not match"):
        verify_raw_qualification_evidence(forged)


def test_top_level_binds_license_and_rollback_to_strict_provenance() -> None:
    evidence = _evidence()
    wrong_license = evidence.license.model_copy(update={"weight_manifest_sha256": "0" * 64})
    payload = evidence.model_dump(mode="python")
    payload["license"] = wrong_license
    with pytest.raises(ValueError, match="license is not bound"):
        RawQualificationEvidence.model_validate(payload, strict=True)

    payload = evidence.model_dump(mode="python")
    payload["rollback_trace"] = evidence.rollback_trace.model_copy(update={"restored_git_sha": "0" * 40})
    with pytest.raises(ValueError, match="reference git"):
        RawQualificationEvidence.model_validate(payload, strict=True)

    payload = evidence.model_dump(mode="python")
    payload["rollback_trace"] = evidence.rollback_trace.model_copy(
        update={"restored_configuration_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="baseline configuration"):
        RawQualificationEvidence.model_validate(payload, strict=True)

    payload = evidence.model_dump(mode="python")
    payload["rollback_trace"] = evidence.rollback_trace.model_copy(
        update={
            "restored_model_weight_manifests": (
                RestoredModelWeightManifest(role="llm", weight_manifest_sha256="0" * 64),
            )
        }
    )
    with pytest.raises(ValueError, match="baseline model weight"):
        RawQualificationEvidence.model_validate(payload, strict=True)


def test_license_contains_exact_local_bytes_and_rejects_symlink(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    license_path = model_root / "LICENSE"
    raw = b"exact\x00license\r\nbytes\n"
    license_path.write_bytes(raw)

    evidence = capture_local_license(
        license_path,
        model_root=model_root,
        role="llm",
        model="candidate-qwen",
        weight_manifest_sha256="2" * 64,
        spdx_license="Apache-2.0",
        source_url="https://huggingface.co/example/candidate-qwen",
        commercial_on_prem_allowed=True,
    )

    assert base64.b64decode(evidence.license_bytes_base64) == raw
    assert evidence.license_text_sha256 == _sha(raw)
    assert evidence.local_relative_path == "LICENSE"

    link = model_root / "LICENSE.link"
    link.symlink_to(license_path)
    with pytest.raises(QualificationEvidenceError, match="opened safely"):
        capture_local_license(
            link,
            model_root=model_root,
            role="llm",
            model="candidate-qwen",
            weight_manifest_sha256="2" * 64,
            spdx_license="Apache-2.0",
            source_url="https://huggingface.co/example/candidate-qwen",
            commercial_on_prem_allowed=True,
        )


def test_license_schema_recomputes_byte_count_and_hash() -> None:
    payload = _license().model_dump(mode="python")
    payload["license_byte_count"] += 1
    with pytest.raises(ValueError, match="byte count mismatch"):
        LocalLicenseEvidence.model_validate(payload, strict=True)

    payload = _license().model_dump(mode="python")
    payload["license_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="byte hash mismatch"):
        LocalLicenseEvidence.model_validate(payload, strict=True)


def test_raw_observations_reject_duplicates_and_inconsistent_outcomes() -> None:
    load = _load()
    with pytest.raises(ValueError, match="request IDs must be unique"):
        LoadRunObservations(
            concurrency=load.concurrency,
            baseline_duration_ms=load.baseline_duration_ms,
            candidate_duration_ms=load.candidate_duration_ms,
            requests=(load.requests[0], load.requests[0]),
        )
    with pytest.raises(ValueError, match="requires only output hash"):
        LongContextObservation(
            case_id="bad-long",
            language="en",
            input_tokens=100,
            model_context_tokens=1000,
            outcome="completed",
            duration_ms=1,
            error_code="unexpected_error",
        )
    with pytest.raises(ValueError, match="passing judgment"):
        JudgeCaseObservation(
            verdict="pass",
            response_sha256=_sha("response"),
            reason_codes=("unsupported_claim",),
        )


def test_load_aggregate_requires_completed_observations_on_both_sides() -> None:
    load = _load()
    failed_requests = tuple(
        item.model_copy(
            update={
                "candidate": LoadAttemptObservation(
                    started_offset_ms=0,
                    finished_offset_ms=10,
                    outcome="error",
                    error_code="runtime_error",
                )
            }
        )
        for item in load.requests
    )
    failed = load.model_copy(update={"requests": failed_requests})

    with pytest.raises(QualificationEvidenceError, match="no completed"):
        aggregate_load(failed)


def test_rollback_trace_requires_ordered_events_and_complete_probe_set() -> None:
    rollback = _rollback()
    reversed_trace = tuple(reversed(rollback.trace))
    with pytest.raises(ValueError, match="complete ordered lifecycle"):
        RollbackRawEvidence(
            reference_report_sha256=rollback.reference_report_sha256,
            restored_git_sha=rollback.restored_git_sha,
            restored_model_weight_manifests=rollback.restored_model_weight_manifests,
            restored_configuration_sha256=rollback.restored_configuration_sha256,
            restored_rag_configuration_sha256=rollback.restored_rag_configuration_sha256,
            restored_runtime_corpus_snapshot_sha256=(rollback.restored_runtime_corpus_snapshot_sha256),
            trace=reversed_trace,
            probes=rollback.probes,
            smoke=rollback.smoke,
        )
    with pytest.raises(ValueError, match="singleton core checks"):
        RollbackRawEvidence(
            reference_report_sha256=rollback.reference_report_sha256,
            restored_git_sha=rollback.restored_git_sha,
            restored_model_weight_manifests=rollback.restored_model_weight_manifests,
            restored_configuration_sha256=rollback.restored_configuration_sha256,
            restored_rag_configuration_sha256=rollback.restored_rag_configuration_sha256,
            restored_runtime_corpus_snapshot_sha256=(rollback.restored_runtime_corpus_snapshot_sha256),
            trace=rollback.trace,
            probes=tuple(item for item in rollback.probes if item.kind != "health"),
            smoke=rollback.smoke,
        )


def test_private_artifact_round_trip_is_fresh_owner_only_and_hash_stable(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir(mode=0o700)
    private = repository_root / ".private"
    private.mkdir(mode=0o700)
    output = private / "qualification.json"
    evidence = _evidence()

    written_hash = write_private_qualification_evidence(output, evidence, repository_root=repository_root)

    assert written_hash == qualification_evidence_sha256(evidence)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert load_private_qualification_evidence(output, repository_root=repository_root) == evidence
    with pytest.raises(QualificationEvidenceError, match="already exists"):
        write_private_qualification_evidence(output, evidence, repository_root=repository_root)


def test_private_loader_rejects_forged_aggregate_and_unsafe_mode(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    private = repository_root / ".private"
    private.mkdir(parents=True, mode=0o700)
    output = private / "forged.json"
    payload = _evidence().model_dump(mode="json")
    payload["aggregates"]["load"]["completed_count"] = 4
    output.write_text(json.dumps(payload), encoding="utf-8")
    output.chmod(0o600)

    with pytest.raises(QualificationEvidenceError, match="do not match"):
        load_private_qualification_evidence(output, repository_root=repository_root)

    output.chmod(0o640)
    with pytest.raises(QualificationEvidenceError, match="input is invalid"):
        load_private_qualification_evidence(output, repository_root=repository_root)


def test_json_schema_is_strict_and_contains_all_raw_evidence_sections() -> None:
    schema = qualification_evidence_json_schema()

    assert schema["properties"]["schema_version"]["const"] == ("rag-model-qualification-raw-v1")
    assert schema["additionalProperties"] is False
    assert {
        "provenance",
        "license",
        "long_context_observations",
        "load_observations",
        "semantic_safety_observations",
        "rollback_trace",
        "aggregates",
    } <= set(schema["properties"])
