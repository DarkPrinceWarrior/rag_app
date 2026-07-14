from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_app.eval.report_attestation import (
    DEFAULT_ATTESTED_SOURCES,
    AttestedCase,
    ReportAttestationError,
    atomic_write_attestation,
    attestation_bytes,
    create_private_artifact_attestation,
    create_report_attestation,
    load_hmac_key,
    load_private_artifact_attestation,
    load_report_attestation,
    private_artifact_attestation_bytes,
    verify_private_artifact_attestation,
    verify_report_attestation,
)


def test_default_attested_sources_cover_evaluation_runtime() -> None:
    required = {
        "pyproject.toml",
        "scripts/evaluate_rag_gold_set.py",
        "src/rag_app/config.py",
        "src/rag_app/eval/baseline.py",
        "src/rag_app/eval/gold_set.py",
        "src/rag_app/eval/private_artifacts.py",
        "src/rag_app/eval/private_sidecar.py",
        "src/rag_app/eval/rag_metrics.py",
        "src/rag_app/eval/report_attestation.py",
        "src/rag_app/llm/embeddings.py",
        "src/rag_app/llm/visual.py",
        "src/rag_app/llm/visual_reranker.py",
        "src/rag_app/pipeline/validate.py",
        "src/rag_app/rag/chat.py",
        "src/rag_app/rag/retrieve.py",
        "src/rag_app/storage/s3.py",
        "uv.lock",
    }

    assert required <= set(DEFAULT_ATTESTED_SOURCES)


def _run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repo"
    repository.mkdir()
    _run_git(repository, "init", "-q")
    _run_git(repository, "config", "user.email", "test@example.invalid")
    _run_git(repository, "config", "user.name", "Test")
    (repository / "producer.py").write_text("PRODUCER_VERSION = 1\n")
    _run_git(repository, "add", "producer.py")
    _run_git(repository, "commit", "-qm", "producer")
    return repository, _run_git(repository, "rev-parse", "HEAD")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _revision(weight_hash: str) -> dict[str, object]:
    return {
        "endpoint_metadata_sha256": "1" * 64,
        "runtime_version_sha256": "2" * 64,
        "runtime_process_sha256": "3" * 64,
        "local_config_manifest_sha256": "4" * 64,
        "weight_manifest_sha256": weight_hash,
        "weight_file_count": 1,
        "weight_bytes": 1024,
        "declared_revision": "revision-1",
    }


def _report_bytes(
    git_sha: str,
    gold_bytes: bytes,
    sidecar_bytes: bytes,
    case_id: str,
) -> bytes:
    configuration = {
        "top_k": 10,
        "dense_top_k": 20,
        "sparse_top_k": 20,
        "rerank_top_k": 10,
        "rerank_min_score": 0.1,
        "embedding_dim": 1024,
        "visual_enabled": False,
        "context_max_chars": 10000,
        "context_window_tokens": 16384,
        "output_tokens": 2048,
        "answer_route": "doc_only",
        "prompt_sha256": "5" * 64,
        "temperature": 0.2,
        "top_p": 0.8,
        "seed_namespace": 2026071300,
        "seed_strategy": "case-id-sha256-v1",
        "enable_thinking": False,
    }
    case = {
        "case_id": case_id,
        "answerable": True,
        "answerability_correct": True,
        "abstained": False,
        "ranked": {
            "gold_evidence": {"recall": {}, "mrr": {}, "ndcg": {}},
        },
        "citation": {},
        "quantities": {},
        "retrieval_ms": 1.0,
        "generation_ms": 2.0,
        "total_ms": 3.0,
    }
    report = {
        "schema_version": "rag-baseline-report-v1",
        "provenance": {
            "runner": "retrieval_direct_answer",
            "evaluation_mode": "release",
            "evaluated_at": "2026-07-13T12:00:00Z",
            "git_sha": git_sha,
            "git_dirty": False,
            "gold_artifact_sha256": _sha256(gold_bytes),
            "sidecar_artifact_sha256": _sha256(sidecar_bytes),
            "corpus_fingerprint_sha256": "6" * 64,
            "runtime_corpus_snapshot_sha256": "7" * 64,
            "scope_count": 1,
            "document_snapshot_count": 1,
            "models": {
                "llm": "llm",
                "embedding": "embedding",
                "reranker": "reranker",
                "visual_embedding": None,
                "visual_reranker": None,
            },
            "model_revisions": {
                "llm": _revision("8" * 64),
                "embedding": _revision("9" * 64),
                "reranker": _revision("a" * 64),
                "visual_embedding": None,
                "visual_reranker": None,
            },
            "configuration": configuration,
            "configuration_sha256": _sha256(
                json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
            ),
        },
        "case_count": 1,
        "answerable_count": 1,
        "no_answer_count": 0,
        "answerability_accuracy": 1.0,
        "mean_recall": {"10": 1.0},
        "mean_mrr": {"10": 1.0},
        "mean_ndcg": {"10": 1.0},
        "mean_citation_precision": 1.0,
        "mean_citation_recall": 1.0,
        "mean_quantity_unit_accuracy": 1.0,
        "mean_quantity_unit_recall": 1.0,
        "unsupported_number_rate": 0.0,
        "latency_ms": {"mean": 3.0, "p50": 3.0, "p95": 3.0},
        "cases": [case],
    }
    return json.dumps(report, sort_keys=True).encode()


def _artifacts(git_sha: str) -> tuple[bytes, bytes, bytes, tuple[AttestedCase, ...]]:
    gold_bytes = b'{"private":"gold"}\n'
    sidecar_bytes = b'{"private":"sidecar"}\n'
    cases = (
        AttestedCase(
            case_id="case-001",
            gold_case_sha256="c" * 64,
            sidecar_case_sha256="d" * 64,
        ),
    )
    return (
        _report_bytes(git_sha, gold_bytes, sidecar_bytes, cases[0].case_id),
        gold_bytes,
        sidecar_bytes,
        cases,
    )


def test_report_attestation_round_trip_and_private_write(tmp_path: Path) -> None:
    repository, git_sha = _repository(tmp_path)
    report, gold, sidecar, cases = _artifacts(git_sha)
    key = b"k" * 32
    attestation = create_report_attestation(
        report_bytes=report,
        gold_bytes=gold,
        sidecar_bytes=sidecar,
        cases=cases,
        key=key,
        repository_root=repository,
        source_paths=("producer.py",),
        created_at=datetime(2026, 7, 13, 12, tzinfo=UTC),
    )
    loaded = load_report_attestation(attestation_bytes(attestation))
    verify_report_attestation(
        loaded,
        report_bytes=report,
        gold_bytes=gold,
        sidecar_bytes=sidecar,
        expected_cases=cases,
        key=key,
        repository_root=repository,
    )

    output = tmp_path / "attestation.json"
    atomic_write_attestation(output, loaded)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert load_report_attestation(output.read_bytes()) == loaded
    with pytest.raises(ReportAttestationError, match="published safely"):
        atomic_write_attestation(output, loaded)


@pytest.mark.parametrize("artifact", ["report", "gold", "sidecar", "cases", "key"])
def test_report_attestation_rejects_tampering(tmp_path: Path, artifact: str) -> None:
    repository, git_sha = _repository(tmp_path)
    report, gold, sidecar, cases = _artifacts(git_sha)
    key = b"k" * 32
    attestation = create_report_attestation(
        report_bytes=report,
        gold_bytes=gold,
        sidecar_bytes=sidecar,
        cases=cases,
        key=key,
        repository_root=repository,
        source_paths=("producer.py",),
    )
    if artifact == "report":
        report += b" "
    elif artifact == "gold":
        gold += b"x"
    elif artifact == "sidecar":
        sidecar += b"x"
    elif artifact == "cases":
        cases = (cases[0].model_copy(update={"gold_case_sha256": "e" * 64}),)
    else:
        key = b"z" * 32
    with pytest.raises(ReportAttestationError):
        verify_report_attestation(
            attestation,
            report_bytes=report,
            gold_bytes=gold,
            sidecar_bytes=sidecar,
            expected_cases=cases,
            key=key,
            repository_root=repository,
        )


def test_report_attestation_verifies_the_attested_revision_not_later_worktree(
    tmp_path: Path,
) -> None:
    repository, git_sha = _repository(tmp_path)
    report, gold, sidecar, cases = _artifacts(git_sha)
    key = b"k" * 32
    attestation = create_report_attestation(
        report_bytes=report,
        gold_bytes=gold,
        sidecar_bytes=sidecar,
        cases=cases,
        key=key,
        repository_root=repository,
        source_paths=("producer.py",),
    )
    (repository / "producer.py").write_text("PRODUCER_VERSION = 2\n")
    verify_report_attestation(
        attestation,
        report_bytes=report,
        gold_bytes=gold,
        sidecar_bytes=sidecar,
        expected_cases=cases,
        key=key,
        repository_root=repository,
    )


def test_create_report_attestation_rejects_dirty_repository(tmp_path: Path) -> None:
    repository, git_sha = _repository(tmp_path)
    report, gold, sidecar, cases = _artifacts(git_sha)
    (repository / "producer.py").write_text("PRODUCER_VERSION = 2\n")

    with pytest.raises(ReportAttestationError, match="clean repository"):
        create_report_attestation(
            report_bytes=report,
            gold_bytes=gold,
            sidecar_bytes=sidecar,
            cases=cases,
            key=b"k" * 32,
            repository_root=repository,
            source_paths=("producer.py",),
        )


def test_hmac_key_must_be_private_owned_external_regular_file(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    key_path = tmp_path / "key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)
    assert load_hmac_key(key_path.resolve(), repository) == b"k" * 32

    key_path.chmod(0o640)
    with pytest.raises(ReportAttestationError, match="read safely"):
        load_hmac_key(key_path.resolve(), repository)

    internal = repository / "key"
    internal.write_bytes(b"k" * 32)
    internal.chmod(0o600)
    with pytest.raises(ReportAttestationError, match="outside"):
        load_hmac_key(internal.resolve(), repository)


def test_hmac_key_rejects_symlink_file(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    key_path = tmp_path / "key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)
    key_link = tmp_path / "key-link"
    key_link.symlink_to(key_path)

    with pytest.raises(ReportAttestationError, match="read safely"):
        load_hmac_key(key_link, repository)


def test_hmac_key_rejects_symlink_parent(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    private_directory = tmp_path / "private"
    private_directory.mkdir(mode=0o700)
    key_path = private_directory / "key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)
    parent_link = tmp_path / "private-link"
    parent_link.symlink_to(private_directory, target_is_directory=True)

    with pytest.raises(ReportAttestationError, match="read safely"):
        load_hmac_key(parent_link / "key", repository)


def test_private_artifact_attestation_binds_type_bytes_and_producer(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    key = b"q" * 32
    artifact = b'{"raw_observations":[]}\n'
    attestation = create_private_artifact_attestation(
        artifact_bytes=artifact,
        artifact_type="rag-model-qualification-v1",
        key=key,
        repository_root=repository,
        source_paths=("producer.py",),
    )
    loaded = load_private_artifact_attestation(private_artifact_attestation_bytes(attestation))
    verify_private_artifact_attestation(
        loaded,
        artifact_bytes=artifact,
        expected_artifact_type="rag-model-qualification-v1",
        key=key,
        repository_root=repository,
    )
    with pytest.raises(ReportAttestationError, match="type"):
        verify_private_artifact_attestation(
            loaded,
            artifact_bytes=artifact,
            expected_artifact_type="rag-model-load-test-v1",
            key=key,
            repository_root=repository,
        )
    with pytest.raises(ReportAttestationError, match="bytes"):
        verify_private_artifact_attestation(
            loaded,
            artifact_bytes=artifact + b"x",
            expected_artifact_type="rag-model-qualification-v1",
            key=key,
            repository_root=repository,
        )


def test_attestation_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    repository, git_sha = _repository(tmp_path)
    report, gold, sidecar, cases = _artifacts(git_sha)
    attestation = create_report_attestation(
        report_bytes=report,
        gold_bytes=gold,
        sidecar_bytes=sidecar,
        cases=cases,
        key=b"k" * 32,
        repository_root=repository,
        source_paths=("producer.py",),
    )
    raw = attestation_bytes(attestation).replace(
        b'{\n  "algorithm"',
        b'{\n  "algorithm": "hmac-sha256",\n  "algorithm"',
        1,
    )
    with pytest.raises(ReportAttestationError, match="invalid"):
        load_report_attestation(raw)
