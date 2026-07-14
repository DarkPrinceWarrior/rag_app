"""Verify signed private RAG evidence and emit a sanitized release decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from rag_app.eval.gold_set import (
    GoldSetValidationError,
    ensure_private_gold_path,
    parse_gold_set_bytes,
)
from rag_app.eval.private_artifacts import (
    PrivateArtifactError,
    PrivateBytesArtifact,
    PrivateJsonArtifact,
    parse_strict_json,
    read_private_bytes,
    read_private_json,
    write_private_json_fresh,
)
from rag_app.eval.private_sidecar import (
    PrivateSidecarError,
    bind_gold_sidecar,
    parse_private_sidecar_bytes,
)
from rag_app.eval.qualification_evidence import (
    QUALIFICATION_ATTESTED_SOURCES,
    RawQualificationEvidence,
    verify_raw_qualification_evidence,
)
from rag_app.eval.release_gate import (
    GateRuntimeProvenance,
    ReleaseGateError,
    ReleaseGatePolicy,
    evaluate_release_gate,
    parse_baseline_report,
)
from rag_app.eval.report_attestation import (
    DEFAULT_ATTESTED_SOURCES,
    ReportAttestation,
    ReportAttestationError,
    build_case_attestations,
    load_hmac_key,
    load_private_artifact_attestation,
    load_report_attestation,
    verify_private_artifact_attestation,
    verify_report_attestation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPOSITORY_ROOT / "deploy" / "rag-eval" / "release-policy-v1.json"
EXPECTED_POLICY_SHA256 = "a8fa047049cd4e0f928a64b0db94d41747e2a7419f610a092c066a0129e79d9d"
QUALIFICATION_ARTIFACT_TYPE = "rag-model-qualification-raw-v1"
_GATE_SOURCE_PATHS = {
    "comparator_sha256": "scripts/compare_rag_baselines.py",
    "release_gate_sha256": "src/rag_app/eval/release_gate.py",
    "private_artifacts_sha256": "src/rag_app/eval/private_artifacts.py",
    "report_attestation_sha256": "src/rag_app/eval/report_attestation.py",
    "qualification_evidence_sha256": "src/rag_app/eval/qualification_evidence.py",
}


class ReleaseArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage()
        self.exit(64, "release gate command line is invalid\n")


def build_parser() -> argparse.ArgumentParser:
    parser = ReleaseArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("--baseline-attestation", type=Path, required=True)
    parser.add_argument("--candidate-attestation", type=Path, required=True)
    parser.add_argument("--qualification-attestation", type=Path, required=True)
    parser.add_argument("--attestation-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _private_location(path: Path) -> None:
    try:
        ensure_private_gold_path(path, REPOSITORY_ROOT)
    except GoldSetValidationError:
        raise ReleaseGateError("private artifact path violates repository policy") from None


def _read_bytes(path: Path, *, max_bytes: int = 256 * 1024 * 1024) -> PrivateBytesArtifact:
    _private_location(path)
    try:
        return read_private_bytes(path, max_bytes=max_bytes)
    except PrivateArtifactError:
        raise ReleaseGateError("private artifact cannot be read safely") from None


def _read_json[T](
    path: Path,
    parser: Callable[[bytes], T],
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> PrivateJsonArtifact[T]:
    _private_location(path)
    try:
        return read_private_json(path, parser=parser, max_bytes=max_bytes)
    except PrivateArtifactError:
        raise ReleaseGateError("private JSON artifact is invalid") from None


def _parse_policy(raw: bytes) -> ReleaseGatePolicy:
    try:
        parse_strict_json(raw)
        return ReleaseGatePolicy.model_validate_json(raw, strict=True)
    except Exception:
        raise ReleaseGateError("release policy is invalid") from None


def _load_pinned_policy() -> tuple[ReleaseGatePolicy, str]:
    try:
        raw = POLICY_PATH.read_bytes()
    except OSError:
        raise ReleaseGateError("pinned release policy is unreadable") from None
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_POLICY_SHA256:
        raise ReleaseGateError("pinned release policy digest is not approved")
    return _parse_policy(raw), digest


def _parse_qualification(raw: bytes) -> RawQualificationEvidence:
    try:
        parse_strict_json(raw)
        evidence = RawQualificationEvidence.model_validate_json(raw, strict=True)
        verify_raw_qualification_evidence(evidence)
        return evidence
    except Exception:
        raise ReleaseGateError("raw qualification evidence is invalid") from None


def _validate_report_attestation_git_bindings(
    baseline: ReportAttestation,
    candidate: ReportAttestation,
    policy: ReleaseGatePolicy,
) -> None:
    if any(
        attestation.repository_git_sha != policy.reference_git_sha
        for attestation in (baseline, candidate)
    ):
        raise ReleaseGateError("report attestation Git binding is invalid")


def _git_output(arguments: list[str], *, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *arguments],  # noqa: S607
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=text,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise ReleaseGateError("release gate repository state is unavailable") from None
    return result.stdout


def _gate_runtime() -> GateRuntimeProvenance:
    revision = str(_git_output(["rev-parse", "HEAD"])).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ReleaseGateError("release gate Git revision is invalid")
    if str(_git_output(["status", "--porcelain", "--untracked-files=all"])).strip():
        raise ReleaseGateError("release gate requires a clean repository")
    hashes: dict[str, str] = {}
    for field, relative in _GATE_SOURCE_PATHS.items():
        try:
            working = (REPOSITORY_ROOT / relative).read_bytes()
        except OSError:
            raise ReleaseGateError("release gate source is unreadable") from None
        committed = _git_output(["show", f"HEAD:{relative}"], text=False)
        if not isinstance(committed, bytes) or committed != working:
            raise ReleaseGateError("release gate source does not match Git")
        hashes[field] = hashlib.sha256(working).hexdigest()
    return GateRuntimeProvenance(git_sha=revision, git_dirty=False, **hashes)


def _atomic_write_fresh(path: Path, payload: dict[str, Any], *, repository_root: Path) -> str:
    try:
        ensure_private_gold_path(path, repository_root)
    except GoldSetValidationError:
        raise ReleaseGateError("release decision path violates repository policy") from None
    parent = path.expanduser().parent
    if not parent.exists():
        raise ReleaseGateError("release decision parent must already exist")
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    try:
        return write_private_json_fresh(path, content).sha256
    except FileExistsError:
        raise ReleaseGateError("release decision output already exists") from None
    except PrivateArtifactError:
        raise ReleaseGateError("release decision output cannot be published safely") from None


def run(args: argparse.Namespace) -> int:
    policy, policy_sha256 = _load_pinned_policy()
    gate_runtime = _gate_runtime()
    evaluated_at = datetime.now(UTC)
    key = load_hmac_key(args.attestation_key, REPOSITORY_ROOT)
    key_id = hashlib.sha256(key).hexdigest()
    if key_id != policy.attestation_key_id:
        raise ReleaseGateError("attestation key is not approved by policy")

    baseline_artifact = _read_json(args.baseline, parse_baseline_report)
    candidate_artifact = _read_json(args.candidate, parse_baseline_report)
    gold_artifact = _read_bytes(args.gold)
    sidecar_artifact = _read_bytes(args.sidecar)
    qualification_artifact = _read_json(args.qualification, _parse_qualification)
    baseline_attestation_artifact = _read_json(
        args.baseline_attestation, load_report_attestation, max_bytes=8 * 1024 * 1024
    )
    candidate_attestation_artifact = _read_json(
        args.candidate_attestation, load_report_attestation, max_bytes=8 * 1024 * 1024
    )
    qualification_attestation_artifact = _read_json(
        args.qualification_attestation,
        load_private_artifact_attestation,
        max_bytes=8 * 1024 * 1024,
    )
    try:
        records, _ = parse_gold_set_bytes(gold_artifact.raw_bytes, mode="release")
        sidecars = parse_private_sidecar_bytes(sidecar_artifact.raw_bytes)
        bound_sidecars = bind_gold_sidecar(records, sidecars)
        case_attestations = build_case_attestations(records, bound_sidecars)
    except (GoldSetValidationError, PrivateSidecarError, ReportAttestationError):
        raise ReleaseGateError("Gold and sidecar binding is invalid") from None

    baseline_attestation = baseline_attestation_artifact.value
    candidate_attestation = candidate_attestation_artifact.value
    expected_report_sources = tuple(sorted(DEFAULT_ATTESTED_SOURCES))
    if (
        tuple(item.path for item in baseline_attestation.sources) != expected_report_sources
        or tuple(item.path for item in candidate_attestation.sources) != expected_report_sources
        or baseline_attestation.source_manifest_sha256 != candidate_attestation.source_manifest_sha256
    ):
        raise ReleaseGateError("report attestation source set is not approved")
    _validate_report_attestation_git_bindings(
        baseline_attestation,
        candidate_attestation,
        policy,
    )
    qualification_attestation = qualification_attestation_artifact.value
    if tuple(item.path for item in qualification_attestation.sources) != tuple(
        sorted(QUALIFICATION_ATTESTED_SOURCES)
    ):
        raise ReleaseGateError("qualification attestation source set is not approved")
    try:
        verify_report_attestation(
            baseline_attestation,
            report_bytes=baseline_artifact.raw_bytes,
            gold_bytes=gold_artifact.raw_bytes,
            sidecar_bytes=sidecar_artifact.raw_bytes,
            expected_cases=case_attestations,
            key=key,
            repository_root=REPOSITORY_ROOT,
        )
        verify_report_attestation(
            candidate_attestation,
            report_bytes=candidate_artifact.raw_bytes,
            gold_bytes=gold_artifact.raw_bytes,
            sidecar_bytes=sidecar_artifact.raw_bytes,
            expected_cases=case_attestations,
            key=key,
            repository_root=REPOSITORY_ROOT,
        )
        verify_private_artifact_attestation(
            qualification_attestation,
            artifact_bytes=qualification_artifact.raw_bytes,
            expected_artifact_type=QUALIFICATION_ARTIFACT_TYPE,
            key=key,
            repository_root=REPOSITORY_ROOT,
        )
    except ReportAttestationError:
        raise ReleaseGateError("private evaluation attestation is invalid") from None
    if qualification_attestation.created_at < qualification_artifact.value.provenance.generated_at:
        raise ReleaseGateError("qualification attestation predates its evidence")
    if any(
        attestation.created_at > evaluated_at
        for attestation in (
            baseline_attestation,
            candidate_attestation,
            qualification_attestation,
        )
    ):
        raise ReleaseGateError("private evaluation attestation is future-dated")
    if (
        qualification_attestation.repository_git_sha
        != qualification_artifact.value.provenance.producer_git_sha
        or qualification_attestation.repository_git_sha != policy.reference_git_sha
    ):
        raise ReleaseGateError("qualification producer Git binding is invalid")

    decision = evaluate_release_gate(
        baseline_artifact.value,
        candidate_artifact.value,
        records,
        qualification_artifact.value,
        policy,
        evaluated_at=evaluated_at,
        baseline_sha256=baseline_artifact.sha256,
        baseline_attestation_sha256=baseline_attestation_artifact.sha256,
        candidate_sha256=candidate_artifact.sha256,
        candidate_attestation_sha256=candidate_attestation_artifact.sha256,
        gold_sha256=gold_artifact.sha256,
        sidecar_sha256=sidecar_artifact.sha256,
        qualification_sha256=qualification_artifact.sha256,
        qualification_attestation_sha256=qualification_attestation_artifact.sha256,
        policy_sha256=policy_sha256,
        gate_runtime=gate_runtime,
    )
    decision_sha256 = _atomic_write_fresh(
        args.output,
        decision.model_dump(mode="json"),
        repository_root=REPOSITORY_ROOT,
    )
    status = "accepted" if decision.accepted else "rejected"
    print(f"release gate {status}; decision_sha256={decision_sha256}")
    return 0 if decision.accepted else 2


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except ReleaseGateError:
        print("release gate invalid or incomparable")
        raise SystemExit(3) from None
    except Exception:  # noqa: BLE001 - sanitize every unexpected operational failure
        print("release gate operational failure")
        raise SystemExit(4) from None


if __name__ == "__main__":
    main()
