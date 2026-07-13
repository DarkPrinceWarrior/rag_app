from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_app.eval.gold_set import (
    DocumentSnapshot,
    GoldRecord,
    gold_record_case_sha256,
    make_document_ref,
    make_scope_id,
    text_sha256,
)
from rag_app.eval.private_checkpoint import (
    CheckpointError,
    CheckpointLineage,
    ContinuationLink,
    PrivateCheckpointStore,
    RunIdentity,
    SlotCheckpoint,
    SlotTarget,
    canonical_sha256,
    checkpoint_lineage_entry,
    checkpoint_tree_sha256,
    read_continuation_link,
    validate_checkpoint_lineage,
    write_continuation_link,
)
from rag_app.eval.private_sidecar import PrivateSidecarRecord


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(*, seed: int = 7) -> RunIdentity:
    return RunIdentity(
        seed=seed,
        corpus_sha256=_sha("corpus"),
        snapshots_sha256=_sha("snapshots"),
        plan_sha256=_sha("plan"),
        model="local-test-model",
        model_revision="revision-1",
        generator_contract_version="generator-v1",
        per_stratum=50,
        min_chars=200,
        trial=False,
    )


def _target() -> SlotTarget:
    return SlotTarget(
        stratum="no_answer",
        slot=0,
        language="en",
        content_type="text",
        source_plan_sha256=_sha("slot-source-plan"),
        source_count=2,
    )


def _case() -> tuple[GoldRecord, PrivateSidecarRecord]:
    source_hash = _sha("checkpoint-source")
    document_ref = make_document_ref(source_hash)
    document_id = uuid.UUID("20000000-0000-0000-0000-000000000001")
    question = "Which value is absent from this document?"
    record = GoldRecord(
        schema_version="rag-gold-v1",
        case_id="ragq-checkpoint-0001",
        status="candidate",
        scope_id=make_scope_id("checkpoint-owner"),
        language="en",
        question=question,
        question_sha256=text_sha256(question),
        answerable=False,
        reference_answer=None,
        reference_answer_sha256=None,
        hop_type="single",
        content_types=("text",),
        challenge_tags=(),
        document_scope=(
            DocumentSnapshot(
                document_ref=document_ref,
                source_sha256=source_hash,
                parsed_content_sha256=_sha("parsed-checkpoint-source"),
                page_count=1,
            ),
        ),
        evidence=(),
        review=None,
    )
    sidecar = PrivateSidecarRecord.model_validate_json(
        json.dumps(
            {
                "schema_version": "private-rag-generator-v1",
                "case_id": record.case_id,
                "gold_case_sha256": gold_record_case_sha256(record),
                "scope_id": record.scope_id,
                "stratum": "no_answer",
                "language": "en",
                "source_documents": [
                    {
                        "document_id": str(document_id),
                        "document_ref": document_ref,
                        "source_lang": "en",
                    }
                ],
                "classification": {
                    "content_types": ["text"],
                    "challenge_tags": [],
                    "has_numbers": False,
                    "has_units": False,
                    "has_standards": False,
                },
                "generation": {"model": "local-test-model", "seed": 7},
                "exact_evidence": [],
                "retrieval_probe": [],
                "quantities": {"expected": [], "supported": []},
                "validation": {"answerable_from_top8": False},
            }
        ),
        strict=True,
    )
    return record, sidecar


def _slot(
    store: PrivateCheckpointStore,
    *,
    target: SlotTarget | None = None,
    source_index: int = 0,
    retry: int = 0,
) -> SlotCheckpoint:
    record, sidecar = _case()
    return SlotCheckpoint.create(
        identity_sha256=store.identity_sha256,
        target=target or _target(),
        source_index=source_index,
        retry=retry,
        call_seed=1234,
        record=record,
        sidecar=sidecar,
    )


def _mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def test_create_save_and_explicit_resume_are_private(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    identity = _identity()
    store = PrivateCheckpointStore.create(root, identity, max_attempts=3)

    assert _mode(root) == 0o700
    assert _mode(root / "slots") == 0o700
    assert _mode(root / "cursors") == 0o700
    assert _mode(root / "manifest.json") == 0o600

    target = _target()
    cursor = store.record_deterministic_reject(target, source_index=0, retry=0, reason_code="language")
    assert cursor.next_attempt(0) == 1
    checkpoint = _slot(store, target=target, retry=1)
    store.save_slot(checkpoint)

    assert _mode(root / "slots" / f"{target.key}.json") == 0o600
    assert _mode(root / "cursors" / f"{target.key}.json") == 0o600
    resumed = PrivateCheckpointStore.resume(root, identity, max_attempts=3)
    assert resumed.accepted_slots == 1
    assert resumed.load_slot(target) == checkpoint
    assert resumed.iter_slots() == (checkpoint,)

    with pytest.raises(CheckpointError, match="already exists"):
        PrivateCheckpointStore.create(root, identity, max_attempts=3)


def test_resume_rejects_identity_decrease_mode_and_hash_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    identity = _identity()
    store = PrivateCheckpointStore.create(root, identity, max_attempts=3)
    store.save_slot(_slot(store))

    with pytest.raises(CheckpointError, match="identity mismatch"):
        PrivateCheckpointStore.resume(root, _identity(seed=8), max_attempts=3)
    with pytest.raises(CheckpointError, match="cannot decrease"):
        PrivateCheckpointStore.resume(root, identity, max_attempts=2)

    slot_path = root / "slots" / f"{_target().key}.json"
    os.chmod(slot_path, 0o640)
    with pytest.raises(CheckpointError, match="0600"):
        PrivateCheckpointStore.resume(root, identity, max_attempts=3)

    os.chmod(slot_path, 0o600)
    payload = json.loads(slot_path.read_text(encoding="utf-8"))
    payload["call_seed"] += 1
    slot_path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(slot_path, 0o600)
    with pytest.raises(CheckpointError, match="corrupt"):
        PrivateCheckpointStore.resume(root, identity, max_attempts=3)


def test_invalidate_slot_advances_cursor_for_targeted_regeneration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    identity = _identity()
    target = _target()
    store = PrivateCheckpointStore.create(root, identity, max_attempts=3)
    store.record_deterministic_reject(
        target, source_index=0, retry=0, reason_code="language"
    )
    store.save_slot(_slot(store, target=target, retry=1))

    cursor = store.invalidate_slot(target, reason_code="duplicate_case")

    assert store.accepted_slots == 0
    assert store.load_slot(target) is None
    assert cursor.next_attempt(0) == 2
    assert not (root / "slots" / f"{target.key}.json").exists()
    store.save_slot(_slot(store, target=target, retry=2))
    assert store.accepted_slots == 1


def test_cursor_is_per_source_contiguous_and_max_attempts_only_increases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    identity = _identity()
    target = _target()
    store = PrivateCheckpointStore.create(root, identity, max_attempts=1)

    store.record_deterministic_reject(target, source_index=0, retry=0, reason_code="language")
    store.record_deterministic_reject(target, source_index=1, retry=0, reason_code="positive_judge")
    with pytest.raises(CheckpointError, match="next_attempt"):
        store.record_deterministic_reject(target, source_index=0, retry=0, reason_code="language")

    resumed = PrivateCheckpointStore.resume(root, identity, max_attempts=2)
    assert resumed.max_attempts == 2
    assert resumed.load_cursor(target).next_attempt(0) == 1
    cursor = resumed.record_deterministic_reject(target, source_index=0, retry=1, reason_code="language")
    assert cursor.next_attempt(0) == 2
    assert cursor.next_attempt(1) == 1


def test_slot_binding_and_cursor_position_are_fail_closed(tmp_path: Path) -> None:
    store = PrivateCheckpointStore.create(tmp_path / "checkpoint", _identity(), max_attempts=3)
    record, sidecar = _case()
    bad_sidecar = sidecar.model_copy(update={"gold_case_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="binding failed"):
        SlotCheckpoint.create(
            identity_sha256=store.identity_sha256,
            target=_target(),
            source_index=0,
            retry=0,
            call_seed=1,
            record=record,
            sidecar=bad_sidecar,
        )

    store.record_deterministic_reject(_target(), source_index=0, retry=0, reason_code="language")
    with pytest.raises(CheckpointError, match="deterministic cursor"):
        store.save_slot(_slot(store, retry=0))
    store.save_slot(_slot(store, retry=1))


def test_cleanup_requires_confirmed_final_write_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    identity = _identity()
    root = tmp_path / "checkpoint"
    store = PrivateCheckpointStore.create(root, identity, max_attempts=3)
    store.save_slot(_slot(store))

    with pytest.raises(CheckpointError, match="confirmation"):
        store.cleanup_after_success(final_artifacts_written=False)
    assert root.exists()
    store.cleanup_after_success(final_artifacts_written=True)
    assert not root.exists()

    real_root = tmp_path / "real-checkpoint"
    PrivateCheckpointStore.create(real_root, identity, max_attempts=3)
    link_root = tmp_path / "linked-checkpoint"
    link_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(CheckpointError, match="non-symlink"):
        PrivateCheckpointStore.resume(link_root, identity, max_attempts=3)


def test_readonly_parent_open_is_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    identity = _identity()
    store = PrivateCheckpointStore.create(root, identity, max_attempts=3)
    store.record_deterministic_reject(
        _target(), source_index=0, retry=0, reason_code="language"
    )
    store.save_slot(_slot(store, retry=1))
    before_tree = checkpoint_tree_sha256(root)
    before_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    readonly = PrivateCheckpointStore.open_readonly(root, identity)

    assert readonly.accepted_slots == 1
    assert readonly.tree_sha256 == before_tree
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before_files


def test_continuation_link_is_private_hashed_and_fail_closed(tmp_path: Path) -> None:
    link = ContinuationLink.create(
        parent_identity_sha256=_sha("parent-identity"),
        parent_tree_sha256=_sha("parent-tree"),
        parent_manifest_sha256=_sha("parent-manifest"),
        parent_slots_sha256=_sha("parent-slots"),
        parent_slot_count=200,
        base_registry_sha256=_sha("base-registry"),
        continuation_identity_sha256=_sha("continuation-identity"),
        missing_targets_sha256=_sha("missing-targets"),
        missing_target_keys=("cross_document-0022", "single_hop-0031"),
        call_seed_namespace=3_000_000_000,
    )
    path = tmp_path / "continuation.link.json"

    write_continuation_link(path, link)

    assert _mode(path) == 0o600
    assert read_continuation_link(path) == link
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["parent_slot_count"] = 199
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(CheckpointError, match="corrupt"):
        read_continuation_link(path)


def test_continuation_link_rejects_unsorted_or_duplicate_targets() -> None:
    common = {
        "parent_identity_sha256": _sha("parent-identity"),
        "parent_tree_sha256": _sha("parent-tree"),
        "parent_manifest_sha256": _sha("parent-manifest"),
        "parent_slots_sha256": _sha("parent-slots"),
        "parent_slot_count": 200,
        "base_registry_sha256": _sha("base-registry"),
        "continuation_identity_sha256": _sha("continuation-identity"),
        "missing_targets_sha256": _sha("missing-targets"),
        "call_seed_namespace": 3_000_000_000,
    }
    with pytest.raises(ValidationError, match="unique and sorted"):
        ContinuationLink.create(
            **common,
            missing_target_keys=("single_hop-0031", "cross_document-0022"),
        )
    with pytest.raises(ValidationError, match="unique and sorted"):
        ContinuationLink.create(
            **common,
            missing_target_keys=("cross_document-0022", "cross_document-0022"),
        )


def test_checkpoint_lineage_recomputes_identity_and_payload_hashes(
    tmp_path: Path,
) -> None:
    base = PrivateCheckpointStore.create(
        tmp_path / "base", _identity(), max_attempts=3
    )
    base.save_slot(_slot(base))
    continuation = PrivateCheckpointStore.create(
        tmp_path / "continuation", _identity(seed=8), max_attempts=48
    )
    link = ContinuationLink.create(
        parent_identity_sha256=base.identity_sha256,
        parent_tree_sha256=base.tree_sha256,
        parent_manifest_sha256=base.manifest_sha256,
        parent_slots_sha256=base.slots_sha256,
        parent_slot_count=base.accepted_slots,
        base_registry_sha256=_sha("base-registry"),
        continuation_identity_sha256=continuation.identity_sha256,
        missing_targets_sha256=_sha("missing-targets"),
        missing_target_keys=("no_answer-0001",),
        call_seed_namespace=1000,
    )
    link_path = tmp_path / "continuation.link.json"
    write_continuation_link(link_path, link)
    lineage = CheckpointLineage.create(
        base=checkpoint_lineage_entry(base),
        continuation=checkpoint_lineage_entry(continuation),
        continuation_link=link,
        merged_slots=1,
    )

    continuation.cleanup_after_success(final_artifacts_written=True)
    link_path.unlink()
    base.cleanup_after_success(final_artifacts_written=True)
    validated = CheckpointLineage.model_validate_json(
        lineage.model_dump_json(), strict=True
    )
    assert validated.base.identity == _identity()
    assert validated.base.max_attempts == 3
    assert validated.continuation.max_attempts == 48
    assert len(validated.base.manifest_sha256) == 64
    assert len(validated.base.slots_sha256) == 64
    assert len(validated.base.tree_sha256) == 64
    assert validated.continuation_link.missing_target_keys == ("no_answer-0001",)
    assert not continuation.root.exists()
    assert not link_path.exists()
    assert not base.root.exists()

    payload = lineage.model_dump(mode="json")
    payload["base"]["identity_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="identity hash mismatch"):
        CheckpointLineage.model_validate(payload, strict=True)


def test_live_lineage_validator_detects_checkpoint_drift(tmp_path: Path) -> None:
    base = PrivateCheckpointStore.create(
        tmp_path / "base", _identity(), max_attempts=3
    )
    base_slot = _slot(base)
    base.save_slot(base_slot)
    continuation = PrivateCheckpointStore.create(
        tmp_path / "continuation", _identity(seed=8), max_attempts=48
    )
    continuation_target = _target().model_copy(update={"slot": 1})
    continuation_slot = _slot(continuation, target=continuation_target)
    continuation.save_slot(continuation_slot)
    link = ContinuationLink.create(
        parent_identity_sha256=base.identity_sha256,
        parent_tree_sha256=base.tree_sha256,
        parent_manifest_sha256=base.manifest_sha256,
        parent_slots_sha256=base.slots_sha256,
        parent_slot_count=base.accepted_slots,
        base_registry_sha256=canonical_sha256(
            [
                {
                    "target_key": base_slot.target.key,
                    "case_id": base_slot.record.case_id,
                    "question_sha256": base_slot.record.question_sha256,
                }
            ]
        ),
        continuation_identity_sha256=continuation.identity_sha256,
        missing_targets_sha256=canonical_sha256(
            [continuation_target.model_dump(mode="json")]
        ),
        missing_target_keys=(continuation_target.key,),
        call_seed_namespace=1000,
    )
    lineage = CheckpointLineage.create(
        base=checkpoint_lineage_entry(base),
        continuation=checkpoint_lineage_entry(continuation),
        continuation_link=link,
        merged_slots=2,
    )

    validate_checkpoint_lineage(
        lineage, base=base, continuation=continuation, link=link
    )
    PrivateCheckpointStore.resume(
        continuation.root, continuation.identity, max_attempts=49
    )
    with pytest.raises(CheckpointError, match="live checkpoint digests"):
        validate_checkpoint_lineage(
            lineage, base=base, continuation=continuation, link=link
        )
