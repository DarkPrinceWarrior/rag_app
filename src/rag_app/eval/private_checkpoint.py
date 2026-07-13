"""Fail-closed private checkpoints for long-running deterministic evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_app.eval.gold_set import ChallengeTag, ContentType, GoldRecord, Language
from rag_app.eval.private_sidecar import PrivateSidecarRecord, bind_gold_sidecar

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_VERSION_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_REASON_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
_MAX_PRIVATE_FILE_BYTES = 4 * 1024 * 1024
_STRATUM_ORDER = {
    "single_hop": 0,
    "multi_hop": 1,
    "cross_document": 2,
    "no_answer": 3,
}

Stratum = Literal["single_hop", "multi_hop", "cross_document", "no_answer"]


class CheckpointError(RuntimeError):
    """Checkpoint failure whose message never includes private payload values."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with stable UTF-8 canonicalization."""

    payload = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RunIdentity(_StrictModel):
    schema_version: Literal["private-checkpoint-run-v1"] = "private-checkpoint-run-v1"
    seed: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    snapshots_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    model: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(min_length=1, max_length=512)
    generator_contract_version: str = Field(pattern=_VERSION_PATTERN)
    per_stratum: int = Field(ge=1, le=500)
    min_chars: int = Field(ge=1)
    trial: bool


class SlotTarget(_StrictModel):
    stratum: Stratum
    slot: int = Field(ge=0)
    language: Language
    content_type: ContentType | None = None
    challenge_tag: ChallengeTag | None = None
    source_plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_count: int = Field(ge=1, le=8)

    @property
    def key(self) -> str:
        return f"{self.stratum}-{self.slot:04d}"


def _hashed_payload(model: BaseModel) -> str:
    return canonical_sha256(model.model_dump(mode="json", exclude={"payload_sha256"}))


class SlotCheckpoint(_StrictModel):
    schema_version: Literal["private-checkpoint-slot-v1"] = "private-checkpoint-slot-v1"
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    target: SlotTarget
    source_index: int = Field(ge=0)
    retry: int = Field(ge=0)
    call_seed: int = Field(ge=0)
    record: GoldRecord
    sidecar: PrivateSidecarRecord
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        identity_sha256: str,
        target: SlotTarget,
        source_index: int,
        retry: int,
        call_seed: int,
        record: GoldRecord,
        sidecar: PrivateSidecarRecord,
    ) -> Self:
        payload = {
            "schema_version": "private-checkpoint-slot-v1",
            "identity_sha256": identity_sha256,
            "target": target.model_dump(mode="json"),
            "source_index": source_index,
            "retry": retry,
            "call_seed": call_seed,
            "record": record.model_dump(mode="json"),
            "sidecar": sidecar.model_dump(mode="json"),
        }
        return cls.model_validate_json(
            json.dumps(
                {**payload, "payload_sha256": canonical_sha256(payload)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            strict=True,
        )

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        if self.payload_sha256 != _hashed_payload(self):
            raise ValueError("slot checkpoint payload hash mismatch")
        if self.source_index >= self.target.source_count:
            raise ValueError("slot checkpoint source index is outside the source plan")
        if self.sidecar.stratum != self.target.stratum:
            raise ValueError("slot checkpoint stratum mismatch")
        if self.record.language != self.target.language:
            raise ValueError("slot checkpoint language mismatch")
        if self.target.content_type is not None and self.target.content_type not in self.record.content_types:
            raise ValueError("slot checkpoint content target mismatch")
        if (
            self.target.challenge_tag is not None
            and self.target.challenge_tag not in self.record.challenge_tags
        ):
            raise ValueError("slot checkpoint challenge target mismatch")
        try:
            bind_gold_sidecar([self.record], [self.sidecar])
        except ValueError:
            raise ValueError("slot checkpoint record/sidecar binding failed") from None
        return self


class DeterministicReject(_StrictModel):
    source_index: int = Field(ge=0)
    retry: int = Field(ge=0)
    reason_code: str = Field(pattern=_REASON_PATTERN)


class RejectCursor(_StrictModel):
    schema_version: Literal["private-checkpoint-cursor-v1"] = "private-checkpoint-cursor-v1"
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    target: SlotTarget
    rejects: tuple[DeterministicReject, ...] = ()
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def empty(cls, *, identity_sha256: str, target: SlotTarget) -> Self:
        payload = {
            "schema_version": "private-checkpoint-cursor-v1",
            "identity_sha256": identity_sha256,
            "target": target.model_dump(mode="json"),
            "rejects": [],
        }
        return cls.model_validate_json(
            json.dumps(
                {**payload, "payload_sha256": canonical_sha256(payload)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            strict=True,
        )

    def next_attempt(self, source_index: int) -> int:
        """Return the first retry not deterministically rejected for one planned source."""

        if not 0 <= source_index < self.target.source_count:
            raise CheckpointError("source index is outside the slot source plan")
        return sum(item.source_index == source_index for item in self.rejects)

    def with_reject(self, *, source_index: int, retry: int, reason_code: str) -> Self:
        reject = DeterministicReject(
            source_index=source_index,
            retry=retry,
            reason_code=reason_code,
        )
        rejects = sorted(
            (*self.rejects, reject),
            key=lambda item: (item.source_index, item.retry),
        )
        payload = {
            "schema_version": self.schema_version,
            "identity_sha256": self.identity_sha256,
            "target": self.target.model_dump(mode="json"),
            "rejects": [item.model_dump(mode="json") for item in rejects],
        }
        return type(self).model_validate_json(
            json.dumps(
                {**payload, "payload_sha256": canonical_sha256(payload)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            strict=True,
        )

    @model_validator(mode="after")
    def validate_cursor(self) -> Self:
        if self.payload_sha256 != _hashed_payload(self):
            raise ValueError("reject cursor payload hash mismatch")
        pairs = [(item.source_index, item.retry) for item in self.rejects]
        if len(pairs) != len(set(pairs)):
            raise ValueError("reject cursor attempts must be unique")
        if pairs != sorted(pairs):
            raise ValueError("reject cursor attempts must be ordered")
        for source_index in range(self.target.source_count):
            retries = [item.retry for item in self.rejects if item.source_index == source_index]
            if retries != list(range(len(retries))):
                raise ValueError("reject cursor retries must be contiguous per source")
        if any(item.source_index >= self.target.source_count for item in self.rejects):
            raise ValueError("reject cursor source index is outside the source plan")
        return self


class _RunManifest(_StrictModel):
    schema_version: Literal["private-checkpoint-manifest-v1"] = "private-checkpoint-manifest-v1"
    identity: RunIdentity
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    max_attempts: int = Field(ge=1)
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def create(cls, identity: RunIdentity, max_attempts: int) -> Self:
        payload = {
            "schema_version": "private-checkpoint-manifest-v1",
            "identity": identity.model_dump(mode="json"),
            "identity_sha256": canonical_sha256(identity),
            "max_attempts": max_attempts,
        }
        return cls.model_validate({**payload, "payload_sha256": canonical_sha256(payload)}, strict=True)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.identity_sha256 != canonical_sha256(self.identity):
            raise ValueError("manifest identity hash mismatch")
        if self.payload_sha256 != _hashed_payload(self):
            raise ValueError("manifest payload hash mismatch")
        return self


class ContinuationLink(_StrictModel):
    """Public lineage link from an immutable parent to a continuation run."""

    schema_version: Literal["private-checkpoint-continuation-v1"] = (
        "private-checkpoint-continuation-v1"
    )
    epoch: Literal[1] = 1
    parent_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_slots_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_slot_count: int = Field(ge=1, le=500)
    base_registry_sha256: str = Field(pattern=_SHA256_PATTERN)
    continuation_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    missing_targets_sha256: str = Field(pattern=_SHA256_PATTERN)
    missing_target_keys: tuple[str, ...] = Field(min_length=1, max_length=500)
    call_seed_namespace: int = Field(ge=0)
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        parent_identity_sha256: str,
        parent_tree_sha256: str,
        parent_manifest_sha256: str,
        parent_slots_sha256: str,
        parent_slot_count: int,
        base_registry_sha256: str,
        continuation_identity_sha256: str,
        missing_targets_sha256: str,
        missing_target_keys: tuple[str, ...],
        call_seed_namespace: int,
    ) -> Self:
        payload = {
            "schema_version": "private-checkpoint-continuation-v1",
            "epoch": 1,
            "parent_identity_sha256": parent_identity_sha256,
            "parent_tree_sha256": parent_tree_sha256,
            "parent_manifest_sha256": parent_manifest_sha256,
            "parent_slots_sha256": parent_slots_sha256,
            "parent_slot_count": parent_slot_count,
            "base_registry_sha256": base_registry_sha256,
            "continuation_identity_sha256": continuation_identity_sha256,
            "missing_targets_sha256": missing_targets_sha256,
            "missing_target_keys": missing_target_keys,
            "call_seed_namespace": call_seed_namespace,
        }
        return cls.model_validate(
            {**payload, "payload_sha256": canonical_sha256(payload)}, strict=True
        )

    @model_validator(mode="after")
    def validate_link(self) -> Self:
        if self.payload_sha256 != _hashed_payload(self):
            raise ValueError("continuation link payload hash mismatch")
        if self.missing_target_keys != tuple(sorted(set(self.missing_target_keys))):
            raise ValueError("continuation missing target keys must be unique and sorted")
        return self


class CheckpointLineageEntry(_StrictModel):
    identity: RunIdentity
    identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    max_attempts: int = Field(ge=1)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    slots_sha256: str = Field(pattern=_SHA256_PATTERN)
    tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    accepted_slots: int = Field(ge=0, le=500)

    @model_validator(mode="after")
    def validate_identity_hash(self) -> Self:
        if self.identity_sha256 != canonical_sha256(self.identity):
            raise ValueError("lineage identity hash mismatch")
        return self


class CheckpointLineage(_StrictModel):
    schema_version: Literal["private-checkpoint-lineage-v1"] = (
        "private-checkpoint-lineage-v1"
    )
    base: CheckpointLineageEntry
    continuation: CheckpointLineageEntry
    continuation_link: ContinuationLink
    merged_slots: int = Field(ge=1, le=500)
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        base: CheckpointLineageEntry,
        continuation: CheckpointLineageEntry,
        continuation_link: ContinuationLink,
        merged_slots: int,
    ) -> Self:
        payload = {
            "schema_version": "private-checkpoint-lineage-v1",
            "base": base.model_dump(mode="json"),
            "continuation": continuation.model_dump(mode="json"),
            "continuation_link": continuation_link.model_dump(mode="json"),
            "merged_slots": merged_slots,
        }
        return cls.model_validate(
            {
                "schema_version": "private-checkpoint-lineage-v1",
                "base": base,
                "continuation": continuation,
                "continuation_link": continuation_link,
                "merged_slots": merged_slots,
                "payload_sha256": canonical_sha256(payload),
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.payload_sha256 != _hashed_payload(self):
            raise ValueError("checkpoint lineage payload hash mismatch")
        if self.base.accepted_slots + self.continuation.accepted_slots != self.merged_slots:
            raise ValueError("checkpoint lineage merged slot count mismatch")
        link = self.continuation_link
        if (
            link.parent_identity_sha256 != self.base.identity_sha256
            or link.parent_tree_sha256 != self.base.tree_sha256
            or link.parent_manifest_sha256 != self.base.manifest_sha256
            or link.parent_slots_sha256 != self.base.slots_sha256
            or link.parent_slot_count != self.base.accepted_slots
            or link.continuation_identity_sha256 != self.continuation.identity_sha256
        ):
            raise ValueError("checkpoint lineage link does not bind its entries")
        return self


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def _require_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir() or _mode(path) != 0o700:
        raise CheckpointError(f"{label} must be a non-symlink directory with mode 0700")


def _require_private_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _mode(path) != 0o600:
        raise CheckpointError(f"{label} must be a non-symlink file with mode 0600")
    if path.stat(follow_symlinks=False).st_size > _MAX_PRIVATE_FILE_BYTES:
        raise CheckpointError(f"{label} exceeds the private checkpoint size limit")


def _atomic_model_write(path: Path, model: BaseModel) -> None:
    payload = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _read_model(path: Path, model_type: type[_StrictModel], *, label: str) -> Any:
    _require_private_file(path, label=label)
    try:
        return model_type.model_validate_json(path.read_bytes(), strict=True)
    except Exception:
        raise CheckpointError(f"{label} is corrupt or violates its strict schema") from None


def checkpoint_tree_sha256(root: Path) -> str:
    """Hash private checkpoint names, modes and bytes without exposing payloads."""

    _require_directory(root, label="checkpoint root")
    expected_root_entries = {"manifest.json", "slots", "cursors"}
    if {entry.name for entry in root.iterdir()} != expected_root_entries:
        raise CheckpointError("checkpoint root contains unknown or missing entries")
    entries: list[dict[str, Any]] = [{"path": ".", "mode": _mode(root)}]
    for directory_name in ("slots", "cursors"):
        directory = root / directory_name
        _require_directory(directory, label=f"checkpoint {directory_name} directory")
        entries.append({"path": directory_name, "mode": _mode(directory)})
        for path in sorted(directory.iterdir()):
            _require_private_file(path, label="checkpoint payload")
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "mode": _mode(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest_path = root / "manifest.json"
    _require_private_file(manifest_path, label="checkpoint manifest")
    entries.append(
        {
            "path": "manifest.json",
            "mode": _mode(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    )
    return canonical_sha256(sorted(entries, key=lambda item: item["path"]))


def write_continuation_link(path: Path, link: ContinuationLink) -> None:
    if path.exists() or path.is_symlink():
        raise CheckpointError("continuation link already exists")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise CheckpointError("continuation link parent must be a non-symlink directory")
    _atomic_model_write(path, link)


def read_continuation_link(path: Path) -> ContinuationLink:
    return _read_model(path, ContinuationLink, label="continuation link")


class PrivateCheckpointStore:
    """Atomic per-slot private store with explicit, eagerly validated resume."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: _RunManifest,
        slots: dict[str, SlotCheckpoint],
        cursors: dict[str, RejectCursor],
    ) -> None:
        self.root = root
        self._manifest = manifest
        self._slots = slots
        self._cursors = cursors

    @property
    def identity(self) -> RunIdentity:
        return self._manifest.identity

    @property
    def identity_sha256(self) -> str:
        return self._manifest.identity_sha256

    @property
    def max_attempts(self) -> int:
        return self._manifest.max_attempts

    @property
    def accepted_slots(self) -> int:
        return len(self._slots)

    @property
    def manifest_sha256(self) -> str:
        _require_private_file(self._manifest_path, label="checkpoint manifest")
        return hashlib.sha256(self._manifest_path.read_bytes()).hexdigest()

    @property
    def slots_sha256(self) -> str:
        return canonical_sha256(
            [
                {"target_key": slot.target.key, "payload_sha256": slot.payload_sha256}
                for slot in self.iter_slots()
            ]
        )

    @property
    def tree_sha256(self) -> str:
        return checkpoint_tree_sha256(self.root)

    @property
    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def _slots_dir(self) -> Path:
        return self.root / "slots"

    @property
    def _cursors_dir(self) -> Path:
        return self.root / "cursors"

    @classmethod
    def create(cls, root: Path, identity: RunIdentity, *, max_attempts: int) -> Self:
        if max_attempts < 1:
            raise CheckpointError("max_attempts must be positive")
        if root.exists() or root.is_symlink():
            raise CheckpointError("checkpoint root already exists; resume must be explicit")
        root.mkdir(parents=True, mode=0o700)
        os.chmod(root, 0o700)
        slots_dir = root / "slots"
        cursors_dir = root / "cursors"
        slots_dir.mkdir(mode=0o700)
        cursors_dir.mkdir(mode=0o700)
        manifest = _RunManifest.create(identity, max_attempts)
        _atomic_model_write(root / "manifest.json", manifest)
        return cls(root=root, manifest=manifest, slots={}, cursors={})

    @classmethod
    def open_readonly(cls, root: Path, identity: RunIdentity) -> Self:
        """Open and validate a checkpoint without rewriting any parent bytes."""

        _require_directory(root, label="checkpoint root")
        manifest = _read_model(
            root / "manifest.json", _RunManifest, label="checkpoint manifest"
        )
        return cls.resume(
            root,
            identity,
            max_attempts=manifest.max_attempts,
        )

    @classmethod
    def resume(cls, root: Path, identity: RunIdentity, *, max_attempts: int) -> Self:
        if max_attempts < 1:
            raise CheckpointError("max_attempts must be positive")
        _require_directory(root, label="checkpoint root")
        _require_directory(root / "slots", label="checkpoint slots directory")
        _require_directory(root / "cursors", label="checkpoint cursors directory")
        expected_root_entries = {"manifest.json", "slots", "cursors"}
        if {entry.name for entry in root.iterdir()} != expected_root_entries:
            raise CheckpointError("checkpoint root contains unknown or missing entries")
        manifest = _read_model(root / "manifest.json", _RunManifest, label="checkpoint manifest")
        if manifest.identity != identity or manifest.identity_sha256 != canonical_sha256(identity):
            raise CheckpointError("checkpoint run identity mismatch")
        if max_attempts < manifest.max_attempts:
            raise CheckpointError("max_attempts cannot decrease on resume")

        slots: dict[str, SlotCheckpoint] = {}
        for path in sorted((root / "slots").iterdir()):
            if path.suffix != ".json":
                raise CheckpointError("checkpoint slots directory contains an unknown entry")
            checkpoint = _read_model(path, SlotCheckpoint, label="slot checkpoint")
            if checkpoint.identity_sha256 != manifest.identity_sha256:
                raise CheckpointError("slot checkpoint run identity mismatch")
            if path.name != f"{checkpoint.target.key}.json":
                raise CheckpointError("slot checkpoint filename does not match its target")
            if checkpoint.target.key in slots:
                raise CheckpointError("duplicate slot checkpoint target")
            if checkpoint.retry >= max_attempts:
                raise CheckpointError("slot checkpoint retry exceeds max_attempts")
            slots[checkpoint.target.key] = checkpoint

        cursors: dict[str, RejectCursor] = {}
        for path in sorted((root / "cursors").iterdir()):
            if path.suffix != ".json":
                raise CheckpointError("checkpoint cursors directory contains an unknown entry")
            cursor = _read_model(path, RejectCursor, label="reject cursor")
            if cursor.identity_sha256 != manifest.identity_sha256:
                raise CheckpointError("reject cursor run identity mismatch")
            if path.name != f"{cursor.target.key}.json":
                raise CheckpointError("reject cursor filename does not match its target")
            if cursor.target.key in cursors:
                raise CheckpointError("duplicate reject cursor target")
            if any(item.retry >= max_attempts for item in cursor.rejects):
                raise CheckpointError("reject cursor retry exceeds max_attempts")
            cursors[cursor.target.key] = cursor

        for key in slots.keys() & cursors.keys():
            if slots[key].target != cursors[key].target:
                raise CheckpointError("slot checkpoint and reject cursor targets differ")

        if max_attempts > manifest.max_attempts:
            manifest = _RunManifest.create(identity, max_attempts)
            _atomic_model_write(root / "manifest.json", manifest)
        return cls(root=root, manifest=manifest, slots=slots, cursors=cursors)

    def iter_slots(self) -> tuple[SlotCheckpoint, ...]:
        return tuple(
            sorted(
                self._slots.values(),
                key=lambda item: (_STRATUM_ORDER[item.target.stratum], item.target.slot),
            )
        )

    def iter_cursors(self) -> tuple[RejectCursor, ...]:
        return tuple(
            sorted(
                self._cursors.values(),
                key=lambda item: (_STRATUM_ORDER[item.target.stratum], item.target.slot),
            )
        )

    def load_slot(self, target: SlotTarget) -> SlotCheckpoint | None:
        checkpoint = self._slots.get(target.key)
        if checkpoint is not None and checkpoint.target != target:
            raise CheckpointError("stored slot target differs from the current plan")
        return checkpoint

    def load_cursor(self, target: SlotTarget) -> RejectCursor:
        cursor = self._cursors.get(target.key)
        if cursor is None:
            return RejectCursor.empty(identity_sha256=self.identity_sha256, target=target)
        if cursor.target != target:
            raise CheckpointError("stored reject cursor target differs from the current plan")
        return cursor

    def save_slot(self, checkpoint: SlotCheckpoint) -> None:
        try:
            checked = SlotCheckpoint.model_validate_json(checkpoint.model_dump_json(), strict=True)
        except Exception:
            raise CheckpointError("slot checkpoint violates its strict schema") from None
        if checked.identity_sha256 != self.identity_sha256:
            raise CheckpointError("slot checkpoint run identity mismatch")
        if checked.retry >= self.max_attempts:
            raise CheckpointError("slot checkpoint retry exceeds max_attempts")
        existing = self._slots.get(checked.target.key)
        if existing is not None:
            if existing != checked:
                raise CheckpointError("accepted slot checkpoint is immutable")
            return
        cursor = self.load_cursor(checked.target)
        if checked.retry != cursor.next_attempt(checked.source_index):
            raise CheckpointError("slot retry does not match the deterministic cursor")
        path = self._slots_dir / f"{checked.target.key}.json"
        if path.exists() or path.is_symlink():
            raise CheckpointError("slot checkpoint path already exists")
        _atomic_model_write(path, checked)
        self._slots[checked.target.key] = checked

    def invalidate_slot(self, target: SlotTarget, *, reason_code: str) -> RejectCursor:
        """Remove one accepted slot and advance its deterministic retry cursor."""

        checkpoint = self.load_slot(target)
        if checkpoint is None:
            raise CheckpointError("cannot invalidate a missing accepted slot")
        cursor = self.load_cursor(target)
        if checkpoint.retry != cursor.next_attempt(checkpoint.source_index):
            raise CheckpointError("accepted slot retry does not match cursor.next_attempt")
        path = self._slots_dir / f"{target.key}.json"
        _require_private_file(path, label="slot checkpoint")
        path.unlink()
        directory_fd = os.open(self._slots_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        del self._slots[target.key]
        return self.record_deterministic_reject(
            target,
            source_index=checkpoint.source_index,
            retry=checkpoint.retry,
            reason_code=reason_code,
        )

    def record_deterministic_reject(
        self,
        target: SlotTarget,
        *,
        source_index: int,
        retry: int,
        reason_code: str,
    ) -> RejectCursor:
        if target.key in self._slots:
            raise CheckpointError("cannot advance the cursor for an accepted slot")
        cursor = self.load_cursor(target)
        if retry != cursor.next_attempt(source_index):
            raise CheckpointError("reject retry does not match cursor.next_attempt")
        if retry >= self.max_attempts:
            raise CheckpointError("reject retry exceeds max_attempts")
        try:
            updated = cursor.with_reject(
                source_index=source_index,
                retry=retry,
                reason_code=reason_code,
            )
        except Exception:
            raise CheckpointError("deterministic reject violates its strict schema") from None
        path = self._cursors_dir / f"{target.key}.json"
        _atomic_model_write(path, updated)
        self._cursors[target.key] = updated
        return updated

    def cleanup_after_success(self, *, final_artifacts_written: bool) -> None:
        if not final_artifacts_written:
            raise CheckpointError("final artifact confirmation is required for cleanup")
        validated = type(self).resume(self.root, self.identity, max_attempts=self.max_attempts)
        for path in sorted(validated._slots_dir.iterdir()):
            if path.is_symlink():
                raise CheckpointError("cleanup refuses a symlink slot checkpoint")
            path.unlink()
        for path in sorted(validated._cursors_dir.iterdir()):
            if path.is_symlink():
                raise CheckpointError("cleanup refuses a symlink reject cursor")
            path.unlink()
        validated._slots_dir.rmdir()
        validated._cursors_dir.rmdir()
        validated._manifest_path.unlink()
        if validated.root.is_symlink():
            raise CheckpointError("cleanup refuses a symlink checkpoint root")
        validated.root.rmdir()


def checkpoint_lineage_entry(store: PrivateCheckpointStore) -> CheckpointLineageEntry:
    return CheckpointLineageEntry(
        identity=store.identity,
        identity_sha256=store.identity_sha256,
        max_attempts=store.max_attempts,
        manifest_sha256=store.manifest_sha256,
        slots_sha256=store.slots_sha256,
        tree_sha256=store.tree_sha256,
        accepted_slots=store.accepted_slots,
    )


def validate_checkpoint_lineage(
    lineage: CheckpointLineage,
    *,
    base: PrivateCheckpointStore,
    continuation: PrivateCheckpointStore,
    link: ContinuationLink,
) -> None:
    """Recompute every live checkpoint digest referenced by final lineage."""

    expected = CheckpointLineage.create(
        base=checkpoint_lineage_entry(base),
        continuation=checkpoint_lineage_entry(continuation),
        continuation_link=link,
        merged_slots=lineage.merged_slots,
    )
    if lineage != expected:
        raise CheckpointError("checkpoint lineage differs from live checkpoint digests")
    base_slots = base.iter_slots()
    continuation_slots = continuation.iter_slots()
    base_registry_sha256 = canonical_sha256(
        [
            {
                "target_key": slot.target.key,
                "case_id": slot.record.case_id,
                "question_sha256": slot.record.question_sha256,
            }
            for slot in sorted(base_slots, key=lambda item: item.target.key)
        ]
    )
    missing_targets_sha256 = canonical_sha256(
        [
            slot.target.model_dump(mode="json")
            for slot in sorted(continuation_slots, key=lambda item: item.target.key)
        ]
    )
    if (
        link.parent_identity_sha256 != base.identity_sha256
        or link.parent_tree_sha256 != base.tree_sha256
        or link.parent_manifest_sha256 != base.manifest_sha256
        or link.parent_slots_sha256 != base.slots_sha256
        or link.parent_slot_count != base.accepted_slots
        or link.base_registry_sha256 != base_registry_sha256
        or link.continuation_identity_sha256 != continuation.identity_sha256
        or link.missing_target_keys
        != tuple(sorted(slot.target.key for slot in continuation_slots))
        or link.missing_targets_sha256 != missing_targets_sha256
        or any(slot.call_seed < link.call_seed_namespace for slot in continuation_slots)
    ):
        raise CheckpointError("continuation link differs from live checkpoint lineage")


__all__ = [
    "CheckpointLineage",
    "CheckpointLineageEntry",
    "CheckpointError",
    "ContinuationLink",
    "DeterministicReject",
    "PrivateCheckpointStore",
    "RejectCursor",
    "RunIdentity",
    "SlotCheckpoint",
    "SlotTarget",
    "canonical_sha256",
    "checkpoint_lineage_entry",
    "checkpoint_tree_sha256",
    "read_continuation_link",
    "validate_checkpoint_lineage",
    "write_continuation_link",
]
