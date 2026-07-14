from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from rag_app.eval import private_artifacts
from rag_app.eval.private_artifacts import (
    PrivateArtifactFormatError,
    PrivateArtifactSecurityError,
    read_private_bytes,
    read_private_json,
    write_private_json_fresh,
)


def _private_directory(tmp_path: Path, name: str = "private") -> Path:
    directory = tmp_path / name
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    return directory


def _private_file(directory: Path, content: bytes, name: str = "artifact.json") -> Path:
    path = directory / name
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def test_read_private_json_returns_exact_bytes_digest_and_parser_value(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path)
    raw = b'{"schema":"v1","items":[1,2,3]}\n'
    path = _private_file(directory, raw)
    parser_inputs: list[bytes] = []

    def parser(content: bytes) -> tuple[str, int]:
        parser_inputs.append(content)
        value = json.loads(content)
        return value["schema"], len(value["items"])

    artifact = read_private_json(path, parser=parser)

    assert artifact.value == ("v1", 3)
    assert artifact.raw_bytes == raw
    assert artifact.sha256 == hashlib.sha256(raw).hexdigest()
    assert artifact.size == len(raw)
    assert parser_inputs == [raw]
    assert artifact.inode == path.stat().st_ino


def test_read_private_bytes_accepts_jsonl_and_returns_exact_digest(tmp_path: Path) -> None:
    raw = b'{"case_id":"one"}\n{"case_id":"two"}\n'
    path = _private_file(_private_directory(tmp_path), raw, "gold.jsonl")

    artifact = read_private_bytes(path, max_bytes=len(raw))

    assert artifact.raw_bytes == raw
    assert artifact.sha256 == hashlib.sha256(raw).hexdigest()
    assert artifact.size == len(raw)
    assert artifact.inode == path.stat().st_ino


def test_read_private_json_opens_the_artifact_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _private_file(_private_directory(tmp_path), b'{"safe":true}')
    original_open = private_artifacts.os.open
    artifact_open_count = 0

    def counting_open(
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal artifact_open_count
        if target == path.name and not flags & os.O_DIRECTORY:
            artifact_open_count += 1
        return original_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(private_artifacts.os, "open", counting_open)

    artifact = read_private_json(path)

    assert artifact.value == {"safe": True}
    assert artifact_open_count == 1


@pytest.mark.parametrize(
    "raw",
    [
        b'{"same":1,"same":2}',
        b'{"nested":{"same":1,"same":2}}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
    ],
)
def test_read_private_json_rejects_non_strict_json(tmp_path: Path, raw: bytes) -> None:
    path = _private_file(_private_directory(tmp_path), raw)

    with pytest.raises(PrivateArtifactFormatError):
        read_private_json(path)


def test_read_private_json_rejects_leaf_and_parent_symlinks(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path, "actual")
    target = _private_file(directory, b'{"safe":true}')
    leaf_link = directory / "leaf-link.json"
    leaf_link.symlink_to(target)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(directory, target_is_directory=True)

    with pytest.raises(PrivateArtifactSecurityError):
        read_private_json(leaf_link)
    with pytest.raises(PrivateArtifactSecurityError):
        read_private_json(parent_link / target.name)


def test_read_private_json_rejects_mode_hard_link_and_shared_parent(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path, "private-mode")
    bad_mode = _private_file(directory, b"{}", "bad-mode.json")
    bad_mode.chmod(0o640)
    hard_link_source = _private_file(directory, b"{}", "hard-link.json")
    os.link(hard_link_source, directory / "hard-link-copy.json")
    shared_directory = _private_directory(tmp_path, "shared")
    shared = _private_file(shared_directory, b"{}")
    shared_directory.chmod(0o750)

    with pytest.raises(PrivateArtifactSecurityError):
        read_private_json(bad_mode)
    with pytest.raises(PrivateArtifactSecurityError):
        read_private_json(hard_link_source)
    with pytest.raises(PrivateArtifactSecurityError):
        read_private_json(shared)


def test_read_private_json_enforces_size_limit(tmp_path: Path) -> None:
    path = _private_file(_private_directory(tmp_path), b'{"value":"large"}')

    with pytest.raises(PrivateArtifactSecurityError, match="size limit"):
        read_private_json(path, max_bytes=4)


def test_read_private_json_detects_same_size_concurrent_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _private_file(_private_directory(tmp_path), b'{"value":1}')
    original_read = private_artifacts.os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        content = original_read(descriptor, count)
        if content and not mutated:
            mutated = True
            before = path.stat()
            with path.open("r+b") as stream:
                stream.write(b'{"value":2}')
                stream.flush()
                os.fsync(stream.fileno())
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        return content

    monkeypatch.setattr(private_artifacts.os, "read", mutating_read)

    with pytest.raises(PrivateArtifactSecurityError, match="changed"):
        read_private_json(path)


def test_write_private_json_fresh_publishes_mode_0600_atomically(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path)
    destination = directory / "decision.json"
    raw = b'{"accepted":false,"failure_codes":["NO_IMPROVEMENT"]}\n'

    artifact = write_private_json_fresh(destination, raw)

    assert destination.read_bytes() == raw
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert destination.stat().st_nlink == 1
    assert artifact.raw_bytes == raw
    assert artifact.sha256 == hashlib.sha256(raw).hexdigest()
    assert artifact.value["accepted"] is False
    assert {entry.name for entry in directory.iterdir()} == {destination.name}


def test_write_private_json_fresh_never_replaces_file_or_symlink(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path)
    existing = _private_file(directory, b'{"original":true}', "existing.json")

    with pytest.raises(FileExistsError):
        write_private_json_fresh(existing, b'{"replacement":true}')
    assert existing.read_bytes() == b'{"original":true}'

    victim = _private_file(directory, b'{"victim":true}', "victim.json")
    symlink = directory / "output.json"
    symlink.symlink_to(victim)
    with pytest.raises(FileExistsError):
        write_private_json_fresh(symlink, b'{"replacement":true}')
    assert symlink.is_symlink()
    assert victim.read_bytes() == b'{"victim":true}'


def test_write_private_json_fresh_validates_before_creating_output(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path)
    destination = directory / "invalid.json"

    with pytest.raises(PrivateArtifactFormatError):
        write_private_json_fresh(destination, b'{"duplicate":1,"duplicate":2}')

    assert not destination.exists()
    assert list(directory.iterdir()) == []


def test_private_json_rejects_path_traversal_and_invalid_limits(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path)
    path = _private_file(directory, b"{}")

    with pytest.raises(PrivateArtifactSecurityError, match="traversal"):
        read_private_json(directory / ".." / directory.name / path.name)
    with pytest.raises(ValueError, match="positive integer"):
        read_private_json(path, max_bytes=0)
    with pytest.raises(TypeError, match="must be bytes"):
        write_private_json_fresh(directory / "new.json", "{}")  # type: ignore[arg-type]
