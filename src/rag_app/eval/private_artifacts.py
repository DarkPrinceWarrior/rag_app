from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW


class PrivateArtifactError(RuntimeError):
    """A private artifact failed an I/O, integrity, or format check."""


class PrivateArtifactSecurityError(PrivateArtifactError):
    """A private artifact or its parent violates the local security policy."""


class PrivateArtifactFormatError(PrivateArtifactError):
    """A private artifact is not strict JSON or failed schema validation."""


@dataclass(frozen=True, slots=True)
class PrivateBytesArtifact:
    raw_bytes: bytes
    sha256: str
    size: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class PrivateJsonArtifact[T](PrivateBytesArtifact):
    value: T


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_strict_json(raw_bytes: bytes) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate keys and non-finite constants."""
    try:
        text = raw_bytes.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise PrivateArtifactFormatError(
            f"private artifact is not strict JSON ({type(error).__name__})"
        ) from None


def _path_parts(path: Path) -> tuple[str, tuple[str, ...], str]:
    if path.name in {"", ".", ".."}:
        raise PrivateArtifactSecurityError("private artifact path must name a file")
    parts = path.parts
    if path.is_absolute():
        start = path.anchor
        parents = parts[1:-1]
    else:
        start = "."
        parents = parts[:-1]
    if any(part in {"", ".."} for part in parents):
        raise PrivateArtifactSecurityError("private artifact path traversal is forbidden")
    return start, tuple(part for part in parents if part != "."), path.name


def _open_private_parent(path: Path, *, expected_uid: int) -> tuple[int, str]:
    start, parents, filename = _path_parts(path)
    try:
        directory_fd = os.open(start, _DIRECTORY_FLAGS)
    except OSError as error:
        raise PrivateArtifactSecurityError(
            f"private artifact parent is inaccessible ({type(error).__name__})"
        ) from None
    try:
        for component in parents:
            next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PrivateArtifactSecurityError("private artifact parent must be a directory")
        if metadata.st_uid != expected_uid:
            raise PrivateArtifactSecurityError("private artifact parent has an unexpected owner")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PrivateArtifactSecurityError("private artifact parent must not be shared")
        return directory_fd, filename
    except PrivateArtifactError:
        os.close(directory_fd)
        raise
    except OSError as error:
        os.close(directory_fd)
        raise PrivateArtifactSecurityError(
            f"private artifact parent is unsafe ({type(error).__name__})"
        ) from None


def _validate_file_metadata(metadata: os.stat_result, *, expected_uid: int) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateArtifactSecurityError("private artifact must be a regular file")
    if metadata.st_uid != expected_uid:
        raise PrivateArtifactSecurityError("private artifact has an unexpected owner")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PrivateArtifactSecurityError("private artifact must have mode 0600")
    if metadata.st_nlink != 1:
        raise PrivateArtifactSecurityError("private artifact must have exactly one hard link")


def _stat_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_all(descriptor: int, *, expected_size: int, max_bytes: int) -> bytes:
    content = bytearray()
    while True:
        block = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(content)))
        if not block:
            break
        content.extend(block)
        if len(content) > max_bytes:
            raise PrivateArtifactSecurityError("private artifact exceeds the size limit")
    if len(content) != expected_size:
        raise PrivateArtifactSecurityError("private artifact changed while being read")
    return bytes(content)


def _validate_limits(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")


def read_private_bytes(
    path: Path,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    expected_uid: int | None = None,
) -> PrivateBytesArtifact:
    """Read a private artifact from one pinned file descriptor."""
    _validate_limits(max_bytes)
    owner = os.geteuid() if expected_uid is None else expected_uid
    directory_fd, filename = _open_private_parent(path, expected_uid=owner)
    try:
        try:
            descriptor = os.open(filename, _READ_FLAGS, dir_fd=directory_fd)
        except OSError as error:
            raise PrivateArtifactSecurityError(
                f"private artifact cannot be opened safely ({type(error).__name__})"
            ) from None
    finally:
        os.close(directory_fd)

    try:
        before = os.fstat(descriptor)
        _validate_file_metadata(before, expected_uid=owner)
        if before.st_size > max_bytes:
            raise PrivateArtifactSecurityError("private artifact exceeds the size limit")
        raw_bytes = _read_all(descriptor, expected_size=before.st_size, max_bytes=max_bytes)
        after = os.fstat(descriptor)
        _validate_file_metadata(after, expected_uid=owner)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise PrivateArtifactSecurityError("private artifact changed while being read")
    except OSError as error:
        raise PrivateArtifactError(f"private artifact read failed ({type(error).__name__})") from None
    finally:
        os.close(descriptor)

    return PrivateBytesArtifact(
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        size=len(raw_bytes),
        device=after.st_dev,
        inode=after.st_ino,
    )


@overload
def read_private_json(
    path: Path,
    *,
    parser: None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    expected_uid: int | None = None,
) -> PrivateJsonArtifact[Any]: ...


@overload
def read_private_json[T](
    path: Path,
    *,
    parser: Callable[[bytes], T],
    max_bytes: int = _DEFAULT_MAX_BYTES,
    expected_uid: int | None = None,
) -> PrivateJsonArtifact[T]: ...


def read_private_json[T](
    path: Path,
    *,
    parser: Callable[[bytes], T] | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    expected_uid: int | None = None,
) -> PrivateJsonArtifact[Any] | PrivateJsonArtifact[T]:
    """Read and validate a private JSON artifact from one pinned file descriptor."""
    artifact = read_private_bytes(path, max_bytes=max_bytes, expected_uid=expected_uid)
    raw_bytes = artifact.raw_bytes

    decoded = parse_strict_json(raw_bytes)
    if parser is None:
        value = decoded
    else:
        try:
            value = parser(raw_bytes)
        except Exception as error:
            raise PrivateArtifactFormatError(
                f"private artifact schema is invalid ({type(error).__name__})"
            ) from None
    return PrivateJsonArtifact(
        value=value,
        raw_bytes=raw_bytes,
        sha256=artifact.sha256,
        size=artifact.size,
        device=artifact.device,
        inode=artifact.inode,
    )


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written < 1:
            raise OSError("short write")
        remaining = remaining[written:]


def write_private_json_fresh(
    path: Path,
    raw_bytes: bytes,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    expected_uid: int | None = None,
) -> PrivateJsonArtifact[Any]:
    """Atomically publish a new mode-0600 JSON file without replacing any entry."""
    _validate_limits(max_bytes)
    if not isinstance(raw_bytes, bytes):
        raise TypeError("raw_bytes must be bytes")
    if len(raw_bytes) > max_bytes:
        raise PrivateArtifactSecurityError("private artifact exceeds the size limit")
    decoded = parse_strict_json(raw_bytes)
    owner = os.geteuid() if expected_uid is None else expected_uid
    directory_fd, filename = _open_private_parent(path, expected_uid=owner)
    temporary_name = f".private-json-{os.getpid()}-{secrets.token_hex(12)}.tmp"
    descriptor = -1
    temporary_exists = False
    try:
        try:
            descriptor = os.open(
                temporary_name,
                _WRITE_FLAGS,
                0o600,
                dir_fd=directory_fd,
            )
            temporary_exists = True
            _write_all(descriptor, raw_bytes)
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            _validate_file_metadata(metadata, expected_uid=owner)
            if metadata.st_size != len(raw_bytes):
                raise PrivateArtifactError("private artifact write size is inconsistent")
            try:
                os.link(
                    temporary_name,
                    filename,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raise FileExistsError("refusing to replace an existing private artifact") from None
            os.unlink(temporary_name, dir_fd=directory_fd)
            temporary_exists = False
            metadata = os.fstat(descriptor)
            _validate_file_metadata(metadata, expected_uid=owner)
            os.fsync(directory_fd)
        except (FileExistsError, PrivateArtifactError):
            raise
        except OSError as error:
            raise PrivateArtifactError(f"private artifact write failed ({type(error).__name__})") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)

    return PrivateJsonArtifact(
        value=decoded,
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        size=len(raw_bytes),
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )
