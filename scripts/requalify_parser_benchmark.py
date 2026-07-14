"""Offline requalification of saved MinerU benchmark outputs.

This command rebuilds schema-v2 derived fields without running a parser. It
binds every result to an immutable source PDF and exactly one saved
``content_list.json`` and refuses partial or ambiguous corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_app.pipeline.parse import (  # noqa: E402
    backfill_text_layer,
    pdf_info,
    read_pdf_text_by_page,
)
from rag_app.pipeline.parse_quality import evaluate_parse, quality_metadata  # noqa: E402
from rag_app.pipeline.segments import content_list_to_segments  # noqa: E402
from scripts.benchmark_complex_parsers import (  # noqa: E402
    _aggregates,
    _benchmark_proxies,
    _stats,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SUMMARY_BYTES = 64 * 1024 * 1024
_MAX_CONTENT_LIST_BYTES = 64 * 1024 * 1024


class RequalificationError(ValueError):
    """The saved benchmark cannot be bound and requalified exactly."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError
        value[key] = item
    return value


def _reject_constant(_: str) -> None:
    raise ValueError


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, size_limit: int | None = None) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise RequalificationError(f"unable to stat required file ({type(error).__name__})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RequalificationError("required file must be regular and non-symlink")
    if size_limit is not None and info.st_size > size_limit:
        raise RequalificationError("required file exceeds size limit")
    return info


def _regular_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise RequalificationError(f"unable to stat required directory ({type(error).__name__})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RequalificationError("required directory must be regular and non-symlink")


def _load_json_file(path: Path, *, size_limit: int) -> tuple[Any, bytes]:
    _regular_file(path, size_limit=size_limit)
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
        raise RequalificationError(f"invalid JSON input ({type(error).__name__})") from None
    return value, raw


def _content_list_path(document_dir: Path) -> Path:
    _regular_directory(document_dir)
    candidates = sorted(document_dir.rglob("*_content_list.json"))
    if len(candidates) != 1:
        raise RequalificationError("each parser output must contain exactly one content_list")
    candidate = candidates[0]
    parent = candidate.parent
    while parent != document_dir:
        _regular_directory(parent)
        if document_dir not in parent.parents:
            raise RequalificationError("content_list escapes its document directory")
        parent = parent.parent
    _regular_file(candidate, size_limit=_MAX_CONTENT_LIST_BYTES)
    return candidate


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RequalificationError("destination already exists")
    _regular_directory(path.parent)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise RequalificationError("destination already exists") from None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def requalify_summary(
    summary_path: Path,
    output_root: Path,
    pdf_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Rebuild one complete MinerU summary and create ``destination`` once."""

    summary_value, summary_raw = _load_json_file(summary_path, size_limit=_MAX_SUMMARY_BYTES)
    if not isinstance(summary_value, dict):
        raise RequalificationError("benchmark summary must be a JSON object")
    summary = cast(dict[str, Any], summary_value)
    results = summary.get("results")
    backends = summary.get("backends")
    if not isinstance(results, dict) or not results:
        raise RequalificationError("benchmark summary results are invalid")
    if not isinstance(backends, list) or "mineru" not in backends:
        raise RequalificationError("benchmark summary must contain MinerU")

    _regular_directory(output_root)
    mineru_root = output_root / "mineru"
    _regular_directory(mineru_root)
    _regular_directory(pdf_root)

    expected_stems: set[str] = set()
    verified_rows: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    source_digests: set[str] = set()
    for filename, row_value in results.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".pdf"
            or not isinstance(row_value, dict)
        ):
            raise RequalificationError("benchmark result identity is invalid")
        row = cast(dict[str, Any], row_value)
        source_sha256 = row.get("source_sha256")
        backend_value = row.get("mineru")
        if (
            not isinstance(source_sha256, str)
            or _SHA256.fullmatch(source_sha256) is None
            or source_sha256 in source_digests
            or not isinstance(backend_value, dict)
            or backend_value.get("status") != "ok"
        ):
            raise RequalificationError("benchmark MinerU result is not exact and complete")
        source_digests.add(source_sha256)
        stem = Path(filename).stem
        if stem in expected_stems:
            raise RequalificationError("benchmark output directory identity is ambiguous")
        expected_stems.add(stem)
        verified_rows.append((filename, row, cast(dict[str, Any], backend_value), stem))

    try:
        output_entries = list(mineru_root.iterdir())
    except OSError as error:
        raise RequalificationError(f"unable to inspect parser outputs ({type(error).__name__})") from None
    if {entry.name for entry in output_entries} != expected_stems:
        raise RequalificationError("parser output root does not contain the exact benchmark results")
    for entry in output_entries:
        _regular_directory(entry)

    migrated = cast(dict[str, Any], json.loads(json.dumps(summary)))
    migrated["benchmark_schema_version"] = 2
    migrated["requalification"] = {
        "schema_version": 1,
        "source_summary_sha256": _sha256_bytes(summary_raw),
        "method": "offline-saved-content-v1",
    }

    for filename, _, old_result, stem in verified_rows:
        source_sha256 = cast(str, results[filename]["source_sha256"])
        pdf_path = pdf_root / filename
        content_path = _content_list_path(mineru_root / stem)
        _regular_file(pdf_path)
        pdf_sha256 = _sha256_file(pdf_path)
        content_sha256 = _sha256_file(content_path)
        if pdf_sha256 != source_sha256:
            raise RequalificationError("source PDF SHA does not match benchmark result")

        content_value, _ = _load_json_file(content_path, size_limit=_MAX_CONTENT_LIST_BYTES)
        if not isinstance(content_value, list) or not all(isinstance(item, dict) for item in content_value):
            raise RequalificationError("content_list must be a JSON array of objects")
        try:
            n_pages, has_text = pdf_info(pdf_path)
            if old_result.get("n_pages") != n_pages:
                raise RequalificationError("source PDF page count does not match benchmark result")
            native_text = read_pdf_text_by_page(pdf_path) if has_text else None
            drafts = content_list_to_segments(cast(list[dict[str, Any]], content_value))
            raw_quality = evaluate_parse(drafts, n_pages=n_pages, native_text_by_page=native_text)
            raw_stats = _stats(drafts)
            final_drafts = drafts
            backfilled_pages: list[int] = []
            if has_text:
                final_drafts, backfilled_pages = backfill_text_layer(
                    pdf_path,
                    drafts,
                    native_text_by_page=native_text,
                )
            final_quality = evaluate_parse(
                final_drafts,
                n_pages=n_pages,
                native_text_by_page=native_text,
            )
            final_stats = _stats(final_drafts)
        except RequalificationError:
            raise
        except Exception as error:
            raise RequalificationError(
                f"unable to rebuild benchmark result ({type(error).__name__})"
            ) from None

        if _sha256_file(pdf_path) != pdf_sha256 or _sha256_file(content_path) != content_sha256:
            raise RequalificationError("benchmark input changed during requalification")
        new_result = dict(old_result)
        new_result.update(
            {
                "status": "ok",
                "n_pages": n_pages,
                "has_text_layer": has_text,
                "quality": quality_metadata(
                    final_quality,
                    backend="mineru",
                    raw_report=raw_quality,
                    backfilled_pages=backfilled_pages,
                ),
                "raw_stats": raw_stats,
                "content_list_sha256": content_sha256,
                **final_stats,
            }
        )
        page = migrated["results"][filename]
        new_result["benchmark"] = _benchmark_proxies(page, new_result)
        page["mineru"] = new_result

    migrated["aggregates"] = _aggregates(migrated)
    _atomic_create_json(destination, migrated)
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("pdf_root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        result = requalify_summary(args.summary, args.output_root, args.pdf_root, args.destination)
    except RequalificationError as error:
        parser.exit(2, f"requalification failed: {error}\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "documents": len(result["results"]),
                "destination_sha256": _sha256_file(args.destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
