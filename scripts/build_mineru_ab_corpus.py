"""Build a byte-pinned, single-page PDF corpus for paired MinerU A/B runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import PIL
from PIL import Image

_CONTRACT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_INPUT_BYTES = 512 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PDF_TIME = time.gmtime(0)
_KNOWN_CATEGORIES = {"table", "chart", "layout", "text_formatting"}

_CATEGORY_BY_DATASET_STRATUM: dict[tuple[str, str], str] = {
    ("pubtables_v2", "pages_2"): "table",
    ("pubtables_v2", "pages_3"): "table",
    ("pubtables_v2", "pages_4"): "table",
    ("varex", "Flat"): "layout",
    ("varex", "Nested"): "layout",
    ("varex", "Table"): "table",
    ("ai2d_rst", "diagram"): "chart",
    ("mws_vision_bench", "document_parsing_ru"): "layout",
    ("mws_vision_bench", "full_page_ocr_ru"): "text_formatting",
    ("mws_vision_bench", "key_information_extraction_ru"): "layout",
    ("mws_vision_bench", "reasoning_vqa_ru"): "layout",
    ("mws_vision_bench", "text_grounding_ru"): "layout",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _strict_json_loads(payload: bytes, *, context: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{context}: non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: invalid JSON") from exc


def _safe_relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError(f"{context}: expected a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or path.as_posix() != value:
        raise ValueError(f"{context}: path traversal or non-canonical path is forbidden")
    return value


def _absolute_root(path: Path, *, context: str) -> Path:
    root = Path(os.path.abspath(path))
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{context}: root does not exist") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"{context}: root must be a real directory, not a symlink")
    return root


def _path_beneath(root: Path, path: Path, *, context: str) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context}: path must stay under its declared root") from exc
    return _safe_relative(relative.as_posix(), context=context)


@contextmanager
def _open_beneath(root: Path, relative: str, *, context: str) -> Iterator[BinaryIO]:
    """Open one regular file using no-follow directory traversal from ``root``."""

    parts = PurePosixPath(_safe_relative(relative, context=context)).parts
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = os.open(root, flags | os.O_DIRECTORY)
    directory_fd = root_fd
    file_fd: int | None = None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory_fd)
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError(f"{context}: file cannot be opened without following links") from exc
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context}: expected a regular file")
        with os.fdopen(file_fd, "rb") as stream:
            file_fd = None
            yield stream
            after = os.fstat(stream.fileno())
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise ValueError(f"{context}: file changed while it was being read")
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd != root_fd:
            os.close(directory_fd)
        os.close(root_fd)


def _read_beneath(
    root: Path,
    relative: str,
    *,
    context: str,
    max_bytes: int,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    with _open_beneath(root, relative, context=context) as stream:
        size = os.fstat(stream.fileno()).st_size
        if expected_size is not None and size != expected_size:
            raise ValueError(f"{context}: size_bytes mismatch")
        if size > max_bytes:
            raise ValueError(f"{context}: file exceeds {max_bytes} bytes")
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"{context}: file exceeds {max_bytes} bytes")
    if expected_size is not None and len(payload) != expected_size:
        raise ValueError(f"{context}: size_bytes mismatch")
    if expected_sha256 is not None and _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{context}: SHA-256 mismatch")
    return payload


def _required_string(value: Mapping[str, Any], key: str, *, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{key}: expected a non-empty string")
    return result


def _required_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{context}: expected lowercase hexadecimal SHA-256")
    return value


def _required_size(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context}: expected a positive integer")
    return value


def _render_single_page_pdf(image_payload: bytes, *, context: str) -> bytes:
    try:
        with Image.open(io.BytesIO(image_payload)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"{context}: multi-frame images are unsupported")
            image.load()
            rgb = image.convert("RGB")
        output = io.BytesIO()
        rgb.save(
            output,
            format="PDF",
            resolution=200.0,
            creationDate=_PDF_TIME,
            modDate=_PDF_TIME,
            producer="DocRAGenslate MinerU A/B corpus builder",
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{context}: Pillow could not decode the image") from exc
    payload = output.getvalue()
    if not payload.startswith(b"%PDF-"):
        raise RuntimeError(f"{context}: Pillow did not produce a PDF")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _store_pdf(stage_dir: Path, payload: bytes, seen: set[str], *, context: str) -> tuple[str, str]:
    sha256 = _sha256_bytes(payload)
    if sha256 in seen:
        raise ValueError(f"{context}: duplicate rendered PDF SHA-256 would collapse benchmark pages")
    seen.add(sha256)
    filename = f"{sha256}.pdf"
    _atomic_write(stage_dir / filename, payload)
    return filename, sha256


def _dataset_sources(manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw = manifest.get("datasets")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("verified manifest datasets must be a non-empty object")
    indexed: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for dataset in sorted(raw):
        metadata = raw[dataset]
        context = f"datasets.{dataset}"
        if not isinstance(dataset, str) or not isinstance(metadata, dict):
            raise ValueError("verified manifest dataset metadata is invalid")
        normalized = {
            "dataset": dataset,
            "source": _required_string(metadata, "source", context=context),
            "source_url": _required_string(metadata, "source_url", context=context),
            "revision": _required_string(metadata, "revision", context=context),
            "license": _required_string(metadata, "license", context=context),
        }
        for optional in ("license_components", "source_components"):
            if optional in metadata:
                normalized[optional] = copy.deepcopy(metadata[optional])
        indexed[dataset] = normalized
        sources.append(copy.deepcopy(normalized))
    return indexed, sources


def _licensed_pages(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    stage_dir: Path,
    sources: Mapping[str, Mapping[str, Any]],
    seen_pdf_sha256: set[str],
) -> list[dict[str, Any]]:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("verified manifest selected must be a non-empty array")
    pages: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(selected):
        context = f"selected[{candidate_index}]"
        if not isinstance(candidate, dict):
            raise ValueError(f"{context}: expected an object")
        dataset = _required_string(candidate, "dataset", context=context)
        stratum = _required_string(candidate, "stratum", context=context)
        canonical_id = _required_string(candidate, "canonical_id", context=context)
        group_id = _required_string(candidate, "group_id", context=context)
        selection_hash = _required_sha256(
            candidate.get("selection_hash"), context=f"{context}.selection_hash"
        )
        category = _CATEGORY_BY_DATASET_STRATUM.get((dataset, stratum))
        if category is None:
            raise ValueError(f"{context}: no benchmark category for {dataset}/{stratum}")
        source = sources.get(dataset)
        if source is None:
            raise ValueError(f"{context}: dataset metadata is missing for {dataset}")
        inputs = candidate.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{context}.inputs: expected a non-empty array")
        for input_index, item in enumerate(inputs):
            input_context = f"{context}.inputs[{input_index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{input_context}: expected an object")
            role = _required_string(item, "role", context=input_context)
            relative = _safe_relative(item.get("materialized_path"), context=input_context)
            source_sha256 = _required_sha256(item.get("sha256"), context=f"{input_context}.sha256")
            source_size = _required_size(item.get("size_bytes"), context=f"{input_context}.size_bytes")
            image_payload = _read_beneath(
                root,
                relative,
                context=input_context,
                max_bytes=_MAX_INPUT_BYTES,
                expected_size=source_size,
                expected_sha256=source_sha256,
            )
            pdf_payload = _render_single_page_pdf(image_payload, context=input_context)
            filename, pdf_sha256 = _store_pdf(stage_dir, pdf_payload, seen_pdf_sha256, context=input_context)
            pages.append(
                {
                    "file": filename,
                    "bytes": len(pdf_payload),
                    "sha256": pdf_sha256,
                    "category": category,
                    "selection": {
                        "origin": "licensed_ocr_manifest",
                        "dataset": dataset,
                        "stratum": stratum,
                        "canonical_id": canonical_id,
                        "group_id": group_id,
                        "role": role,
                        "selection_hash": selection_hash,
                    },
                    "source": source["source"],
                    "source_revision": source["revision"],
                    "source_license": source["license"],
                    "source_image_sha256": source_sha256,
                    "source_image_size_bytes": source_size,
                    "source_materialized_path": relative,
                }
            )
    return pages


def _append_parsebench_pages(
    pages: list[dict[str, Any]],
    *,
    corpus_dir: Path,
    stage_dir: Path,
    seen_pdf_sha256: set[str],
) -> dict[str, Any]:
    root = _absolute_root(corpus_dir, context="ParseBench")
    manifest_payload = _read_beneath(
        root,
        _MANIFEST_NAME,
        context="ParseBench manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = _strict_json_loads(manifest_payload, context="ParseBench manifest")
    if not isinstance(manifest, dict):
        raise ValueError("ParseBench manifest must be an object")
    source = _required_string(manifest, "source", context="ParseBench manifest")
    revision = _required_string(manifest, "source_revision", context="ParseBench manifest")
    license_ = _required_string(manifest, "source_license", context="ParseBench manifest")
    raw_pages = manifest.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("ParseBench manifest pages must be a non-empty array")
    for index, source_page in enumerate(raw_pages):
        context = f"ParseBench pages[{index}]"
        if not isinstance(source_page, dict):
            raise ValueError(f"{context}: expected an object")
        relative = _safe_relative(source_page.get("file"), context=f"{context}.file")
        if PurePosixPath(relative).name != relative:
            raise ValueError(f"{context}.file: nested paths are unsupported")
        sha256 = _required_sha256(source_page.get("sha256"), context=f"{context}.sha256")
        size_value = source_page.get("bytes")
        size = _required_size(size_value, context=f"{context}.bytes") if size_value is not None else None
        payload = _read_beneath(
            root,
            relative,
            context=context,
            max_bytes=_MAX_INPUT_BYTES,
            expected_size=size,
            expected_sha256=sha256,
        )
        if not payload.startswith(b"%PDF-"):
            raise ValueError(f"{context}: source is not a PDF")
        category = source_page.get("category")
        if category not in _KNOWN_CATEGORIES:
            raise ValueError(f"{context}.category: unsupported benchmark category")
        selection = source_page.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"{context}.selection: expected an object")
        filename, stored_sha256 = _store_pdf(stage_dir, payload, seen_pdf_sha256, context=context)
        if stored_sha256 != sha256:
            raise RuntimeError(f"{context}: copied PDF digest changed")
        pages.append(
            {
                "file": filename,
                "bytes": len(payload),
                "sha256": sha256,
                "category": category,
                "selection": copy.deepcopy(selection),
                "source": source,
                "source_revision": revision,
                "source_license": license_,
                "source_file": relative,
                "origin": "parsebench",
            }
        )
    return {
        "source": source,
        "revision": revision,
        "license": license_,
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "page_count": len(raw_pages),
    }


def build_corpus(
    verified_manifest: Path,
    materialized_root: Path,
    output_dir: Path,
    *,
    parsebench_corpus: Path | None = None,
) -> Path:
    """Build the corpus atomically and return its final manifest path."""

    root = _absolute_root(materialized_root, context="materialized corpus")
    manifest_relative = _path_beneath(root, verified_manifest, context="verified manifest")
    manifest_payload = _read_beneath(
        root,
        manifest_relative,
        context="verified manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = _strict_json_loads(manifest_payload, context="verified manifest")
    if not isinstance(manifest, dict):
        raise ValueError("verified manifest must be an object")
    if manifest.get("manifest_version") != 1:
        raise ValueError("verified manifest_version must be 1")
    if manifest.get("verification_state") != "bytes_verified":
        raise ValueError("verified manifest must have verification_state=bytes_verified")
    source_manifest_sha256 = _required_sha256(
        manifest.get("source_manifest_sha256"),
        context="verified manifest source_manifest_sha256",
    )
    source_index, sources = _dataset_sources(manifest)

    output = Path(os.path.abspath(output_dir))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    seen_pdf_sha256: set[str] = set()
    try:
        pages = _licensed_pages(
            manifest,
            root=root,
            stage_dir=stage_dir,
            sources=source_index,
            seen_pdf_sha256=seen_pdf_sha256,
        )
        parsebench_provenance = None
        if parsebench_corpus is not None:
            parsebench_provenance = _append_parsebench_pages(
                pages,
                corpus_dir=parsebench_corpus,
                stage_dir=stage_dir,
                seen_pdf_sha256=seen_pdf_sha256,
            )

        verified_manifest_sha256 = _sha256_bytes(manifest_payload)
        revision_inputs: dict[str, Any] = {
            "verified_manifest_sha256": verified_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
        }
        if parsebench_provenance is not None:
            revision_inputs["parsebench_manifest_sha256"] = parsebench_provenance["manifest_sha256"]
        source_revision = _sha256_bytes(_canonical_json_bytes(revision_inputs))
        output_manifest: dict[str, Any] = {
            "manifest_version": _CONTRACT_VERSION,
            "source": (
                "DocRAGenslate licensed OCR corpus + llamaindex/ParseBench"
                if parsebench_provenance is not None
                else "DocRAGenslate licensed OCR corpus"
            ),
            "source_revision": source_revision,
            "source_license": "MIXED; see provenance.sources and page-level licenses",
            "selection_policy": (
                "all byte-verified selected image inputs in source order; optional ParseBench pages appended"
            ),
            "provenance": {
                "builder": "scripts/build_mineru_ab_corpus.py",
                "contract_version": _CONTRACT_VERSION,
                "verified_manifest": {
                    "file": manifest_relative,
                    "sha256": verified_manifest_sha256,
                    "source_manifest_sha256": source_manifest_sha256,
                    "verification_state": "bytes_verified",
                },
                "sources": sources,
                "pdf_conversion": {
                    "library": "Pillow",
                    "version": PIL.__version__,
                    "mode": "RGB",
                    "resolution_dpi": 200,
                    "metadata_time_unix": 0,
                    "single_page": True,
                },
                "parsebench": parsebench_provenance,
            },
            "pages": pages,
        }
        _atomic_write(
            stage_dir / _MANIFEST_NAME,
            json.dumps(
                output_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode()
            + b"\n",
        )
        stage_fd = os.open(stage_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        os.replace(stage_dir, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return output / _MANIFEST_NAME


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verified_manifest", type=Path)
    parser.add_argument("materialized_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--parsebench-corpus", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_corpus(
        args.verified_manifest,
        args.materialized_root,
        args.output_dir,
        parsebench_corpus=args.parsebench_corpus,
    )
    print(manifest)


if __name__ == "__main__":
    main()
