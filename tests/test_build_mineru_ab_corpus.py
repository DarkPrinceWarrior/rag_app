from __future__ import annotations

import hashlib
import io
import json
import runpy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

_SCRIPT = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "build_mineru_ab_corpus.py"))
build_corpus = _SCRIPT["build_corpus"]
render_single_page_pdf = _SCRIPT["_render_single_page_pdf"]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _png(colour: tuple[int, int, int], *, size: tuple[int, int] = (32, 24)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, colour).save(output, format="PNG")
    return output.getvalue()


def _verified_manifest(inputs: list[tuple[str, str, bytes]]) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "verification_state": "bytes_verified",
        "source_manifest_sha256": "a" * 64,
        "datasets": {
            "varex": {
                "source": "ibm-research/VAREX",
                "source_url": "https://huggingface.co/datasets/ibm-research/VAREX",
                "revision": "2dfc3386a4567c7d56bf1abf4d12ff42afed27b6",
                "license": "CDLA-Permissive-2.0",
                "quotas": {"Table": 1},
            }
        },
        "selected": [
            {
                "dataset": "varex",
                "stratum": "Table",
                "canonical_id": "document-1",
                "group_id": "document-1",
                "selection_hash": "b" * 64,
                "inputs": [
                    {
                        "role": role,
                        "materialized_path": relative,
                        "size_bytes": len(payload),
                        "sha256": _sha(payload),
                    }
                    for role, relative, payload in inputs
                ],
                "metadata": {},
            }
        ],
    }


def _write_verified_root(tmp_path: Path) -> tuple[Path, Path, list[bytes]]:
    root = tmp_path / "materialized"
    objects = root / "objects"
    objects.mkdir(parents=True)
    payloads = [_png((255, 0, 0)), _png((0, 255, 0), size=(18, 30))]
    inputs = []
    for index, payload in enumerate(payloads):
        relative = f"objects/image-{index}.png"
        (root / relative).write_bytes(payload)
        inputs.append((f"image_{index}", relative, payload))
    manifest = root / "manifest.verified.json"
    manifest.write_text(json.dumps(_verified_manifest(inputs)), encoding="utf-8")
    return root, manifest, payloads


def _write_parsebench(tmp_path: Path) -> tuple[Path, bytes, dict[str, Any]]:
    root = tmp_path / "parsebench"
    root.mkdir(parents=True)
    payload = render_single_page_pdf(_png((0, 0, 255)), context="test")
    (root / "hard-chart.pdf").write_bytes(payload)
    selection = {"tags": ["hard", "3d_chart"], "rules": 17}
    manifest = {
        "source": "llamaindex/ParseBench",
        "source_revision": "main",
        "source_license": "Apache-2.0",
        "selection_policy": "hard pages",
        "pages": [
            {
                "file": "hard-chart.pdf",
                "bytes": len(payload),
                "sha256": _sha(payload),
                "category": "chart",
                "selection": selection,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, payload, selection


def test_builds_reproducible_content_addressed_corpus_and_appends_parsebench(
    tmp_path: Path,
) -> None:
    root, source_manifest, source_images = _write_verified_root(tmp_path)
    parsebench, parsebench_pdf, parsebench_selection = _write_parsebench(tmp_path)

    first_path = build_corpus(
        source_manifest,
        root,
        tmp_path / "corpus-a",
        parsebench_corpus=parsebench,
    )
    second_path = build_corpus(
        source_manifest,
        root,
        tmp_path / "corpus-b",
        parsebench_corpus=parsebench,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    first = json.loads(first_path.read_text(encoding="utf-8"))
    assert first["manifest_version"] == 1
    assert first["provenance"]["verified_manifest"]["verification_state"] == "bytes_verified"
    assert first["provenance"]["sources"][0]["revision"].startswith("2dfc")
    assert first["provenance"]["sources"][0]["license"] == "CDLA-Permissive-2.0"
    assert first["provenance"]["parsebench"]["manifest_sha256"]
    assert len(first["pages"]) == 3

    for page in first["pages"]:
        assert page["file"] == f"{page['sha256']}.pdf"
        first_pdf = first_path.parent / page["file"]
        second_pdf = second_path.parent / page["file"]
        assert first_pdf.read_bytes() == second_pdf.read_bytes()
        assert _sha(first_pdf.read_bytes()) == page["sha256"]
        assert first_pdf.read_bytes().startswith(b"%PDF-")
        assert b"/Count 1" in first_pdf.read_bytes()

    licensed = first["pages"][:2]
    assert [page["category"] for page in licensed] == ["table", "table"]
    assert [page["source_image_sha256"] for page in licensed] == [_sha(payload) for payload in source_images]
    assert licensed[0]["selection"]["dataset"] == "varex"
    assert licensed[0]["selection"]["stratum"] == "Table"

    appended = first["pages"][2]
    assert appended["selection"] == parsebench_selection
    assert appended["sha256"] == _sha(parsebench_pdf)
    assert appended["source_revision"] == "main"
    assert appended["source_license"] == "Apache-2.0"


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_rejects_materialized_byte_mismatch_and_rolls_back(tmp_path: Path, field: str) -> None:
    root, source_manifest, _ = _write_verified_root(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    item = manifest["selected"][0]["inputs"][0]
    item[field] = item[field] + 1 if field == "size_bytes" else "0" * 64
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "corpus"

    with pytest.raises(ValueError, match="mismatch"):
        build_corpus(source_manifest, root, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".corpus.tmp-*"))


def test_rejects_traversal_and_symlinks(tmp_path: Path) -> None:
    root, source_manifest, _ = _write_verified_root(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["selected"][0]["inputs"][0]["materialized_path"] = "../outside.png"
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="traversal"):
        build_corpus(source_manifest, root, tmp_path / "traversal")

    root, source_manifest, _ = _write_verified_root(tmp_path / "symlink-case")
    target = root / "objects" / "image-0.png"
    link = root / "objects" / "link.png"
    link.symlink_to(target)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["selected"][0]["inputs"][0]["materialized_path"] = "objects/link.png"
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="without following links"):
        build_corpus(source_manifest, root, tmp_path / "symlink-output")


def test_rejects_manifest_outside_declared_root(tmp_path: Path) -> None:
    root, source_manifest, _ = _write_verified_root(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(source_manifest.read_bytes())

    with pytest.raises(ValueError, match="stay under"):
        build_corpus(outside, root, tmp_path / "corpus")


def test_rejects_unknown_mapping_and_invalid_image_atomically(tmp_path: Path) -> None:
    root, source_manifest, _ = _write_verified_root(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["selected"][0]["stratum"] = "Unknown"
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no benchmark category"):
        build_corpus(source_manifest, root, tmp_path / "unknown")

    root, source_manifest, _ = _write_verified_root(tmp_path / "bad-image-case")
    bad_path = root / "objects" / "image-0.png"
    bad_path.write_bytes(b"not-an-image")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    item = manifest["selected"][0]["inputs"][0]
    item["size_bytes"] = len(b"not-an-image")
    item["sha256"] = _sha(b"not-an-image")
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "bad-image"
    with pytest.raises(ValueError, match="could not decode"):
        build_corpus(source_manifest, root, output)
    assert not output.exists()


def test_parsebench_sha_and_symlink_are_revalidated(tmp_path: Path) -> None:
    root, source_manifest, _ = _write_verified_root(tmp_path)
    parsebench, _, _ = _write_parsebench(tmp_path)
    parsebench_manifest = json.loads((parsebench / "manifest.json").read_text())
    parsebench_manifest["pages"][0]["sha256"] = "0" * 64
    (parsebench / "manifest.json").write_text(json.dumps(parsebench_manifest))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_corpus(
            source_manifest,
            root,
            tmp_path / "bad-parsebench-sha",
            parsebench_corpus=parsebench,
        )

    parsebench, _, _ = _write_parsebench(tmp_path / "linked")
    original = parsebench / "hard-chart.pdf"
    linked = parsebench / "linked.pdf"
    linked.symlink_to(original)
    manifest_path = parsebench / "manifest.json"
    parsebench_manifest = json.loads(manifest_path.read_text())
    parsebench_manifest["pages"][0]["file"] = linked.name
    manifest_path.write_text(json.dumps(parsebench_manifest))
    with pytest.raises(ValueError, match="without following links"):
        build_corpus(
            source_manifest,
            root,
            tmp_path / "bad-parsebench-link",
            parsebench_corpus=parsebench,
        )
