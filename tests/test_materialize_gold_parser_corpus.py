from __future__ import annotations

import asyncio
import hashlib
import json
import runpy
import stat
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "materialize_gold_parser_corpus.py"
_SCRIPT = runpy.run_path(str(_SCRIPT_PATH))
CorpusMaterializationError = _SCRIPT["CorpusMaterializationError"]
CorpusPlan = _SCRIPT["CorpusPlan"]
ControlChunkRow = _SCRIPT["ControlChunkRow"]
DocumentBinding = _SCRIPT["DocumentBinding"]
DocumentRow = _SCRIPT["DocumentRow"]
ResolvedScope = _SCRIPT["ResolvedScope"]
ScopePlan = _SCRIPT["ScopePlan"]
collect_corpus_plan = _SCRIPT["collect_corpus_plan"]
load_bound_corpus_plan = _SCRIPT["load_bound_corpus_plan"]
materialize_gold_parser_corpus = _SCRIPT["materialize_gold_parser_corpus"]
register_scope_owner = _SCRIPT["_register_scope_owner"]
single_anchor_owner = _SCRIPT["_single_anchor_owner"]

from rag_app.eval.gold_set import (  # noqa: E402
    DocumentSnapshot,
    make_document_ref,
    make_scope_id,
    parsed_chunks_sha256,
)
from rag_app.eval.private_sidecar import SidecarDocument  # noqa: E402


def _pdf(*, page_count: int = 1, label: str) -> bytes:
    output = BytesIO()
    document = canvas.Canvas(output, pageCompression=0)
    for page in range(page_count):
        document.drawString(72, 760, f"Private parser {label} page {page + 1}")
        document.showPage()
    document.save()
    return output.getvalue()


def _ooxml(*, label: str) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
            "<Override PartName='/word/document.xml' "
            "ContentType='application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document.main+xml'/></Types>",
        )
        archive.writestr("word/document.xml", f"<document><body>{label}</body></document>")
    return output.getvalue()


def _binding(
    document_id: uuid.UUID,
    payload: bytes,
    *,
    page_count: int = 1,
    parsed_content_sha256: str = "f" * 64,
) -> Any:
    source_sha256 = hashlib.sha256(payload).hexdigest()
    return DocumentBinding(
        document_id,
        DocumentSnapshot(
            document_ref=make_document_ref(source_sha256),
            source_sha256=source_sha256,
            parsed_content_sha256=parsed_content_sha256,
            page_count=page_count,
        ),
    )


def _scope(owner: str, bindings: list[Any], *, anchor_count: int) -> Any:
    return ScopePlan(
        scope_id=make_scope_id(owner),
        snapshots=tuple(item.snapshot for item in bindings),
        anchors=tuple(bindings[:anchor_count]),
    )


def _row(
    binding: Any,
    key: str,
    *,
    document_id: uuid.UUID | None = None,
    page_count: int | None = None,
    status: str = "done",
) -> Any:
    return DocumentRow(
        document_id or binding.document_id,
        key,
        binding.snapshot.page_count if page_count is None else page_count,
        status,
    )


class _FakeResolver:
    def __init__(self, resolved: list[Any], chunks: list[Any] | None = None) -> None:
        self.resolved = resolved
        self.chunks = chunks or []
        self.requested: tuple[Any, ...] | None = None
        self.requested_controls: tuple[uuid.UUID, ...] | None = None

    async def resolve(self, scopes: tuple[Any, ...]) -> list[Any]:
        self.requested = scopes
        return self.resolved

    async def resolve_control_chunks(self, document_ids: tuple[uuid.UUID, ...]) -> list[Any]:
        self.requested_controls = document_ids
        return self.chunks


class _FakeStorage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        assert bucket == "originals"
        return self.objects[key]


def _ten_document_fixture() -> tuple[list[bytes], list[Any], list[Any], Any, Any, Any]:
    page_counts = [2, 3, 3, 10, 12, 15, 16, 20, 29, 30]
    payloads = [
        *(
            _pdf(page_count=page_counts[index], label=f"document-{index}")
            for index in range(7)
        ),
        *(_ooxml(label=f"document-{index}") for index in range(7, 10)),
    ]
    bindings: list[Any] = []
    control_chunks: list[Any] = []
    for index, (payload, page_count) in enumerate(zip(payloads, page_counts, strict=True)):
        document_id = uuid.UUID(int=index + 1)
        parsed_hash = "f" * 64
        chunks: list[dict[str, Any]] = []
        if index >= 7:
            chunks = [
                {
                    "idx": 0,
                    "kind": "section",
                    "heading_path": f"Control {index}",
                    "page_start": 0,
                    "page_end": page_count - 1,
                    "text": f"Private control text {index}",
                }
            ]
            parsed_hash = parsed_chunks_sha256(chunks)
        bindings.append(
            _binding(
                document_id,
                payload,
                page_count=page_count,
                parsed_content_sha256=parsed_hash,
            )
        )
        control_chunks.extend(
            ControlChunkRow(
                document_id=document_id,
                idx=chunk["idx"],
                kind=chunk["kind"],
                heading_path=chunk["heading_path"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                text_en=chunk["text"],
            )
            for chunk in chunks
        )
    scope_a = _scope("owner-a", bindings[:3], anchor_count=2)
    scope_b = _scope("owner-b", bindings[3:], anchor_count=7)
    return payloads, bindings, control_chunks, scope_a, scope_b, CorpusPlan((scope_a, scope_b))


def _resolved_scopes(bindings: list[Any]) -> tuple[list[Any], dict[str, bytes]]:
    keys = [f"private/object-{index}.pdf" for index in range(10)]
    rows_a = tuple(_row(item, key) for item, key in zip(bindings[:3], keys[:3], strict=True))
    rows_b = tuple(_row(item, key) for item, key in zip(bindings[3:], keys[3:], strict=True))
    return [
        ResolvedScope(make_scope_id("owner-a"), rows_a),
        ResolvedScope(make_scope_id("owner-b"), rows_b),
    ], dict.fromkeys(keys, b"")


def test_materializes_two_exact_scopes_with_ten_snapshots_and_nine_anchors(tmp_path: Path) -> None:
    payloads, bindings, control_chunks, scope_a, scope_b, plan = _ten_document_fixture()
    resolved, objects = _resolved_scopes(bindings)
    rows_b = list(resolved[1].rows)
    last_control = rows_b[-1]
    rows_b[-1] = DocumentRow(
        document_id=last_control.document_id,
        s3_key_original=last_control.s3_key_original,
        page_count=None,
        status=last_control.status,
    )
    resolved[1] = ResolvedScope(resolved[1].scope_id, tuple(rows_b))
    for index, payload in enumerate(payloads):
        objects[f"private/object-{index}.pdf"] = payload
    resolver = _FakeResolver(resolved, control_chunks)
    output = tmp_path / "private-corpus"

    manifest_path = asyncio.run(
        materialize_gold_parser_corpus(
            plan,
            output,
            resolver=resolver,
            storage=_FakeStorage(objects),
            bucket_originals="originals",
            expected_documents=10,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["pages"]) == 7
    assert [page["sha256"] for page in manifest["pages"]] == [
        item.source_sha256 for item in plan.snapshots[:7]
    ]
    assert resolver.requested is not None
    assert [(item.scope_id, len(item.document_ids)) for item in resolver.requested] == sorted(
        [(scope_a.scope_id, 2), (scope_b.scope_id, 7)]
    )
    assert manifest["source"] == "private-rag-gold-release"
    controls_path = output / "controls.json"
    controls_payload = controls_path.read_bytes()
    controls = json.loads(controls_payload)
    assert manifest["controls"] == {
        "file": "controls.json",
        "sha256": hashlib.sha256(controls_payload).hexdigest(),
        "count": 3,
    }
    assert len(controls["controls"]) == 3
    assert sum(page["selection"]["page_count"] for page in manifest["pages"]) == 61
    assert sum(control["page_count"] for control in controls["controls"]) == 79
    assert stat.S_IMODE(controls_path.stat().st_mode) == 0o600
    assert resolver.requested_controls == tuple(item.document_id for item in bindings[7:])
    assert all(page["category"] == "layout" for page in manifest["pages"])
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    serialized = manifest_path.read_text(encoding="utf-8")
    assert "private/" not in serialized
    assert "owner-a" not in serialized and "owner-b" not in serialized
    assert scope_a.scope_id not in serialized and scope_b.scope_id not in serialized
    assert all(str(item.document_id) not in serialized for item in bindings)
    controls_serialized = controls_payload.decode("utf-8")
    assert "Private control text" in controls_serialized
    assert "private/" not in controls_serialized
    assert "owner-a" not in controls_serialized and "owner-b" not in controls_serialized
    assert all(str(item.document_id) not in controls_serialized for item in bindings)
    for page in manifest["pages"]:
        target = output / page["file"]
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert hashlib.sha256(target.read_bytes()).hexdigest() == page["sha256"]


def test_collect_plan_preserves_scope_unions_and_distractor() -> None:
    _, bindings, _, scope_a, scope_b, _ = _ten_document_fixture()
    records = [
        SimpleNamespace(
            case_id="case-a",
            scope_id=scope_a.scope_id,
            document_scope=tuple(item.snapshot for item in bindings[:3]),
        ),
        SimpleNamespace(
            case_id="case-b",
            scope_id=scope_b.scope_id,
            document_scope=tuple(item.snapshot for item in bindings[3:]),
        ),
    ]
    sidecars = {
        "case-a": SimpleNamespace(
            source_documents=tuple(
                SidecarDocument(
                    document_id=item.document_id,
                    document_ref=item.snapshot.document_ref,
                    source_lang="ru",
                )
                for item in bindings[:2]
            )
        ),
        "case-b": SimpleNamespace(
            source_documents=tuple(
                SidecarDocument(
                    document_id=item.document_id,
                    document_ref=item.snapshot.document_ref,
                    source_lang="ru",
                )
                for item in bindings[3:]
            )
        ),
    }

    plan = collect_corpus_plan(records, sidecars)

    assert sorted((len(scope.snapshots), len(scope.anchors)) for scope in plan.scopes) == [
        (3, 2),
        (7, 7),
    ]
    assert len(plan.snapshots) == 10


def test_owner_resolution_rejects_foreign_or_colliding_scopes() -> None:
    document_ids = (uuid.UUID(int=1), uuid.UUID(int=2))
    assert single_anchor_owner(
        document_ids, [(item, "owner-a") for item in document_ids]
    ) == "owner-a"
    with pytest.raises(CorpusMaterializationError, match="one owner"):
        single_anchor_owner(document_ids, [(document_ids[0], "owner-a"), (document_ids[1], "owner-b")])

    owners: dict[str, str] = {}
    register_scope_owner(owners, make_scope_id("owner-a"), "owner-a")
    with pytest.raises(CorpusMaterializationError, match="same owner"):
        register_scope_owner(owners, make_scope_id("owner-b"), "owner-a")
    with pytest.raises(CorpusMaterializationError, match="does not match"):
        register_scope_owner({}, make_scope_id("owner-b"), "owner-a")


@pytest.mark.parametrize("mutation", ["missing", "extra", "cross-scope-transfer"])
def test_rejects_missing_extra_or_scope_expansion(tmp_path: Path, mutation: str) -> None:
    payloads, bindings, control_chunks, scope_a, scope_b, plan = _ten_document_fixture()
    resolved, objects = _resolved_scopes(bindings)
    for index, payload in enumerate(payloads):
        objects[f"private/object-{index}.pdf"] = payload
    rows_a = list(resolved[0].rows)
    rows_b = list(resolved[1].rows)
    if mutation == "missing":
        rows_a.pop()
    elif mutation == "extra":
        extra_payload = _pdf(label="extra")
        extra = _binding(uuid.UUID(int=99), extra_payload)
        rows_b.append(_row(extra, "extra"))
        objects["extra"] = extra_payload
    else:
        rows_a.append(rows_b.pop())
    bad_resolved = [
        ResolvedScope(scope_a.scope_id, tuple(rows_a)),
        ResolvedScope(scope_b.scope_id, tuple(rows_b)),
    ]

    with pytest.raises(CorpusMaterializationError, match="exact Gold scope"):
        asyncio.run(
            materialize_gold_parser_corpus(
                plan,
                tmp_path / mutation,
                resolver=_FakeResolver(bad_resolved, control_chunks),
                storage=_FakeStorage(objects),
                bucket_originals="originals",
            )
        )


def test_rejects_cross_scope_payload_swap_even_when_counts_are_unchanged(tmp_path: Path) -> None:
    payloads, bindings, control_chunks, scope_a, scope_b, plan = _ten_document_fixture()
    resolved, objects = _resolved_scopes(bindings)
    for index, payload in enumerate(payloads):
        objects[f"private/object-{index}.pdf"] = payload
    objects["private/object-0.pdf"], objects["private/object-3.pdf"] = (
        objects["private/object-3.pdf"],
        objects["private/object-0.pdf"],
    )

    with pytest.raises(CorpusMaterializationError, match="outside its exact Gold scope"):
        asyncio.run(
            materialize_gold_parser_corpus(
                plan,
                tmp_path / "scope-swap",
                resolver=_FakeResolver(
                    [
                        ResolvedScope(scope_a.scope_id, resolved[0].rows),
                        ResolvedScope(scope_b.scope_id, resolved[1].rows),
                    ],
                    control_chunks,
                ),
                storage=_FakeStorage(objects),
                bucket_originals="originals",
            )
        )


def test_rejects_cross_scope_anchor_id_swap(tmp_path: Path) -> None:
    payloads, bindings, control_chunks, scope_a, scope_b, plan = _ten_document_fixture()
    resolved, objects = _resolved_scopes(bindings)
    for index, payload in enumerate(payloads):
        objects[f"private/object-{index}.pdf"] = payload
    rows_a = list(resolved[0].rows)
    rows_b = list(resolved[1].rows)
    rows_a[0] = _row(bindings[0], "private/object-0.pdf", document_id=bindings[3].document_id)
    rows_b[0] = _row(bindings[3], "private/object-3.pdf", document_id=bindings[0].document_id)

    with pytest.raises(CorpusMaterializationError, match="missing a sidecar anchor"):
        asyncio.run(
            materialize_gold_parser_corpus(
                plan,
                tmp_path / "anchor-swap",
                resolver=_FakeResolver(
                    [
                        ResolvedScope(scope_a.scope_id, tuple(rows_a)),
                        ResolvedScope(scope_b.scope_id, tuple(rows_b)),
                    ],
                    control_chunks,
                ),
                storage=_FakeStorage(objects),
                bucket_originals="originals",
            )
        )


def test_rejects_wrong_page_count_and_non_done_status(tmp_path: Path) -> None:
    payloads, bindings, control_chunks, scope_a, scope_b, plan = _ten_document_fixture()
    resolved, objects = _resolved_scopes(bindings)
    for index, payload in enumerate(payloads):
        objects[f"private/object-{index}.pdf"] = payload
    rows_a = list(resolved[0].rows)
    rows_a[0] = _row(bindings[0], "private/object-0.pdf", page_count=99)
    bad_pages = [
        ResolvedScope(scope_a.scope_id, tuple(rows_a)),
        ResolvedScope(scope_b.scope_id, resolved[1].rows),
    ]
    with pytest.raises(CorpusMaterializationError, match="database page count"):
        asyncio.run(
            materialize_gold_parser_corpus(
                plan,
                tmp_path / "pages",
                resolver=_FakeResolver(bad_pages, control_chunks),
                storage=_FakeStorage(objects),
                bucket_originals="originals",
            )
        )

    rows_a[0] = _row(bindings[0], "private/object-0.pdf", status="parsed")
    bad_status = [
        ResolvedScope(scope_a.scope_id, tuple(rows_a)),
        ResolvedScope(scope_b.scope_id, resolved[1].rows),
    ]
    with pytest.raises(CorpusMaterializationError, match="done status"):
        asyncio.run(
            materialize_gold_parser_corpus(
                plan,
                tmp_path / "status",
                resolver=_FakeResolver(bad_status, control_chunks),
                storage=_FakeStorage(objects),
                bucket_originals="originals",
            )
        )


def test_control_hash_mismatch_removes_pdf_and_control_outputs_atomically(tmp_path: Path) -> None:
    payloads, bindings, control_chunks, scope_a, scope_b, plan = _ten_document_fixture()
    resolved, objects = _resolved_scopes(bindings)
    for index, payload in enumerate(payloads):
        objects[f"private/object-{index}.pdf"] = payload
    first = control_chunks[0]
    control_chunks[0] = ControlChunkRow(
        document_id=first.document_id,
        idx=first.idx,
        kind=first.kind,
        heading_path=first.heading_path,
        page_start=first.page_start,
        page_end=first.page_end,
        text_en="changed private control text",
    )
    output = tmp_path / "bad-controls"

    with pytest.raises(CorpusMaterializationError, match="parsed snapshot"):
        asyncio.run(
            materialize_gold_parser_corpus(
                plan,
                output,
                resolver=_FakeResolver(
                    [
                        ResolvedScope(scope_a.scope_id, resolved[0].rows),
                        ResolvedScope(scope_b.scope_id, resolved[1].rows),
                    ],
                    control_chunks,
                ),
                storage=_FakeStorage(objects),
                bucket_originals="originals",
            )
        )
    assert not output.exists()


def test_unknown_magic_bytes_fail_and_cleanup(tmp_path: Path) -> None:
    payload = b"unsupported-private-document"
    binding = _binding(uuid.UUID(int=1), payload)
    scope = _scope("owner-a", [binding], anchor_count=1)
    output = tmp_path / "unknown"

    with pytest.raises(CorpusMaterializationError, match="neither PDF nor supported OOXML"):
        asyncio.run(
            materialize_gold_parser_corpus(
                CorpusPlan((scope,)),
                output,
                resolver=_FakeResolver(
                    [ResolvedScope(scope.scope_id, (_row(binding, "unknown"),))]
                ),
                storage=_FakeStorage({"unknown": payload}),
                bucket_originals="originals",
                expected_documents=1,
                expected_pdfs=0,
                expected_controls=1,
            )
        )
    assert not output.exists()


def test_loader_uses_release_binding_and_private_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold.jsonl"
    sidecar = tmp_path / "sidecar.jsonl"
    gold.write_text("gold\n", encoding="utf-8")
    sidecar.write_text("sidecar\n", encoding="utf-8")
    gold.chmod(0o600)
    sidecar.chmod(0o600)
    first = _binding(uuid.UUID(int=1), _pdf(label="first"))
    distractor = _binding(uuid.UUID(int=2), _pdf(label="distractor"))
    scope_id = make_scope_id("owner-a")
    record = SimpleNamespace(
        case_id="case-a",
        scope_id=scope_id,
        document_scope=(first.snapshot, distractor.snapshot),
    )
    document = SidecarDocument(
        document_id=first.document_id,
        document_ref=first.snapshot.document_ref,
        source_lang="en",
    )
    private_record = SimpleNamespace(case_id="case-a", source_documents=(document,))
    calls: list[Any] = []

    def fake_gold(path: Path, *, mode: str, repository_root: Path) -> tuple[list[Any], object]:
        calls.append(("gold", path, mode, repository_root))
        return [record], object()

    def fake_sidecar(path: Path, *, repository_root: Path) -> list[Any]:
        calls.append(("sidecar", path, repository_root))
        return [private_record]

    def fake_bind(records: list[Any], sidecars: list[Any]) -> dict[str, Any]:
        calls.append(("bind", records, sidecars))
        return {"case-a": private_record}

    monkeypatch.setitem(load_bound_corpus_plan.__globals__, "load_gold_set", fake_gold)
    monkeypatch.setitem(load_bound_corpus_plan.__globals__, "load_private_sidecar", fake_sidecar)
    monkeypatch.setitem(load_bound_corpus_plan.__globals__, "bind_gold_sidecar", fake_bind)

    plan = load_bound_corpus_plan(gold, sidecar, repository_root=tmp_path)
    assert len(plan.scopes) == 1
    assert len(plan.scopes[0].snapshots) == 2
    assert plan.scopes[0].anchors == (first,)
    assert calls[0][2] == "release"

    sidecar.chmod(0o644)
    with pytest.raises(CorpusMaterializationError, match="0600"):
        load_bound_corpus_plan(gold, sidecar, repository_root=tmp_path)
