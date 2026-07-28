from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from rag_app.workers import tasks


class _ScalarResult:
    def __init__(self, value: int | None = None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> int | None:
        return self.value


class _Session:
    def __init__(self, revision: int) -> None:
        self.revision = revision
        self.statements: list[Any] = []
        self.added: list[Any] = []
        self.committed = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def execute(self, statement: Any) -> _ScalarResult:
        self.statements.append(statement)
        if len(self.statements) == 1:
            return _ScalarResult(self.revision)
        return _ScalarResult()

    def add_all(self, values: Any) -> None:
        self.added.extend(values)

    async def commit(self) -> None:
        self.committed = True


class _Storage:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def download_to(self, _bucket: str, _key: str, destination: Path) -> None:
        if self.fail:
            raise RuntimeError("storage unavailable")
        destination.write_bytes(b"pdf")


class _Visual:
    async def embed_page(self, _image: bytes) -> list[float]:
        return [0.25, 0.75]


def _doc(revision: int) -> SimpleNamespace:
    return SimpleNamespace(
        parse_revision=revision,
        kind="pdf_scan",
        s3_key_original="original.pdf",
    )


def _ctx(session: _Session, *, storage: _Storage | None = None) -> dict[str, Any]:
    return {
        "sessionmaker": lambda: session,
        "storage": storage or _Storage(),
        "visual": _Visual(),
    }


def test_visual_index_legacy_job_without_revision_is_skipped() -> None:
    result = asyncio.run(tasks.index_pages_visual({}, str(uuid4())))

    assert result == "skipped visual index: missing parse_revision"


def test_visual_index_stale_revision_is_skipped_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_doc(_ctx: dict[str, Any], _doc_id: object) -> SimpleNamespace:
        return _doc(8)

    monkeypatch.setattr(tasks.settings, "visual_enabled", True)
    monkeypatch.setattr(tasks, "_get_doc", get_doc)

    result = asyncio.run(tasks.index_pages_visual({}, str(uuid4()), 7))

    assert result == "skipped visual revision=7: current=8"


def test_visual_index_stale_revision_cannot_mutate_page_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(revision=8)

    async def get_doc(_ctx: dict[str, Any], _doc_id: object) -> SimpleNamespace:
        return _doc(7)

    async def render_in_thread(_func: object, *_args: object, **_kwargs: object) -> list[tuple[int, bytes]]:
        return [(0, b"jpeg")]

    monkeypatch.setattr(tasks.settings, "visual_enabled", True)
    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks.asyncio, "to_thread", render_in_thread)

    result = asyncio.run(tasks.index_pages_visual(_ctx(session), str(uuid4()), 7))

    assert result == "skipped visual revision=7: current=8"
    assert len(session.statements) == 1  # только SELECT ... FOR UPDATE
    assert session.added == []
    assert session.committed is False


def test_visual_index_stale_failure_cannot_write_index_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(revision=8)

    async def get_doc(_ctx: dict[str, Any], _doc_id: object) -> SimpleNamespace:
        return _doc(7)

    monkeypatch.setattr(tasks.settings, "visual_enabled", True)
    monkeypatch.setattr(tasks, "_get_doc", get_doc)

    result = asyncio.run(
        tasks.index_pages_visual(_ctx(session, storage=_Storage(fail=True)), str(uuid4()), 7)
    )

    assert result == "skipped failed visual revision=7: current=8"
    assert len(session.statements) == 1  # только SELECT ... FOR UPDATE
    assert session.added == []
    assert session.committed is False


def test_visual_index_current_revision_replaces_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(revision=7)

    async def get_doc(_ctx: dict[str, Any], _doc_id: object) -> SimpleNamespace:
        return _doc(7)

    async def render_in_thread(_func: object, *_args: object, **_kwargs: object) -> list[tuple[int, bytes]]:
        return [(3, b"jpeg")]

    monkeypatch.setattr(tasks.settings, "visual_enabled", True)
    monkeypatch.setattr(tasks, "_get_doc", get_doc)
    monkeypatch.setattr(tasks.asyncio, "to_thread", render_in_thread)

    result = asyncio.run(tasks.index_pages_visual(_ctx(session), str(uuid4()), 7))

    assert result == "visual indexed: 1 pages"
    assert len(session.statements) == 3  # revision lock, DELETE embeddings, UPDATE error
    assert len(session.added) == 1
    assert session.added[0].page_idx == 3
    assert session.committed is True


def test_all_visual_index_enqueues_pass_parse_revision() -> None:
    sources = [
        Path("src/rag_app/workers/tasks.py").read_text(encoding="utf-8"),
        Path("scripts/reindex_visual.py").read_text(encoding="utf-8"),
    ]
    calls: list[ast.Call] = []
    for source in sources:
        tree = ast.parse(source)
        calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "enqueue_job"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "index_pages_visual"
        )

    assert len(calls) == 3
    revisions = [ast.unparse(call.args[2]) for call in calls if len(call.args) >= 3]
    assert revisions == ["parse_revision", "parse_revision", "doc.parse_revision"]
