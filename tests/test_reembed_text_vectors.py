from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag_app.llm import embeddings as app_embeddings

_SCRIPT = Path(__file__).parents[1] / "scripts" / "reembed_text_vectors.py"
_SPEC = importlib.util.spec_from_file_location("reembed_text_vectors_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
target = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = target
_SPEC.loader.exec_module(target)


def test_nullable_uuid_bind_parameters_are_typed_before_null_checks() -> None:
    """asyncpg cannot infer ``$1`` from ``$1 IS NULL`` even if it is cast later."""

    for item in target.TARGETS:
        assert ":after_id IS NULL" not in item.fetch_sql
        assert ":last_id IS NOT NULL" not in item.resume_sql
        assert "CAST(:after_id AS uuid) IS NULL" in item.fetch_sql
        assert "CAST(:last_id AS uuid) IS NOT NULL" in item.resume_sql


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:8002/v1/", "http://127.0.0.1:8002/v1"),
        ("http://localhost:18002/v1", "http://localhost:18002/v1"),
        ("https://[::1]:18002/v1", "https://[::1]:18002/v1"),
    ],
)
def test_normalize_embedding_base_url_accepts_loopback_v1(
    value: str, expected: str
) -> None:
    assert target.normalize_embedding_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://models.example/v1",
        "http://token@127.0.0.1:8002/v1",
        "http://127.0.0.1:8002/v1?token=secret",
        "http://127.0.0.1:8002/v1/models",
        "http://127.0.0.1/v1",
        "file:///tmp/model",
    ],
)
def test_normalize_embedding_base_url_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(target.ReembeddingError):
        target.normalize_embedding_base_url(value)


def test_env_file_is_plain_data_and_does_not_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / "runtime.env"
    env_file.write_text(
        "# comment\nRAG_EMBED_MODEL='candidate'\n"
        'RAG_EMBED_INPUT_PROFILE="nemotron3"\n'
        "LITERAL=$(touch /tmp/must-not-run)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_EMBED_MODEL", "already-set")
    monkeypatch.delenv("RAG_EMBED_INPUT_PROFILE", raising=False)
    monkeypatch.delenv("LITERAL", raising=False)

    loaded = target.load_env_file(env_file)

    assert "RAG_EMBED_MODEL" not in loaded
    assert os.environ["RAG_EMBED_MODEL"] == "already-set"
    assert os.environ["RAG_EMBED_INPUT_PROFILE"] == "nemotron3"
    assert os.environ["LITERAL"] == "$(touch /tmp/must-not-run)"


@pytest.mark.parametrize(
    "content",
    [
        "export RAG_EMBED_MODEL=x\n",
        "NOT-A-KEY=x\n",
        'RAG_EMBED_MODEL="unterminated\n',
    ],
)
def test_env_file_rejects_shell_or_malformed_lines(
    content: str, tmp_path: Path
) -> None:
    env_file = tmp_path / "bad.env"
    env_file.write_text(content, encoding="utf-8")
    with pytest.raises(target.ReembeddingError):
        target.load_env_file(env_file)


def test_env_file_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.env"
    real.write_text("A=B\n", encoding="utf-8")
    link = tmp_path / "link.env"
    link.symlink_to(real)
    with pytest.raises(target.ReembeddingError, match="non-symlink"):
        target.load_env_file(link)


def test_document_input_profiles_are_exact_and_do_not_double_prefix() -> None:
    assert target.format_document_input("  text  ", "qwen3") == "text"
    assert target.format_document_input("", "qwen3") == "."
    assert target.format_document_input("text", "nemotron3") == "passage: text"
    assert (
        target.format_document_input("PASSAGE: already", "nemotron3")
        == "passage: already"
    )


def test_nemotron_long_prefixed_input_has_exact_application_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_embeddings.settings, "embed_input_profile", "nemotron3")
    value = "  PASSAGE:   " + ("x" * 8_050) + "  "
    assert target.format_document_input(
        value, "nemotron3"
    ) == app_embeddings._document_input(value)
    assert len(target.format_document_input(value, "nemotron3")) == len("passage: ") + 8_000


def test_database_identity_excludes_credentials() -> None:
    first = target.database_identity_sha256(
        "postgresql+asyncpg://worker:first@127.0.0.1:5433/rag_app"
    )
    second = target.database_identity_sha256(
        "postgresql+asyncpg://other:second@127.0.0.1:5433/rag_app"
    )
    different_database = target.database_identity_sha256(
        "postgresql+asyncpg://worker:first@127.0.0.1:5433/other"
    )
    assert first == second
    assert first != different_database


def test_target_filters_match_production_eligibility() -> None:
    by_key = {item.key: item for item in target.TARGETS}
    assert "d.status = 'done'" in by_key["chunks.emb_en"].fetch_sql
    assert "d.status = 'done'" in by_key["chunks.emb_ru"].fetch_sql
    assert (
        "status = 'active' AND deleted_at IS NULL"
        in by_key["memory_items.embedding"].fetch_sql
    )
    assert (
        "status = 'approved' AND revoked_at IS NULL"
        in by_key["translation_memory.source_embedding"].fetch_sql
    )
    for item in target.TARGETS:
        assert "SET id" not in item.update_sql


def test_normalize_vector_truncates_and_normalizes() -> None:
    vector = target.normalize_vector([3.0, 4.0, 99.0], 2)
    assert vector == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    ("vector", "dimension"),
    [
        ([1.0], 2),
        ([0.0, 0.0], 2),
        ([float("nan"), 1.0], 2),
        ([True, 1.0], 2),
    ],
)
def test_normalize_vector_rejects_invalid_vectors(
    vector: list[object], dimension: int
) -> None:
    with pytest.raises(target.ReembeddingError):
        target.normalize_vector(vector, dimension)


def test_embedding_client_preflight_and_batch_validate_model_and_dimension() -> None:
    async def exercise() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200, json={"data": [{"id": "candidate"}]}, request=request
                )
            payload = json.loads(request.content)
            assert payload["model"] == "candidate"
            return httpx.Response(
                200,
                json={
                    "model": "candidate",
                    "data": [
                        {"index": index, "embedding": [3.0, 4.0, 99.0]}
                        for index, _ in enumerate(payload["input"])
                    ],
                },
                request=request,
            )

        client = target.SafeEmbeddingClient(
            base_url="http://127.0.0.1:8002/v1",
            model="candidate",
            profile="nemotron3",
            dimension=2,
            timeout_s=1.0,
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        )
        try:
            await client.preflight()
            vectors = await client.embed(["one", "two"])
        finally:
            await client.close()
        assert len(vectors) == 2
        assert all(vector == pytest.approx([0.6, 0.8]) for vector in vectors)
        assert client.native_dimension == 3

    asyncio.run(exercise())


def test_embedding_client_rejects_response_model_drift() -> None:
    async def exercise() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "other",
                    "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                },
                request=request,
            )

        client = target.SafeEmbeddingClient(
            base_url="http://127.0.0.1:8002/v1",
            model="candidate",
            profile="qwen3",
            dimension=2,
            timeout_s=1.0,
        )
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        )
        try:
            with pytest.raises(target.ReembeddingError, match="different model"):
                await client.embed(["one"])
        finally:
            await client.close()

    asyncio.run(exercise())


def test_checkpoint_is_private_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    identity = _identity()
    checkpoint = target._new_checkpoint(
        identity, {"one": target.SourceManifest(2, "b" * 32)}
    )

    target.save_checkpoint(path, checkpoint)
    loaded = target.load_checkpoint(path)

    assert loaded["payload_sha256"] == target.checkpoint_payload_sha256(loaded)
    unsigned = dict(loaded)
    unsigned.pop("payload_sha256")
    assert unsigned == checkpoint
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_rejects_public_permissions(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "schema": target._CHECKPOINT_SCHEMA,
                "identity": {
                    "model": "m",
                    "model_revision": "1" * 40,
                    "profile": "qwen3",
                    "dimension": 2,
                    "native_dimension": 3,
                    "endpoint_sha256": "a" * 64,
                    "database_sha256": "b" * 64,
                },
                "targets": {},
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o644)
    with pytest.raises(target.ReembeddingError, match="permissions"):
        target.load_checkpoint(path)


@pytest.mark.parametrize("field", ["identity", "progress"])
def test_checkpoint_payload_hash_rejects_tampering(
    field: str, tmp_path: Path
) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = target._new_checkpoint(
        _identity(), {"one": target.SourceManifest(1, "b" * 32)}
    )
    target.save_checkpoint(path, checkpoint)
    tampered = json.loads(path.read_bytes())
    if field == "identity":
        tampered["identity"]["model"] = "replaced"
    else:
        tampered["targets"]["one"]["processed"] = 1
    path.write_text(
        json.dumps(tampered, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    path.chmod(0o600)

    with pytest.raises(target.ReembeddingError, match="payload hash mismatch"):
        target.load_checkpoint(path)


@pytest.mark.parametrize("revision", ["main", "latest", "abc123", "1" * 39, "g" * 40])
def test_model_revision_rejects_mutable_or_non_sha_values(revision: str) -> None:
    with pytest.raises(target.ReembeddingError, match="40 hexadecimal"):
        target.validate_model_revision(revision, required=True)


def test_model_revision_normalizes_uppercase_commit_sha() -> None:
    assert target.validate_model_revision("A" * 40, required=True) == "a" * 40


class FakeRepository:
    def __init__(
        self,
        rows: dict[str, list[Any]],
        *,
        fail_commit_once: bool = False,
        existing_vectors: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        self.rows = rows
        self.fail_commit_once = fail_commit_once
        self.committed: dict[str, dict[str, list[float]]] = {
            key: dict((existing_vectors or {}).get(key, {})) for key in rows
        }
        self.commit_calls = 0

    async def source_manifest(self, spec: Any) -> Any:
        rows = self.rows[spec.key]
        digest_input = ",".join(f"{row.id}:{row.text}" for row in rows)
        return target.SourceManifest(
            len(rows),
            target.hashlib.md5(digest_input.encode(), usedforsecurity=False).hexdigest(),
        )

    async def fetch_batch(
        self, spec: Any, *, after_id: str | None, limit: int
    ) -> list[Any]:
        rows = self.rows[spec.key]
        return [row for row in rows if after_id is None or row.id > after_id][:limit]

    async def commit_batch(
        self,
        spec: Any,
        rows: Sequence[Any],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self.commit_calls += 1
        if self.fail_commit_once:
            self.fail_commit_once = False
            raise target.ReembeddingError("simulated transaction rollback")
        self.committed[spec.key].update(
            {
                row.id: list(vector)
                for row, vector in zip(rows, vectors, strict=True)
            }
        )

    async def verify(self, spec: Any, dimension: int) -> Any:
        vectors = self.committed[spec.key]
        return target.Verification(
            eligible=len(self.rows[spec.key]),
            non_null=len(vectors),
            wrong_dimension=sum(len(vector) != dimension for vector in vectors.values()),
        )

    async def resume_state(self, spec: Any, last_id: str | None) -> Any:
        rows = self.rows[spec.key]
        through = [
            row for row in rows if last_id is not None and row.id <= last_id
        ]
        return target.ResumeState(
            count_through_id=len(through),
            max_eligible_id=rows[-1].id if rows else None,
            cursor_exists=last_id is not None
            and any(row.id == last_id for row in rows),
        )

    async def vector_manifest(self, spec: Any) -> str:
        return self._vector_manifest(spec, last_id=None)

    async def prefix_vector_manifest(self, spec: Any, last_id: str) -> str:
        return self._vector_manifest(spec, last_id=last_id)

    def _vector_manifest(self, spec: Any, last_id: str | None) -> str:
        vectors = self.committed[spec.key]
        payload = ",".join(
            f"{row_id}:{json.dumps(vector, separators=(',', ':'))}"
            for row_id, vector in sorted(vectors.items())
            if last_id is None or row_id <= last_id
        )
        return target.hashlib.md5(
            payload.encode(), usedforsecurity=False
        ).hexdigest()


def _test_specs() -> tuple[Any, ...]:
    return (
        target.TargetSpec("one", "", "", "", ""),
        target.TargetSpec("two", "", "", "", ""),
    )


def _identity(
    model: str = "model",
    *,
    revision: str = "1" * 40,
    profile: str = "qwen3",
    dimension: int = 2,
    native_dimension: int = 3,
) -> Any:
    return target.RunIdentity(
        model=model,
        model_revision=revision,
        profile=profile,
        dimension=dimension,
        native_dimension=native_dimension,
        endpoint_sha256="a" * 64,
        database_sha256="b" * 64,
    )


async def _fake_embed(texts: Sequence[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


def test_apply_reembedding_preserves_ids_batches_and_completes(
    tmp_path: Path,
) -> None:
    specs = _test_specs()
    rows = {
        "one": [
            target.SourceRow("00000000-0000-0000-0000-000000000001", "a"),
            target.SourceRow("00000000-0000-0000-0000-000000000002", "b"),
            target.SourceRow("00000000-0000-0000-0000-000000000003", "c"),
        ],
        "two": [
            target.SourceRow("00000000-0000-0000-0000-000000000004", "d")
        ],
    }
    repository = FakeRepository(rows)
    checkpoint = tmp_path / "checkpoint.json"
    identity = _identity()

    result = asyncio.run(
        target.apply_reembedding(
            repository=repository,
            embed_batch=_fake_embed,
            identity=identity,
            targets=specs,
            batch_size=2,
            checkpoint_path=checkpoint,
        )
    )

    assert set(repository.committed["one"]) == {row.id for row in rows["one"]}
    assert set(repository.committed["two"]) == {row.id for row in rows["two"]}
    assert result["one"].non_null == 3
    saved = target.load_checkpoint(checkpoint)
    assert all(item["complete"] for item in saved["targets"].values())
    assert saved["targets"]["one"]["processed"] == 3
    commit_calls = repository.commit_calls
    resumed = asyncio.run(
        target.apply_reembedding(
            repository=repository,
            embed_batch=_fake_embed,
            identity=identity,
            targets=specs,
            batch_size=2,
            checkpoint_path=checkpoint,
        )
    )
    assert resumed == result
    assert repository.commit_calls == commit_calls


def test_failed_batch_does_not_advance_checkpoint_and_resume_is_idempotent(
    tmp_path: Path,
) -> None:
    spec = (target.TargetSpec("one", "", "", "", ""),)
    rows = {
        "one": [
            target.SourceRow("00000000-0000-0000-0000-000000000001", "a"),
            target.SourceRow("00000000-0000-0000-0000-000000000002", "b"),
        ]
    }
    repository = FakeRepository(rows, fail_commit_once=True)
    checkpoint = tmp_path / "checkpoint.json"
    identity = _identity(profile="nemotron3")

    with pytest.raises(target.ReembeddingError, match="rollback"):
        asyncio.run(
            target.apply_reembedding(
                repository=repository,
                embed_batch=_fake_embed,
                identity=identity,
                targets=spec,
                batch_size=2,
                checkpoint_path=checkpoint,
            )
        )
    saved = target.load_checkpoint(checkpoint)
    assert saved["targets"]["one"]["processed"] == 0
    assert saved["targets"]["one"]["last_id"] is None

    result = asyncio.run(
        target.apply_reembedding(
            repository=repository,
            embed_batch=_fake_embed,
            identity=identity,
            targets=spec,
            batch_size=2,
            checkpoint_path=checkpoint,
        )
    )
    assert result["one"].non_null == 2


def test_resume_rejects_model_or_source_drift(tmp_path: Path) -> None:
    spec = (target.TargetSpec("one", "", "", "", ""),)
    repository = FakeRepository(
        {
            "one": [
                target.SourceRow(
                    "00000000-0000-0000-0000-000000000001", "original"
                )
            ]
        }
    )
    checkpoint = tmp_path / "checkpoint.json"
    original = _identity("model-a")
    manifests = asyncio.run(target.collect_manifests(repository, spec))
    target.save_checkpoint(checkpoint, target._new_checkpoint(original, manifests))

    with pytest.raises(target.ReembeddingError, match="another model"):
        asyncio.run(
            target.apply_reembedding(
                repository=repository,
                embed_batch=_fake_embed,
                identity=_identity("model-a", revision="2" * 40),
                targets=spec,
                batch_size=1,
                checkpoint_path=checkpoint,
            )
        )

    repository.rows["one"][0] = target.SourceRow(
        "00000000-0000-0000-0000-000000000001", "changed"
    )
    with pytest.raises(target.ReembeddingError, match="source rows changed"):
        asyncio.run(
            target.apply_reembedding(
                repository=repository,
                embed_batch=_fake_embed,
                identity=original,
                targets=spec,
                batch_size=1,
                checkpoint_path=checkpoint,
            )
        )


def test_resume_rejects_skipped_cursor_even_when_checkpoint_count_looks_valid(
    tmp_path: Path,
) -> None:
    spec = (target.TargetSpec("one", "", "", "", ""),)
    rows = {
        "one": [
            target.SourceRow("00000000-0000-0000-0000-000000000001", "a"),
            target.SourceRow("00000000-0000-0000-0000-000000000002", "b"),
            target.SourceRow("00000000-0000-0000-0000-000000000003", "c"),
        ]
    }
    repository = FakeRepository(rows)
    identity = _identity()
    manifests = asyncio.run(target.collect_manifests(repository, spec))
    checkpoint_data = target._new_checkpoint(identity, manifests)
    progress = checkpoint_data["targets"]["one"]
    progress["processed"] = 1
    progress["last_id"] = rows["one"][1].id
    progress["prefix_vector_manifest_md5"] = "0" * 32
    checkpoint = tmp_path / "checkpoint.json"
    target.save_checkpoint(checkpoint, checkpoint_data)

    with pytest.raises(target.ReembeddingError, match="cursor does not match"):
        asyncio.run(
            target.apply_reembedding(
                repository=repository,
                embed_batch=_fake_embed,
                identity=identity,
                targets=spec,
                batch_size=1,
                checkpoint_path=checkpoint,
            )
        )
    assert repository.commit_calls == 0


def test_forged_complete_checkpoint_cannot_accept_old_1024_vectors(
    tmp_path: Path,
) -> None:
    spec = (target.TargetSpec("one", "", "", "", ""),)
    row = target.SourceRow("00000000-0000-0000-0000-000000000001", "old")
    repository = FakeRepository(
        {"one": [row]},
        existing_vectors={"one": {row.id: [1.0] + [0.0] * 1023}},
    )
    identity = _identity(dimension=1024, native_dimension=4096)
    manifests = asyncio.run(target.collect_manifests(repository, spec))
    checkpoint_data = target._new_checkpoint(identity, manifests)
    progress = checkpoint_data["targets"]["one"]
    progress.update(
        {
            "processed": 1,
            "last_id": row.id,
            "complete": True,
            "prefix_vector_manifest_md5": "0" * 32,
            "vector_manifest_md5": "0" * 32,
        }
    )
    checkpoint = tmp_path / "checkpoint.json"
    target.save_checkpoint(checkpoint, checkpoint_data)

    with pytest.raises(target.ReembeddingError, match="prefix vector attestation changed"):
        asyncio.run(
            target.apply_reembedding(
                repository=repository,
                embed_batch=_fake_embed,
                identity=identity,
                targets=spec,
                batch_size=1,
                checkpoint_path=checkpoint,
            )
        )
    assert repository.commit_calls == 0


def test_incomplete_resume_rejects_replaced_processed_prefix_vectors(
    tmp_path: Path,
) -> None:
    spec = (target.TargetSpec("one", "", "", "", ""),)
    first = target.SourceRow("00000000-0000-0000-0000-000000000001", "a")
    second = target.SourceRow("00000000-0000-0000-0000-000000000002", "b")
    candidate_vector = [0.0, 1.0] + [0.0] * 1022
    repository = FakeRepository(
        {"one": [first, second]},
        existing_vectors={"one": {first.id: candidate_vector}},
    )
    identity = _identity(dimension=1024, native_dimension=4096)
    manifests = asyncio.run(target.collect_manifests(repository, spec))
    checkpoint_data = target._new_checkpoint(identity, manifests)
    progress = checkpoint_data["targets"]["one"]
    progress.update(
        {
            "processed": 1,
            "last_id": first.id,
            "prefix_vector_manifest_md5": asyncio.run(
                repository.prefix_vector_manifest(spec[0], first.id)
            ),
        }
    )
    checkpoint = tmp_path / "checkpoint.json"
    target.save_checkpoint(checkpoint, checkpoint_data)

    repository.committed["one"][first.id] = [1.0, 0.0] + [0.0] * 1022
    with pytest.raises(
        target.ReembeddingError, match="prefix vector attestation changed"
    ):
        asyncio.run(
            target.apply_reembedding(
                repository=repository,
                embed_batch=_fake_embed,
                identity=identity,
                targets=spec,
                batch_size=1,
                checkpoint_path=checkpoint,
            )
        )
    assert repository.commit_calls == 0


def test_dry_run_is_default_and_apply_requires_checkpoint() -> None:
    parser = target.build_parser()
    args = parser.parse_args([])
    assert args.apply is False
    assert args.checkpoint is None
    assert args.model_revision is None

    args = parser.parse_args(["--apply"])
    with pytest.raises(target.ReembeddingError, match="checkpoint"):
        asyncio.run(target.run(args))

    args = parser.parse_args(["--apply", "--checkpoint", "/tmp/checkpoint.json"])
    with pytest.raises(target.ReembeddingError, match="model-revision"):
        asyncio.run(target.run(args))
