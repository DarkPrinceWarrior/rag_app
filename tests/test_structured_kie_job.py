from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import uuid
from typing import Any

import pytest
from arq import Retry

from rag_app.config import settings
from rag_app.pipeline.structured_extraction_executor import (
    InferenceRequest,
    InferenceResponse,
    TransientInferenceError,
)
from rag_app.pipeline.structured_extraction_protocol import (
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    canonical_json_sha256,
)
from rag_app.workers import structured_kie
from rag_app.workers.structured_lifecycle import StructuredArtifactClaim

_DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_ARTIFACT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_CLAIM_TOKEN = uuid.UUID("33333333-3333-3333-3333-333333333333")
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _Session:
    def __init__(self) -> None:
        self.committed = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def commit(self) -> None:
        self.committed = True


class _SessionMaker:
    def __init__(self) -> None:
        self.sessions: list[_Session] = []

    def __call__(self) -> _Session:
        session = _Session()
        self.sessions.append(session)
        return session


class _Storage:
    def __init__(self, source: bytes = _PNG) -> None:
        self.source = source
        self.reads: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, bytes, str]] = []
        self.removed: list[tuple[str, str]] = []

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        self.reads.append((bucket, key))
        return self.source

    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        self.writes.append((bucket, key, data, content_type))

    async def remove_object(self, bucket: str, key: str) -> None:
        self.removed.append((bucket, key))


def _claim(
    *,
    schema: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> StructuredArtifactClaim:
    request_schema = schema or {
        "type": "object",
        "properties": {"tag": {"type": "string"}},
    }
    return StructuredArtifactClaim(
        artifact_id=_ARTIFACT_ID,
        document_id=_DOC_ID,
        parse_revision=4,
        page_idx=7,
        artifact_type="kie",
        backend="granite",
        model=settings.structured_model_name,
        model_revision=settings.structured_model_revision,
        prompt_version=PROMPT_VERSION,
        protocol_version=f"structured-v{PROTOCOL_VERSION}",
        schema_version=2,
        request_hash="b" * 64,
        request_schema=request_schema,
        schema_sha256=canonical_json_sha256(request_schema),
        request_options=options or {"temperature": 0, "max_tokens": 4096},
        source_key=f"{_DOC_ID}/sidecars/r4/p000007/source.png",
        source_sha256=hashlib.sha256(_PNG).hexdigest(),
        attempt_count=1,
        max_attempts=settings.structured_job_max_attempts,
        claim_token=_CLAIM_TOKEN,
    )


def _ctx(storage: _Storage, inference: Any) -> dict[str, Any]:
    return {
        "sessionmaker": _SessionMaker(),
        "storage": storage,
        "structured_client": inference,
    }


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "structured_extraction_enabled", True)


def _install_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    claim: StructuredArtifactClaim,
    *,
    publish: bool = True,
    fail_status: str = "error",
) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {"claim": [], "publish": [], "fail": []}

    async def fake_claim(session: Any, artifact_id: uuid.UUID, **kwargs: Any):
        calls["claim"].append({"artifact_id": artifact_id, **kwargs})
        return claim

    async def fake_publish(session: Any, value: StructuredArtifactClaim, **kwargs: Any):
        calls["publish"].append(kwargs)
        return publish

    async def fake_fail(session: Any, value: StructuredArtifactClaim, **kwargs: Any):
        calls["fail"].append(kwargs)
        return fail_status

    monkeypatch.setattr(structured_kie, "claim_structured_artifact", fake_claim)
    monkeypatch.setattr(structured_kie, "publish_structured_artifact", fake_publish)
    monkeypatch.setattr(structured_kie, "fail_structured_artifact", fake_fail)
    return calls


def test_feature_flag_returns_before_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "structured_extraction_enabled", False)

    async def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("disabled job must not claim")

    monkeypatch.setattr(structured_kie, "claim_structured_artifact", forbidden)
    result = asyncio.run(structured_kie.run_structured_kie({}, str(_ARTIFACT_ID)))
    assert result == {"status": "disabled"}


def test_nested_kie_claim_is_published_conditionally(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    claim = _claim()
    calls = _install_lifecycle(monkeypatch, claim)
    storage = _Storage()

    async def inference(request: InferenceRequest) -> InferenceResponse:
        assert request.image_url.startswith("data:image/png;base64,")
        return InferenceResponse('{"tag":"A-17"}', "stop", 0.2)

    result = asyncio.run(structured_kie.run_structured_kie(_ctx(storage, inference), str(_ARTIFACT_ID)))

    assert result["status"] == "ready"
    assert len(storage.writes) == 1
    bucket, key, payload, content_type = storage.writes[0]
    assert bucket == settings.bucket_artifacts
    assert key.endswith(f"attempt-{_CLAIM_TOKEN}.json")
    assert content_type == "application/json"
    artifact = json.loads(payload)
    assert artifact["schema_version"] == 2
    assert artifact["payload"] == {
        "fields": [],
        "result": {"tag": "A-17"},
        "schema_sha256": claim.schema_sha256,
    }
    published = calls["publish"][0]
    assert published["artifact_key"] == key
    assert published["content_sha256"] == hashlib.sha256(payload).hexdigest()
    assert published["summary"]["model_calls"] == 1
    assert "raw_response" not in published["summary"]
    assert calls["fail"] == []


def test_table_kie_uses_table_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
            }
        },
    }
    claim = _claim(schema=schema, options={"schema_mode": "table", "max_tokens": 4096})
    _install_lifecycle(monkeypatch, claim)
    storage = _Storage()

    async def inference(request: InferenceRequest) -> InferenceResponse:
        fields = request.schema["properties"]["rows"]["items"]["properties"]
        row = {field: {"id": "P-1", "value": "16.5 MPa"}[field] for field in fields}
        return InferenceResponse(json.dumps({"rows": [row]}), "stop", 0.1)

    result = asyncio.run(structured_kie.run_structured_kie(_ctx(storage, inference), str(_ARTIFACT_ID)))

    assert result["status"] == "ready"
    artifact = json.loads(storage.writes[0][2])
    assert artifact["payload"]["result"] == {"rows": [{"id": "P-1", "value": "16.5 MPa"}]}


def test_lost_publish_claim_removes_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    claim = _claim()
    _install_lifecycle(monkeypatch, claim, publish=False)
    storage = _Storage()

    async def inference(request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse('{"tag":"A-17"}', "stop")

    result = asyncio.run(structured_kie.run_structured_kie(_ctx(storage, inference), str(_ARTIFACT_ID)))

    assert result == {"status": "stale"}
    assert storage.removed == [(settings.bucket_artifacts, storage.writes[0][1])]


def test_invalid_model_output_fails_terminal_without_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    claim = _claim()
    calls = _install_lifecycle(monkeypatch, claim)
    storage = _Storage()

    async def inference(request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse("not-json", "stop")

    result = asyncio.run(structured_kie.run_structured_kie(_ctx(storage, inference), str(_ARTIFACT_ID)))

    assert result == {"status": "error", "error": "model_output_invalid"}
    assert storage.writes == []
    assert calls["publish"] == []
    assert calls["fail"] == [
        {
            "error_code": "model_output_invalid",
            "retryable": False,
            "retry_delay_seconds": 30,
        }
    ]


def test_transient_model_failure_requeues_with_bounded_arq_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    claim = _claim()
    calls = _install_lifecycle(monkeypatch, claim, fail_status="queued")
    storage = _Storage()
    model_calls = 0

    async def inference(request: InferenceRequest) -> InferenceResponse:
        nonlocal model_calls
        model_calls += 1
        raise TransientInferenceError("503")

    with pytest.raises(Retry) as retry:
        asyncio.run(structured_kie.run_structured_kie(_ctx(storage, inference), str(_ARTIFACT_ID)))

    assert retry.value.defer_score == 30_000
    assert model_calls == 2
    assert storage.writes == []
    assert calls["fail"][0]["error_code"] == "model_transient"
    assert calls["fail"][0]["retryable"] is True


def test_source_hash_mismatch_is_terminal_before_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    claim = _claim()
    calls = _install_lifecycle(monkeypatch, claim)
    storage = _Storage(source=_PNG + b"tampered")

    async def forbidden(request: InferenceRequest) -> InferenceResponse:
        raise AssertionError("bad source must not reach model")

    result = asyncio.run(structured_kie.run_structured_kie(_ctx(storage, forbidden), str(_ARTIFACT_ID)))

    assert result == {"status": "error", "error": "source_hash_mismatch"}
    assert storage.writes == []
    assert calls["fail"][0]["retryable"] is False
