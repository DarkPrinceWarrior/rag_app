from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from openai import APITimeoutError, AsyncOpenAI
from pydantic import ValidationError

from rag_app.config import Settings
from rag_app.llm.structured import GraniteStructuredClient
from rag_app.pipeline.structured_extraction_executor import (
    InferenceRequest,
    TransientInferenceError,
)


class _Completions:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


class _FakeClient:
    def __init__(self, completions: _Completions) -> None:
        self.chat = SimpleNamespace(completions=completions)

    async def close(self) -> None:
        return None


def _client(completions: _Completions) -> GraniteStructuredClient:
    client = object.__new__(GraniteStructuredClient)
    client._model = "granite-test"
    client._client = cast(AsyncOpenAI, _FakeClient(completions))
    return client


def _request() -> InferenceRequest:
    return InferenceRequest(
        chunk_id="c000",
        prompt="extract",
        schema={"type": "object", "properties": {"number": {"type": "string"}}},
        image_url="data:image/png;base64,AA==",
        max_tokens=1024,
    )


def test_structured_settings_are_disabled_and_pinned_by_default() -> None:
    configured = Settings(_env_file=None)

    assert configured.structured_extraction_enabled is False
    assert configured.structured_model_base_url == "http://127.0.0.1:8132/v1"
    assert configured.structured_model_revision == "82472ca3a4905fff5e4daa481c0b9cd530859c79"
    assert configured.structured_model_timeout_s < configured.parser_sidecar_timeout_s
    assert configured.structured_job_lease_s > configured.parser_sidecar_timeout_s


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://example.com/v1",
        "http://user:secret@127.0.0.1:8132/v1",
        "http://127.0.0.1:8132/other",
        "http://127.0.0.1:8132/v1?token=secret",
    ),
)
def test_structured_settings_reject_non_loopback_or_credentialed_endpoint(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="structured model endpoint"):
        Settings(_env_file=None, structured_model_base_url=endpoint)


def test_structured_settings_reject_unsafe_timing_and_unpinned_revision() -> None:
    with pytest.raises(ValidationError, match="full commit SHA"):
        Settings(_env_file=None, structured_model_revision="main")
    with pytest.raises(ValidationError, match="below sidecar job timeout"):
        Settings(_env_file=None, structured_model_timeout_s=180)
    with pytest.raises(ValidationError, match="exceed sidecar job timeout"):
        Settings(_env_file=None, structured_job_lease_s=180)


def test_sends_multimodal_json_schema_request() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"number":"A-1"}'), finish_reason="stop")]
    )
    completions = _Completions(response)
    client = _client(completions)

    result = asyncio.run(client(_request()))

    assert result.content == '{"number":"A-1"}'
    assert result.finish_reason == "stop"
    assert result.latency_s is not None and result.latency_s >= 0
    assert completions.kwargs["model"] == "granite-test"
    assert completions.kwargs["messages"][0]["content"][0]["type"] == "image_url"
    assert completions.kwargs["messages"][0]["content"][1] == {
        "type": "text",
        "text": "extract",
    }
    assert completions.kwargs["response_format"]["json_schema"]["schema"] == _request().schema


def test_maps_sdk_timeout_to_bounded_executor_error() -> None:
    error = APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:8132/v1"))
    client = _client(_Completions(error=error))

    with pytest.raises(TransientInferenceError, match="APITimeoutError"):
        asyncio.run(client(_request()))


def test_rejects_missing_content_and_multiple_choices() -> None:
    invalid = (
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")]
        ),
        SimpleNamespace(choices=[]),
    )
    for response in invalid:
        with pytest.raises(ValueError):
            asyncio.run(_client(_Completions(response))(_request()))
