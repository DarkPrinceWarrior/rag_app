"""Локальный Granite client для строгого schema-KIE без скрытых повторов SDK."""

from __future__ import annotations

import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema

from rag_app.pipeline.structured_extraction_executor import (
    InferenceRequest,
    InferenceResponse,
    TransientInferenceError,
)


class GraniteStructuredClient:
    """Adapter OpenAI-compatible vLLM к чистому structured executor."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float,
    ) -> None:
        if not model.strip() or timeout_s <= 0:
            raise ValueError("model and positive timeout are required")
        self._model = model
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
        )

    async def close(self) -> None:
        await self._client.close()

    async def __call__(self, request: InferenceRequest) -> InferenceResponse:
        messages: list[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": request.image_url, "detail": "auto"},
                    },
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]
        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": "document_fields",
                "schema": _json_schema_dict(request.schema),
            },
        }
        started = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=response_format,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as exc:
            raise TransientInferenceError(type(exc).__name__) from exc
        except APIStatusError as exc:
            if exc.status_code in {408, 429} or exc.status_code >= 500:
                raise TransientInferenceError(f"http_{exc.status_code}") from exc
            raise

        if len(response.choices) != 1:
            raise ValueError("structured model must return exactly one choice")
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):
            raise ValueError("structured model returned non-text content")
        return InferenceResponse(
            content=content,
            finish_reason=str(choice.finish_reason),
            latency_s=time.monotonic() - started,
        )


def _json_schema_dict(schema: dict[str, Any]) -> dict[str, object]:
    """Сохранить точный JSON Schema и удовлетворить инвариант SDK typing."""

    return {key: value for key, value in schema.items()}
