#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import hashlib
import json
import math
import os
import shlex
import socket
import struct
import sys
import time
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from rag_app.eval.private_artifacts import (  # type: ignore[import-untyped]
    PrivateArtifactError,
    write_private_json_fresh,
)

DEFAULT_TESTS = (
    "health",
    "version",
    "models",
    "ru",
    "en",
    "zh",
    "json_schema",
    "forced_tool",
    "auto_tool",
    "streaming",
    "vision",
    "long_4k",
    "long_8k",
    "concurrency",
)

MAX_JSON_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_STATUS_RESPONSE_BYTES = 16 * 1024
MAX_STREAM_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_SSE_EVENTS = 10_000
MAX_TIMEOUT_S = 3_600.0
MAX_CONCURRENT_REQUESTS = 128
MAX_CONCURRENCY = 32
REPORT_MAX_BYTES = 8 * 1024 * 1024
QUALIFICATION_CONTRACT = {
    "schema_version": 2,
    "tests": DEFAULT_TESTS,
    "exact_text_markers": True,
    "long_context_prompt_token_floor": True,
    "served_model_must_match": True,
    "vision_order": ["red", "blue"],
    "response_byte_limits": {
        "json": MAX_JSON_RESPONSE_BYTES,
        "status": MAX_STATUS_RESPONSE_BYTES,
        "stream": MAX_STREAM_RESPONSE_BYTES,
    },
}


class QualificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    try:
        port = parsed.port
    except ValueError as exc:
        raise QualificationError("base URL contains an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1"
    ):
        raise QualificationError(
            "base URL must be a credential-free loopback HTTP(S) endpoint with an "
            "explicit port and exact /v1 path"
        )
    return normalized


def redacted_command(executable: str, argv: list[str]) -> str:
    rendered = [executable]
    redact_next = False
    for argument in argv:
        if redact_next:
            rendered.append("<redacted>")
            redact_next = False
        elif argument == "--api-key":
            rendered.append(argument)
            redact_next = True
        elif argument.startswith("--api-key="):
            rendered.append("--api-key=<redacted>")
        else:
            rendered.append(argument)
    return shlex.join(rendered)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise QualificationError(f"cannot hash runner source: {type(exc).__name__}") from exc


def write_report_create_only(path: Path, report: dict[str, Any]) -> str:
    raw = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    try:
        artifact = write_private_json_fresh(path, raw, max_bytes=REPORT_MAX_BYTES)
    except FileExistsError:
        raise QualificationError(f"refusing to overwrite existing report: {path}") from None
    except PrivateArtifactError as exc:
        raise QualificationError(f"cannot publish private report: {exc}") from exc
    return artifact.sha256


def endpoint(base_url: str, path: str) -> str:
    return f"{base_url}/{path.lstrip('/')}"


def limited_text(value: bytes | str, limit: int = 32_000) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…[truncated {len(text) - limit} chars]"


def http_client(timeout_s: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout_s,
        trust_env=False,
        follow_redirects=False,
    )


def reject_redirect(response: httpx.Response, *, url: str) -> None:
    if response.is_redirect:
        raise QualificationError(f"redirect refused for {url}")


def check_deadline(deadline: float, *, operation: str) -> None:
    if time.monotonic() >= deadline:
        raise QualificationError(f"{operation} exceeded absolute wall deadline")


def read_bounded_response(
    response: httpx.Response,
    *,
    max_bytes: int,
    deadline: float,
) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise QualificationError("response has an invalid Content-Length") from exc
        if declared_size < 0 or declared_size > max_bytes:
            raise QualificationError(f"response exceeds {max_bytes} bytes")

    raw = bytearray()
    for chunk in response.iter_bytes():
        check_deadline(deadline, operation="response read")
        if len(raw) + len(chunk) > max_bytes:
            raise QualificationError(f"response exceeds {max_bytes} bytes")
        raw.extend(chunk)
    check_deadline(deadline, operation="response read")
    return bytes(raw)


def request_json(
    *,
    url: str,
    api_key: str,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any], str]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    deadline = time.monotonic() + timeout_s
    try:
        with http_client(timeout_s) as client:
            check_deadline(deadline, operation="request")
            with client.stream(
                "POST" if payload is not None else "GET",
                url,
                headers=headers,
                json=payload,
            ) as response:
                reject_redirect(response, url=url)
                raw = read_bounded_response(
                    response,
                    max_bytes=MAX_JSON_RESPONSE_BYTES,
                    deadline=deadline,
                )
                status = response.status_code
    except httpx.RequestError as exc:
        raise QualificationError(f"request failed for {url}: {exc}") from exc

    if status >= 400:
        raise QualificationError(f"HTTP {status} {url}: {limited_text(raw, 8_000)}")
    raw_text = limited_text(raw)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QualificationError(f"non-JSON response from {url}: {limited_text(raw, 8_000)}") from exc
    if not isinstance(body, dict):
        raise QualificationError(f"non-object JSON response from {url}")
    return status, body, raw_text


def request_status(*, url: str, api_key: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    try:
        with http_client(timeout_s) as client:
            check_deadline(deadline, operation="request")
            with client.stream(
                "GET",
                url,
                headers={
                    "Accept": "*/*",
                    "Authorization": f"Bearer {api_key}",
                },
            ) as response:
                reject_redirect(response, url=url)
                raw = read_bounded_response(
                    response,
                    max_bytes=MAX_STATUS_RESPONSE_BYTES,
                    deadline=deadline,
                )
                status = response.status_code
                content_type = response.headers.get("content-type")
    except httpx.RequestError as exc:
        raise QualificationError(f"request failed for {url}: {exc}") from exc
    if status >= 400:
        raise QualificationError(f"HTTP {status} {url}: {limited_text(raw, 8_000)}")
    return {
        "http_status": status,
        "content_type": content_type,
        "body": limited_text(raw, 8_000),
    }


def message_from_response(body: dict[str, Any]) -> dict[str, Any]:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QualificationError("response has no choices[0].message") from exc
    if not isinstance(message, dict):
        raise QualificationError("choices[0].message is not an object")
    return message


def message_text(body: dict[str, Any]) -> str:
    content = message_from_response(body).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "".join(parts)
    return ""


def validate_served_model(body: dict[str, Any], expected_model: str) -> None:
    actual_model = body.get("model")
    if actual_model != expected_model:
        raise QualificationError(f"served model mismatch: expected {expected_model!r}, got {actual_model!r}")


def version_probe(*, base_url: str, api_key: str, timeout_s: float) -> dict[str, Any]:
    status, body, _ = request_json(
        url=endpoint(base_url.removesuffix("/v1"), "version"),
        api_key=api_key,
        timeout_s=timeout_s,
    )
    version = body.get("version")
    if not isinstance(version, str) or not version.strip() or len(version) > 128:
        raise QualificationError(f"invalid endpoint version: {version!r}")
    return {"http_status": status, "version": version.strip()}


def models_probe(*, base_url: str, api_key: str, timeout_s: float, model: str) -> dict[str, Any]:
    status, body, _ = request_json(
        url=endpoint(base_url, "models"),
        api_key=api_key,
        timeout_s=timeout_s,
    )
    rows = body.get("data")
    if not isinstance(rows, list):
        raise QualificationError("models response has no data array")
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == model]
    if len(matches) != 1:
        model_ids = [row.get("id") for row in rows if isinstance(row, dict) and "id" in row]
        raise QualificationError(f"expected exactly one served model {model!r}, got {model_ids!r}")
    max_model_len = matches[0].get("max_model_len")
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
        raise QualificationError(f"invalid max_model_len: {max_model_len!r}")
    return {
        "http_status": status,
        "model_id": model,
        "max_model_len": max_model_len,
    }


def chat_payload(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def synthetic_vision_page() -> bytes:
    width, height = 320, 180
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            color = (250, 250, 250)
            if 28 <= x < 132 and 38 <= y < 142:
                color = (220, 28, 36)
            elif 188 <= x < 292 and 38 <= y < 142:
                color = (30, 80, 210)
            row.extend(color)
        rows.append(b"\x00" + bytes(row))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
        + png_chunk(b"IEND", b"")
    )


def run_probe(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    try:
        details = operation()
        return {
            "name": name,
            "status": "pass",
            "started_at": started_at,
            "duration_s": round(time.monotonic() - started, 3),
            "details": details,
        }
    except Exception as exc:  # noqa: BLE001 - qualification must capture every failure
        return {
            "name": name,
            "status": "fail",
            "started_at": started_at,
            "duration_s": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def language_probe(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float,
    model: str,
    prompt: str,
    marker: str,
) -> dict[str, Any]:
    status, body, _ = request_json(
        url=endpoint(base_url, "chat/completions"),
        api_key=api_key,
        timeout_s=timeout_s,
        payload=chat_payload(model, prompt, max_tokens=64),
    )
    validate_served_model(body, model)
    content = message_text(body).strip()
    if content != marker:
        raise QualificationError(f"expected marker {marker!r}, got {content!r}")
    return {"http_status": status, "marker": marker, "content": content}


def json_schema_probe(*, base_url: str, api_key: str, timeout_s: float, model: str) -> dict[str, Any]:
    payload = chat_payload(
        model,
        "Верни объект: document_id DOC-42, pressure_mpa 16, approved true.",
        max_tokens=128,
    )
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "qualification_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "const": "DOC-42"},
                    "pressure_mpa": {"type": "integer", "const": 16},
                    "approved": {"type": "boolean", "const": True},
                },
                "required": ["document_id", "pressure_mpa", "approved"],
                "additionalProperties": False,
            },
        },
    }
    status, body, _ = request_json(
        url=endpoint(base_url, "chat/completions"),
        api_key=api_key,
        timeout_s=timeout_s,
        payload=payload,
    )
    validate_served_model(body, model)
    content = message_text(body).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise QualificationError(f"schema response is not JSON: {content!r}") from exc
    expected = {"document_id": "DOC-42", "pressure_mpa": 16, "approved": True}
    if parsed != expected:
        raise QualificationError(f"unexpected schema object: {parsed!r}")
    return {"http_status": status, "parsed": parsed}


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_document",
            "description": "Find a document by its exact identifier.",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
                "additionalProperties": False,
            },
        },
    }


def tool_probe(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float,
    model: str,
    automatic: bool,
) -> dict[str, Any]:
    payload = chat_payload(
        model,
        "Use lookup_document to find the document with identifier DOC-42. Do not answer from memory.",
        max_tokens=256,
    )
    payload["tools"] = [tool_definition()]
    payload["tool_choice"] = (
        "auto" if automatic else {"type": "function", "function": {"name": "lookup_document"}}
    )
    status, body, _ = request_json(
        url=endpoint(base_url, "chat/completions"),
        api_key=api_key,
        timeout_s=timeout_s,
        payload=payload,
    )
    validate_served_model(body, model)
    calls = message_from_response(body).get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise QualificationError(f"expected exactly one tool call, got {calls!r}")
    call = calls[0]
    if not isinstance(call, dict) or call.get("type") != "function":
        raise QualificationError(f"tool call must have type='function': {call!r}")
    call_id = call.get("id")
    if (
        not isinstance(call_id, str)
        or not call_id.strip()
        or call_id != call_id.strip()
        or len(call_id) > 256
    ):
        raise QualificationError(f"tool call id must contain 1 to 256 characters: {call_id!r}")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "lookup_document":
        function_name = function.get("name") if isinstance(function, dict) else None
        raise QualificationError(f"unexpected tool name: {function_name!r}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise QualificationError("tool arguments must be a JSON string on the wire")
    if len(arguments.encode("utf-8")) > 4096:
        raise QualificationError("tool arguments exceed 4096 UTF-8 bytes")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QualificationError(f"duplicate tool argument key: {key!r}")
            result[key] = value
        return result

    try:
        parsed_arguments = json.loads(
            arguments,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                QualificationError(f"non-finite tool argument constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise QualificationError(f"invalid tool arguments: {arguments!r}") from exc
    expected_arguments = {"document_id": "DOC-42"}
    if parsed_arguments != expected_arguments:
        raise QualificationError(f"unexpected tool arguments: {parsed_arguments!r}")
    return {
        "http_status": status,
        "tool_call_id": call_id,
        "function": "lookup_document",
        "arguments": parsed_arguments,
    }


def streaming_probe(*, base_url: str, api_key: str, timeout_s: float, model: str) -> dict[str, Any]:
    payload = chat_payload(
        model,
        "Ответь точно маркером STREAM_OK.",
        max_tokens=64,
    )
    payload["stream"] = True
    chunks: list[str] = []
    served_models: set[str] = set()
    event_count = 0
    saw_done = False
    total_bytes = 0
    pending = bytearray()
    deadline = time.monotonic() + timeout_s

    def consume_line(raw_line: bytes) -> None:
        nonlocal event_count, saw_done
        line = raw_line.rstrip(b"\r").decode("utf-8", errors="strict").strip()
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            saw_done = True
            return
        event_count += 1
        if event_count > MAX_SSE_EVENTS:
            raise QualificationError(f"stream exceeds {MAX_SSE_EVENTS} SSE events")
        event = json.loads(data)
        if not isinstance(event, dict):
            raise QualificationError("stream event is not a JSON object")
        event_model = event.get("model")
        if event_model is not None:
            if not isinstance(event_model, str):
                raise QualificationError("stream event model is not a string")
            served_models.add(event_model)
        try:
            delta = event["choices"][0]["delta"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QualificationError("stream event has no choices[0].delta") from exc
        if not isinstance(delta, dict):
            raise QualificationError("stream delta is not an object")
        content = delta.get("content")
        if isinstance(content, str):
            chunks.append(content)

    try:
        with http_client(timeout_s) as client:
            check_deadline(deadline, operation="streaming request")
            with client.stream(
                "POST",
                endpoint(base_url, "chat/completions"),
                headers={
                    "Accept": "text/event-stream",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                reject_redirect(response, url=str(response.request.url))
                status = response.status_code
                if status >= 400:
                    raw = read_bounded_response(
                        response,
                        max_bytes=MAX_STATUS_RESPONSE_BYTES,
                        deadline=deadline,
                    )
                    raise QualificationError(f"HTTP {status} streaming: {limited_text(raw, 8_000)}")
                for raw_chunk in response.iter_bytes():
                    check_deadline(deadline, operation="streaming response")
                    total_bytes += len(raw_chunk)
                    if total_bytes > MAX_STREAM_RESPONSE_BYTES:
                        raise QualificationError(f"stream exceeds {MAX_STREAM_RESPONSE_BYTES} bytes")
                    pending.extend(raw_chunk)
                    while b"\n" in pending:
                        raw_line, _, remainder = pending.partition(b"\n")
                        pending = bytearray(remainder)
                        consume_line(bytes(raw_line))
                        if saw_done:
                            break
                    if saw_done:
                        break
                check_deadline(deadline, operation="streaming response")
                if pending and not saw_done:
                    consume_line(bytes(pending))
    except httpx.RequestError as exc:
        raise QualificationError(f"streaming request failed: {exc}") from exc

    content = "".join(chunks).strip()
    if not saw_done:
        raise QualificationError("stream ended without [DONE]")
    if served_models != {model}:
        raise QualificationError(
            f"stream served model mismatch: expected {model!r}, got {sorted(served_models)!r}"
        )
    if content != "STREAM_OK":
        raise QualificationError(f"stream marker mismatch: {content!r}")
    return {
        "http_status": status,
        "event_count": event_count,
        "saw_done": saw_done,
        "served_model": model,
        "response_bytes": total_bytes,
        "content": content,
    }


def vision_probe(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float,
    model: str,
    image_bytes: bytes,
    image_sha256: str,
) -> dict[str, Any]:
    image_data = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Назови два основных цвета прямоугольников слева направо. "
                            "Ответь только двумя английскими словами."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, body, _ = request_json(
        url=endpoint(base_url, "chat/completions"),
        api_key=api_key,
        timeout_s=timeout_s,
        payload=payload,
    )
    validate_served_model(body, model)
    content = message_text(body).strip().casefold()
    normalized_content = " ".join(content.split())
    if normalized_content != "red blue":
        raise QualificationError(f"expected exactly 'red blue', got {content!r}")
    return {
        "http_status": status,
        "image_sha256": image_sha256,
        "content": normalized_content,
    }


def long_context_probe(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float,
    model: str,
    target_tokens: int,
) -> dict[str, Any]:
    marker = f"LONG_OK_{target_tokens}"
    filler_words = target_tokens
    prompt = (
        f"{marker}\nRemember the unique marker from the first line. "
        + ("calibration " * filler_words)
        + "\nReturn exactly the unique marker from the first line and nothing else."
    )
    if prompt.count(marker) != 1 or not prompt.startswith(f"{marker}\n"):
        raise QualificationError("long-context prompt marker placement is invalid")
    payload = chat_payload(model, prompt, max_tokens=64)
    status, body, _ = request_json(
        url=endpoint(base_url, "chat/completions"),
        api_key=api_key,
        timeout_s=timeout_s,
        payload=payload,
    )
    validate_served_model(body, model)
    content = message_text(body).strip()
    if content != marker:
        raise QualificationError(f"long-context marker mismatch: {content!r}")
    raw_usage = body.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < target_tokens:
        raise QualificationError(f"reported prompt_tokens must be >= {target_tokens}, got {prompt_tokens!r}")
    return {
        "http_status": status,
        "minimum_prompt_tokens": target_tokens,
        "filler_words": filler_words,
        "reported_prompt_tokens": prompt_tokens,
        "content": content,
    }


def concurrency_probe(
    *,
    base_url: str,
    api_key: str,
    timeout_s: float,
    model: str,
    request_count: int,
    workers: int,
) -> dict[str, Any]:
    def one(index: int) -> dict[str, Any]:
        marker = f"PARALLEL_OK_{index:02d}"
        started = time.monotonic()
        status, body, _ = request_json(
            url=endpoint(base_url, "chat/completions"),
            api_key=api_key,
            timeout_s=timeout_s,
            payload=chat_payload(
                model,
                f"Return exactly {marker}.",
                max_tokens=48,
            ),
        )
        validate_served_model(body, model)
        content = message_text(body).strip()
        return {
            "index": index,
            "http_status": status,
            "duration_s": round(time.monotonic() - started, 3),
            "marker": marker,
            "content": content,
            "passed": content == marker,
        }

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(one, range(request_count)))
    failed = [row for row in rows if not row["passed"]]
    if failed:
        raise QualificationError(f"{len(failed)}/{request_count} concurrent markers missing: {failed!r}")
    durations = [float(row["duration_s"]) for row in rows]
    return {
        "request_count": request_count,
        "workers": workers,
        "wall_duration_s": round(time.monotonic() - started, 3),
        "request_duration_s": {
            "min": min(durations),
            "max": max(durations),
            "mean": round(sum(durations) / len(durations), 3),
        },
        "requests": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify an OpenAI-compatible chat endpoint without corpus access.",
        allow_abbrev=False,
    )
    parser.add_argument("--base-url", required=True, help="For example http://127.0.0.1:8006/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "local"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--concurrent-requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--exclude",
        default="",
        help=f"Comma-separated tests to skip. Available: {','.join(DEFAULT_TESTS)}",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.timeout) or not 0.1 <= args.timeout <= MAX_TIMEOUT_S:
        raise SystemExit(f"timeout must be finite and in [0.1, {MAX_TIMEOUT_S}]")
    if not 1 <= args.concurrent_requests <= MAX_CONCURRENT_REQUESTS:
        raise SystemExit(f"concurrent requests must be in [1, {MAX_CONCURRENT_REQUESTS}]")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise SystemExit(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    if args.concurrency > args.concurrent_requests:
        raise SystemExit("concurrency must not exceed concurrent requests")
    model = args.model.strip()
    if not model or len(model) > 256:
        raise SystemExit("model identifier must contain 1 to 256 non-whitespace characters")

    try:
        base_url = normalize_base_url(args.base_url)
    except QualificationError as exc:
        raise SystemExit(str(exc)) from None

    excluded = {item.strip() for item in args.exclude.split(",") if item.strip()}
    unknown = excluded.difference(DEFAULT_TESTS)
    if unknown:
        raise SystemExit(f"unknown excluded tests: {sorted(unknown)}")

    output_path = args.output.resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    vision_bytes = synthetic_vision_page() if "vision" not in excluded else None
    vision_sha256 = sha256_bytes(vision_bytes) if vision_bytes is not None else None
    operations: dict[str, Callable[[], dict[str, Any]]] = {
        "health": lambda: request_status(
            url=endpoint(base_url.removesuffix("/v1"), "health"),
            api_key=args.api_key,
            timeout_s=args.timeout,
        ),
        "version": lambda: version_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
        ),
        "models": lambda: models_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
        ),
        "ru": lambda: language_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            prompt=("Клапан имеет рабочее давление 16 МПа. Ответь точно маркером ДАВЛЕНИЕ_16_МПа."),
            marker="ДАВЛЕНИЕ_16_МПа",
        ),
        "en": lambda: language_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            prompt=("The valve operating pressure is 16 MPa. Return exactly the marker PRESSURE_16_MPA."),
            marker="PRESSURE_16_MPA",
        ),
        "zh": lambda: language_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            prompt="设备额定电压为10千伏。请只返回标记：额定电压_10千伏",
            marker="额定电压_10千伏",
        ),
        "json_schema": lambda: json_schema_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
        ),
        "forced_tool": lambda: tool_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            automatic=False,
        ),
        "auto_tool": lambda: tool_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            automatic=True,
        ),
        "streaming": lambda: streaming_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
        ),
        "long_4k": lambda: long_context_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            target_tokens=4096,
        ),
        "long_8k": lambda: long_context_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            target_tokens=8192,
        ),
        "concurrency": lambda: concurrency_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            request_count=args.concurrent_requests,
            workers=args.concurrency,
        ),
    }
    if vision_bytes is not None and vision_sha256 is not None:
        operations["vision"] = lambda: vision_probe(
            base_url=base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            model=model,
            image_bytes=vision_bytes,
            image_sha256=vision_sha256,
        )

    contract_bytes = json.dumps(
        QUALIFICATION_CONTRACT,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    report: dict[str, Any] = {
        "schema_version": 2,
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "command": redacted_command(sys.executable, sys.argv),
        "configuration": {
            "base_url": base_url,
            "model": model,
            "timeout_s": args.timeout,
            "concurrent_requests": args.concurrent_requests,
            "concurrency": args.concurrency,
            "vision_source": ("generated-safe-synthetic-page" if vision_bytes is not None else None),
            "excluded": sorted(excluded),
            "corpus_access": False,
            "signed_corpus_gates_modified": False,
        },
        "provenance": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "contract_sha256": sha256_bytes(contract_bytes),
            "vision_image_sha256": vision_sha256,
            "endpoint_version": None,
            "served_model_id": None,
            "max_model_len": None,
        },
        "probes": [],
    }

    for name in DEFAULT_TESTS:
        if name in excluded:
            report["probes"].append({"name": name, "status": "skipped"})
            continue
        print(f"[qualification] {name} ...", flush=True)
        result = run_probe(name, operations[name])
        report["probes"].append(result)
        print(f"[qualification] {name}: {result['status']}", flush=True)

    passed_details = {
        probe["name"]: probe["details"]
        for probe in report["probes"]
        if probe.get("status") == "pass" and isinstance(probe.get("details"), dict)
    }
    version_details = passed_details.get("version", {})
    model_details = passed_details.get("models", {})
    report["provenance"].update(
        {
            "endpoint_version": version_details.get("version"),
            "served_model_id": model_details.get("model_id"),
            "max_model_len": model_details.get("max_model_len"),
        }
    )

    failed = [probe["name"] for probe in report["probes"] if probe["status"] == "fail"]
    passed = [probe["name"] for probe in report["probes"] if probe["status"] == "pass"]
    skipped = [probe["name"] for probe in report["probes"] if probe["status"] == "skipped"]
    overall_status = "fail" if failed else "incomplete" if skipped else "pass"
    report["finished_at"] = utc_now()
    report["summary"] = {
        "status": overall_status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
    }
    try:
        report_sha256 = write_report_create_only(output_path, report)
    except QualificationError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    print(f"report: {output_path} sha256={report_sha256}", flush=True)
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
