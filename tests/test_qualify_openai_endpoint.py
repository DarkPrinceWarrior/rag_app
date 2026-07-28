from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "qualify_openai_endpoint.py"
_SPEC = importlib.util.spec_from_file_location("qualify_openai_endpoint", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)
assert isinstance(qualification, ModuleType)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:8006/v1/", "http://127.0.0.1:8006/v1"),
        ("http://localhost:18016/v1", "http://localhost:18016/v1"),
        ("https://[::1]:18017/v1", "https://[::1]:18017/v1"),
    ],
)
def test_normalize_base_url_accepts_only_loopback_v1(value: str, expected: str) -> None:
    assert qualification.normalize_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://models.example/v1",
        "http://token@127.0.0.1:8006/v1",
        "http://127.0.0.1:8006/v1?token=secret",
        "http://127.0.0.1:8006/health",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1:8006/v1/extra",
        "file:///tmp/model",
    ],
)
def test_normalize_base_url_rejects_external_or_ambiguous_urls(value: str) -> None:
    with pytest.raises(qualification.QualificationError, match="loopback|port|/v1"):
        qualification.normalize_base_url(value)


def test_parser_rejects_abbreviated_api_key_option() -> None:
    parser = qualification.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--base-url",
                "http://127.0.0.1:8006/v1",
                "--model",
                "candidate",
                "--api-k",
                "secret",
                "--output",
                "/tmp/report.json",
            ]
        )


def test_redacted_command_never_records_api_key() -> None:
    command = qualification.redacted_command(
        "/usr/bin/python",
        [
            "qualify.py",
            "--api-key",
            "first-secret",
            "--model",
            "candidate",
            "--api-key=second-secret",
        ],
    )

    assert "first-secret" not in command
    assert "second-secret" not in command
    assert command.count("<redacted>") == 2


def test_private_report_writer_is_atomic_create_only_and_mode_0600(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    digest = qualification.write_report_create_only(output, {"result": "first"})

    assert json.loads(output.read_bytes()) == {"result": "first"}
    assert digest == qualification.sha256_bytes(output.read_bytes())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".private-json-*.tmp"))
    with pytest.raises(qualification.QualificationError, match="overwrite"):
        qualification.write_report_create_only(output, {"result": "second"})
    assert json.loads(output.read_bytes()) == {"result": "first"}


def test_main_rejects_existing_report_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify_openai_endpoint.py",
            "--base-url",
            "http://127.0.0.1:8006/v1",
            "--model",
            "candidate",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        qualification.main()
    assert output.read_text(encoding="utf-8") == "keep"


def test_all_excluded_is_incomplete_and_does_not_create_vision_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"

    def fail_if_called() -> bytes:
        raise AssertionError("vision asset must not be created when vision is excluded")

    monkeypatch.setattr(qualification, "synthetic_vision_page", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify_openai_endpoint.py",
            "--base-url",
            "http://127.0.0.1:8006/v1",
            "--model",
            " candidate ",
            "--output",
            str(output),
            "--exclude",
            ",".join(qualification.DEFAULT_TESTS),
        ],
    )

    assert qualification.main() == 1
    report = json.loads(output.read_bytes())
    assert report["summary"]["status"] == "incomplete"
    assert report["summary"]["passed_count"] == 0
    assert report["summary"]["skipped_count"] == len(qualification.DEFAULT_TESTS)
    assert report["configuration"]["model"] == "candidate"
    assert report["provenance"]["vision_image_sha256"] is None


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("--timeout", "nan", "timeout"),
        ("--timeout", "3601", "timeout"),
        ("--concurrent-requests", "129", "concurrent requests"),
        ("--concurrency", "33", "concurrency"),
    ],
)
def test_main_rejects_non_finite_or_unbounded_work(
    option: str,
    value: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify_openai_endpoint.py",
            "--base-url",
            "http://127.0.0.1:8006/v1",
            "--model",
            "candidate",
            "--output",
            str(tmp_path / "report.json"),
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit, match=error):
        qualification.main()


def test_request_json_ignores_proxy_environment_and_refuses_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.Client
    captured_options: dict[str, Any] = {}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": "http://127.0.0.1:18018/v1/models"},
        )

    def client_factory(**kwargs: Any) -> httpx.Client:
        captured_options.update(kwargs)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(qualification.httpx, "Client", client_factory)
    with pytest.raises(qualification.QualificationError, match="redirect refused"):
        qualification.request_json(
            url="http://127.0.0.1:18017/v1/models",
            api_key="top-secret",
            timeout_s=1,
        )

    assert captured_options["trust_env"] is False
    assert captured_options["follow_redirects"] is False
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer top-secret"


def test_request_json_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            content=b"x" * (qualification.MAX_JSON_RESPONSE_BYTES + 1),
        )

    def client_factory(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(qualification.httpx, "Client", client_factory)
    with pytest.raises(qualification.QualificationError, match="exceeds"):
        qualification.request_json(
            url="http://127.0.0.1:18017/v1/models",
            api_key="local",
            timeout_s=1,
        )


def test_request_json_enforces_absolute_wall_deadline_between_drip_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.Client
    clock_values = iter((0.0, 0.5, 1.1))

    class DripStream(httpx.SyncByteStream):
        def __iter__(self) -> Any:
            yield b'{"result":'
            yield b'"late"}'

    def monotonic() -> float:
        return next(clock_values, 2.0)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=DripStream())

    def client_factory(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(qualification.time, "monotonic", monotonic)
    monkeypatch.setattr(qualification.httpx, "Client", client_factory)
    with pytest.raises(qualification.QualificationError, match="wall deadline"):
        qualification.request_json(
            url="http://127.0.0.1:18017/v1/models",
            api_key="local",
            timeout_s=1,
        )


def test_models_probe_requires_exact_model_and_max_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_json(**kwargs: Any) -> tuple[int, dict[str, Any], str]:
        del kwargs
        return (
            200,
            {
                "data": [
                    {"id": "other", "max_model_len": 1},
                    {"id": "candidate", "max_model_len": 16384},
                ]
            },
            "",
        )

    monkeypatch.setattr(qualification, "request_json", fake_request_json)
    result = qualification.models_probe(
        base_url="http://127.0.0.1:18017/v1",
        api_key="local",
        timeout_s=1,
        model="candidate",
    )
    assert result["model_id"] == "candidate"
    assert result["max_model_len"] == 16384


def _tool_response(calls: list[dict[str, Any]]) -> tuple[int, dict[str, Any], str]:
    return (
        200,
        {
            "model": "candidate",
            "choices": [{"message": {"tool_calls": calls}}],
        },
        "",
    )


def _valid_tool_call() -> dict[str, Any]:
    return {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "lookup_document",
            "arguments": '{"document_id":"DOC-42"}',
        },
    }


def test_tool_probe_accepts_one_strict_wire_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "request_json",
        lambda **kwargs: _tool_response([_valid_tool_call()]),
    )

    result = qualification.tool_probe(
        base_url="http://127.0.0.1:18017/v1",
        api_key="local",
        timeout_s=1,
        model="candidate",
        automatic=True,
    )

    assert result["tool_call_id"] == "call-1"
    assert result["arguments"] == {"document_id": "DOC-42"}


@pytest.mark.parametrize(
    "calls",
    [
        [_valid_tool_call(), _valid_tool_call()],
        [{**_valid_tool_call(), "type": "custom"}],
        [{**_valid_tool_call(), "id": ""}],
        [{**_valid_tool_call(), "id": "x" * 257}],
        [
            {
                **_valid_tool_call(),
                "function": {
                    "name": "lookup_document",
                    "arguments": {"document_id": "DOC-42"},
                },
            }
        ],
        [
            {
                **_valid_tool_call(),
                "function": {
                    "name": "lookup_document",
                    "arguments": '{"document_id":"DOC-42","extra":true}',
                },
            }
        ],
        [
            {
                **_valid_tool_call(),
                "function": {
                    "name": "lookup_document",
                    "arguments": '{"document_id":"DOC-42","document_id":"DOC-42"}',
                },
            }
        ],
    ],
    ids=[
        "second-call",
        "wrong-type",
        "empty-id",
        "oversized-id",
        "non-string-wire-arguments",
        "extra-argument",
        "duplicate-key",
    ],
)
def test_tool_probe_rejects_noncanonical_calls(
    calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "request_json",
        lambda **kwargs: _tool_response(calls),
    )

    with pytest.raises(qualification.QualificationError):
        qualification.tool_probe(
            base_url="http://127.0.0.1:18017/v1",
            api_key="local",
            timeout_s=1,
            model="candidate",
            automatic=False,
        )


def test_language_probe_requires_exact_marker_and_served_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_json(**kwargs: Any) -> tuple[int, dict[str, Any], str]:
        del kwargs
        return (
            200,
            {
                "model": "candidate",
                "choices": [{"message": {"content": "MARKER extra"}}],
            },
            "",
        )

    monkeypatch.setattr(qualification, "request_json", fake_request_json)
    with pytest.raises(qualification.QualificationError, match="expected marker"):
        qualification.language_probe(
            base_url="http://127.0.0.1:18017/v1",
            api_key="local",
            timeout_s=1,
            model="candidate",
            prompt="prompt",
            marker="MARKER",
        )


def test_long_context_probe_requires_reported_prompt_token_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_json(**kwargs: Any) -> tuple[int, dict[str, Any], str]:
        payload = kwargs["payload"]
        marker = "LONG_OK_4096"
        assert marker in payload["messages"][0]["content"]
        return (
            200,
            {
                "model": "candidate",
                "choices": [{"message": {"content": marker}}],
                "usage": {"prompt_tokens": 4095},
            },
            "",
        )

    monkeypatch.setattr(qualification, "request_json", fake_request_json)
    with pytest.raises(qualification.QualificationError, match="prompt_tokens"):
        qualification.long_context_probe(
            base_url="http://127.0.0.1:18017/v1",
            api_key="local",
            timeout_s=1,
            model="candidate",
            target_tokens=4096,
        )


def test_long_context_marker_occurs_once_at_head_and_tail_only_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompt = ""

    def fake_request_json(**kwargs: Any) -> tuple[int, dict[str, Any], str]:
        nonlocal captured_prompt
        captured_prompt = kwargs["payload"]["messages"][0]["content"]
        marker = "LONG_OK_4096"
        tail_only_answer = marker if marker in captured_prompt[-512:] else "TAIL_ONLY_MISS"
        return (
            200,
            {
                "model": "candidate",
                "choices": [{"message": {"content": tail_only_answer}}],
                "usage": {"prompt_tokens": 4096},
            },
            "",
        )

    monkeypatch.setattr(qualification, "request_json", fake_request_json)
    with pytest.raises(qualification.QualificationError, match="marker mismatch"):
        qualification.long_context_probe(
            base_url="http://127.0.0.1:18017/v1",
            api_key="local",
            timeout_s=1,
            model="candidate",
            target_tokens=4096,
        )

    assert captured_prompt.startswith("LONG_OK_4096\n")
    assert captured_prompt.count("LONG_OK_4096") == 1


@pytest.mark.parametrize(
    "answer",
    ["blue red", "red blue extra", "not red blue", "red and blue"],
)
def test_vision_probe_requires_exact_normalized_red_blue(
    answer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request_json(**kwargs: Any) -> tuple[int, dict[str, Any], str]:
        del kwargs
        return (
            200,
            {
                "model": "candidate",
                "choices": [{"message": {"content": answer}}],
            },
            "",
        )

    monkeypatch.setattr(qualification, "request_json", fake_request_json)
    image = qualification.synthetic_vision_page()
    with pytest.raises(qualification.QualificationError, match="exactly 'red blue'"):
        qualification.vision_probe(
            base_url="http://127.0.0.1:18017/v1",
            api_key="local",
            timeout_s=1,
            model="candidate",
            image_bytes=image,
            image_sha256=qualification.sha256_bytes(image),
        )


def test_streaming_probe_is_bounded_and_requires_exact_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.Client
    sse = b'data: {"model":"candidate","choices":[{"delta":{"content":"STREAM_OK"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=sse)

    def client_factory(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(qualification.httpx, "Client", client_factory)
    result = qualification.streaming_probe(
        base_url="http://127.0.0.1:18017/v1",
        api_key="local",
        timeout_s=1,
        model="candidate",
    )
    assert result["content"] == "STREAM_OK"
    assert result["response_bytes"] == len(sse)


def test_streaming_probe_enforces_absolute_wall_deadline_between_drip_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.Client
    clock_values = iter((0.0, 0.5, 1.1))

    class DripStream(httpx.SyncByteStream):
        def __iter__(self) -> Any:
            yield b'data: {"model":"candidate","choices":[{"delta":'
            yield b'{"content":"STREAM_OK"}}]}\n\ndata: [DONE]\n\n'

    def monotonic() -> float:
        return next(clock_values, 2.0)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=DripStream())

    def client_factory(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(qualification.time, "monotonic", monotonic)
    monkeypatch.setattr(qualification.httpx, "Client", client_factory)
    with pytest.raises(qualification.QualificationError, match="wall deadline"):
        qualification.streaming_probe(
            base_url="http://127.0.0.1:18017/v1",
            api_key="local",
            timeout_s=1,
            model="candidate",
        )
