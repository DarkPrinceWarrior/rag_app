from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rag_app.eval.qualification_evidence import RestoredModelWeightManifest


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_rag_model_qualification.py"
    spec = importlib.util.spec_from_file_location("run_rag_model_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SCRIPT = _script_module()


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _record(
    index: int,
    *,
    challenge_tags: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        case_id=f"ragq-qualification-{index:04d}",
        language=("ru", "en", "zh")[index % 3],
        question=f"Question {index}?",
        answerable=True,
        reference_answer=f"Answer {index}.",
        challenge_tags=challenge_tags,
    )


def _sidecar(index: int) -> Any:
    evidence = SimpleNamespace(exact_quote=f"Evidence {index}.")
    return SimpleNamespace(exact_evidence=(evidence,))


def _completion(content: str, *, prompt_tokens: int = 10) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt_tokens},
        },
    )


def test_endpoint_validation_is_loopback_only() -> None:
    endpoint = _SCRIPT.OpenAIEndpoint.validated(
        "http://127.0.0.1:8006/v1",
        "baseline",
        name="baseline",
    )
    assert endpoint.api_url("models") == "http://127.0.0.1:8006/v1/models"
    assert endpoint.server_url("metrics") == "http://127.0.0.1:8006/metrics"
    with pytest.raises(_SCRIPT.QualificationProducerError, match="loopback"):
        _SCRIPT.OpenAIEndpoint.validated(
            "https://models.example.test/v1",
            "candidate",
            name="candidate",
        )
    with pytest.raises(_SCRIPT.QualificationProducerError, match="end with /v1"):
        _SCRIPT.OpenAIEndpoint.validated(
            "http://127.0.0.1:8007/api",
            "candidate",
            name="candidate",
        )


def test_long_context_collects_exactly_thirty_multilingual_cases() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 14_000, "max_model_len": 16_384})
        payload = json.loads(request.content)
        text = payload["messages"][-1]["content"]
        if "只回答准备完毕" in text:
            answer = "准备完毕"
        elif "ГОТОВО" in text:
            answer = "ГОТОВО"
        else:
            answer = "READY"
        return _completion(answer, prompt_tokens=14_000)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _SCRIPT.collect_long_context_observations(
                client,
                _SCRIPT.OpenAIEndpoint("http://127.0.0.1:8007/v1", "candidate"),
                context_window_tokens=16_384,
            )

    observations = asyncio.run(run())
    assert len(observations) == 30
    assert Counter(item.language for item in observations) == {"ru": 10, "en": 10, "zh": 10}
    assert {item.outcome for item in observations} == {"completed"}
    assert all(13_927 <= item.input_tokens <= 15_564 for item in observations)


def test_paired_load_is_exact_seeded_alternating_and_concurrency_ten() -> None:
    records = [_record(index) for index in range(201)]
    sidecars = {record.case_id: _sidecar(index) for index, record in enumerate(records)}
    active = Counter()
    peak = Counter()
    calls: dict[str, list[tuple[int, int]]] = defaultdict(list)

    async def handler(request: httpx.Request) -> httpx.Response:
        port = request.url.port or 0
        if request.url.path == "/metrics":
            return httpx.Response(200, text="process_start_time_seconds 12345\n")
        payload = json.loads(request.content)
        question = json.loads(payload["messages"][-1]["content"])["question"]
        active[port] += 1
        peak[port] = max(peak[port], active[port])
        calls[question].append((port, payload["seed"]))
        await asyncio.sleep(0.001)
        active[port] -= 1
        return _completion(f"answer-{port}")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _SCRIPT.collect_paired_answers(
                client,
                _SCRIPT.OpenAIEndpoint("http://127.0.0.1:8101/v1", "baseline"),
                _SCRIPT.OpenAIEndpoint("http://127.0.0.1:8102/v1", "candidate"),
                records,
                sidecars,
            )

    load, baseline_answers, candidate_answers = asyncio.run(run())
    assert load.concurrency == 10
    assert len(load.requests) == 200
    assert len(baseline_answers) == len(candidate_answers) == 201
    assert peak == {8101: 10, 8102: 10}
    assert load.runtime_events == ()
    first_ports = set()
    for record in records:
        case_calls = calls[record.question]
        assert len(case_calls) == 2
        assert case_calls[0][1] == case_calls[1][1] == _SCRIPT._case_seed(record.case_id)
        expected_first = 8101 if int(_sha(record.case_id)[-1], 16) % 2 == 0 else 8102
        assert case_calls[0][0] == expected_first
        first_ports.add(case_calls[0][0])
    assert first_ports == {8101, 8102}


def test_semantic_judge_covers_every_gold_case_and_required_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record(0, challenge_tags=("prompt_injection",)),
        _record(1, challenge_tags=("standards",)),
        _record(2),
    ]
    sidecars = {record.case_id: _sidecar(index) for index, record in enumerate(records)}
    baseline_answers = {
        record.case_id: _SCRIPT.ChatResult("baseline", _sha("baseline"), None, 10) for record in records
    }
    candidate_answers = {
        record.case_id: _SCRIPT.ChatResult("candidate", _sha("candidate"), None, 10) for record in records
    }
    monkeypatch.setattr(_SCRIPT, "gold_record_case_sha256", lambda record: _sha(record.case_id))

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        return _completion(json.dumps({"verdict": "pass", "reason_codes": []}))

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _SCRIPT.collect_semantic_safety_observations(
                client,
                _SCRIPT.OpenAIEndpoint("http://127.0.0.1:8103/v1", "judge"),
                b"judge prompt",
                records,
                sidecars,
                baseline_answers,
                candidate_answers,
            )

    observations = asyncio.run(run())
    assert {item.case_id for item in observations} == {record.case_id for record in records}
    assert observations[0].categories == ("semantic", "safety")
    assert observations[1].categories == ("semantic", "standards")
    assert observations[2].categories == ("semantic",)
    assert all(item.baseline.verdict == item.candidate.verdict == "pass" for item in observations)


def test_rollback_uses_no_shell_and_captures_complete_actual_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_sha = "1" * 64
    weight_sha = "2" * 64
    rag_sha = "3" * 64
    corpus_sha = "4" * 64
    git_sha = "5" * 40
    identity = _SCRIPT.ModelIdentity(
        root=tmp_path,
        weight_manifest_sha256=weight_sha,
        config_sha256=config_sha,
        declared_revision="revision",
        weight_file_count=1,
        weight_bytes=1024,
    )
    target = _SCRIPT.RollbackModelTarget(
        role="llm",
        endpoint=_SCRIPT.OpenAIEndpoint("http://127.0.0.1:8101/v1", "baseline"),
        identity=identity,
        expected_process_sha256="6" * 64,
    )
    records = [_record(index) for index in range(10)]
    sidecars = {record.case_id: _sidecar(index) for index, record in enumerate(records)}
    provenance = SimpleNamespace(
        configuration_sha256=rag_sha,
        runtime_corpus_snapshot_sha256=corpus_sha,
    )
    baseline_report = SimpleNamespace(provenance=provenance)
    subprocess_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        subprocess_calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"ok", stderr=b"")

    async def fake_verify(*args: Any, **kwargs: Any) -> None:
        return None

    async def fake_measure(*args: Any, **kwargs: Any) -> tuple[str, str]:
        return rag_sha, corpus_sha

    monkeypatch.setattr(_SCRIPT.subprocess, "run", fake_run)
    monkeypatch.setattr(_SCRIPT, "inspect_model_root", lambda path: identity)
    monkeypatch.setattr(_SCRIPT, "_clean_repository_sha", lambda path: git_sha)
    monkeypatch.setattr(_SCRIPT, "_verify_served_model", fake_verify)
    monkeypatch.setattr(_SCRIPT, "_measure_runtime_rag_state", fake_measure)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/documents":
            return httpx.Response(401, content=b"protected")
        if request.url.path == "/api/config":
            return httpx.Response(200, json={"auth_enabled": True})
        if request.url.path == "/v1/chat/completions":
            return _completion("restored")
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "baseline", "root": str(tmp_path)}]})
        return httpx.Response(200, content=b"ok")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await _SCRIPT.collect_rollback_evidence(
                client,
                command=("/trusted/rollback", "--restore"),
                command_sha256="7" * 64,
                command_timeout_seconds=60,
                app_base_url="http://127.0.0.1:8100",
                smoke_records=records,
                sidecars=sidecars,
                records=records,
                baseline_report=baseline_report,
                reference_report_sha256="8" * 64,
                expected_git_sha=git_sha,
                targets=(target,),
                deployment_root=tmp_path,
            )

    evidence = asyncio.run(run())
    assert len(subprocess_calls) == 1
    command, kwargs = subprocess_calls[0]
    assert command == ["/trusted/rollback", "--restore"]
    assert kwargs["shell"] is False
    assert [item.kind for item in evidence.trace] == [
        "rollback_started",
        "config_restored",
        "code_restored",
        "services_restarted",
        "verification_started",
        "rollback_completed",
    ]
    assert all(item.success for item in evidence.trace)
    assert len(evidence.smoke) == 10
    assert all(item.passed for item in evidence.smoke)
    assert evidence.restored_git_sha == git_sha
    assert evidence.restored_configuration_sha256 == config_sha
    assert evidence.restored_rag_configuration_sha256 == rag_sha
    assert evidence.restored_runtime_corpus_snapshot_sha256 == corpus_sha
    assert evidence.restored_model_weight_manifests == (
        RestoredModelWeightManifest(role="llm", weight_manifest_sha256=weight_sha),
    )


def test_cli_argument_errors_use_exit_64() -> None:
    with pytest.raises(SystemExit) as error:
        _SCRIPT.main([])
    assert error.value.code == 64
