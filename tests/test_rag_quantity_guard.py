from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from rag_app.api.routes import chat as chat_route
from rag_app.config import Settings
from rag_app.rag.digest import session_digest
from rag_app.rag.quantity_guard import (
    QUANTITY_WARNING_MARKDOWN,
    QUANTITY_WARNING_MARKER,
    RAG_QUANTITY_GUARD_ERRORS,
    RAG_QUANTITY_MENTIONS,
    RAG_QUANTITY_UNSUPPORTED,
    annotate_quantity_warning,
    evaluate_quantity_support,
    private_quantity_guard_artifact,
    quantity_warning_markdown,
    record_quantity_guard_metrics,
    strip_quantity_warning,
)
from rag_app.rag.retrieve import RetrievedChunk


def _chunk(text: str, *, lang: str = "ru", kind: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="synthetic.txt",
        heading_path="",
        kind=kind,
        page_start=0,
        page_end=0,
        text_en=text if lang == "en" else "",
        text_ru=text if lang != "en" else "",
        meta={},
    )


def test_quantity_guard_normalizes_ru_en_zh_signed_decimal_and_citations() -> None:
    chunks = [
        _chunk("Расчётное давление составляет -16,5 МПа."),
        _chunk("The design temperature is 40 °C.", lang="en"),
        _chunk("管道长度为 120 米。"),
    ]

    result = evaluate_quantity_support(
        "Давление -16.5 MPa [1], температура 40 °C [2], 长度 120 米 [3].",
        chunks,
    )

    assert result == {
        "mentioned_count": 3,
        "supported_count": 3,
        "unsupported_count": 0,
        "unsupported_pair_count": 0,
        "unsupported_value_count": 0,
        "invalid_unit_count": 0,
        "unsupported_rate": 0.0,
    }


def test_quantity_guard_counts_unsupported_pair_and_value_without_raw_payload() -> None:
    result = evaluate_quantity_support(
        "Давление 16,5 bar, температура 55 °C [2].",
        [_chunk("Давление 16,5 МПа, температура 40 °C.")],
    )
    artifact = private_quantity_guard_artifact("gold-0236", result)

    assert result["mentioned_count"] == 2
    assert result["supported_count"] == 0
    assert result["unsupported_pair_count"] == 2
    assert result["unsupported_value_count"] == 1
    assert result["unsupported_rate"] == 1.0
    assert artifact["schema_version"] == "rag-quantity-guard/v1"
    assert set(artifact) == {
        "schema_version",
        "case_id",
        "mentioned_count",
        "supported_count",
        "unsupported_count",
        "unsupported_pair_count",
        "unsupported_value_count",
        "invalid_unit_count",
        "unsupported_rate",
    }
    assert "answer" not in artifact
    assert "text" not in artifact


def test_quantity_guard_ignores_catalog_pseudo_chunk() -> None:
    result = evaluate_quantity_support(
        "Рабочее давление 25 MPa.",
        [_chunk("catalog item: 25 MPa", lang="en", kind="catalog")],
    )

    assert result["unsupported_count"] == 1
    assert result["unsupported_value_count"] == 1


def test_quantity_guard_uses_both_bilingual_chunk_fields() -> None:
    chunk = _chunk("Перевод без величины")
    chunk.text_en = "The source pressure is 16.5 MPa."

    result = evaluate_quantity_support("Рабочее давление 16,5 МПа.", [chunk])

    assert result["unsupported_count"] == 0


def test_private_quantity_guard_artifact_rejects_empty_case_id() -> None:
    with pytest.raises(ValueError, match="case_id"):
        private_quantity_guard_artifact(
            " ",
            evaluate_quantity_support("No quantities.", []),
        )


def test_quantity_guard_config_allows_warning_but_not_enforce_mode() -> None:
    assert Settings(rag_quantity_guard_mode="shadow").rag_quantity_guard_mode == "shadow"
    assert Settings(rag_quantity_guard_mode="warn").rag_quantity_guard_mode == "warn"
    with pytest.raises(ValidationError):
        Settings.model_validate({"rag_quantity_guard_mode": "enforce"})


def test_quantity_guard_is_fail_open_and_logs_no_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    before = RAG_QUANTITY_GUARD_ERRORS.labels("warn")._value.get()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("private answer payload")

    monkeypatch.setattr(chat_route, "evaluate_quantity_support", fail)

    result = chat_route._evaluate_quantity_guard(
        "private answer", [_chunk("private evidence")], mode="warn"
    )

    assert result is None
    assert "RuntimeError" in caplog.text
    assert "private answer" not in caplog.text
    assert "private evidence" not in caplog.text
    assert RAG_QUANTITY_GUARD_ERRORS.labels("warn")._value.get() == before + 1
    assert chat_route._apply_quantity_guard(
        "private answer", [_chunk("private evidence")], mode="warn"
    ) == ("private answer", "")


def test_quantity_warning_is_user_visible_only_for_absent_values() -> None:
    unsupported = evaluate_quantity_support(
        "Рабочее давление 50 МПа.",
        [_chunk("Рабочее давление 5 МПа.")],
    )
    supported = evaluate_quantity_support(
        "Рабочее давление 5 МПа.",
        [_chunk("Рабочее давление 5 МПа.")],
    )

    assert quantity_warning_markdown(unsupported) == QUANTITY_WARNING_MARKDOWN
    assert quantity_warning_markdown(supported) == ""
    annotated = annotate_quantity_warning("Ответ с незакрытым блоком ```", unsupported)
    assert annotated == f"Ответ с незакрытым блоком ```{QUANTITY_WARNING_MARKDOWN}"
    assert annotate_quantity_warning(annotated, unsupported) == annotated
    assert strip_quantity_warning(annotated) == "Ответ с незакрытым блоком ```"
    assert strip_quantity_warning(f"{annotated} ") == f"{annotated} "
    assert strip_quantity_warning(f"prefix {QUANTITY_WARNING_MARKER} suffix") == (
        f"prefix {QUANTITY_WARNING_MARKER} suffix"
    )
    assert QUANTITY_WARNING_MARKER in annotated
    assert not any(character.isdigit() for character in QUANTITY_WARNING_MARKDOWN)


def test_pair_only_mismatch_is_measured_without_user_warning() -> None:
    answer = "Рабочее давление 16,5 bar."
    chunks = [_chunk("Рабочее давление 16,5 МПа.")]

    result = evaluate_quantity_support(answer, chunks)

    assert result["unsupported_pair_count"] == 1
    assert result["unsupported_value_count"] == 0
    assert quantity_warning_markdown(result) == ""
    assert chat_route._apply_quantity_guard(answer, chunks, mode="warn") == (answer, "")


def test_quantity_guard_application_preserves_model_answer_and_modes() -> None:
    answer = "Рабочее давление 50 МПа [1]."
    chunks = [_chunk("Рабочее давление 5 МПа.")]

    assert chat_route._apply_quantity_guard(answer, chunks, mode="off") == (answer, "")
    assert chat_route._apply_quantity_guard(answer, chunks, mode="shadow") == (answer, "")
    annotated, suffix = chat_route._apply_quantity_guard(answer, chunks, mode="warn")
    assert annotated == answer + QUANTITY_WARNING_MARKDOWN
    assert suffix == QUANTITY_WARNING_MARKDOWN
    assert chat_route._apply_quantity_guard(annotated, chunks, mode="warn") == (annotated, "")
    model_citations = chat_route.extract_citations(answer, chunks)
    assert len(model_citations) == 1
    assert chat_route.extract_citations(annotated, chunks) == model_citations


def test_quantity_guard_metrics_failure_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Рабочее давление 50 МПа."
    before = RAG_QUANTITY_GUARD_ERRORS.labels("warn")._value.get()

    def fail(_result: object) -> None:
        raise RuntimeError("metrics unavailable")

    monkeypatch.setattr(chat_route, "record_quantity_guard_metrics", fail)
    assert chat_route._apply_quantity_guard(answer, [_chunk("5 МПа")], mode="warn") == (answer, "")
    assert RAG_QUANTITY_GUARD_ERRORS.labels("warn")._value.get() == before + 1


@pytest.mark.asyncio
async def test_session_digest_strips_warning_only_from_summary_input() -> None:
    model_answer = "Рабочее давление 50 МПа [1]."
    annotated = f"{model_answer}{QUANTITY_WARNING_MARKDOWN}"
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="- Краткая выжимка"))]
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    messages = [
        SimpleNamespace(role="user", content="Какое давление?", citations=[]),
        SimpleNamespace(
            role="assistant",
            content=annotated,
            citations=[{"document_id": "doc-1", "page_start": 0, "heading_path": ""}],
        ),
    ]

    digest = await session_digest(client, "Синтетический чат", messages)

    call = create.await_args
    assert call is not None
    summary_prompt = call.kwargs["messages"][0]["content"]
    assert model_answer in summary_prompt
    assert QUANTITY_WARNING_MARKER not in summary_prompt
    assert digest["messages"][1]["content"] == annotated
    assert QUANTITY_WARNING_MARKER in digest["messages"][1]["content"]
    assert digest["sources"] == [messages[1].citations[0]]


def test_quantity_guard_publishes_aggregate_only_prometheus_counters() -> None:
    before_mentions = RAG_QUANTITY_MENTIONS._value.get()
    before_pairs = RAG_QUANTITY_UNSUPPORTED.labels("pair")._value.get()
    before_values = RAG_QUANTITY_UNSUPPORTED.labels("value")._value.get()
    result = evaluate_quantity_support(
        "Давление 16,5 bar, температура 55 °C.",
        [_chunk("Давление 16,5 МПа, температура 40 °C.")],
    )

    record_quantity_guard_metrics(result)

    assert RAG_QUANTITY_MENTIONS._value.get() == before_mentions + 2
    assert RAG_QUANTITY_UNSUPPORTED.labels("pair")._value.get() == before_pairs + 2
    assert RAG_QUANTITY_UNSUPPORTED.labels("value")._value.get() == before_values + 1
