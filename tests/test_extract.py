"""Spec-интенты / BabelDOC-глоссарий: чистые проверки без сети/БД."""

from __future__ import annotations

from pathlib import Path

from rag_app.pipeline.babeldoc import write_glossary_csv


def test_glossary_csv_format(tmp_path: Path) -> None:
    p = write_glossary_csv(
        [("pressure vessel", "сосуд под давлением"), ("", ""), ("x", "")],
        tmp_path / "g.csv",
    )
    assert p is not None
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source,target,tgt_lng"
    assert lines[1] == "pressure vessel,сосуд под давлением,ru"
    assert len(lines) == 2  # пустые/неполные пары отфильтрованы


def test_glossary_csv_empty_returns_none(tmp_path: Path) -> None:
    assert write_glossary_csv([], tmp_path / "g.csv") is None
