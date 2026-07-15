from __future__ import annotations

import io

from openpyxl import load_workbook

from rag_app.api.routes.extract import XlsxIn, _build_xlsx, _xlsx_safe_cell


def test_xlsx_safe_cell_neutralizes_formula_prefixes() -> None:
    for value in ('=HYPERLINK("https://example.test")', "+1+1", "-2+3", "@SUM(A1:A2)"):
        assert _xlsx_safe_cell(value) == "'" + value


def test_xlsx_safe_cell_preserves_non_strings_and_plain_text() -> None:
    assert _xlsx_safe_cell("plain") == "plain"
    assert _xlsx_safe_cell(42) == 42
    assert _xlsx_safe_cell(None) is None


def test_extract_xlsx_writes_untrusted_values_as_text_on_every_sheet() -> None:
    payload = _build_xlsx(
        XlsxIn(
            columns=["=COLUMN()"],
            rows=[["=HYPERLINK(\"https://example.test\")"]],
            sources=[{"n": 1, "filename": "=WEBSERVICE(\"https://example.test\")"}],
        )
    )
    workbook = load_workbook(io.BytesIO(payload.getvalue()), data_only=False)

    assert workbook["Спецификации"]["A1"].value == "'=COLUMN()"
    assert workbook["Спецификации"]["A2"].value.startswith("'=HYPERLINK")
    assert workbook["Источники"]["B2"].value.startswith("'=WEBSERVICE")
    assert workbook["Спецификации"]["A2"].data_type == "s"
