"""Спец-интенты § 5 п.6: экстракция таблиц + экспорт XLSX.

POST /api/extract/table — запрос → таблица {title, columns, rows, sources}
(structured output). POST /api/extract/xlsx — та же таблица → файл .xlsx
(openpyxl, потоком; без хранения — stateless).
"""

from __future__ import annotations

import io
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from openpyxl import Workbook
from pydantic import BaseModel, Field, model_validator

from rag_app.api.auth import require_user
from rag_app.rag.extract import extract_table

router = APIRouter(prefix="/api/extract", tags=["extract"], dependencies=[require_user])

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ExtractIn(BaseModel):
    query: str
    document_id: uuid.UUID | None = None
    folder_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
    folder_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)

    @model_validator(mode="after")
    def validate_scope(self) -> ExtractIn:
        document_ids = self.document_ids or []
        folder_ids = self.folder_ids or []
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_ids must be unique")
        if len(folder_ids) != len(set(folder_ids)):
            raise ValueError("folder_ids must be unique")
        legacy_selected = sum((self.document_id is not None, self.folder_id is not None))
        if legacy_selected > 1 or (legacy_selected and (document_ids or folder_ids)):
            raise ValueError("extract scope fields are mutually exclusive")
        return self


@router.post("/table")
async def extract_table_route(request: Request, body: ExtractIn) -> dict:
    if not body.query.strip():
        raise HTTPException(422, "пустой запрос")
    user = request.state.user
    async with request.app.state.sessionmaker() as db:
        return await extract_table(
            request.app.state.chat_engine.client,
            request.app.state.retriever,
            db,
            body.query,
            document_id=body.document_id,
            folder_id=body.folder_id,
            document_ids=body.document_ids or None,
            folder_ids=body.folder_ids,
            owner_sub=None if user.is_admin else user.sub,
        )


class XlsxIn(BaseModel):
    title: str = "Таблица"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    sources: list[dict] | None = None


def _xlsx_safe_cell(value: Any) -> Any:
    """Force potentially executable spreadsheet values to remain literal text."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _build_xlsx(body: XlsxIn) -> io.BytesIO:
    wb = Workbook()
    ws = wb.active
    if ws is None:
        raise RuntimeError("openpyxl не создал активный лист")
    ws.title = "Спецификации"
    if body.columns:
        ws.append([_xlsx_safe_cell(value) for value in body.columns])
    for row in body.rows:
        ws.append([_xlsx_safe_cell(str(value)) for value in row])
    if body.sources:
        s2 = wb.create_sheet("Источники")
        s2.append(["#", "Файл", "Раздел", "Стр."])
        for src in body.sources:
            s2.append(
                [
                    _xlsx_safe_cell(src.get("n")),
                    _xlsx_safe_cell(src.get("filename")),
                    _xlsx_safe_cell(src.get("heading_path")),
                    _xlsx_safe_cell(src.get("page")),
                ]
            )
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


@router.post("/xlsx")
async def extract_xlsx(body: XlsxIn) -> StreamingResponse:
    return StreamingResponse(
        _build_xlsx(body),
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": 'attachment; filename="extract.xlsx"'},
    )
