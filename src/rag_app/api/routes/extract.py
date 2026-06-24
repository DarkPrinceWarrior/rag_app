"""Спец-интенты § 5 п.6: экстракция таблиц + экспорт XLSX.

POST /api/extract/table — запрос → таблица {title, columns, rows, sources}
(structured output). POST /api/extract/xlsx — та же таблица → файл .xlsx
(openpyxl, потоком; без хранения — stateless).
"""

from __future__ import annotations

import io
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from openpyxl import Workbook
from pydantic import BaseModel

from rag_app.api.auth import require_user
from rag_app.rag.extract import extract_table

router = APIRouter(prefix="/api/extract", tags=["extract"], dependencies=[require_user])

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ExtractIn(BaseModel):
    query: str
    document_id: uuid.UUID | None = None
    folder_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = None


@router.post("/table")
async def extract_table_route(request: Request, body: ExtractIn) -> dict:
    if not body.query.strip():
        raise HTTPException(422, "пустой запрос")
    async with request.app.state.sessionmaker() as db:
        return await extract_table(
            request.app.state.chat_engine.client,
            request.app.state.retriever,
            db,
            body.query,
            document_id=body.document_id,
            folder_id=body.folder_id,
            document_ids=body.document_ids or None,
        )


class XlsxIn(BaseModel):
    title: str = "Таблица"
    columns: list[str] = []
    rows: list[list] = []
    sources: list[dict] | None = None


@router.post("/xlsx")
async def extract_xlsx(body: XlsxIn) -> StreamingResponse:
    wb = Workbook()
    ws = wb.active
    ws.title = "Спецификации"
    if body.columns:
        ws.append(body.columns)
    for row in body.rows:
        ws.append([str(x) for x in row])
    if body.sources:
        s2 = wb.create_sheet("Источники")
        s2.append(["#", "Файл", "Раздел", "Стр."])
        for src in body.sources:
            s2.append([src.get("n"), src.get("filename"), src.get("heading_path"), src.get("page")])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": 'attachment; filename="extract.xlsx"'},
    )
