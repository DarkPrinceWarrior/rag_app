"""Библиотека: папки и гибридный поиск по чанкам (roadmap § 11 этап 3)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

from rag_app.db.models import Document, Folder

router = APIRouter(prefix="/api", tags=["library"])


class FolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=256)


@router.get("/folders")
async def list_folders(request: Request) -> list[dict]:
    async with request.app.state.sessionmaker() as db:
        rows = (
            await db.execute(
                select(Folder, func.count(Document.id))
                .outerjoin(Document, Document.folder_id == Folder.id)
                .group_by(Folder.id)
                .order_by(Folder.name)
            )
        ).all()
    return [
        {"id": str(f.id), "name": f.name, "documents": count}
        for f, count in rows
    ]


@router.post("/folders", status_code=201)
async def create_folder(request: Request, body: FolderIn) -> dict:
    async with request.app.state.sessionmaker() as db:
        existing = (
            await db.execute(select(Folder).where(Folder.name == body.name.strip()))
        ).scalar_one_or_none()
        if existing:
            return {"id": str(existing.id), "name": existing.name}
        folder = Folder(name=body.name.strip())
        db.add(folder)
        await db.commit()
        return {"id": str(folder.id), "name": folder.name}


class DocumentFolderIn(BaseModel):
    folder_id: uuid.UUID | None


@router.patch("/documents/{doc_id}/folder")
async def move_document(request: Request, doc_id: uuid.UUID, body: DocumentFolderIn) -> dict:
    async with request.app.state.sessionmaker() as db:
        doc = await db.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, "документ не найден")
        if body.folder_id is not None and await db.get(Folder, body.folder_id) is None:
            raise HTTPException(404, "папка не найдена")
        await db.execute(
            update(Document).where(Document.id == doc_id).values(folder_id=body.folder_id)
        )
        await db.commit()
    return {"status": "ok"}


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(min_length=2),
    document_id: uuid.UUID | None = None,
    folder_id: uuid.UUID | None = None,
    top_k: int = Query(default=10, le=30),
) -> list[dict]:
    """Поиск по библиотеке: гибрид + reranker, сниппеты с привязкой к документу."""
    async with request.app.state.sessionmaker() as db:
        chunks = await request.app.state.retriever.retrieve(
            db, q, document_id=document_id, folder_id=folder_id, top_k=top_k
        )
    return [
        {
            "chunk_id": str(c.id),
            "document_id": str(c.document_id),
            "filename": c.filename,
            "heading_path": c.heading_path,
            "kind": c.kind,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "snippet": (c.text_ru or c.text_en)[:400],
            "score": round(c.score, 4),
        }
        for c in chunks
    ]
