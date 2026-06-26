"""Библиотека: папки и гибридный поиск по чанкам (roadmap § 11 этап 3)."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

from rag_app.api.auth import User, require_user
from rag_app.config import settings
from rag_app.db.models import Document, Folder

router = APIRouter(prefix="/api", tags=["library"], dependencies=[require_user])


class FolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=256)


@router.get("/folders")
async def list_folders(request: Request) -> list[dict]:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as db:
        stmt = (
            select(Folder, func.count(Document.id))
            .outerjoin(Document, Document.folder_id == Folder.id)
            .group_by(Folder.id)
            .order_by(Folder.name)
        )
        if not user.is_admin:  # RBAC §4.7.1: свои папки + dev-папки (owner NULL)
            stmt = stmt.where((Folder.owner_sub == user.sub) | (Folder.owner_sub.is_(None)))
        rows = (await db.execute(stmt)).all()
    return [
        {"id": str(f.id), "name": f.name, "documents": count}
        for f, count in rows
    ]


@router.post("/folders", status_code=201)
async def create_folder(request: Request, body: FolderIn) -> dict:
    user: User = request.state.user
    name = body.name.strip()
    async with request.app.state.sessionmaker() as db:
        # дедуп в пределах владельца (составной ключ owner_sub+name, §4.7.1)
        existing = (
            await db.execute(
                select(Folder).where(Folder.name == name, Folder.owner_sub == user.sub)
            )
        ).scalar_one_or_none()
        if existing:
            return {"id": str(existing.id), "name": existing.name}
        folder = Folder(name=name, owner_sub=user.sub)
        db.add(folder)
        await db.commit()
        return {"id": str(folder.id), "name": folder.name}


class DocumentFolderIn(BaseModel):
    folder_id: uuid.UUID | None


@router.patch("/documents/{doc_id}/folder")
async def move_document(request: Request, doc_id: uuid.UUID, body: DocumentFolderIn) -> dict:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as db:
        doc = await db.get(Document, doc_id)
        if doc is None or (
            not user.is_admin and doc.owner_sub is not None and doc.owner_sub != user.sub
        ):
            raise HTTPException(404, "документ не найден")  # не раскрываем существование
        if body.folder_id is not None:
            folder = await db.get(Folder, body.folder_id)
            if folder is None or (
                not user.is_admin and folder.owner_sub is not None and folder.owner_sub != user.sub
            ):
                raise HTTPException(404, "папка не найдена")
        await db.execute(
            update(Document).where(Document.id == doc_id).values(folder_id=body.folder_id)
        )
        await db.commit()
    return {"status": "ok"}


@router.delete("/folders/{folder_id}", status_code=204)
async def delete_folder(request: Request, folder_id: uuid.UUID) -> None:
    """Удаление папки. Документы НЕ удаляются: их folder_id обнуляется (FK
    ondelete=SET NULL) — остаются в библиотеке без папки."""
    user: User = request.state.user
    async with request.app.state.sessionmaker() as db:
        folder = await db.get(Folder, folder_id)
        if folder is None or (
            not user.is_admin and folder.owner_sub is not None and folder.owner_sub != user.sub
        ):
            raise HTTPException(404, "папка не найдена")
        await db.delete(folder)
        await db.commit()


@router.get("/search/visual")
async def search_visual(
    request: Request,
    q: str = Query(min_length=2),
    top_k: int = Query(default=5, le=20),
) -> list[dict]:
    """Визуальный поиск по страницам сканов (§ 12.1 шаг 4): печати, штампы,
    чертежи — текстовый запрос в общем пространстве с изображениями страниц.

    Кириллический запрос прозрачно переводится на английский быстрым контуром:
    кросс-языковость VL-эмбеддера заметно слабее (0.33 против 0.56 на контроле).
    """
    # Визуальный контур запаркован (vllm-visual-embedding погашен 2026-06-18,
    # фича почти не использовалась) → не дёргаем мёртвый :8007.
    if not settings.visual_enabled:
        raise HTTPException(503, "визуальный поиск выключен (visual_enabled=false)")
    if any("а" <= ch.lower() <= "я" for ch in q):
        q, _ = await request.app.state.fast_translator.translate(q, target_lang="en")
    q_emb = await request.app.state.visual.embed_text_query(q)
    user: User = request.state.user
    owner = None if user.is_admin else user.sub  # RBAC §4.7.1
    sql_text = """
        SELECT p.document_id, d.filename, p.page_idx, 1 - (p.emb <=> CAST(:qe AS vector)) AS score
        FROM page_embeddings p JOIN documents d ON d.id = p.document_id
        WHERE (CAST(:owner AS text) IS NULL OR d.owner_sub = :owner OR d.owner_sub IS NULL)
        ORDER BY p.emb <=> CAST(:qe AS vector)
        LIMIT :k
    """
    from sqlalchemy import text as sql

    async with request.app.state.sessionmaker() as db:
        rows = (await db.execute(sql(sql_text), {"qe": str(q_emb), "owner": owner, "k": top_k})).all()
    return [
        {
            "document_id": str(r.document_id),
            "filename": r.filename,
            "page": r.page_idx + 1,
            "score": round(float(r.score), 4),
        }
        for r in rows
    ]


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(min_length=2),
    document_id: uuid.UUID | None = None,
    folder_id: uuid.UUID | None = None,
    top_k: int = Query(default=10, le=30),
    kind: str | None = Query(None, description="тип файла (ТЗ §4.7.3)"),
    date_from: str | None = Query(None, description="дата загрузки от (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="дата загрузки до (YYYY-MM-DD)"),
) -> list[dict]:
    """Единый поиск по библиотеке (ТЗ §4.7.3): гибрид + reranker по СОДЕРЖИМОМУ +
    совпадение по ИМЕНИ ФАЙЛА. Фильтры тип/дата сужают набор документов."""
    user: User = request.state.user
    owner = None if user.is_admin else user.sub
    ql = q.strip().lower()
    async with request.app.state.sessionmaker() as db:
        # набор документов под фильтры (тип/дата/папка/документ + владелец)
        dstmt = select(Document.id, Document.filename, Document.kind)
        if not user.is_admin:
            dstmt = dstmt.where((Document.owner_sub == user.sub) | (Document.owner_sub.is_(None)))
        if document_id:
            dstmt = dstmt.where(Document.id == document_id)
        if folder_id:
            dstmt = dstmt.where(Document.folder_id == folder_id)
        if kind and kind.strip():
            dstmt = dstmt.where(Document.kind == kind.strip())
        for raw, op in ((date_from, ">="), (date_to, "<=")):
            if raw:
                try:
                    d = date.fromisoformat(raw)
                except ValueError:
                    continue
                col = func.date(Document.created_at)
                dstmt = dstmt.where(col >= d if op == ">=" else col <= d)
        docrows = (await db.execute(dstmt)).all()
        narrowed = bool((kind and kind.strip()) or date_from or date_to)
        if narrowed and not docrows:
            return []  # фильтры исключили все документы
        scope_ids = [r.id for r in docrows] if narrowed else None
        # совпадение по имени файла
        fname = [r for r in docrows if ql in (r.filename or "").lower()][:10]
        # поиск по содержимому (в пределах отфильтрованного набора, если есть фильтры)
        chunks = await request.app.state.retriever.retrieve(
            db, q, document_id=document_id, folder_id=folder_id,
            document_ids=scope_ids, top_k=top_k, owner_sub=owner,
        )
    fname_ids = {str(r.id) for r in fname}
    results: list[dict] = [
        {
            "chunk_id": "",
            "document_id": str(r.id),
            "filename": r.filename,
            "heading_path": "совпадение по имени файла",
            "kind": r.kind,
            "page_start": None,
            "page_end": None,
            "snippet": "",
            "score": 1.0,
            "match": "filename",
        }
        for r in fname
    ]
    for c in chunks:
        if str(c.document_id) in fname_ids:
            continue  # документ уже показан как совпадение по имени
        results.append(
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
                "match": "content",
            }
        )
    return results
