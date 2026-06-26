"""RAG-чат: SSE-стрим ответа с цитатами, сессии и история (roadmap § 5)."""

from __future__ import annotations

import io
import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy import text as sql

from rag_app.api.audit import audit
from rag_app.api.auth import User, require_user
from rag_app.config import settings
from rag_app.db.models import ChatMessage, ChatSession
from rag_app.rag.memory.rls import apply_scope_guc
from rag_app.observability import log_agent_trace, log_chat_trace
from rag_app.rag.agent import AgentLoop, classify
from rag_app.rag.chat import extract_citations, make_session_title
from rag_app.rag.digest import render_docx, render_md, session_digest
from rag_app.rag.tools import AgentTools

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[require_user])


class ChatIn(BaseModel):
    message: str
    session_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    folder_id: uuid.UUID | None = None
    # произвольный набор документов (мульти-выбор в чате); область запроса —
    # шлётся каждым ходом, в сессии не персистится (активный чат держит фронт)
    document_ids: list[uuid.UUID] | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _owner_ok(session: ChatSession, user: User) -> bool:
    """RBAC: admin — любой чат; user — свои + сессии dev-периода (owner NULL)."""
    return user.is_admin or session.owner_sub is None or session.owner_sub == user.sub


@router.post("")
async def chat(request: Request, body: ChatIn, memory: bool = True) -> StreamingResponse:
    if not body.message.strip():
        raise HTTPException(422, "пустой вопрос")
    app = request.app
    user: User = request.state.user
    # временный чат (?memory=off): не пишем/не читаем память (§9, Этап 3)
    mem_on = memory and settings.memory_enabled
    await audit(
        request,
        "chat_query",
        "chat_session",
        str(body.session_id) if body.session_id else None,
        {"document_id": str(body.document_id) if body.document_id else None, "q": body.message[:200]},
    )

    async def gen():
        sessionmaker = app.state.sessionmaker
        # 1) сессия + сообщение пользователя
        async with sessionmaker() as db:
            if body.session_id:
                chat_session = await db.get(ChatSession, body.session_id)
                if chat_session is None:
                    yield _sse({"type": "error", "detail": "сессия не найдена"})
                    return
                if not _owner_ok(chat_session, user):
                    yield _sse({"type": "error", "detail": "сессия не найдена"})
                    return
            else:
                chat_session = ChatSession(
                    id=uuid.uuid4(),  # default колонки срабатывает только на flush
                    title=make_session_title(body.message),
                    document_id=body.document_id,
                    owner_sub=user.sub,
                    folder_id=body.folder_id,
                )
                db.add(chat_session)
                await db.flush()  # сессия должна попасть в INSERT раньше сообщения (FK)
            db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.message))
            session_id = chat_session.id
            doc_filter = chat_session.document_id or body.document_id
            project_id = chat_session.folder_id or body.folder_id
            doc_ids = body.document_ids or None  # набор документов (мульти-выбор)
            # scope памяти треда: user=owner, project=папка, document=документ, thread=сессия
            mem_scope = app.state.memory.scope_for(
                user.sub, project_id=project_id, document_id=doc_filter, thread_id=session_id
            )
            # входящая реплика → memory_events (ground-truth) в том же коммите
            if mem_on:
                await app.state.memory.record_message(db, mem_scope, "user", body.message)
            await db.commit()

            all_msgs = (
                (
                    await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.session_id == session_id)
                        .order_by(ChatMessage.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            # prior — без только что записанного вопроса; recent N идут как
            # история, вытесненные старые — в инкрементальную сводку (§ 5 п.5)
            prior = all_msgs[:-1]
            recent_n = settings.rag_history_messages
            recent = prior[-recent_n:]
            older = prior[:-recent_n] if len(prior) > recent_n else []
            summary = chat_session.summary
            new_old = older[chat_session.summary_msg_count :]
            if new_old:
                summary = await app.state.chat_engine.summarize_history(summary, new_old)
                await db.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(summary=summary, summary_msg_count=len(older))
                )
                # §5 п.5 поглощается памятью: сводка треда → memory_item(kind=summary)
                if mem_on:
                    await app.state.memory.write_summary(db, mem_scope, summary)
                await db.commit()
        # Обрезаем аномально длинные реплики в истории (ассистент вьювера вшивает
        # текст открытой страницы в сообщение — без обрезки история раздувает
        # контекст и переполняет окно модели на 2-м ходу). Обычные вопросы короче
        # лимита и не страдают; актуальный контекст страницы шлётся текущим ходом.
        history = [
            {"role": m.role, "content": (m.content or "")[: settings.rag_history_msg_chars]}
            for m in recent
        ]
        yield _sse({"type": "session", "session_id": str(session_id)})

        # 2) роутинг по сложности (§ 5 п.7): single-hop hybrid+rerank или
        #    multi-hop агентный цикл сбора контекста
        t_retr = time.monotonic()
        route_info = await classify(app.state.chat_engine.client, body.message, history)
        mode, reason = route_info.mode, route_info.reason
        yield _sse({"type": "mode", "mode": mode, "reason": reason, "route": route_info.route})

        # память (§4): ретрив по разрешённому scope → gate → отдельный блок промпта
        memory_block: str | None = None
        if mem_on and route_info.needs_memory:
            try:
                async with sessionmaker() as mdb:
                    memory_block, mem_hits = await app.state.memory.retrieve_block(
                        mdb, body.message, mem_scope
                    )
                if memory_block:
                    yield _sse({"type": "memory", "count": len(mem_hits)})
            except Exception as exc:
                logger.warning("memory retrieve failed: %s", exc)

        # memory_only (§2.3.1): вопрос о пользователе/проекте — документный поиск
        # пропускаем, отвечаем из памяти.
        needs_docs = route_info.route not in ("memory_only", "out_of_scope")
        agent: AgentLoop | None = None
        chunks = []
        try:
            if not needs_docs:
                pass  # без документного контекста
            elif mode == "multi_hop":
                tools = AgentTools(
                    sessionmaker,
                    app.state.retriever,
                    document_id=doc_filter,
                    folder_id=project_id,
                    document_ids=doc_ids,
                    session_id=session_id,
                    # как _owner_filter в documents.py: admin видит все документы
                    owner_sub=None if user.is_admin else user.sub,
                )
                agent = AgentLoop(app.state.chat_engine.client, tools)
                async for ev in agent.gather(body.message, history):
                    yield _sse(ev)
                chunks = agent.chunks
            else:
                async with sessionmaker() as db:
                    chunks = await app.state.retriever.retrieve(
                        db, body.message, document_id=doc_filter,
                        folder_id=project_id, document_ids=doc_ids,
                        # RBAC §4.7.1: single-hop поиск — только свои документы
                        owner_sub=None if user.is_admin else user.sub,
                    )
        except Exception as exc:
            logger.exception("retrieve/agent failed")
            yield _sse({"type": "error", "detail": f"поиск не удался: {exc}"})
            return
        if agent is not None:
            log_agent_trace(
                question=body.message,
                mode=mode,
                steps=agent.steps,
                stop_reason=agent.stop_reason,
                tokens=agent.tokens,
                n_chunks=len(chunks),
                ms=int((time.monotonic() - t_retr) * 1000),
                session_id=str(session_id),
                user_sub=user.sub,
            )
        # 3) ответ. Нет ни документов, ни памяти → канонический отказ (но дальше
        #    всё равно фиксируем реплику и ставим экстракцию: входящее сообщение
        #    могло нести факты для памяти). Иначе — стрим из LLM.
        t_gen = time.monotonic()
        if needs_docs and not chunks and not memory_block:
            answer = "В библиотеке не нашлось проиндексированных фрагментов по этому запросу."
            yield _sse({"type": "delta", "text": answer})
        else:
            parts: list[str] = []
            try:
                async for delta in app.state.chat_engine.stream_answer(
                    body.message, chunks, history, summary=summary,
                    memory_block=memory_block, route=route_info.route,
                ):
                    parts.append(delta)
                    yield _sse({"type": "delta", "text": delta})
            except Exception as exc:
                logger.exception("LLM stream failed")
                yield _sse({"type": "error", "detail": f"генерация не удалась: {exc}"})
                return
            answer = "".join(parts).strip()

        citations = extract_citations(answer, chunks) if chunks else []
        async with sessionmaker() as db:
            msg = ChatMessage(
                session_id=session_id, role="assistant", content=answer, citations=citations
            )
            db.add(msg)
            await db.flush()
            await db.execute(
                update(ChatSession).where(ChatSession.id == session_id).values(updated_at=func.now())
            )
            # ответ → memory_events (ground-truth для асинхронной экстракции, Этап 2)
            if mem_on:
                await app.state.memory.record_message(
                    db, mem_scope, "assistant", answer, source_message_id=msg.id
                )
            await db.commit()
            message_id = msg.id
        # асинхронная экстракция кандидатов памяти (вне latency ответа, §4)
        if mem_on:
            try:
                await app.state.arq.enqueue_job("extract_memory", str(session_id))
            except Exception as exc:
                logger.warning("extract_memory enqueue failed: %s", exc)
        log_chat_trace(
            question=body.message,
            chunks=chunks,
            answer=answer,
            model=settings.llm_model,
            ms=int((time.monotonic() - t_gen) * 1000),
            session_id=str(session_id),
            user_sub=user.sub,
        )
        yield _sse({"type": "done", "citations": citations, "message_id": str(message_id)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions")
async def list_sessions(request: Request) -> list[dict]:
    user: User = request.state.user
    stmt = select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(100)
    if not user.is_admin:
        # свои сессии + сессии dev-периода (owner NULL), как _owner_filter в documents.py
        stmt = stmt.where(
            (ChatSession.owner_sub == user.sub) | (ChatSession.owner_sub.is_(None))
        )
    async with request.app.state.sessionmaker() as db:
        rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": str(s.id),
            "title": s.title,
            "document_id": str(s.document_id) if s.document_id else None,
            "folder_id": str(s.folder_id) if s.folder_id else None,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in rows
    ]


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: uuid.UUID) -> None:
    """Удаление чата: сессия + сообщения (FK ondelete=CASCADE) + мягкая чистка
    памяти этого треда (расцеплена с FK; под RLS FORCE нужен GUC скоупа)."""
    user: User = request.state.user
    async with request.app.state.sessionmaker() as db:
        session = await db.get(ChatSession, session_id)
        if session is None or not _owner_ok(session, user):
            raise HTTPException(404, "сессия не найдена")
        await db.delete(session)
        await db.commit()
    try:
        memory = request.app.state.memory
        async with request.app.state.sessionmaker() as db:
            await apply_scope_guc(db, memory.scope_for(user.sub, thread_id=session_id))
            await db.execute(
                sql(
                    "UPDATE memory_items SET status='deleted', deleted_at=now()"
                    " WHERE user_id=:u AND thread_id=:t AND status<>'deleted'"
                ),
                {"u": user.sub, "t": str(session_id)},
            )
            await db.commit()
    except Exception:  # noqa: BLE001 — чистка памяти не должна валить удаление
        pass
    await audit(request, "delete", "chat_session", str(session_id))


@router.get("/sessions/{session_id}/messages")
async def session_messages(request: Request, session_id: uuid.UUID) -> list[dict]:
    user: User = request.state.user
    async with request.app.state.sessionmaker() as db:
        session = await db.get(ChatSession, session_id)
        if session is None or not _owner_ok(session, user):
            raise HTTPException(404, "сессия не найдена")
        rows = (
            (
                await db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at)
                )
            )
            .scalars()
            .all()
        )
    return [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "citations": m.citations or [],
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]


@router.get("/sessions/{session_id}/export")
async def export_session(
    request: Request, session_id: uuid.UUID, format: str = "md"
) -> StreamingResponse:
    """Выжимка/история сессии (§ 5): LLM-сводка + транскрипт + источники → md/docx."""
    user: User = request.state.user
    async with request.app.state.sessionmaker() as db:
        session = await db.get(ChatSession, session_id)
        if session is None or not _owner_ok(session, user):
            raise HTTPException(404, "сессия не найдена")
        messages = (
            (
                await db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at)
                )
            )
            .scalars()
            .all()
        )
    digest = await session_digest(request.app.state.chat_engine.client, session.title, messages)
    await audit(request, "chat_export", "chat_session", str(session_id), {"format": format})
    stem = (session.title or "chat")[:40].strip().replace("/", "-")
    ascii_name = stem.encode("ascii", "ignore").decode() or "chat"
    if format == "docx":
        return StreamingResponse(
            io.BytesIO(render_docx(digest)),
            media_type=_DOCX_MIME,
            headers={"Content-Disposition": f'attachment; filename="{ascii_name}.docx"'},
        )
    return StreamingResponse(
        io.BytesIO(render_md(digest).encode("utf-8")),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{ascii_name}.md"'},
    )
