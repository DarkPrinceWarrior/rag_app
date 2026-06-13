"""RAG-чат: SSE-стрим ответа с цитатами, сессии и история (roadmap § 5)."""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, update

from rag_app.api.audit import audit
from rag_app.api.auth import require_user
from rag_app.config import settings
from rag_app.db.models import ChatMessage, ChatSession
from rag_app.observability import log_agent_trace, log_chat_trace
from rag_app.rag.agent import AgentLoop, classify
from rag_app.rag.chat import extract_citations, make_session_title
from rag_app.rag.tools import AgentTools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[require_user])


class ChatIn(BaseModel):
    message: str
    session_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    folder_id: uuid.UUID | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(request: Request, body: ChatIn) -> StreamingResponse:
    if not body.message.strip():
        raise HTTPException(422, "пустой вопрос")
    app = request.app
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
            else:
                chat_session = ChatSession(
                    id=uuid.uuid4(),  # default колонки срабатывает только на flush
                    title=make_session_title(body.message),
                    document_id=body.document_id,
                )
                db.add(chat_session)
                await db.flush()  # сессия должна попасть в INSERT раньше сообщения (FK)
            db.add(ChatMessage(session_id=chat_session.id, role="user", content=body.message))
            await db.commit()
            session_id = chat_session.id
            doc_filter = chat_session.document_id or body.document_id

            history_rows = (
                (
                    await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.session_id == session_id)
                        .order_by(ChatMessage.created_at.desc())
                        .limit(12)
                    )
                )
                .scalars()
                .all()
            )
        # history_rows[0] — только что записанный вопрос, в историю не дублируем
        history = [{"role": m.role, "content": m.content} for m in reversed(history_rows[1:])]
        yield _sse({"type": "session", "session_id": str(session_id)})

        # 2) роутинг по сложности (§ 5 п.7): single-hop hybrid+rerank или
        #    multi-hop агентный цикл сбора контекста
        t_retr = time.monotonic()
        mode, reason = await classify(app.state.chat_engine.client, body.message, history)
        yield _sse({"type": "mode", "mode": mode, "reason": reason})
        agent: AgentLoop | None = None
        try:
            if mode == "multi_hop":
                tools = AgentTools(
                    sessionmaker,
                    app.state.retriever,
                    document_id=doc_filter,
                    folder_id=body.folder_id,
                    session_id=session_id,
                )
                agent = AgentLoop(app.state.chat_engine.client, tools)
                async for ev in agent.gather(body.message, history):
                    yield _sse(ev)
                chunks = agent.chunks
            else:
                async with sessionmaker() as db:
                    chunks = await app.state.retriever.retrieve(
                        db, body.message, document_id=doc_filter, folder_id=body.folder_id
                    )
        except Exception as exc:
            logger.exception("retrieve/agent failed")
            yield _sse({"type": "error", "detail": f"поиск не удался: {exc}"})
            return
        if agent is not None:
            user = getattr(request.state, "user", None)
            log_agent_trace(
                question=body.message,
                mode=mode,
                steps=agent.steps,
                stop_reason=agent.stop_reason,
                tokens=agent.tokens,
                n_chunks=len(chunks),
                ms=int((time.monotonic() - t_retr) * 1000),
                session_id=str(session_id),
                user_sub=user.sub if user else None,
            )
        if not chunks:
            answer = "В библиотеке не нашлось проиндексированных фрагментов по этому запросу."
            async with sessionmaker() as db:
                db.add(ChatMessage(session_id=session_id, role="assistant", content=answer))
                await db.commit()
            yield _sse({"type": "delta", "text": answer})
            yield _sse({"type": "done", "citations": []})
            return

        # 3) стрим ответа
        parts: list[str] = []
        t_gen = time.monotonic()
        try:
            async for delta in app.state.chat_engine.stream_answer(body.message, chunks, history):
                parts.append(delta)
                yield _sse({"type": "delta", "text": delta})
        except Exception as exc:
            logger.exception("LLM stream failed")
            yield _sse({"type": "error", "detail": f"генерация не удалась: {exc}"})
            return

        answer = "".join(parts).strip()
        citations = extract_citations(answer, chunks)
        async with sessionmaker() as db:
            msg = ChatMessage(
                session_id=session_id, role="assistant", content=answer, citations=citations
            )
            db.add(msg)
            await db.execute(
                update(ChatSession).where(ChatSession.id == session_id).values(updated_at=func.now())
            )
            await db.commit()
            message_id = msg.id
        user = getattr(request.state, "user", None)
        log_chat_trace(
            question=body.message,
            chunks=chunks,
            answer=answer,
            model=settings.llm_model,
            ms=int((time.monotonic() - t_gen) * 1000),
            session_id=str(session_id),
            user_sub=user.sub if user else None,
        )
        yield _sse({"type": "done", "citations": citations, "message_id": str(message_id)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions")
async def list_sessions(request: Request) -> list[dict]:
    async with request.app.state.sessionmaker() as db:
        rows = (
            (await db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(100)))
            .scalars()
            .all()
        )
    return [
        {
            "id": str(s.id),
            "title": s.title,
            "document_id": str(s.document_id) if s.document_id else None,
            "created_at": s.created_at.isoformat(),
        }
        for s in rows
    ]


@router.get("/sessions/{session_id}/messages")
async def session_messages(request: Request, session_id: uuid.UUID) -> list[dict]:
    async with request.app.state.sessionmaker() as db:
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
