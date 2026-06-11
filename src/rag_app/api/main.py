"""FastAPI-приложение. Запуск: uv run uvicorn rag_app.api.main:app --port 8100"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_app.api.routes.chat import router as chat_router
from rag_app.api.routes.documents import router as documents_router
from rag_app.api.routes.glossary import router as glossary_router
from rag_app.api.routes.library import router as library_router
from rag_app.api.routes.segments import router as segments_router
from rag_app.api.routes.widget import router as widget_router
from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.llm.fast import FastTranslator
from rag_app.rag.chat import ChatEngine
from rag_app.rag.retrieve import Retriever
from rag_app.storage.s3 import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = create_engine()
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
    app.state.storage = Storage()
    await app.state.storage.ensure_buckets()
    app.state.arq = await create_pool(
        RedisSettings(host=settings.redis_host, port=settings.redis_port, database=settings.redis_db)
    )
    app.state.retriever = Retriever(Embedder(), Reranker())
    app.state.chat_engine = ChatEngine()
    app.state.fast_translator = FastTranslator()
    yield
    await app.state.arq.aclose()
    await app.state.engine.dispose()


app = FastAPI(title="rag_app — перевод документации EN→RU", lifespan=lifespan)
# Запросы из расширения (chrome-extension://) и страниц; auth — этап 5 (Keycloak)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)
app.include_router(documents_router)
app.include_router(widget_router)
app.include_router(segments_router)
app.include_router(glossary_router)
app.include_router(chat_router)
app.include_router(library_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/view", include_in_schema=False)
async def view() -> FileResponse:
    return FileResponse(STATIC_DIR / "view.html")


@app.get("/chat", include_in_schema=False)
async def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
