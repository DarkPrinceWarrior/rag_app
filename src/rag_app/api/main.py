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
from prometheus_fastapi_instrumentator import Instrumentator

from rag_app.api.routes.chat import router as chat_router
from rag_app.api.routes.documents import router as documents_router
from rag_app.api.routes.extract import router as extract_router
from rag_app.api.routes.glossary import router as glossary_router
from rag_app.api.routes.library import router as library_router
from rag_app.api.routes.memory import router as memory_router
from rag_app.api.routes.segments import router as segments_router
from rag_app.api.routes.widget import router as widget_router
from rag_app.config import settings
from rag_app.db.engine import create_engine, create_sessionmaker
from rag_app.llm.client import Translator
from rag_app.llm.embeddings import Embedder, Reranker
from rag_app.llm.fast import FastTranslator, HyMTDocTranslator
from rag_app.llm.visual import VisualEmbedder
from rag_app.llm.visual_reranker import VisualReranker
from rag_app.rag.chat import ChatEngine
from rag_app.rag.memory import MemoryService
from rag_app.rag.retrieve import Retriever
from rag_app.storage.s3 import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Прод-сборка веб-SPA (web/dist; собирается локально, deploy/build_web.sh).
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = create_engine()
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
    app.state.storage = Storage()
    await app.state.storage.ensure_buckets()
    app.state.arq = await create_pool(
        RedisSettings(host=settings.redis_host, port=settings.redis_port, database=settings.redis_db)
    )
    app.state.retriever = Retriever(
        Embedder(), Reranker(), VisualEmbedder(), VisualReranker(), app.state.storage
    )
    app.state.chat_engine = ChatEngine()
    # слой памяти — на тех же embedder/reranker (без лишних клиентов), §15.0
    app.state.memory = MemoryService(app.state.retriever.embedder, app.state.retriever.reranker)
    app.state.visual = VisualEmbedder()
    app.state.fast_translator = FastTranslator()
    # перевод фрагмента — тем же документным движком, что и пайплайн (по умолчанию
    # Hy-MT2; тумблер doc_translate_backend). Qwen3.5 переводчиком больше не работает.
    app.state.translator = (
        HyMTDocTranslator() if settings.doc_translate_backend == "hymt2" else Translator()
    )
    yield
    await app.state.arq.aclose()
    await app.state.engine.dispose()


app = FastAPI(title="rag_app — перевод документации EN→RU", lifespan=lifespan)
# CORS без wildcard (этап 5): явные origin'ы веб-приложения + регулярка для
# страниц расширения (chrome-extension://). Фоновый SW расширения ходит по
# host_permissions, к нему CORS не применяется.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)
app.include_router(documents_router)
app.include_router(widget_router)
app.include_router(segments_router)
app.include_router(glossary_router)
app.include_router(chat_router)
app.include_router(library_router)
app.include_router(extract_router)
app.include_router(memory_router)

# Метрики Prometheus (§ 10): HTTP-метрики по эндпоинтам (rate/latency/errors).
# /metrics — публичный (вне require_user-роутеров), Prometheus скрейпит без токена;
# наружу не выставляется (скрейп с localhost). Скрейпит deploy/monitoring/.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
async def api_config() -> dict:
    """Публичная конфигурация для UI и расширения (auth-параметры)."""
    return {
        "auth_enabled": settings.auth_enabled,
        "oidc_authority": settings.oidc_public_url,
        "oidc_client_id": settings.oidc_client_id,
    }


# --- Веб-приложение (SPA, roadmap § 7) -----------------------------------
# Прод: React-SPA из web/dist (собирается локально, deploy/build_web.sh).
# Клиентские маршруты (/chat, /extract, /view/...) отдают index.html (catch-all
# регистрируется ПОСЛЕДНИМ — /api/*, /assets, /healthz, /metrics уже разобраны).
# Без сборки (web/dist) фронта нет — соберите deploy/build_web.sh.
if (WEB_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        # клиентские маршруты (/chat, /extract, /view/...) → index.html;
        # реальные файлы из dist-корня (favicon.svg и пр.) отдаём как есть
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        # index.html НЕ кэшировать: он ссылается на хэшированные /assets/* (те
        # иммутабельны и кэшируются вечно), поэтому новый деплой виден сразу,
        # без cache-buster/жёсткой перезагрузки.
        return FileResponse(
            WEB_DIST / "index.html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )
else:
    logging.getLogger(__name__).warning("web/dist не найден — соберите SPA: deploy/build_web.sh")
