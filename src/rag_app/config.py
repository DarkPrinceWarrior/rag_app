"""Конфигурация приложения (env-префикс RAG_, файл .env в корне)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore"
    )

    # --- PostgreSQL (порт 5433 на a100: 5432 занят чужим контейнером) ---
    database_url: str = "postgresql+asyncpg://rag:rag-pg-2026@127.0.0.1:5433/rag_app"

    # --- Redis / ARQ ---
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0

    # --- MinIO (S3) ---
    s3_endpoint: str = "127.0.0.1:9000"
    s3_access_key: str = "rag-minio"
    s3_secret_key: str = "rag-minio-secret-2026"
    s3_secure: bool = False
    bucket_originals: str = "originals"
    bucket_artifacts: str = "artifacts"
    bucket_translated: str = "translated"
    bucket_exports: str = "exports"

    # --- LLM (vLLM, OpenAI-совместимый endpoint) ---
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen3-32b-awq"
    llm_max_tokens: int = 4096
    translate_concurrency: int = 12
    translate_max_retries: int = 3

    # --- Быстрый контур виджета: Hunyuan-MT-7B, GPU3 (roadmap § 4.1/4.3) ---
    fast_llm_enabled: bool = True
    fast_llm_base_url: str = "http://127.0.0.1:8004/v1"
    fast_llm_model: str = "hunyuan-mt-7b"
    selection_max_chars: int = 4000
    web_translate_max_items: int = 300
    web_translate_concurrency: int = 16

    # --- Эмбеддинги и reranker (GPU4, vLLM; roadmap § 4.3, § 12.1 шаг 1) ---
    embed_base_url: str = "http://127.0.0.1:8002/v1"
    embed_model: str = "qwen3-embedding-0.6b"
    embed_batch_size: int = 32
    # Серия Qwen3-Embedding instruction-aware: запрос идёт с инструкцией
    # («Instruct: …\nQuery: …»), документы — без; пустая строка отключает префикс
    # (так работал BGE-M3).
    embed_query_instruction: str = (
        "Given a search query, retrieve relevant passages from technical documentation"
        " that answer the query"
    )
    # --- Визуальный контур сканов (§ 12.1 шаг 4): эмбеддинг страниц-картинок ---
    visual_enabled: bool = True
    visual_embed_base_url: str = "http://127.0.0.1:8007"
    visual_embed_model: str = "qwen3-vl-embedding-8b"
    # Полный dim: усечение ломает ранжирование (серия не MRL-обученная).
    # HNSW при >2000 невозможен — страницы ищутся последовательным сканом.
    visual_embed_dim: int = 4096
    visual_query_instruction: str = (
        "Given a search query, retrieve document pages that answer the query"
    )
    visual_render_scale: float = 2.0  # 144 DPI (требует --enforce-eager у сервиса)

    rerank_base_url: str = "http://127.0.0.1:8003"
    rerank_model: str = "qwen3-reranker-4b"
    rerank_instruction: str = (
        "Given a search query, retrieve relevant passages from technical documentation"
        " that answer the query"
    )

    # --- RAG (roadmap § 5) ---
    rag_dense_top_k: int = 50
    rag_sparse_top_k: int = 50
    rag_rerank_top_k: int = 20  # после RRF — в reranker
    rag_context_top_k: int = 5  # в промпт
    rag_history_messages: int = 8
    chunk_max_chars: int = 4000  # ~1K токенов
    chunk_min_chars: int = 200  # секции короче — клеим к соседней

    # --- Agentic-RAG (§ 5 п.7): роутинг по сложности + multi-hop tool-цикл ---
    agent_enabled: bool = True
    # Стоп-условия (главный провал agentic RAG — незавершающийся цикл):
    agent_max_iters: int = 5  # ≤5 итераций tool-цикла
    agent_token_budget: int = 30_000  # ≤30K токенов на запрос (суммарно usage)
    agent_timeout_s: int = 60  # ≤60 c wall-clock на сбор контекста
    agent_search_top_k: int = 8  # сколько чанков возвращает один search_chunks
    agent_max_context_chunks: int = 12  # union evidence → reranker → столько в ответ

    # --- Спец-интенты § 5 п.6: экстракция таблиц (structured output → XLSX) ---
    extract_context_top_k: int = 10  # фрагментов в контекст экстракции

    # --- MinerU (парсинг) ---
    mineru_device: str = "cuda:2"  # GPU2 — контур парсинга/OCR (roadmap § 4.3)
    mineru_backend: str = "pipeline"
    mineru_method: str = "auto"  # auto: текстовый слой / OCR постранично (roadmap § 3.1)
    mineru_lang: str = "en"  # подсказка OCR (pipeline-бэкенд)
    mineru_timeout_s: int = 1800
    # Бэкенд для форс-OCR (битый cmap): vlm-engine = MinerU 3.3 VLM
    # (MinerU2.5-Pro-2605-1.2B, multilingual) — чинит кириллицу/таблицы/надстрочные,
    # где PaddleOCR-пайплайн искажает. pipeline = быстрый PP-OCR (нужен -l).
    mineru_force_ocr_backend: str = "vlm-engine"

    # --- Оверлей сканов (перевод поверх изображения по bbox) ---
    scan_font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # --- BabelDOC (PDF→PDF с вёрсткой; AGPL-изоляция в отдельном venv) ---
    babeldoc_enabled: bool = True
    babeldoc_bin: str = "/root/services/babeldoc/.venv/bin/babeldoc"
    babeldoc_qps: int = 8
    babeldoc_timeout_s: int = 3600
    # --auto-enable-ocr-workaround на ветке pdf_text: для обычных PDF детект
    # не срабатывает (поведение прежнее), для searchable-сканов (растр +
    # текстовый слой стороннего OCR) BabelDOC включит белые плашки вместо
    # отказа «Scanned PDF detected». Image-only сканы (pdf_scan) это не лечит
    # («no paragraphs», проверено) — для них наш оверлей pipeline/scan_pdf.py.
    babeldoc_auto_ocr_workaround: bool = True

    # --- CORS (этап 5, прод): без wildcard ---
    # Веб-приложение отдаётся тем же origin (:8100) — ему CORS не нужен; список
    # нужен для сторонних браузерных вызовов. Расширение ходит из фонового SW по
    # host_permissions (CORS к ним не применяется), а его страницы имеют origin
    # chrome-extension://<id> — покрыт регуляркой. Переопределяется env
    # RAG_CORS_ORIGINS (JSON-список) при развёртывании за корпоративным доменом.
    cors_origins: list[str] = [
        "http://localhost:8100",
        "http://127.0.0.1:8100",
    ]
    cors_origin_regex: str = r"chrome-extension://.*"

    # --- Аутентификация (Keycloak OIDC, этап 5) ---
    auth_enabled: bool = False  # false — dev-режим без токенов
    # issuer зафиксирован KC_HOSTNAME в compose; бэкенд резолвит localhost сам
    oidc_issuer: str = "http://localhost:8180/realms/rag-app"
    # адрес для браузера (если Keycloak за другим адресом, чем видит бэкенд)
    oidc_public_url: str = "http://localhost:8180/realms/rag-app"
    oidc_client_id: str = "rag-web"

    # --- Прочее ---
    max_upload_mb: int = 200
    job_timeout_s: int = 3600


settings = Settings()
