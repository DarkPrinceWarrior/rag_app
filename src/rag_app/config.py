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

    # --- MinerU (парсинг) ---
    mineru_device: str = "cuda:2"  # GPU2 — контур парсинга/OCR (roadmap § 4.3)
    mineru_backend: str = "pipeline"
    mineru_method: str = "auto"  # auto: текстовый слой / OCR постранично (roadmap § 3.1)
    mineru_lang: str = "en"  # подсказка OCR (pipeline-бэкенд)
    mineru_timeout_s: int = 1800

    # --- BabelDOC (PDF→PDF с вёрсткой; AGPL-изоляция в отдельном venv) ---
    babeldoc_enabled: bool = True
    babeldoc_bin: str = "/root/services/babeldoc/.venv/bin/babeldoc"
    babeldoc_qps: int = 8
    babeldoc_timeout_s: int = 3600

    # --- Прочее ---
    max_upload_mb: int = 200
    job_timeout_s: int = 3600


settings = Settings()
