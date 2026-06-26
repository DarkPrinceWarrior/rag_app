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
    # Воркхорс перевода + RAG-чата — Qwen3.5-35B-A3B (GPU3:8006). Qwen3-32B-AWQ
    # (:8001) ретайрнут 2026-06-18 (GPU0 освобождена) — на дефолт его не возвращаем.
    llm_base_url: str = "http://127.0.0.1:8006/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen3.5-35b-a3b"
    llm_max_tokens: int = 4096
    translate_concurrency: int = 12
    translate_max_retries: int = 3
    # Рендер OOXML (docx/xlsx/pptx) в PDF через LibreOffice headless — для просмотра
    # «как в Microsoft» (оригинал и перевод в pdf.js-вьювере, а не плоским текстом).
    office_render_enabled: bool = True
    office_render_timeout_s: int = 150
    # Потолок сегментов на XLSX-лист-дамп: химические/числовые таблицы (коды,
    # единицы, значения) после фильтра «только проза» + дедупа по тексту обычно
    # дают тысячи, но патологический дата-дамп может выдать и больше — обрезаем,
    # чтобы перевод и вьювер не вешались. Превышение логируется (warning).
    xlsx_max_segments: int = 5000

    # --- Быстрый контур виджета: Hy-MT2-7B, GPU1 :8005 (roadmap § 12.1 п.5) ---
    # Hy-MT2-7B принят 2026-06-19 по COMET-A/B (COMETKiwi-22, 300 сегм.): средний
    # 0.7790 vs 0.7716 у HY-MT1.5 и 0.7516 у прод-Qwen3.5; p10 0.618. bf16 (FP8 на
    # A100 даёт мусор). Та же arch hunyuan_v1_dense, тот же промпт-формат и сэмплинг.
    fast_llm_enabled: bool = True
    fast_llm_base_url: str = "http://127.0.0.1:8005/v1"
    fast_llm_model: str = "hy-mt2-7b"
    # Движок перевода ДОКУМЕНТОВ (translate_document): hymt2 → Hy-MT2-7B (спец-MT,
    # нативный шаблон + term-anchored глоссарий, БЕЗ Qwen3.5-фолбэка; выиграл COMET-A/B
    # 2026-06-19) | qwen3 → воркхорс Qwen3.5-35B (scaffolded). Qwen3.5 остаётся под
    # RAG-чат/анализ/VL, но не как переводчик документов.
    doc_translate_backend: str = "hymt2"  # hymt2 | qwen3
    selection_max_chars: int = 4000
    web_translate_max_items: int = 300
    web_translate_concurrency: int = 16

    # --- Эмбеддинги и reranker (GPU4, vLLM; roadmap § 4.3, § 12.1 шаг 1) ---
    embed_base_url: str = "http://127.0.0.1:8002/v1"
    # Qwen3-Embedding-8B (2026-06-19): на реальной библиотеке recall@5 0.975 vs 0.887
    # у 0.6B (eval_retrieval.py, 80 вопросов). MRL-усечение до embed_dim=1024 (drop-in
    # в pgvector(1024), HNSW-совместимо; качество как у полного 4096). Сервинг 8B на
    # Ampere ТРЕБУЕТ --enforce-eager --dtype float16 (иначе NaN → recall≈0).
    embed_model: str = "qwen3-embedding-8b"
    embed_dim: int = 1024  # MRL-усечение выхода эмбеддера (нативный 8B = 4096)
    embed_batch_size: int = 32
    # Серия Qwen3-Embedding instruction-aware: запрос идёт с инструкцией
    # («Instruct: …\nQuery: …»), документы — без; пустая строка отключает префикс
    # (так работал BGE-M3).
    embed_query_instruction: str = (
        "Given a search query, retrieve relevant passages from technical documentation"
        " that answer the query"
    )
    # --- Визуальный контур сканов (§ 12.1 шаг 4): эмбеддинг страниц-картинок ---
    # ЗАПАРКОВАН 2026-06-18: фича почти не использовалась (page_embeddings = 1 строка,
    # RAG-чат по ней не ходит — только /api/search/visual), а сервис держал ~23 ГБ
    # GPU5. `vllm-visual-embedding` погашен, флаг выключен (index_pages_visual и ручка
    # визпоиска становятся no-op/503). Ревайв — отдельной задачей (визуальный retrieval
    # в чат), модель остаётся Qwen3-VL-Embedding (Apache-2.0), полный dim 4096.
    visual_enabled: bool = False
    visual_embed_base_url: str = "http://127.0.0.1:8007"
    visual_embed_model: str = "qwen3-vl-embedding-8b"
    # Полный dim: усечение ломает ранжирование (серия не MRL-обученная).
    # HNSW при >2000 невозможен — страницы ищутся последовательным сканом.
    visual_embed_dim: int = 4096
    visual_query_instruction: str = (
        "Given a search query, retrieve document pages that answer the query"
    )
    visual_render_scale: float = 2.0  # 144 DPI (требует --enforce-eager у сервиса)

    # --- Визуальный реранкер Qwen3-VL-Reranker-2B (GPU2 :8009) ---
    # Cross-encoder (query, страница-картинка) → relevance score; раздаётся
    # отдельным FastAPI-сервисом через transformers (НЕ vLLM: vllm#35412 даёт
    # реверсивные скоры, а vLLM 0.22/0.23 не знают Qwen3VLForSequenceClassification).
    visual_rerank_base_url: str = "http://127.0.0.1:8009"
    visual_rerank_model: str = "qwen3-vl-reranker-2b"

    # --- Генеративный VL для описания/объяснения рисунков (GPU2 :8008) ---
    # Qwen3-VL-8B-Instruct: для сканов-чертежей, P&ID, схем, графиков, фото —
    # раскрывает СМЫСЛ изображения текстом (на русском). В отличие от visual_*
    # (только эмбеддинги для поиска) — этот генерирует описание.
    vl_enabled: bool = True
    # Генеративный VL — воркхорс Qwen3.5-35B-A3B (:8006, мультимодальный; отдельный
    # Qwen3-VL-8B на GPU2 ретайрнут 2026-06-19). Картинка капается до vl_max_side px
    # (vision.py) — GPU3 тесная (ctx 8192), большой чертёж иначе переполняет контекст.
    vl_base_url: str = "http://127.0.0.1:8006/v1"
    vl_model: str = "qwen3.5-35b-a3b"
    vl_max_tokens: int = 1200
    vl_max_side: int = 1400  # макс. сторона картинки-страницы (кап vision-токенов)
    vl_render_scale: float = 1.6  # рендер страницы PDF в картинку для VL
    vl_max_pages: int = 12  # потолок страниц-картинок на документ (латентность)
    # figure-sweep для pdf_text/docx/pptx: обход страниц с поиском рисунков —
    # потолок выше (текстовые страницы дёшевы: VL быстро отвечает EMPTY, ~0.8 с/стр)
    vl_sweep_max_pages: int = 200

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
    # порог релевантности реранкера [0..1]: если лучший фрагмент ниже — считаем,
    # что запрос не про эти документы, и не вываливаем случайные чанки (пусто)
    rag_rerank_min_score: float = 0.02
    rag_context_top_k: int = 5  # в промпт
    # сколько вырезанных рисунков (img_s3 среди найденных чанков) приложить кропами
    # в мультимодальный запрос Qwen3.5 (vision on-demand в чате) — кап под ctx 8192
    rag_vision_max_images: int = 3
    # визуальный контур (§12.1 шаг4): сколько страниц поднимает page_embeddings и
    # сколько image-чанков добавить в контекст после визуального реранка
    rag_visual_pages_k: int = 10
    rag_visual_top_k: int = 3
    rag_history_messages: int = 8
    # обрезка одной реплики истории (ассистент вьювера вшивает текст страницы в
    # сообщение — без лимита история переполняет окно модели на 2-м ходу)
    rag_history_msg_chars: int = 1200
    # бюджет блока фрагментов в финальном промпте (символы) — backstop против
    # переполнения окна модели при multi-hop сборе (Qwen3.5 max-model-len 16384)
    rag_context_max_chars: int = 28000
    # §4.5 защита окна модели: вход (контекст+история+картинки) не должен
    # переполнять окно. Бюджет = window − ответ − запас − картинки; если перебор,
    # чат режет историю (старейшую), затем усекает контекст — вместо падения.
    chat_context_window: int = 16384
    chat_output_tokens: int = 2048
    chat_image_tokens: int = 1300  # ~стоимость одного приложенного кропа (1400px)
    chat_chars_per_token: int = 3  # грубая оценка ru/en (символов на токен)
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

    # --- Слой памяти (docs/MEMORY_rev4_mem0_articles.md §15) ---
    memory_enabled: bool = True
    # single-org: tenant_id — константа (env RAG_TENANT_ID); multi-tenant не разворачиваем
    tenant_id: str = "00000000-0000-0000-0000-000000000000"
    memory_provider: str = "internal"  # internal | mem0 (свопается за MemoryAdapter, Этап 4)
    mem0_base_url: str = "http://127.0.0.1:8088"
    # gate-пороги (§5): доверие источника (confidence) и релевантность (rerank).
    # rerank-порог=0: Qwen3-Reranker заточен под документные пассажи и даёт
    # коротким фактам памяти скоры ~0 (паре «как обращаться»↔«зовут Руслан» ≈0),
    # поэтому жёсткий relevance-блок убивал всю выдачу. Релевантность держим
    # порядком reranker'а + dense/sparse-поиском + cap по scope (memory_max_*).
    memory_gate_min_confidence: float = 0.5
    memory_gate_min_rerank: float = 0.0
    # сколько items впрыскивается в промпт после gate, по scope
    memory_max_user: int = 5
    memory_max_project: int = 5
    memory_max_document: int = 5
    memory_max_summary: int = 3
    memory_raw_limit: int = 20  # кандидатов из поиска ДО gate
    memory_rerank_top_k: int = 20  # после RRF — в reranker
    # Этап 2: автоэкстракция и consolidation
    memory_auto_accept_confidence: float = 0.80  # выше — кандидат принимается без очереди
    memory_extract_window: int = 12  # последних реплик в окно экстракции

    # --- MinerU (парсинг) ---
    # GPU2 отдан генеративному VL (:8008) — парсинг MinerU уводим на GPU4
    # (embed/rerank, есть запас памяти), иначе MinerU и VL конфликтуют по памяти.
    mineru_device: str = "cuda:4"
    # Путь к бинарю mineru: пусто → сосед текущего python (общий venv). Для VLM
    # через vllm — изолированный venv (.venv-mineru, torch 2.11), чтобы не понижать
    # torch 2.12 в рабочем venv воркера/API.
    mineru_bin: str | None = None
    # Бэкенд парсинга pdf_text. vlm-http-client → MinerU2.5-Pro (VLM, OmniDocBench-
    # лидер): на сложных макетах (договоры с нумерованными пунктами в «висячем»
    # столбце) НЕ плодит ложные таблицы/склейки страниц и не теряет оглавление —
    # в отличие от CNN-layout pipeline. Сервер mineru-vllm-server на GPU5 (:30010),
    # capped-память. При недоступности сервера парсинг падает обратно на pipeline.
    mineru_backend: str = "vlm-http-client"
    mineru_vlm_url: str = "http://127.0.0.1:30010"
    # Детекция таблиц: на текстовых договорах нумерация пунктов читается как
    # «таблица» и склеивается через страницы → выключаем (текст идёт абзацами
    # постранично). Настоящие таблицы в pdf_text редки (схемы/спеки — сканы/xlsx).
    mineru_table_enable: bool = False
    mineru_method: str = "auto"  # auto: текстовый слой / OCR постранично (roadmap § 3.1)
    mineru_lang: str = "en"  # подсказка OCR (pipeline-бэкенд)
    mineru_timeout_s: int = 1800
    # Бэкенд для форс-OCR (битый cmap): hybrid-engine (MinerU 3.3) — нативный
    # текст для тела/оглавления (полнота, без пропусков) + VLM для таблиц/сложных
    # шрифтов (кириллица/надстрочные). vlm-engine — чистый VLM (теряет плотные
    # списки/оглавления); pipeline — быстрый PP-OCR, но искажает битый cmap.
    mineru_force_ocr_backend: str = "hybrid-engine"

    # --- Выбор парсера pdf_text (на документе можно переопределить) ---
    # mineru → MinerU2.5-Pro (VLM) + добор из текстового слоя (дефолт, единственный
    # с middle.json-геометрией для bbox/цитат); dots_mocr → rednote-hilab/dots.mocr
    # (3B, чистые слитые таблицы); paddle_vl → PaddleOCR-VL 1.6 (0.9B). dots/paddle —
    # альтернативные движки для сравнения, грузятся на GPU4.
    pdf_parser_backend: str = "mineru"  # mineru | dots_mocr | paddle_vl
    # dots.mocr: постоянный vLLM-сервис на GPU4 (deploy/dots-mocr.service) + CLI parser.py
    dots_url: str = "http://127.0.0.1:8120"
    dots_model_name: str = "model"
    dots_repo: str = "/root/parser_trials/dots.mocr"
    dots_venv_python: str = "/root/parser_trials/dots.mocr/.venv_client/bin/python"
    dots_prompt: str = "prompt_layout_all_en"
    dots_num_thread: int = 8
    dots_timeout_s: int = 1800
    # PaddleOCR-VL-0.9B: VLM-инференс на ПОСТОЯННОМ genai vLLM-сервере (paddlex_
    # genai_server, GPU0:8118 — on-demand в 3.7 виснет без inference-движка);
    # layout-детекция локально на paddle_device. Изолированный paddle-venv
    # (vllm 0.10.2 — пин PaddleOCR genai-plugin, не трогает основной стек).
    paddle_venv_python: str = "/root/parser_trials/paddle/.venv_paddle/bin/python"
    paddle_runner: str = "/root/projects/rag_app/deploy/parsers/run_paddle_cli.py"
    paddle_vl_server_url: str = "http://127.0.0.1:8118/v1"
    paddle_device: str = "0"
    paddle_timeout_s: int = 1800

    # --- Оверлей сканов (перевод поверх изображения по bbox) ---
    scan_font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # --- BabelDOC (PDF→PDF с вёрсткой; AGPL-изоляция в отдельном venv) ---
    babeldoc_enabled: bool = True
    babeldoc_bin: str = "/root/services/babeldoc/.venv/bin/babeldoc"
    # BabelDOC переводит каждый текстовый блок отдельным LLM-вызовом. Уводим его
    # на БЫСТРЫЙ контур HY-MT-7B (:8005, заточен под перевод, отдельный GPU) и
    # поднимаем параллелизм — иначе на больших PDF он не успевает за таймаут.
    # Пусто → берётся основной llm_* (как раньше).
    babeldoc_base_url: str = "http://127.0.0.1:8005/v1"
    babeldoc_model: str = "hy-mt2-7b"
    babeldoc_qps: int = 30
    # потолок BabelDOC: на тяжёлых image-PDF он очень медленный — по таймауту
    # подпроцесс убивается, экспорт довольствуется DOCX и идёт в индекс (документ
    # не блокируется; вёрстку-PDF можно дособрать позже). Вьювер pdf_text и без
    # BabelDOC рендерит оригинал через pdf.js.
    babeldoc_timeout_s: int = 1200
    # --auto-enable-ocr-workaround на ветке pdf_text: для обычных PDF детект
    # не срабатывает (поведение прежнее), для searchable-сканов (растр +
    # текстовый слой стороннего OCR) BabelDOC включит белые плашки вместо
    # отказа «Scanned PDF detected». Image-only сканы (pdf_scan) это не лечит
    # («no paragraphs», проверено) — для них наш оверлей pipeline/scan_pdf.py.
    babeldoc_auto_ocr_workaround: bool = True
    # --enhance-compatibility (= --skip-clean --dual-translate-first
    # --disable-rich-text-translate). Гасит по-символьную rich-text стилизацию,
    # которая на леттерспейс-капсе шапок («S T A N D A R D …») ломала перевод:
    # русский шёл белым за пределы цветной плашки и обрезался (white-on-white).
    # С флагом текст плашки рендерится нормальным цветом/кеглем одной строкой и
    # читается. Жертвуем тонкой стилизацией (цвет/жирность рунов) ради читаемости
    # — для технических PDF это выигрыш. Проверено на EPC-контракте.
    babeldoc_enhance_compatibility: bool = True
    # «Вёрстка» перевода pdf_text = чистый reflow-PDF из нашего DOCX
    # (build_docx → LibreOffice) ВМЕСТО пиксель-подгонки BabelDOC. Reflow исключает
    # overflow / «скачущий шрифт» / утечки тегов и непереведённых связок, которыми
    # BabelDOC сыпет на плотных таблицах и шаблонных блоках (стратегия
    # «геометрия-в-геометрию» — предел для русского). True → BabelDOC для pdf_text
    # отключён (код остаётся за этим флагом, реверсивно).
    translated_pdf_from_docx: bool = True

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
    # Split-horizon за внешним прокси: issuer/public — публичный домен, а ключи
    # (JWKS) API берёт по ВНУТРЕННЕМУ URL Keycloak, не ходя наружу. Пусто →
    # выводится из oidc_issuer (текущее поведение). См. deploy/PUBLIC.md.
    oidc_jwks_url: str = ""
    # адрес для браузера (если Keycloak за другим адресом, чем видит бэкенд)
    oidc_public_url: str = "http://localhost:8180/realms/rag-app"
    oidc_client_id: str = "rag-web"

    # --- Прочее ---
    max_upload_mb: int = 200
    job_timeout_s: int = 3600


settings = Settings()
