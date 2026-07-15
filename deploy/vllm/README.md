# Обновление основного vLLM без смешивания с MinerU

Цель первого окна — квалифицировать `vLLM 0.24.0`; переход на 0.25.x выполняется
отдельно. Скрипт `prepare_candidate_env.sh` создает новые Python 3.12-окружения
`main` и `visual` и не изменяет текущие venv, приложение или
`/root/services/mineru/current`. MinerU 3.4.4 остается на vLLM 0.21, а
PaddleOCR-VL — на собственном vLLM 0.10.2.

## Матрица приемки

| Профиль | GPU | Кандидатный порт | Обязательная проверка | Откат |
| --- | ---: | ---: | --- | --- |
| Qwen3.5-35B-A3B | 3 | 18006 | текстовый и image_url chat, JSON schema, длинный контекст 16K, защита от падения на некорректном multimodal-вводе | запустить старый `vllm-qwen35` |
| Hy-MT2-7B | 1 | 18005 | `smoke_candidate.py`, COMET-A/B, числа/термины, bf16 | `vllm-hymt2` |
| Qwen3-Embedding-8B | 4 | 18002 | конечные векторы, dim 4096 до MRL, recall@5; сохранить eager+fp16 | `vllm-embedding` |
| Qwen3-Reranker-4B | 4 | 18003 | порядок релевантной/нерелевантной пары, Gold ranking; проверить применение шаблона | `vllm-reranker` |
| Qwen3-VL-Embedding-8B | 2 | 18007 | реальные страницы, отсутствие NaN, visual recall; сохранить eager+fp16 | `vllm-visual-embedding` |
| dots.mocr | 0 | 18120 | закрытый parser-корпус и сложные таблицы | `dots-mocr` |
| MinerU 3.4.4 | 5 | не меняется | контрольный parse после окна | отдельный vLLM 0.21, не обновлять |
| PaddleOCR-VL 1.6 | 0 | не меняется | контрольный parse после окна | отдельный vLLM 0.10.2, не обновлять |

## Порядок окна

1. Подготовить окружения без переключения: `VLLM_VERSION=0.24.0
   deploy/vllm/prepare_candidate_env.sh main` и `... visual`.
2. Установить только шаблон кандидата: `install -m 0644
   deploy/vllm/vllm-candidate@.service /etc/systemd/system/` и выполнить
   `systemctl daemon-reload`.
3. Для одного профиля остановить соответствующий production-юнит. Запуск
   кандидата откажется работать, пока production-юнит активен.
4. Запустить `systemctl start vllm-candidate@<профиль>`; проверить `/v1/models`,
   затем `deploy/vllm/smoke_candidate.py <профиль>` и профильный Gold/A-B.
5. Сохранить `requirements.freeze.txt`, логи, версии CUDA/драйвера и метрики.
   После успешной матрицы отдельным изменением перевести production-юниты на
   новый immutable-path. При любой регрессии остановить кандидат и запустить
   прежний unit; существующее окружение не менялось.

Нельзя обновлять symlink `mineru/current`, выполнять `uv sync --extra parse` в
этом окне или устанавливать vLLM в `.venv` приложения. Одновременный запуск
production и кандидата на одной карте не поддерживается.

Статическая проверка репозитория 15 июля 2026 года не нашла использования
`prompt_embeds` в коде приложения: клиенты передают `messages` или текстовый
`input`. Повторить эту проверку перед переключением и приложить результат к
протоколу окна.
