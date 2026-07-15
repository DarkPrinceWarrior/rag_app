# Аттестация offline red-team harness — 2026-07-15

## Область проверки

Проверен локальный Promptfoo-контур без обращения к API DocRAGenslate, внешним
сетям, production-документам и секретам. Использован детерминированный
`mock_provider.mjs`; через реальный Promptfoo CLI прошли рабочие `cases.yaml` и
`assertions.mjs`.

## Окружение и воспроизведение

- Node.js: `22.20.0`.
- Promptfoo: `0.120.19`, локальная pinned dev-зависимость, MIT.
- Команды: `npm run contract`, `npm run eval:mock`.
- Телеметрия, sharing, update-check и cache отключены; remote provider отсутствует.

## Результат

- Promptfoo mock-eval: `8 passed, 0 failed, 0 errors`.
- Python scaffold/deploy contracts: `20 passed`.
- JSON evidence: `/tmp/docragenslate-promptfoo-mock/mock-results.json`, режим
  `0600`, 7875 байт, SHA-256
  `7c12ffc61ac2a4435d11c49342d9e8969dbc9e9dc21cb8095847dd77e2036006`.
- `package-lock.json` SHA-256:
  `bc1c5a514b1b04fc36c13c98726ed17968e9cfd1a0fc70a9564a79026f50715f`.
- `promptfooconfig.yaml` SHA-256:
  `19191ce9241c57e602e7049cb3b476cbbfc3cf86224ed3b42f82ff52be61f74e`.
- `cases.yaml` SHA-256:
  `a41e3415bfa4d7bb3e5ce9bc565110dd7e9fd41b6d513f368fece1600d03e7ba`.

## Ограничение вывода

Эта аттестация подтверждает воспроизводимость и fail-closed свойства harness,
но не безопасность production RAG. Реальный red-team не запускался: для него
нужны одноразовые БД/индекс/MinIO, два тестовых пользователя, короткоживущий
token и только синтетические canary-fixtures по `README.md`. После такого
прогона raw evidence нельзя публиковать; в общий отчёт переносятся только хеш,
агрегаты и обезличенные категории отказов.
