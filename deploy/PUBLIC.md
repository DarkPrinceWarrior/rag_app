# Внешний доступ — один субдомен (функциональная проверка)

Цель: открыть DocRAGenslate по одному субдомену, чтобы приложение И вход
(Keycloak) работали под одним origin. TLS терминирует внешний прокси (Захар).

## Заявка Захару

```
Привет, мне нужно это развернуть
Сабдомен:    translate.ds-mind-lab.ru
Адрес тачки: 192.168.101.12
Порт:        8090
```

На 8090 — внутренний обратный прокси (Caddy, `deploy/proxy/`), который под одним
origin раздаёт:
- `/realms*`, `/resources*`, `/admin*`, `/js*` → Keycloak (127.0.0.1:8180);
- всё остальное (SPA, `/api`, `/healthz`, `/metrics`) → приложение (127.0.0.1:8100).

Внешний TLS-прокси Захара: `https://translate.ds-mind-lab.ru` → `192.168.101.12:8090`,
с пробросом `X-Forwarded-Proto: https`.

## Запуск внутреннего прокси (уже сделано)

```bash
docker compose -f deploy/proxy/docker-compose.yml up -d
# проверка: curl localhost:8090/healthz → 200; /realms/rag-app/.well-known/... → KC
```

## Переключение на публичный домен (ПОСЛЕ того, как Захар поднимет свою сторону)

Пока не применять — иначе сломается текущий доступ через SSH-туннель (Keycloak
начнёт редиректить на публичный домен, который локально не резолвится).

1. **Keycloak** (env, перезапуск контейнера) — внешнее имя + доверие прокси:
   ```
   RAG_KC_HOSTNAME=https://translate.ds-mind-lab.ru
   KC_PROXY_HEADERS=xforwarded
   ```
   (в `docker-compose.yml` keycloak: добавить `KC_PROXY_HEADERS`; `KC_HOSTNAME`
   уже параметризован `RAG_KC_HOSTNAME`). Затем `docker compose up -d keycloak`.

2. **API** (env `.env.api.local`, перезапуск tmux rag_api) — split-horizon:
   ```
   RAG_OIDC_ISSUER=https://translate.ds-mind-lab.ru/realms/rag-app
   RAG_OIDC_PUBLIC_URL=https://translate.ds-mind-lab.ru/realms/rag-app
   RAG_OIDC_JWKS_URL=http://127.0.0.1:8180/realms/rag-app/protocol/openid-connect/certs
   ```
   issuer/public — публичный домен (для браузера и проверки токена), ключи API
   берёт по внутреннему URL Keycloak (не ходит наружу).

3. **Redirect URI** уже в `deploy/keycloak/rag-app-realm.json` (`https://translate.ds-mind-lab.ru/*`).
   Realm импортирован — добавить URI в РАБОТАЮЩИЙ Keycloak: админ-консоль
   (`/admin`, admin/kc-admin-2026) → Clients → rag-web → Valid redirect URIs, либо
   re-import realm.

## Проверка после перехода

- `https://translate.ds-mind-lab.ru/` → страница входа Keycloak → логин
  `ruslan / rag-dev-2026` → библиотека.
- `https://translate.ds-mind-lab.ru/api/config` → `oidc_authority` = публичный URL.
- Перевод RU→EN/ZH, поиск, чат, история правок — функционально.

## Замечание по 152-ФЗ

Сервер остаётся on-prem, наружу торчит только TLS-прокси Захара. Для боевого
контура заказчика — доступ через корпоративный VPN, а не публичный IP.
