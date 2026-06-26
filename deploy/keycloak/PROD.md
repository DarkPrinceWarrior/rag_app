# Keycloak — перевод в прод-режим (roadmap § 9)

Dev-стенд поднимается корневым `docker-compose.yml` (`start-dev --import-realm`,
HTTP `:8180`, H2-issuer `http://localhost:8180`). Ниже — что меняется при
развёртывании у заказчика. Часть шагов (TLS-сертификат, координаты AD)
зависит от инфраструктуры заказчика и **не может быть проверена на нашем
стенде** — это подготовленный scaffold, доводится при пусконаладке.

## 1. TLS

1. Получить сертификат на хост SSO (корпоративный CA или Let's Encrypt во
   внутреннем контуре), положить в `deploy/keycloak/certs/tls.crt` и `tls.key`
   (каталог в `.gitignore`, в репозиторий не коммитится).
2. Прод-адрес задаётся `RAG_KC_HOSTNAME=https://sso.example.corp` в `.env`.
   Это фиксирует `issuer` в токенах — он должен совпасть с `RAG_OIDC_ISSUER`
   и `RAG_OIDC_PUBLIC_URL` бэкенда (`src/rag_app/config.py`).

## 2. Прод-режим запуска

```bash
docker compose -f docker-compose.yml -f deploy/keycloak/docker-compose.prod.yml up -d keycloak
```

Оверлей (`docker-compose.prod.yml`) переключает на `start --optimized`, гасит
HTTP, открывает TLS `:8443`. `--optimized` не импортирует realm на каждом
старте — realm уже в БД (Postgres, отдельная база `keycloak`). **Первичный
импорт realm `rag-app`** делается один раз: либо оставить dev-команду с
`--import-realm` на первый запуск, либо `kc.sh import --file
/opt/keycloak/data/import/rag-app-realm.json`.

После смены адреса обновить redirect-URI клиентов на прод-домен:
- `rag-web` → `https://app.example.corp/*`;
- `rag-extension` → `https://*.chromiumapp.org/*` (не меняется — это
  фиксированный redirect chrome.identity).

## 3. Федерация AD/LDAP

Шаблон — `deploy/keycloak/ldap-federation.example.json.tmpl` (расширение
`.tmpl`, а не `.json`, чтобы Keycloak `start-dev --import-realm` не пытался
импортировать его как realm и не падал на старте). Координаты AD
(`connectionUrl`, `bindDn`, `usersDn`, сервисная учётка) даёт заказчик;
без них компонент не создать, поэтому локально не проверялось.

```bash
# через kcadm (внутри контейнера keycloak)
kcadm.sh config credentials --server http://localhost:8080 \
  --realm master --user "$RAG_KC_ADMIN" --password "$RAG_KC_ADMIN_PASSWORD"
kcadm.sh create components -r rag-app -f /path/ldap-federation.example.json.tmpl
```

Роли `user`/`admin` маппятся из групп AD отдельными `role-ldap-mapper`
(`RAG-Admins → admin`, `RAG-Users → user`) — см. `_mappers_note` в шаблоне.
Бэкенд берёт роли из `realm_access.roles` (`api/auth.py`), доступ — owner_sub
+ роль admin; без членства в группах rag-app пользователь получает 403.

## 4. Чек-лист пусконаладки

- [ ] TLS-сертификат на хост SSO, `certs/` заполнен
- [ ] `RAG_KC_HOSTNAME` = прод-https, синхронен с `RAG_OIDC_ISSUER`/`_PUBLIC_URL`
- [ ] сильные `RAG_KC_ADMIN_PASSWORD`, `RAG_PG_PASSWORD`
- [ ] realm импортирован один раз, redirect-URI на прод-домен
- [ ] LDAP-федерация создана, маппинг групп AD → роли проверен тестовым входом
- [ ] `RAG_AUTH_ENABLED=true` на бэкенде (см. roadmap § 12 / журнал)
