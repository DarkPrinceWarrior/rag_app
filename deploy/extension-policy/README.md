# Раздача расширения Chrome (roadmap § 8)

## Текущий пилотный канал: готовый ZIP

Для первых пользователей используется готовый артефакт
`web/public/downloads/DocRAGenslate-Chrome.zip`. Он также доступен для
скачивания в профиле веб-приложения. Пользователю не требуется настраивать API:
production-сборка по умолчанию обращается к
`https://doc-rag-translate.ds-mind-lab.ru`.

Порядок установки:

1. Скачать ZIP и распаковать его в постоянную папку. Эту папку нельзя удалять
   или перемещать после установки.
2. Открыть `chrome://extensions` в Google Chrome.
3. Включить «Режим разработчика».
4. Нажать «Загрузить распакованное» (`Load unpacked`) и выбрать папку, в
   которой находится `manifest.json`.
5. Закрепить DocRAGenslate на панели, открыть расширение и войти под выданной
   корпоративной учётной записью.

Стабильный ID production-сборки:
`mhjdfiggibjmaiomlggloepafgphgdpa`. Собрать и проверить релизный архив можно
из `extension/` командой `pnpm build:prod`; чистый smoke — `pnpm smoke:prod`.

Unpacked-расширение не обновляется автоматически. При выпуске нового ZIP
пользователь заменяет содержимое постоянной папки и нажимает «Обновить» у
расширения на `chrome://extensions`.

## Будущий управляемый канал с автообновлением

Для автоматических обновлений можно отдельно выпустить расширение через Chrome
Web Store (например, Unlisted) либо настроить self-hosted `.crx` и managed
политику браузера. Шаблоны ниже предназначены для второго варианта:
корпоративный хост отдаёт `.crx` + `update_manifest.xml`, а политика
принудительно его устанавливает. Фактическая раскладка требует MDM/GPO
заказчика и на пилотном стенде не проверялась.

## 1. Стабильный extension ID для managed-раздачи

ExtensionInstallForcelist ключуется по ID, поэтому ID должен быть
детерминированным (иначе после каждой пересборки — новый). ID выводится из
публичного ключа в `manifest.key`.

```bash
# сгенерировать приватный ключ и получить base64 публичного для manifest.key
openssl genrsa 2048 | openssl pkcs8 -topk8 -nocrypt -out rag_app.pem
KEY=$(openssl rsa -in rag_app.pem -pubout -outform DER 2>/dev/null | base64 -w0)
echo "RAG_EXT_KEY=$KEY"
```

`rag_app.pem` — приватный ключ подписи CRX, хранить в секрете (вне репозитория).

## 2. Сборка корпоративного артефакта

```bash
cd extension
RAG_EXT_HOST=https://rag.example.corp \
RAG_EXT_KEY="$KEY" \
  pnpm wxt build           # host_permissions получит прод-хост, ID станет фиксированным
pnpm wxt zip               # .output/*.zip → распаковать/упаковать в .crx
# упаковка .crx с тем же ключом (стабильный ID):
google-chrome --pack-extension=.output/chrome-mv3 --pack-extension-key=rag_app.pem
```

ID можно заранее вычислить из публичного ключа (Chrome берёт SHA-256 от DER
и кодирует первые 16 байт в a–p) — он и подставляется в политику и
`update_manifest.xml`.

## 3. Хостинг

На корпоративном хосте `https://<CORP_HOST>/ext/`:
- `rag_app-<version>.crx`
- `update_manifest.xml` (codebase → адрес .crx; см. шаблон)

## 4. Раскладка политики

`<EXTENSION_ID>` и `<CORP_HOST>` подставить в `chrome-managed-policy.json` и
`update_manifest.xml`.

- **Chrome/Linux:** `chrome-managed-policy.json` → `/etc/opt/chrome/policies/managed/rag_app.json`
- **Chromium/Linux:** `/etc/chromium/policies/managed/rag_app.json`
- **Edge/Linux:** `/etc/opt/edge/policies/managed/rag_app.json`
- **Chrome/Windows (GPO):** `HKLM\Software\Policies\Google\Chrome\ExtensionInstallForcelist`
  → строка `<EXTENSION_ID>;https://<CORP_HOST>/ext/update_manifest.xml`
  (или ADMX-шаблон Chrome, политика ExtensionSettings).
- **Edge/Windows (GPO):** `HKLM\Software\Policies\Microsoft\Edge\ExtensionInstallForcelist`

Проверка после применения: `chrome://policy` → Reload policies → расширение
в списке Force-installed; `chrome://extensions` → установлено, снять нельзя.

## 5. Связь с бэкендом

- `RAG_EXT_HOST` в сборке = адрес веб-приложения; он же в
  `host_permissions` (фоновый SW ходит туда в обход CORS).
- OIDC-вход (chrome.identity) использует redirect `https://*.chromiumapp.org/*`
  — он уже в клиенте `rag-extension` realm'а, от хоста не зависит.
- После раздачи включить `RAG_AUTH_ENABLED=true` на бэкенде (см. roadmap § 12).
