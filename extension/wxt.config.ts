import { defineConfig } from 'wxt';

// MV3, единая кодовая база Chrome/Edge/Firefox (roadmap § 8).
// Для корпоративного артефакта прод-хост и стабильный ключ (детерминированный
// extension ID, нужен для ExtensionInstallForcelist) подставляются из env при
// сборке — см. deploy/extension-policy/README.md. Без env собирается dev-вариант
// (localhost), поведение прежнее.
const PROD_HOST = process.env.RAG_EXT_HOST; // напр. https://rag.example.corp
const KC_HOST = process.env.RAG_EXT_KC_HOST; // хост Keycloak/SSO, напр. https://sso.example.corp
const EXT_KEY = process.env.RAG_EXT_KEY; // base64 публичного ключа CRX → фикс. ID
const normalizeHost = (value: string | undefined): string | undefined => value?.replace(/\/+$/, '');
const DEFAULT_API_BASE = normalizeHost(PROD_HOST) ?? 'http://localhost:8100';
const hostPermissions = PROD_HOST
  ? [...new Set([normalizeHost(PROD_HOST), normalizeHost(KC_HOST ?? PROD_HOST)].filter(Boolean).map((host) => `${host}/*`))]
  : [
      'http://localhost:8100/*',
      'http://127.0.0.1:8100/*',
      'http://localhost:8180/*',
      'http://127.0.0.1:8180/*',
    ];

export default defineConfig({
  modules: ['@wxt-dev/module-react'],
  vite: () => ({
    define: {
      __RAG_EXT_DEFAULT_API__: JSON.stringify(DEFAULT_API_BASE),
    },
  }),
  manifest: {
    name: 'DocRAGenslate — переводчик',
    short_name: 'DocRAG',
    description: 'Перевод выделенного текста и страниц через корпоративный on-prem контур',
    permissions: ['storage', 'activeTab', 'tabs', 'identity'],
    ...(EXT_KEY ? { key: EXT_KEY } : {}),
    icons: {
      16: 'icon/16.png',
      32: 'icon/32.png',
      48: 'icon/48.png',
      128: 'icon/128.png',
    },
    action: {
      default_icon: {
        16: 'icon/16.png',
        32: 'icon/32.png',
      },
    },
    // service worker ходит по host_permissions в обход CORS — нужен и хост API,
    // и хост Keycloak (обмен/refresh токена идёт напрямую к нему), иначе на проде
    // token endpoint падает по CORS. В production остаются только хосты из env;
    // localhost разрешён исключительно в локальной сборке без RAG_EXT_HOST.
    host_permissions: hostPermissions,
  },
});
