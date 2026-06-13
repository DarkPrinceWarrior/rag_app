import { defineConfig } from 'wxt';

// MV3, единая кодовая база Chrome/Edge/Firefox (roadmap § 8).
// Для корпоративного артефакта прод-хост и стабильный ключ (детерминированный
// extension ID, нужен для ExtensionInstallForcelist) подставляются из env при
// сборке — см. deploy/extension-policy/README.md. Без env собирается dev-вариант
// (localhost), поведение прежнее.
const PROD_HOST = process.env.RAG_EXT_HOST; // напр. https://rag.example.corp
const EXT_KEY = process.env.RAG_EXT_KEY; // base64 публичного ключа CRX → фикс. ID

export default defineConfig({
  modules: ['@wxt-dev/module-react'],
  manifest: {
    name: 'rag_app — переводчик EN→RU',
    description: 'Перевод выделенного текста и страниц через корпоративный on-prem контур',
    permissions: ['storage', 'activeTab', 'tabs', 'identity'],
    ...(EXT_KEY ? { key: EXT_KEY } : {}),
    host_permissions: [
      'http://localhost:8100/*',
      'http://127.0.0.1:8100/*',
      ...(PROD_HOST ? [`${PROD_HOST.replace(/\/+$/, '')}/*`] : []),
    ],
  },
});
