import { defineConfig } from 'wxt';

// MV3, единая кодовая база Chrome/Edge/Firefox (roadmap § 8).
// host_permissions обходят CORS для заявленных хостов; корпоративный хост
// добавляется сюда при сборке прод-артефакта (этап 5).
export default defineConfig({
  modules: ['@wxt-dev/module-react'],
  manifest: {
    name: 'rag_app — переводчик EN→RU',
    description: 'Перевод выделенного текста и страниц через корпоративный on-prem контур',
    permissions: ['storage', 'activeTab', 'tabs'],
    host_permissions: [
      'http://localhost:8100/*',
      'http://127.0.0.1:8100/*',
    ],
  },
});
