// Фоновый service worker (MV3): вся сеть, история и OIDC-логин — здесь (§ 8, § 9).
// Интерактивный вход идёт через SW (а не попап), чтобы окно chrome.identity
// переживало закрытие попапа.

import { pushHistory, translateNodes, translateSelection, type NodeItem } from '@/utils/api';
import { authStatus, login, logout } from '@/utils/auth';

export default defineBackground(() => {
  browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === 'login') {
      login()
        .then(() => sendResponse({ ok: true }))
        .catch((e) => sendResponse({ error: String(e) }));
      return true;
    }
    if (msg?.type === 'logout') {
      logout().then(() => sendResponse({ ok: true }));
      return true;
    }
    if (msg?.type === 'auth-status') {
      authStatus()
        .then(sendResponse)
        .catch(() => sendResponse({ enabled: false, loggedIn: false }));
      return true;
    }
    if (msg?.type === 'selection') {
      translateSelection(msg.text as string)
        .then(async (res) => {
          await pushHistory({
            source: (msg.text as string).slice(0, 500),
            translated: res.text.slice(0, 500),
            engine: res.engine,
            ts: Date.now(),
          });
          sendResponse(res);
        })
        .catch((e) => sendResponse({ error: String(e) }));
      return true; // асинхронный sendResponse
    }
    if (msg?.type === 'web-batch') {
      translateNodes(msg.items as NodeItem[])
        .then(sendResponse)
        .catch((e) => sendResponse({ error: String(e) }));
      return true;
    }
  });
});
