// Фоновый service worker (MV3): вся сеть и история — здесь (roadmap § 8).
// OIDC PKCE через chrome.identity → Keycloak добавится на этапе 5.

import { pushHistory, translateNodes, translateSelection, type NodeItem } from '@/utils/api';

export default defineBackground(() => {
  browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
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
