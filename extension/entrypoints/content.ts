// Content script: кнопка у выделения + полностраничный перевод (roadmap § 8).
// UI живёт в Shadow DOM — стили страницы не протекают. Разметка страницы
// на сервер не уходит: только тексты узлов (§ 3.3.C).

import { browser } from 'wxt/browser';

const BATCH_SIZE = 60;
const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'CODE', 'PRE', 'TEXTAREA', 'INPUT']);

interface OriginalText {
  node: Text;
  original: string;
}

export default defineContentScript({
  matches: ['<all_urls>'],
  main() {
    // ---------- Shadow DOM UI ----------
    const host = document.createElement('div');
    host.id = 'rag-app-widget-host';
    host.style.cssText = 'all: initial; position: absolute; top: 0; left: 0; z-index: 2147483647;';
    const shadow = host.attachShadow({ mode: 'closed' });
    shadow.innerHTML = `
      <style>
        .btn { position: absolute; background: #2563eb; color: #fff; border: 0; border-radius: 7px;
               padding: 5px 11px; font: 12.5px system-ui, sans-serif; cursor: pointer;
               box-shadow: 0 2px 8px rgba(0,0,0,.25); display: none; }
        .tip { position: absolute; max-width: 440px; background: #101725; color: #eef1f6;
               border-radius: 9px; padding: 10px 13px; font: 13.5px/1.5 system-ui, sans-serif;
               box-shadow: 0 4px 16px rgba(0,0,0,.35); display: none; white-space: pre-wrap; }
        .tip .meta { color: #8b97ad; font-size: 11px; margin-top: 6px; }
        .badge { position: fixed; right: 14px; bottom: 14px; background: #101725; color: #eef1f6;
                 border-radius: 99px; padding: 7px 14px; font: 12.5px system-ui, sans-serif;
                 box-shadow: 0 2px 10px rgba(0,0,0,.3); display: none; }
      </style>
      <button class="btn">Перевести</button>
      <div class="tip"></div>
      <div class="badge"></div>`;
    document.documentElement.appendChild(host);
    const btn = shadow.querySelector('.btn') as HTMLButtonElement;
    const tip = shadow.querySelector('.tip') as HTMLDivElement;
    const badge = shadow.querySelector('.badge') as HTMLDivElement;

    let selectionText = '';

    const hide = (el: HTMLElement) => (el.style.display = 'none');
    const placeAt = (el: HTMLElement, x: number, y: number) => {
      el.style.left = `${x + window.scrollX}px`;
      el.style.top = `${y + window.scrollY}px`;
      el.style.display = 'block';
    };

    // ---------- перевод выделения ----------
    document.addEventListener('mouseup', (e) => {
      if (e.composedPath().includes(host)) return;
      setTimeout(() => {
        const sel = window.getSelection();
        const text = sel?.toString().trim() ?? '';
        hide(tip);
        if (!text || text.length < 2 || !sel?.rangeCount) {
          hide(btn);
          return;
        }
        selectionText = text;
        const rect = sel.getRangeAt(0).getBoundingClientRect();
        placeAt(btn, rect.left + rect.width / 2 - 36, rect.bottom + 8);
      }, 0);
    });

    btn.addEventListener('click', async () => {
      const rect = { x: parseFloat(btn.style.left), y: parseFloat(btn.style.top) };
      hide(btn);
      tip.textContent = 'Перевожу…';
      tip.style.left = `${rect.x}px`;
      tip.style.top = `${rect.y}px`;
      tip.style.display = 'block';
      const res = await browser.runtime.sendMessage({ type: 'selection', text: selectionText });
      if (res?.error) {
        tip.textContent = `Ошибка: ${res.error}`;
        return;
      }
      tip.innerHTML = '';
      tip.append(document.createTextNode(res.text));
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = `${res.engine} · ${res.ms} мс`;
      tip.append(meta);
    });

    document.addEventListener('mousedown', (e) => {
      if (!e.composedPath().includes(host)) {
        hide(tip);
        hide(btn);
      }
    });

    // ---------- полностраничный перевод ----------
    const originals = new Map<string, OriginalText>();
    let translating = false;

    function collectTextNodes(): Map<string, Text> {
      const out = new Map<string, Text>();
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          const text = node.textContent ?? '';
          if (text.trim().length < 2) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent || SKIP_TAGS.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
          if (host.contains(parent)) return NodeFilter.FILTER_REJECT;
          if (!/[A-Za-z]{2,}/.test(text)) return NodeFilter.FILTER_REJECT; // нечего переводить
          const style = window.getComputedStyle(parent);
          if (style.display === 'none' || style.visibility === 'hidden') return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        },
      });
      let i = 0;
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        out.set(`n${i++}`, n as Text);
      }
      return out;
    }

    async function translatePage(): Promise<{ ok: boolean; detail?: string }> {
      if (translating) return { ok: false, detail: 'уже идёт' };
      translating = true;
      const nodes = collectTextNodes();
      const entries = [...nodes.entries()];
      let done = 0;
      badge.style.display = 'block';
      try {
        for (let i = 0; i < entries.length; i += BATCH_SIZE) {
          const batch = entries.slice(i, i + BATCH_SIZE);
          badge.textContent = `Перевод страницы… ${done}/${entries.length}`;
          const res = await browser.runtime.sendMessage({
            type: 'web-batch',
            items: batch.map(([id, node]) => ({ id, text: node.textContent ?? '' })),
          });
          if (res?.error) throw new Error(res.error);
          for (const item of res.items as { id: string; text: string }[]) {
            const node = nodes.get(item.id);
            if (node && item.text && node.textContent !== item.text) {
              if (!originals.has(item.id)) originals.set(item.id, { node, original: node.textContent ?? '' });
              node.textContent = item.text; // прогрессивная замена
            }
          }
          done += batch.length;
        }
        badge.textContent = `Готово: переведено узлов — ${originals.size}`;
        setTimeout(() => hide(badge), 4000);
        return { ok: true };
      } catch (e) {
        badge.textContent = `Ошибка перевода: ${e}`;
        setTimeout(() => hide(badge), 6000);
        return { ok: false, detail: String(e) };
      } finally {
        translating = false;
      }
    }

    function restorePage(): { ok: boolean } {
      for (const { node, original } of originals.values()) node.textContent = original;
      originals.clear();
      hide(badge);
      return { ok: true };
    }

    browser.runtime.onMessage.addListener((msg: any, _sender: unknown, sendResponse: (r: unknown) => void) => {
      if (msg?.type === 'translate-page') {
        translatePage().then(sendResponse);
        return true;
      }
      if (msg?.type === 'restore-page') {
        sendResponse(restorePage());
      }
      if (msg?.type === 'page-state') {
        sendResponse({ translated: originals.size > 0, translating });
      }
    });
  },
});
