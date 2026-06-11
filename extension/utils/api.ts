// Общий клиент API (фоновый SW — единственный, кто ходит в сеть).
import { browser } from 'wxt/browser';

export interface NodeItem {
  id: string;
  text: string;
}

export interface HistoryEntry {
  source: string;
  translated: string;
  engine: string;
  ts: number;
}

export async function getApiBase(): Promise<string> {
  const { apiBase } = await browser.storage.sync.get({ apiBase: 'http://localhost:8100' });
  return (apiBase as string).replace(/\/+$/, '');
}

async function postJson<T>(path: string, body: unknown, retries = 2): Promise<T> {
  const base = await getApiBase();
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const resp = await fetch(`${base}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const detail = await resp.json().then((d) => d.detail).catch(() => resp.statusText);
        throw new Error(`${resp.status}: ${detail}`);
      }
      return (await resp.json()) as T;
    } catch (e) {
      lastErr = e;
      if (attempt < retries) await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
    }
  }
  throw lastErr;
}

export function translateSelection(text: string) {
  return postJson<{ text: string; engine: string; ms: number }>('/api/selection/translate', {
    text,
    target_lang: 'ru',
  });
}

export function translateNodes(items: NodeItem[]) {
  return postJson<{ items: NodeItem[]; engine: string; ms: number }>('/api/web/translate', {
    items,
    target_lang: 'ru',
  });
}

export async function pushHistory(entry: HistoryEntry): Promise<void> {
  const { history } = await browser.storage.local.get({ history: [] as HistoryEntry[] });
  const next = [entry, ...(history as HistoryEntry[])].slice(0, 20);
  await browser.storage.local.set({ history: next });
}
