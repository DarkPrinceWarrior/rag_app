import { useEffect, useRef, useState } from 'react';
import { browser } from 'wxt/browser';
import { getApiBase, type HistoryEntry } from '@/utils/api';
import { getAccessToken } from '@/utils/auth';

interface AuthState {
  enabled: boolean;
  loggedIn: boolean;
}

async function activeTabId(): Promise<number | undefined> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  return tab?.id;
}

export default function App() {
  const [status, setStatus] = useState('');
  const [pageTranslated, setPageTranslated] = useState(false);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [apiBase, setApiBase] = useState('http://localhost:8100');
  const [auth, setAuth] = useState<AuthState>({ enabled: false, loggedIn: false });
  const fileInput = useRef<HTMLInputElement>(null);

  function refreshAuth() {
    browser.runtime
      .sendMessage({ type: 'auth-status' })
      .then((s) => setAuth(s as AuthState))
      .catch(() => setAuth({ enabled: false, loggedIn: false }));
  }

  useEffect(() => {
    browser.storage.local
      .get({ history: [] })
      .then(({ history }) => setHistory(history as HistoryEntry[]));
    getApiBase().then(setApiBase);
    refreshAuth();
    activeTabId().then(async (id) => {
      if (!id) return;
      try {
        const state = await browser.tabs.sendMessage(id, { type: 'page-state' });
        setPageTranslated(Boolean(state?.translated));
      } catch {
        /* content script ещё не загружен на этой вкладке */
      }
    });
  }, []);

  async function translatePage() {
    const id = await activeTabId();
    if (!id) return;
    setBusy(true);
    setStatus('Перевожу страницу…');
    try {
      const res = await browser.tabs.sendMessage(id, { type: 'translate-page' });
      if (res?.ok) {
        setPageTranslated(true);
        setStatus('Страница переведена');
      } else {
        setStatus(`Ошибка: ${res?.detail ?? 'нет ответа content script'}`);
      }
    } catch {
      setStatus('На этой странице перевод недоступен (системная вкладка?)');
    }
    setBusy(false);
  }

  async function restorePage() {
    const id = await activeTabId();
    if (!id) return;
    await browser.tabs.sendMessage(id, { type: 'restore-page' });
    setPageTranslated(false);
    setStatus('Показан оригинал');
  }

  async function uploadFile(file: File) {
    setStatus(`Загружаю «${file.name}»…`);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const token = await getAccessToken(false);
      const resp = await fetch(`${apiBase}/api/documents`, {
        method: 'POST',
        body: fd,
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (resp.status === 401) {
        setStatus('Нужен вход — нажмите «Войти» выше');
        return;
      }
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(d.detail);
      }
      setStatus('Файл в очереди на перевод — смотрите веб-приложение');
    } catch (e) {
      setStatus(`Ошибка загрузки: ${e}`);
    }
  }

  async function doLogin() {
    setBusy(true);
    setStatus('Открываю окно входа…');
    const res = await browser.runtime.sendMessage({ type: 'login' });
    setStatus(res?.ok ? 'Вход выполнен' : `Ошибка входа: ${res?.error ?? 'неизвестно'}`);
    refreshAuth();
    setBusy(false);
  }

  async function doLogout() {
    await browser.runtime.sendMessage({ type: 'logout' });
    setStatus('Вы вышли');
    refreshAuth();
  }

  async function saveApiBase(value: string) {
    setApiBase(value);
    await browser.storage.sync.set({ apiBase: value });
  }

  return (
    <>
      <header>
        <b>rag_app</b>
        <span>перевод EN→RU · on-prem</span>
      </header>
      <main>
        {auth.enabled && (
          <div className="row">
            {auth.loggedIn ? (
              <button className="secondary" onClick={doLogout}>
                Выйти
              </button>
            ) : (
              <button onClick={doLogin} disabled={busy}>
                Войти
              </button>
            )}
            <span className="status">{auth.loggedIn ? 'вход выполнен' : 'требуется вход'}</span>
          </div>
        )}
        <div className="row">
          <button onClick={translatePage} disabled={busy}>
            Перевести страницу
          </button>
          {pageTranslated && (
            <button className="secondary" onClick={restorePage}>
              Оригинал
            </button>
          )}
        </div>
        <div className="row">
          <button className="secondary" onClick={() => window.open(apiBase, '_blank')}>
            Открыть веб-приложение
          </button>
        </div>
        <div
          className="filebox"
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) void uploadFile(f);
          }}
        >
          Документ в библиотеку: клик или перетащите файл
          <input
            ref={fileInput}
            type="file"
            hidden
            accept=".pdf,.docx,.xlsx,.pptx"
            onChange={(e) => e.target.files?.[0] && void uploadFile(e.target.files[0])}
          />
        </div>
        <div className="status">{status}</div>

        <div className="section">История переводов</div>
        {history.length === 0 && <div className="empty">Выделите текст на странице — кнопка «Перевести» появится рядом</div>}
        {history.map((h, i) => (
          <div
            key={i}
            className="hist"
            title="Скопировать перевод"
            onClick={() => navigator.clipboard.writeText(h.translated)}
          >
            <div className="src">{h.source}</div>
            <div className="dst">{h.translated}</div>
          </div>
        ))}

        <div className="section">Адрес API</div>
        <input
          type="text"
          value={apiBase}
          onChange={(e) => void saveApiBase(e.target.value)}
          spellCheck={false}
        />
      </main>
    </>
  );
}
