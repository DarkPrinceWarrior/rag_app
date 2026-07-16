import { useRef, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, Copy, Download, Puzzle } from 'lucide-react'
import { api, type MemoryItem } from '@/lib/api'
import { authFetch, currentUser, logout } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'

export const Route = createFileRoute('/account')({ component: AccountPage })

const SCOPES = ['user', 'project', 'document', 'thread', 'org']
const KINDS = ['preference', 'fact', 'glossary', 'rule', 'task', 'correction', 'summary']

// русские подписи областей и типов памяти (значения в БД остаются английскими)
const SCOPE_RU: Record<string, string> = {
  user: 'Пользователь',
  project: 'Проект',
  document: 'Документ',
  thread: 'Диалог',
  org: 'Организация',
}
const KIND_RU: Record<string, string> = {
  preference: 'Предпочтение',
  fact: 'Факт',
  glossary: 'Глоссарий',
  rule: 'Правило',
  task: 'Задача',
  correction: 'Исправление',
  summary: 'Сводка',
}
const scopeRu = (s: string) => SCOPE_RU[s] ?? s
const kindRu = (k: string) => KIND_RU[k] ?? k

const CHROME_EXTENSIONS_ADDRESS = 'chrome://extensions'
const EXTENSION_INSTALL_STEPS = [
  {
    title: 'Скачайте расширение',
    description: 'Нажмите кнопку «Скачать расширение» в верхней части карточки и дождитесь окончания загрузки.',
  },
  {
    title: 'Распакуйте архив',
    description:
      'Откройте папку «Загрузки», нажмите на ZIP-архив правой кнопкой мыши и выберите «Извлечь всё…», затем «Извлечь». Полученную папку не удаляйте и не перемещайте.',
  },
  {
    title: 'Откройте страницу расширений Chrome',
    description: 'Вставьте адрес ниже в адресную строку Google Chrome и нажмите Enter.',
    showAddress: true,
  },
  {
    title: 'Включите режим разработчика',
    description: 'В правом верхнем углу страницы включите переключатель «Режим разработчика».',
  },
  {
    title: 'Установите расширение',
    description:
      'Нажмите «Загрузить распакованное» и выберите папку, полученную после распаковки. Выбирайте папку, а не ZIP-архив.',
  },
  {
    title: 'Закрепите DocRAGenslate',
    description:
      'Нажмите значок пазла «Расширения» справа от адресной строки Chrome и нажмите булавку рядом с DocRAGenslate.',
  },
  {
    title: 'Войдите под своей учётной записью',
    description: 'Откройте DocRAGenslate, нажмите «Войти» и введите выданные вам логин и пароль.',
  },
  {
    title: 'Обновите страницу и проверьте перевод',
    description:
      'На уже открытой английской странице нажмите Ctrl+R. Выделите текст и нажмите появившуюся кнопку «Перевести» — либо откройте расширение и выберите «Перевести страницу».',
  },
]

function AccountPage() {
  const user = currentUser()
  const chromeAddressRef = useRef<HTMLElement>(null)
  const [addressCopyStatus, setAddressCopyStatus] = useState<'idle' | 'copied' | 'manual'>('idle')
  const [isInstallGuideOpen, setIsInstallGuideOpen] = useState(false)

  function copyAddressWithFallback(): boolean {
    const field = document.createElement('textarea')
    try {
      field.value = CHROME_EXTENSIONS_ADDRESS
      field.style.position = 'fixed'
      field.style.opacity = '0'
      document.body.appendChild(field)
      field.select()
      return document.execCommand('copy')
    } catch {
      return false
    } finally {
      field.remove()
    }
  }

  async function copyChromeExtensionsAddress() {
    let copied: boolean
    try {
      await navigator.clipboard.writeText(CHROME_EXTENSIONS_ADDRESS)
      copied = true
    } catch {
      copied = copyAddressWithFallback()
    }
    if (copied) {
      setAddressCopyStatus('copied')
    } else if (chromeAddressRef.current) {
      const range = document.createRange()
      range.selectNodeContents(chromeAddressRef.current)
      window.getSelection()?.removeAllRanges()
      window.getSelection()?.addRange(range)
      setAddressCopyStatus('manual')
    }
    window.setTimeout(() => setAddressCopyStatus('idle'), 2200)
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-5">
      {/* Карточка пользователя */}
      <div className="mb-5 flex items-center gap-3 rounded-xl border bg-card p-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
          {user.username.slice(0, 1).toUpperCase()}
        </div>
        <div className="min-w-0">
          <div className="text-base font-semibold">{user.username}</div>
          <div className="mt-0.5 flex flex-wrap gap-1">
            {(user.roles.length ? user.roles : ['user']).map((r) => (
              <span key={r} className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                {r === 'admin' ? 'администратор' : 'пользователь'}
              </span>
            ))}
          </div>
        </div>
        <Button variant="outline" size="sm" className="ml-auto" onClick={logout}>
          Выйти
        </Button>
      </div>

      <section className="mb-5 rounded-2xl border bg-card p-4 sm:p-5" aria-labelledby="chrome-extension-title">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
          <div className="flex min-w-0 items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <Puzzle className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <h2 id="chrome-extension-title" className="text-base font-semibold">
                Расширение для Google Chrome
              </h2>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">
                Переводите выделенный текст и целые страницы через корпоративный контур.
              </p>
            </div>
          </div>
          <a
            href="/downloads/DocRAGenslate-Chrome.zip"
            download
            aria-label="Скачать расширение для Google Chrome"
            className="flex min-h-11 w-full shrink-0 items-center justify-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 sm:ml-auto sm:w-auto"
          >
            <Download className="h-4 w-4" aria-hidden="true" />
            Скачать расширение
          </a>
        </div>

        <div className="mt-5 overflow-hidden rounded-xl border bg-card">
          <button
            type="button"
            aria-expanded={isInstallGuideOpen}
            aria-controls="chrome-install-guide-content"
            aria-label={`${isInstallGuideOpen ? 'Свернуть' : 'Развернуть'} инструкцию «Как установить расширение»`}
            onClick={() => setIsInstallGuideOpen((isOpen) => !isOpen)}
            className="flex min-h-11 w-full items-center gap-2 px-4 py-3 text-left transition hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary"
          >
            <span className="shrink-0 text-muted-foreground" aria-hidden="true">
              {isInstallGuideOpen ? '▾' : '▸'}
            </span>
            <span id="chrome-install-guide-title" role="heading" aria-level={3} className="shrink-0 text-sm font-semibold">
              Как установить расширение
            </span>
            <span className="min-w-0 truncate text-xs text-muted-foreground">
              Займёт 2–3 минуты · только один раз
            </span>
          </button>

          {isInstallGuideOpen && (
            <div id="chrome-install-guide-content" className="border-t px-4 py-4">
            <ol className="space-y-3" aria-label="Инструкция по установке расширения">
              {EXTENSION_INSTALL_STEPS.map((step, index) => (
                <li
                  key={step.title}
                  className="grid grid-cols-[2rem_minmax(0,1fr)] gap-3 rounded-xl border border-border/80 bg-muted/20 p-3 sm:p-4"
                >
                  <span
                    className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground"
                    aria-hidden="true"
                  >
                    {index + 1}
                  </span>
                  <div className="min-w-0 pt-0.5">
                    <p className="text-sm font-semibold text-foreground">{step.title}</p>
                    <p className="mt-1 text-sm leading-6 text-muted-foreground">{step.description}</p>
                    {step.showAddress && (
                      <div className="mt-3 flex flex-col gap-2 rounded-lg border bg-background p-2 sm:flex-row sm:items-center">
                        <code
                          ref={chromeAddressRef}
                          className="min-w-0 flex-1 select-all overflow-x-auto px-2 py-1 text-sm font-semibold text-foreground"
                        >
                          {CHROME_EXTENSIONS_ADDRESS}
                        </code>
                        <button
                          type="button"
                          onClick={() => void copyChromeExtensionsAddress()}
                          className="flex min-h-11 shrink-0 items-center justify-center gap-2 rounded-md border bg-card px-3 text-xs font-semibold transition hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                          aria-label="Скопировать адрес страницы расширений Chrome"
                        >
                          {addressCopyStatus === 'copied' ? (
                            <Check className="h-4 w-4 text-emerald-600" aria-hidden="true" />
                          ) : (
                            <Copy className="h-4 w-4" aria-hidden="true" />
                          )}
                          {addressCopyStatus === 'copied'
                            ? 'Скопировано'
                            : addressCopyStatus === 'manual'
                              ? 'Адрес выделен — нажмите Ctrl+C'
                              : 'Скопировать адрес'}
                        </button>
                        <span className="sr-only" role="status" aria-live="polite">
                          {addressCopyStatus === 'copied'
                            ? 'Адрес страницы расширений скопирован'
                            : addressCopyStatus === 'manual'
                              ? 'Адрес выделен. Нажмите Ctrl+C'
                              : ''}
                        </span>
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ol>

            <div className="mt-4 rounded-xl border border-primary/20 bg-primary/[0.04] p-4 text-sm leading-6 text-foreground">
              <span className="font-semibold">Если позже выйдет новая версия:</span> замените файлы в сохранённой папке,
              нажмите «Обновить» на странице{' '}
              <code className="rounded bg-background px-1.5 py-0.5 text-xs">chrome://extensions</code>, затем обновите
              открытые вкладки через <kbd className="rounded border bg-background px-1.5 py-0.5 text-xs">Ctrl+R</kbd>.
            </div>
          </div>
          )}
        </div>
      </section>

      <MemorySection isAdmin={user.isAdmin} />
    </div>
  )
}

function MemorySection({ isAdmin }: { isAdmin: boolean }) {
  const qc = useQueryClient()
  const [q, setQ] = useState('')
  const [scope, setScope] = useState('user')
  const [kind, setKind] = useState('fact')
  const [content, setContent] = useState('')
  const [editId, setEditId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [open, setOpen] = useState(false) // «Память» свёрнута по умолчанию (не на весь экран)
  const [fScope, setFScope] = useState('') // фильтр списка по области ('' = все)
  const [fKind, setFKind] = useState('') // фильтр списка по типу ('' = все)

  const itemsQ = useQuery({ queryKey: ['memory', q], queryFn: () => api.listMemory({ q: q || undefined }) })
  const candQ = useQuery({ queryKey: ['memory-candidates'], queryFn: () => api.listMemoryCandidates('pending') })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['memory'] })
    qc.invalidateQueries({ queryKey: ['memory-candidates'] })
  }

  const createM = useMutation({
    mutationFn: () => api.createMemory({ scope, kind, content: content.trim() }),
    onSuccess: () => {
      setContent('')
      invalidate()
    },
  })
  const delM = useMutation({ mutationFn: (id: string) => api.deleteMemory(id), onSuccess: invalidate })
  const editM = useMutation({
    mutationFn: (v: { id: string; content: string }) => api.updateMemory(v.id, { content: v.content }),
    onSuccess: () => {
      setEditId(null)
      invalidate()
    },
  })
  const acceptM = useMutation({ mutationFn: (id: string) => api.acceptCandidate(id), onSuccess: invalidate })
  const rejectM = useMutation({ mutationFn: (id: string) => api.rejectCandidate(id), onSuccess: invalidate })
  const purgeM = useMutation({ mutationFn: () => api.purgeMemory(), onSuccess: invalidate })

  async function exportMemory() {
    const r = await authFetch('/api/memory/export')
    if (!r.ok) return
    const blob = new Blob([JSON.stringify(await r.json(), null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'memory.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const count = itemsQ.data?.length
  const pending = candQ.data?.length ?? 0
  // список с учётом фильтров области/типа (поиск q идёт на сервер)
  const items = (itemsQ.data ?? []).filter(
    (it) => (!fScope || it.scope === fScope) && (!fKind || it.kind === fKind),
  )

  return (
    <div className="rounded-xl border bg-card">
      {/* Заголовок-переключатель: «Память» свёрнута, разворачивается по клику */}
      <div className="flex items-center gap-2 px-4 py-3">
        <button onClick={() => setOpen((v) => !v)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
          <span className="text-muted-foreground">{open ? '▾' : '▸'}</span>
          <h1 className="text-base font-semibold">Память</h1>
          {pending > 0 && (
            <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-xs text-amber-800">
              {pending} на подтверждение
            </span>
          )}
          <span className="truncate text-xs text-muted-foreground">
            {count != null ? `${count} запис${count % 10 === 1 && count % 100 !== 11 ? 'ь' : 'ей'} · ` : ''}
            что приложение помнит о вас и проектах{isAdmin ? ' (админ: видны все)' : ''}
          </span>
        </button>
        {open && (
          <div className="flex shrink-0 gap-1.5">
            <Button variant="outline" size="sm" onClick={exportMemory}>
              Экспорт
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (confirm('Удалить всю вашу память без возможности восстановления (152-ФЗ)?')) purgeM.mutate()
              }}
            >
              Очистить всё
            </Button>
          </div>
        )}
      </div>

      {!open ? null : (
        <div className="border-t px-4 py-4">
          {/* Кандидаты на подтверждение */}
      {(candQ.data?.length ?? 0) > 0 && (
        <div className="mb-4 rounded-lg border border-amber-300/50 bg-amber-50/40 p-3">
          <div className="mb-1.5 text-sm font-medium">На подтверждение ({candQ.data!.length})</div>
          <div className="space-y-1.5">
            {candQ.data!.map((c) => (
              <div key={c.id} className="flex items-center gap-2 text-sm">
                <span className="rounded bg-muted px-1.5 py-0.5 text-xs">{String(c.proposed.kind ?? '')}</span>
                <span className="min-w-0 flex-1 truncate">{String(c.proposed.content ?? '')}</span>
                <span className="text-xs text-muted-foreground">{(c.confidence * 100).toFixed(0)}%</span>
                <Button size="sm" variant="outline" onClick={() => acceptM.mutate(c.id)}>
                  ✓
                </Button>
                <Button size="sm" variant="ghost" onClick={() => rejectM.mutate(c.id)}>
                  ✕
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Добавление */}
      <div className="mb-4 flex flex-wrap items-end gap-2 rounded-lg border p-3">
        <Select
          value={scope}
          onChange={setScope}
          options={SCOPES.map((s) => ({ value: s, label: scopeRu(s) }))}
          className="min-w-[7rem]"
        />
        <Select
          value={kind}
          onChange={setKind}
          options={KINDS.map((k) => ({ value: k, label: kindRu(k) }))}
          className="min-w-[8rem]"
        />
        <input
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Например: отчёты присылать в формате XLSX"
          className="h-9 min-w-[16rem] flex-1 rounded-md border bg-card px-3 text-sm"
        />
        <Button size="sm" disabled={!content.trim() || createM.isPending} onClick={() => createM.mutate()}>
          Добавить
        </Button>
      </div>

      {/* Поиск + фильтры (область / тип) */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Поиск по памяти…"
          className="h-9 min-w-[12rem] flex-1 rounded-md border bg-card px-3 text-sm"
        />
        <Select
          value={fScope}
          onChange={setFScope}
          options={[{ value: '', label: 'Все области' }, ...SCOPES.map((s) => ({ value: s, label: scopeRu(s) }))]}
          className="min-w-[9rem]"
        />
        <Select
          value={fKind}
          onChange={setFKind}
          options={[{ value: '', label: 'Все типы' }, ...KINDS.map((k) => ({ value: k, label: kindRu(k) }))]}
          className="min-w-[9rem]"
        />
      </div>
      <div className="space-y-1.5">
        {items.length === 0 && (
          <p className="py-6 text-center text-sm text-muted-foreground">
            {itemsQ.data?.length ? 'Ничего не найдено по фильтрам' : 'Память пуста'}
          </p>
        )}
        {items.map((it: MemoryItem) => (
          <div key={it.id} className="flex items-start gap-2 rounded-lg border px-3 py-2 text-sm">
            <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-0.5 text-xs">
              {kindRu(it.kind)} · {scopeRu(it.scope)}
            </span>
            {editId === it.id ? (
              <>
                <input
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  className="min-w-0 flex-1 rounded-md border bg-card px-2 py-1 text-sm"
                />
                <Button size="sm" variant="outline" onClick={() => editM.mutate({ id: it.id, content: draft })}>
                  Сохранить
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setEditId(null)}>
                  Отмена
                </Button>
              </>
            ) : (
              <>
                <span className="min-w-0 flex-1 whitespace-pre-wrap">{it.content}</span>
                <button
                  className="text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setEditId(it.id)
                    setDraft(it.content)
                  }}
                >
                  ✎
                </button>
                <button className="text-xs text-destructive hover:underline" onClick={() => delM.mutate(it.id)}>
                  Удалить
                </button>
              </>
            )}
          </div>
        ))}
          </div>
        </div>
      )}
    </div>
  )
}
