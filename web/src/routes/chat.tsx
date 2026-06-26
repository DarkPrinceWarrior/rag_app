import { useEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Trash2, FileText, X, Table as TableIcon, ChevronDown, Download, Folder as FolderIcon } from 'lucide-react'
import { api, type ChatSession, type Citation, type Document, type Folder, type ExtractTable } from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { cn } from '@/lib/utils'
import { streamChat } from '@/lib/sse'
import { Button } from '@/components/ui/button'
import { Markdown } from '@/components/Markdown'
import { SegmentBody } from '@/components/SegmentBody'
import { dedupeCitations } from '@/lib/citations'

export const Route = createFileRoute('/chat')({
  validateSearch: (s: Record<string, unknown>): { doc?: string; sid?: string } => ({
    doc: typeof s.doc === 'string' ? s.doc : undefined,
    sid: typeof s.sid === 'string' ? s.sid : undefined,
  }),
  component: Chat,
})

// Область чата: вся библиотека / папка / произвольный набор документов (мультивыбор).
type Scope =
  | { kind: 'all' }
  | { kind: 'folder'; folderId: string }
  | { kind: 'docs'; docIds: string[] }

function scopeToBody(scope: Scope): {
  document_id?: string | null
  folder_id?: string
  document_ids?: string[]
} {
  if (scope.kind === 'folder') return { folder_id: scope.folderId }
  if (scope.kind === 'docs')
    return scope.docIds.length === 1
      ? { document_id: scope.docIds[0] }
      : { document_ids: scope.docIds }
  return {}
}

interface Msg {
  role: 'user' | 'assistant'
  content: string
  trace: string[]
  citations: Citation[]
  table?: ExtractTable
  error?: string
}

function Chat() {
  const { doc, sid: sidParam } = Route.useSearch()
  const navigate = Route.useNavigate()
  const queryClient = useQueryClient()
  const [scope, setScope] = useState<Scope>(doc ? { kind: 'docs', docIds: [doc] } : { kind: 'all' })
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [temporary, setTemporary] = useState(false) // временный чат — без памяти
  const [source, setSource] = useState<Citation | null>(null) // открытая панель источника
  const [sid, setSid] = useState<string | null>(sidParam ?? null) // активная сессия
  const sessionId = useRef<string | null>(sidParam ?? null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const loadedSid = useRef<string | null>(null) // какую сессию уже подняли из истории

  const sessionsQ = useQuery({ queryKey: ['chat-sessions'], queryFn: api.listSessions })
  const foldersQ = useQuery({ queryKey: ['folders'], queryFn: api.listFolders })
  const docsQ = useQuery({
    queryKey: ['documents'],
    queryFn: api.listDocuments,
    select: (ds) => ds.filter((d) => d.status === 'done'),
  })

  // Восстановление чата при заходе по ?sid= (фикс «чат пропадает при выходе в меню»).
  useEffect(() => {
    if (!sidParam || loadedSid.current === sidParam) return
    loadedSid.current = sidParam
    sessionId.current = sidParam
    setSid(sidParam)
    api
      .getSessionMessages(sidParam)
      .then((msgs) =>
        setMessages(
          msgs.map((m) => ({ role: m.role, content: m.content, trace: [], citations: m.citations })),
        ),
      )
      .catch(() => setMessages([]))
  }, [sidParam])

  function openSession(s: ChatSession) {
    if (busy) return
    setScope(
      s.document_id
        ? { kind: 'docs', docIds: [s.document_id] }
        : s.folder_id
          ? { kind: 'folder', folderId: s.folder_id }
          : { kind: 'all' },
    )
    loadedSid.current = null // заставить эффект перечитать сообщения
    navigate({ search: (prev) => ({ ...prev, sid: s.id }) })
  }

  function newChat() {
    if (busy) return
    sessionId.current = null
    loadedSid.current = null
    setSid(null)
    setMessages([])
    navigate({ search: (prev) => ({ ...prev, sid: undefined }) })
  }

  // Смена области → новый чат (сессия создаётся с новой областью; контекст другой).
  function onScopeChange(next: Scope) {
    setScope(next)
    if (messages.length > 0 || sessionId.current) newChat()
  }

  async function exportChat(fmt: 'md' | 'docx') {
    if (!sid) return
    const r = await authFetch(`/api/chat/sessions/${sid}/export?format=${fmt}`)
    if (!r.ok) return
    const blob = await r.blob()
    const cd = r.headers.get('Content-Disposition') || ''
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = /filename="?([^"]+)"?/.exec(cd)?.[1] || `chat.${fmt}`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  async function deleteSession(s: ChatSession, e: React.MouseEvent) {
    e.stopPropagation()
    if (busy) return
    if (!confirm(`Удалить чат «${s.title}»?`)) return
    await api.deleteSession(s.id)
    queryClient.invalidateQueries({ queryKey: ['chat-sessions'] })
    if (sid === s.id) newChat()
  }

  function patchLast(fn: (m: Msg) => Msg) {
    setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)))
  }

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    const isNew = !sessionId.current
    setInput('')
    setBusy(true)
    setMessages((ms) => [
      ...ms,
      { role: 'user', content: text, trace: [], citations: [] },
      { role: 'assistant', content: '', trace: [], citations: [] },
    ])
    try {
      await streamChat(
        { message: text, session_id: sessionId.current, ...scopeToBody(scope) },
        (ev) => {
          if (ev.type === 'session') {
            sessionId.current = ev.session_id
            loadedSid.current = ev.session_id // эту сессию уже держим в state — не перечитывать
            setSid(ev.session_id)
            if (isNew) {
              navigate({ search: (prev) => ({ ...prev, sid: ev.session_id }) })
              queryClient.invalidateQueries({ queryKey: ['chat-sessions'] })
            }
          } else if (ev.type === 'mode' && ev.mode === 'multi_hop')
            patchLast((m) => ({ ...m, trace: [...m.trace, '🧭 углублённый разбор запроса'] }))
          else if (ev.type === 'memory')
            patchLast((m) => ({ ...m, trace: [...m.trace, `🧠 учтено из памяти: ${ev.count}`] }))
          else if (ev.type === 'step')
            patchLast((m) => ({ ...m, trace: [...m.trace, `🔧 ${ev.tool}${ev.arg ? ': ' + ev.arg : ''}`] }))
          else if (ev.type === 'agent_summary')
            patchLast((m) => ({ ...m, trace: [...m.trace, `✓ собрано фрагментов: ${ev.chunks} (шагов ${ev.iters}, стоп: ${ev.stop})`] }))
          else if (ev.type === 'delta') patchLast((m) => ({ ...m, content: m.content + ev.text }))
          else if (ev.type === 'done') patchLast((m) => ({ ...m, citations: ev.citations ?? [] }))
          else if (ev.type === 'error') patchLast((m) => ({ ...m, error: ev.detail }))
        },
        undefined,
        temporary,
      )
    } catch (e) {
      patchLast((m) => ({ ...m, error: String(e) }))
    }
    setBusy(false)
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }

  // Извлечь таблицу из той же области (спец-интент § 5 п.6, интегрирован в чат).
  async function runTable() {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)
    setMessages((ms) => [
      ...ms,
      { role: 'user', content: text, trace: [], citations: [] },
      { role: 'assistant', content: '', trace: ['⊞ извлекаю таблицу из источников…'], citations: [] },
    ])
    try {
      const t = await api.extractTable(text, scopeToBody(scope))
      const cites: Citation[] = (t.sources ?? []).map((s) => ({
        n: s.n,
        chunk_id: '',
        document_id: s.document_id,
        filename: s.filename,
        heading_path: s.heading_path,
        page_start: s.page != null ? s.page - 1 : null,
        page_end: null,
        segment_ids: s.segment_ids,
        bboxes: [],
      }))
      patchLast((m) => ({
        ...m,
        trace: [],
        table: t,
        citations: cites,
        content: t.rows.length ? '' : 'По запросу не удалось собрать таблицу — уточните формулировку.',
      }))
    } catch (e) {
      patchLast((m) => ({ ...m, trace: [], error: String(e) }))
    }
    setBusy(false)
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }

  const started = messages.length > 0

  return (
    <div className="mx-auto flex h-[calc(100vh-49px)] max-w-5xl gap-3 px-4">
      {/* Сайдбар истории чатов */}
      <aside className="hidden w-60 shrink-0 flex-col py-3 md:flex">
        <Button variant="outline" size="sm" className="mb-2" onClick={newChat} disabled={busy}>
          + Новый чат
        </Button>
        <div className="flex-1 space-y-0.5 overflow-auto pr-1">
          {sessionsQ.data?.length === 0 && (
            <p className="px-1 pt-2 text-xs text-muted-foreground">История пуста</p>
          )}
          {sessionsQ.data?.map((s) => (
            <div
              key={s.id}
              className={cn(
                'group flex items-center rounded-md',
                s.id === sid ? 'bg-accent' : 'hover:bg-accent/60',
              )}
            >
              <button
                onClick={() => openSession(s)}
                title={s.title}
                className={cn(
                  'min-w-0 flex-1 truncate px-2 py-1.5 text-left text-xs',
                  s.id === sid ? 'font-medium text-accent-foreground' : 'text-muted-foreground',
                )}
              >
                {s.title}
              </button>
              <button
                onClick={(e) => deleteSession(s, e)}
                title="Удалить чат"
                className="mr-1 shrink-0 rounded p-1 text-muted-foreground opacity-0 transition hover:bg-background hover:text-destructive group-hover:opacity-70 hover:!opacity-100"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Колонка чата */}
      <div className="flex min-w-0 flex-1 flex-col">
        {!started ? (
          /* Пустой чат: ввод по центру, как в Claude/ChatGPT */
          <div className="flex flex-1 flex-col items-center justify-center px-2 pb-12">
            <h2 className="text-xl font-semibold">Чат с документами</h2>
            <p className="mb-5 mt-1 text-sm text-muted-foreground">
              Задайте вопрос или извлеките таблицу — со ссылками на источники.
            </p>
            <div className="w-full max-w-2xl">
              <Composer
                value={input}
                setValue={setInput}
                onSend={send}
                onTable={runTable}
                busy={busy}
                autoFocus
                placeholder="Например: сравни требования к испытаниям и сведи в таблицу"
              />
              <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
                <ScopePicker scope={scope} onChange={onScopeChange} docs={docsQ.data ?? []} folders={foldersQ.data ?? []} />
                <TempToggle temporary={temporary} setTemporary={setTemporary} />
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-2 py-3">
              <ScopePicker scope={scope} onChange={onScopeChange} docs={docsQ.data ?? []} folders={foldersQ.data ?? []} />
              {/* «Временный» выбирается только при старте нового чата */}
              {!sid && <TempToggle temporary={temporary} setTemporary={setTemporary} />}
              {sid && (
                <div className="ml-auto flex items-center gap-1.5">
                  <span className="text-xs text-muted-foreground">Сохранить:</span>
                  <Button variant="outline" size="sm" onClick={() => exportChat('md')}>
                    MD
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => exportChat('docx')}>
                    DOCX
                  </Button>
                </div>
              )}
            </div>

            <div className="flex-1 space-y-3 overflow-auto pb-4">
              {messages.map((m, i) => (
                <Bubble key={i} m={m} onCite={setSource} activeCite={source} />
              ))}
              <div ref={bottomRef} />
            </div>

            <div className="border-t pt-3 pb-6">
              <Composer
                value={input}
                setValue={setInput}
                onSend={send}
                onTable={runTable}
                busy={busy}
                placeholder="Спросите ещё что-нибудь или соберите таблицу…"
              />
            </div>
          </>
        )}
      </div>

      {source && <SourcePanel citation={source} onClose={() => setSource(null)} />}
    </div>
  )
}

/** Боковая панель источника цитаты: текст процитированного фрагмента + переход
 *  во вьювер. Открывается прямо в чате — не надо уходить и возвращаться. */
function SourcePanel({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  const segsQ = useQuery({
    queryKey: ['segments', citation.document_id],
    queryFn: () => api.getSegments(citation.document_id),
  })
  const cited = (segsQ.data ?? []).filter((s) => citation.segment_ids?.includes(s.id))
  const page = citation.page_start != null ? citation.page_start + 1 : undefined
  return (
    <div className="fixed bottom-0 right-0 top-[49px] z-30 flex w-[min(92vw,420px)] flex-col border-l bg-card shadow-2xl">
      <header className="flex items-start gap-2 border-b px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Источник [{citation.n}]
          </div>
          <div className="truncate text-sm font-medium">{citation.filename}</div>
          <div className="mt-0.5 truncate text-xs text-muted-foreground">
            {citation.heading_path}
            {page ? ` · стр. ${page}` : ''}
          </div>
        </div>
        <button onClick={onClose} title="Закрыть" className="shrink-0 rounded p-1 hover:bg-accent">
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="flex-1 overflow-auto px-4 py-3">
        {segsQ.isLoading ? (
          <p className="text-sm text-muted-foreground">Загрузка…</p>
        ) : cited.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Текст фрагмента не найден в документе — откройте во вьювере.
          </p>
        ) : (
          cited.map((s) => (
            <div key={s.id} className="mb-4">
              <SegmentBody s={s} />
            </div>
          ))
        )}
      </div>

      <div className="border-t p-3">
        <Link
          to="/view/$id"
          params={{ id: citation.document_id }}
          search={{ seg: citation.segment_ids?.[0], page }}
        >
          <Button variant="outline" size="sm" className="w-full">
            Открыть во вьювере
          </Button>
        </Link>
      </div>
    </div>
  )
}

/** Поле ввода: многострочное, Enter — отправка, Shift+Enter — перенос.
 *  Доп. действие «Таблица» — извлечь структурированную таблицу из источников. */
function Composer({
  value,
  setValue,
  onSend,
  onTable,
  busy,
  placeholder,
  autoFocus,
}: {
  value: string
  setValue: (v: string) => void
  onSend: () => void
  onTable: () => void
  busy: boolean
  placeholder: string
  autoFocus?: boolean
}) {
  return (
    <div className="flex items-end gap-2 rounded-xl border bg-card p-2 shadow-sm transition focus-within:ring-2 focus-within:ring-ring">
      <textarea
        autoFocus={autoFocus}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            onSend()
          }
        }}
        rows={1}
        placeholder={placeholder}
        className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none"
      />
      <Button
        variant="outline"
        onClick={onTable}
        disabled={busy || !value.trim()}
        title="Собрать структурированную таблицу из найденных фрагментов (с экспортом в XLSX)"
      >
        <TableIcon className="h-4 w-4" />
        Таблица
      </Button>
      <Button onClick={onSend} disabled={busy || !value.trim()}>
        {busy ? '…' : 'Спросить'}
      </Button>
    </div>
  )
}

/** Выбор области чата: вся библиотека / папка / произвольный набор документов. */
function ScopePicker({
  scope,
  onChange,
  docs,
  folders,
}: {
  scope: Scope
  onChange: (s: Scope) => void
  docs: Document[]
  folders: Folder[]
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const label =
    scope.kind === 'folder'
      ? `Папка: ${folders.find((f) => f.id === scope.folderId)?.name ?? '…'}`
      : scope.kind === 'docs'
        ? scope.docIds.length === 1
          ? (docs.find((d) => d.id === scope.docIds[0])?.filename ?? '1 документ')
          : `Выбрано документов: ${scope.docIds.length}`
        : 'Вся библиотека'

  const checked = (id: string) => scope.kind === 'docs' && scope.docIds.includes(id)
  function toggleDoc(id: string) {
    const cur = scope.kind === 'docs' ? scope.docIds : []
    const next = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]
    onChange(next.length ? { kind: 'docs', docIds: next } : { kind: 'all' })
  }

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-[280px] items-center gap-2 rounded-lg border bg-card px-3 py-1.5 text-sm transition-colors hover:bg-accent"
        title="Область поиска для чата и таблиц"
      >
        <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate text-left">{label}</span>
        <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-40 mt-1 max-h-[62vh] w-[330px] overflow-auto rounded-lg border bg-card p-1.5 shadow-2xl">
          <button
            onClick={() => {
              onChange({ kind: 'all' })
              setOpen(false)
            }}
            className={cn(
              'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent',
              scope.kind === 'all' && 'bg-accent font-medium',
            )}
          >
            <FileText className="h-4 w-4 text-muted-foreground" />
            Вся библиотека
          </button>

          {folders.length > 0 && (
            <>
              <div className="px-2 pb-0.5 pt-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Папки
              </div>
              {folders.map((f) => (
                <button
                  key={f.id}
                  onClick={() => {
                    onChange({ kind: 'folder', folderId: f.id })
                    setOpen(false)
                  }}
                  className={cn(
                    'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent',
                    scope.kind === 'folder' && scope.folderId === f.id && 'bg-accent font-medium',
                  )}
                >
                  <FolderIcon className="h-4 w-4 text-muted-foreground" />
                  <span className="min-w-0 flex-1 truncate">{f.name}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">{f.documents}</span>
                </button>
              ))}
            </>
          )}

          <div className="px-2 pb-0.5 pt-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Документы (можно несколько)
          </div>
          {docs.length === 0 && <div className="px-2 py-1 text-xs text-muted-foreground">Нет готовых документов</div>}
          {docs.map((d) => (
            <label
              key={d.id}
              className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent"
            >
              <input
                type="checkbox"
                checked={checked(d.id)}
                onChange={() => toggleDoc(d.id)}
                className="h-3.5 w-3.5 shrink-0"
              />
              <span className="min-w-0 flex-1 truncate" title={d.filename}>
                {d.filename}
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

function TempToggle({
  temporary,
  setTemporary,
}: {
  temporary: boolean
  setTemporary: (v: boolean) => void
}) {
  return (
    <label
      className="flex cursor-pointer items-center gap-1.5 rounded-lg border bg-card px-3 py-1.5 text-xs text-muted-foreground"
      title="Не сохранять и не использовать долговременную память в этом чате"
    >
      <input type="checkbox" checked={temporary} onChange={(e) => setTemporary(e.target.checked)} />
      Временный чат
    </label>
  )
}

/** Экспорт извлечённой таблицы в XLSX (POST /api/extract/xlsx, без хранения). */
async function downloadTableXlsx(t: ExtractTable) {
  const r = await authFetch('/api/extract/xlsx', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: t.title, columns: t.columns, rows: t.rows, sources: t.sources }),
  })
  if (!r.ok) return
  const blob = await r.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${(t.title || 'таблица').slice(0, 40).replace(/[^\wа-яёА-ЯЁ -]/gi, '')}.xlsx`
  a.click()
  URL.revokeObjectURL(a.href)
}

function TableCard({ t }: { t: ExtractTable }) {
  return (
    <div className="rounded-lg border bg-card shadow-sm">
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <TableIcon className="h-4 w-4 text-primary" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{t.title}</span>
        <Button variant="outline" size="sm" onClick={() => downloadTableXlsx(t)}>
          <Download className="h-4 w-4" />
          XLSX
        </Button>
      </div>
      <div className="max-h-[60vh] overflow-auto">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 bg-muted/70 backdrop-blur">
            <tr>
              {t.columns.map((c, i) => (
                <th key={i} className="border-b border-r px-2.5 py-1.5 text-left font-medium last:border-r-0">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {t.rows.map((row, ri) => (
              <tr key={ri} className="even:bg-muted/30">
                {t.columns.map((_, ci) => (
                  <td key={ci} className="border-b border-r px-2.5 py-1.5 align-top last:border-r-0">
                    {row[ci] ?? ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Bubble({
  m,
  onCite,
  activeCite,
}: {
  m: Msg
  onCite: (c: Citation) => void
  activeCite: Citation | null
}) {
  if (m.role === 'user')
    return (
      <div className="ml-auto max-w-[80%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
        {m.content}
      </div>
    )
  return (
    <div className="max-w-[90%]">
      {m.trace.length > 0 && (
        <div className="mb-1 border-l-2 border-border pl-2 text-xs text-muted-foreground">
          {m.trace.map((t, i) => (
            <div key={i}>{t}</div>
          ))}
        </div>
      )}
      {m.error ? (
        <div className="text-sm text-destructive">Ошибка: {m.error}</div>
      ) : m.table && m.table.rows.length > 0 ? (
        <TableCard t={m.table} />
      ) : (
        <div className="rounded-lg bg-card px-3 py-2 shadow-sm">
          <Markdown content={m.content || '…'} />
        </div>
      )}
      {m.citations.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {dedupeCitations(m.citations).map((c) => (
            <button
              key={c.n}
              onClick={() => onCite(c)}
              title={c.heading_path}
              className={cn(
                'rounded-md border px-2 py-0.5 text-xs transition-colors hover:bg-accent',
                activeCite?.n === c.n && activeCite?.document_id === c.document_id
                  ? 'border-primary bg-accent text-accent-foreground'
                  : 'bg-accent/40 text-accent-foreground',
              )}
            >
              [{c.n}] {c.filename}
              {c.page_start != null ? ` · стр. ${c.page_start + 1}` : ''}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
