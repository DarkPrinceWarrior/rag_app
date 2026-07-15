import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowUp,
  Check,
  Download,
  Eye,
  Folder as FolderIcon,
  Loader2,
  Table as TableIcon,
  Timer,
  Trash2,
  X,
} from 'lucide-react'
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
  const [sideTab, setSideTab] = useState<'docs' | 'sessions'>('docs')
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
    queryFn: () => api.listDocuments(),
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
    <div className="mx-auto flex h-[calc(100vh-97px)] max-w-[1136px] gap-6 px-4 py-8">
      {/* Сайдбар: переключатель «Документы» (область чата) / «Мои чаты» (история) */}
      <aside className="hidden h-full w-[320px] shrink-0 flex-col gap-5 rounded-[24px] bg-[#222226]/[0.02] p-6 md:flex">
        <div className="flex shrink-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setSideTab('docs')}
            className={cn(
              'text-[16px] font-semibold leading-[1.5] tracking-[-0.16px] transition',
              sideTab === 'docs' ? 'text-[#222226]' : 'text-[#222226]/50 hover:text-[#222226]/70',
            )}
          >
            Документы
          </button>
          <span className="h-4 w-px shrink-0 bg-[#e5e5e5]" />
          <button
            type="button"
            onClick={() => setSideTab('sessions')}
            className={cn(
              'text-[16px] font-semibold leading-[1.5] tracking-[-0.16px] transition',
              sideTab === 'sessions' ? 'text-[#222226]' : 'text-[#222226]/50 hover:text-[#222226]/70',
            )}
          >
            Мои чаты
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {sideTab === 'docs' ? (
            <DocPicker scope={scope} onChange={onScopeChange} docs={docsQ.data ?? []} folders={foldersQ.data ?? []} />
          ) : (
            <SessionList
              sessions={sessionsQ.data ?? []}
              activeSid={sid}
              onOpen={openSession}
              onDelete={deleteSession}
              onNewChat={newChat}
              busy={busy}
            />
          )}
        </div>

        {sideTab === 'docs' && (
          <button
            type="button"
            onClick={() => onScopeChange({ kind: 'all' })}
            disabled={scope.kind === 'all'}
            className="flex shrink-0 items-center justify-center rounded-2xl bg-[#222226]/5 px-6 py-3 text-[16px] font-semibold text-[#424247] transition hover:bg-[#222226]/10 disabled:opacity-40"
          >
            Очистить выбор
          </button>
        )}
      </aside>

      {/* Колонка чата */}
      <div className="flex min-w-0 flex-1 flex-col">
        {!started ? (
          <div className="flex flex-1 flex-col items-center justify-center px-2 pb-12 text-center">
            <h2 className="text-xl font-semibold text-[#222226]">Чат с документами</h2>
            <p className="mb-5 mt-1 text-sm text-[#222226]/50">
              Задайте вопрос или извлеките таблицу — со ссылками на источники.
            </p>
          </div>
        ) : (
          <>
            {sid && (
              <div className="flex shrink-0 items-center justify-end gap-1.5 pb-3 text-xs text-muted-foreground">
                Сохранить:
                <Button variant="ghost" size="sm" onClick={() => exportChat('md')}>
                  MD
                </Button>
                <Button variant="ghost" size="sm" onClick={() => exportChat('docx')}>
                  DOCX
                </Button>
              </div>
            )}
            <div className="flex-1 space-y-4 overflow-auto pb-4">
              {messages.map((m, i) => (
                <Bubble key={i} m={m} onCite={setSource} activeCite={source} />
              ))}
              <div ref={bottomRef} />
            </div>
          </>
        )}

        <div className="shrink-0 pt-3">
          <Composer
            value={input}
            setValue={setInput}
            onSend={send}
            onTable={runTable}
            busy={busy}
            autoFocus={!started}
            placeholder={started ? 'Спросите ещё что-нибудь или соберите таблицу…' : 'Введите запрос'}
            temporary={temporary}
            setTemporary={setTemporary}
            showTempToggle={!sid}
          />
        </div>
      </div>

      {source && <SourcePanel citation={source} onClose={() => setSource(null)} />}
    </div>
  )
}

/** Область чата (сайдбар, вкладка «Документы»): папки (быстрый выбор всей папки)
 *  + плоский чек-лист документов (мультивыбор), в духе Figma 54:2342. */
function DocPicker({
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
  const checked = (id: string) => scope.kind === 'docs' && scope.docIds.includes(id)
  function toggleDoc(id: string) {
    const cur = scope.kind === 'docs' ? scope.docIds : []
    const next = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]
    onChange(next.length ? { kind: 'docs', docIds: next } : { kind: 'all' })
  }

  return (
    <div className="flex flex-col gap-4">
      {folders.length > 0 && (
        <div className="flex flex-col gap-0.5">
          <div className="px-1 pb-1 text-[11px] font-medium uppercase tracking-wide text-[#c1c1c1]">Папки</div>
          {folders.map((f) => {
            const active = scope.kind === 'folder' && scope.folderId === f.id
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => onChange(active ? { kind: 'all' } : { kind: 'folder', folderId: f.id })}
                className={cn(
                  'flex items-center gap-2 rounded-lg px-1 py-1.5 text-left text-[14.3px] font-medium leading-[1.5] tracking-[-0.2145px] transition',
                  active ? 'text-[#222226]' : 'text-[#222226]/70 hover:bg-[#222226]/[0.04]',
                )}
              >
                <FolderIcon className={cn('h-4 w-4 shrink-0', active ? 'text-[#4b4ce6]' : 'text-[#222226]/35')} />
                <span className="min-w-0 flex-1 truncate">{f.name}</span>
                <span className="shrink-0 text-[11px] text-[#c1c1c1]">{f.documents}</span>
              </button>
            )
          })}
        </div>
      )}

      <div className="flex flex-col gap-3">
        {docs.length === 0 && <p className="px-1 text-[13px] text-[#c1c1c1]">Нет готовых документов</p>}
        {docs.map((d) => (
          <label key={d.id} className="flex cursor-pointer items-center gap-2.5">
            <span
              className={cn(
                'flex h-5 w-5 shrink-0 items-center justify-center rounded border transition',
                checked(d.id) ? 'border-[#4b4ce6] bg-[#4b4ce6]' : 'border-[#e5e5e5] bg-white',
              )}
            >
              {checked(d.id) && <Check className="h-3 w-3 text-white" />}
            </span>
            <input
              type="checkbox"
              checked={checked(d.id)}
              onChange={() => toggleDoc(d.id)}
              className="sr-only"
            />
            <span
              className="min-w-0 flex-1 truncate text-[14.3px] font-medium leading-[1.5] tracking-[-0.2145px] text-[#222226]"
              title={d.filename}
            >
              {d.filename}
            </span>
          </label>
        ))}
      </div>
    </div>
  )
}

/** Сайдбар, вкладка «Мои чаты»: история сессий + создание нового чата. */
function SessionList({
  sessions,
  activeSid,
  onOpen,
  onDelete,
  onNewChat,
  busy,
}: {
  sessions: ChatSession[]
  activeSid: string | null
  onOpen: (s: ChatSession) => void
  onDelete: (s: ChatSession, e: React.MouseEvent) => void
  onNewChat: () => void
  busy: boolean
}) {
  return (
    <div className="flex flex-col gap-3">
      <Button variant="outline" size="sm" onClick={onNewChat} disabled={busy}>
        + Новый чат
      </Button>
      <div className="flex flex-col gap-0.5">
        {sessions.length === 0 && <p className="px-1 pt-2 text-[13px] text-[#c1c1c1]">История пуста</p>}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={cn(
              'group flex items-center rounded-lg',
              s.id === activeSid ? 'bg-[#4b4ce6]/10' : 'hover:bg-[#222226]/[0.04]',
            )}
          >
            <button
              type="button"
              onClick={() => onOpen(s)}
              title={s.title}
              className={cn(
                'min-w-0 flex-1 truncate px-2 py-1.5 text-left text-[13px]',
                s.id === activeSid ? 'font-medium text-[#222226]' : 'text-[#222226]/60',
              )}
            >
              {s.title}
            </button>
            <button
              type="button"
              onClick={(e) => onDelete(s, e)}
              title="Удалить чат"
              className="mr-1 shrink-0 rounded p-1 text-[#c1c1c1] opacity-0 transition hover:bg-white hover:text-destructive group-hover:opacity-70 hover:!opacity-100"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
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

const COMPOSER_MAX_H = 160 // px

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
  temporary,
  setTemporary,
  showTempToggle,
}: {
  value: string
  setValue: (v: string) => void
  onSend: () => void
  onTable: () => void
  busy: boolean
  placeholder: string
  autoFocus?: boolean
  temporary: boolean
  setTemporary: (v: boolean) => void
  showTempToggle: boolean
}) {
  const taRef = useRef<HTMLTextAreaElement>(null)
  const [thumb, setThumb] = useState<{ top: number; height: number } | null>(null)

  // Авто-рост textarea (до COMPOSER_MAX_H) + свой тонкий скроллбар вместо
  // нативного — нативный уродливо смотрится в узком однострочном поле
  // (Figma 54:2519: трек 4px + плавающий thumb вместо browser-scrollbar).
  const sync = () => {
    const el = taRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, COMPOSER_MAX_H)}px`
    const { scrollTop, scrollHeight, clientHeight } = el
    if (scrollHeight <= clientHeight + 1) {
      setThumb(null)
      return
    }
    const h = Math.max((clientHeight / scrollHeight) * 100, 15)
    const t = (scrollTop / (scrollHeight - clientHeight)) * (100 - h)
    setThumb({ top: t, height: h })
  }
  useLayoutEffect(sync, [value])

  return (
    <div className="flex flex-col gap-3 rounded-[16px] border border-[#e5e5e5] bg-white px-4 pb-3.5 pt-4 shadow-sm transition focus-within:border-[#6269f3]/40">
      <div className="flex items-start gap-2">
        <textarea
          ref={taRef}
          autoFocus={autoFocus}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onScroll={sync}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSend()
            }
          }}
          rows={1}
          placeholder={placeholder}
          style={{ maxHeight: COMPOSER_MAX_H }}
          className="min-h-[24px] w-full flex-1 resize-none overflow-y-auto bg-transparent text-[16px] font-medium leading-[1.5] tracking-[-0.16px] text-[#222226] outline-none placeholder:text-[#222226]/22 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        />
        {thumb && (
          <div className="relative w-1 shrink-0 self-stretch rounded-full bg-[#222226]/[0.12]">
            <div
              className="absolute left-0 w-1 rounded-full bg-[#424247]"
              style={{ top: `${thumb.top}%`, height: `${thumb.height}%` }}
            />
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onTable}
            disabled={busy || !value.trim()}
            title="Собрать структурированную таблицу из найденных фрагментов (с экспортом в XLSX)"
            className="flex min-h-11 items-center gap-1.5 rounded-lg bg-[#222226]/[0.02] px-3 py-1.5 text-[13px] font-medium text-[#222226]/70 transition hover:bg-[#222226]/[0.05] disabled:opacity-40"
          >
            <TableIcon className="h-4 w-4" />
            Таблица
          </button>
          {showTempToggle && (
            <button
              type="button"
              onClick={() => setTemporary(!temporary)}
              title="Не сохранять и не использовать долговременную память в этом чате"
              className={cn(
                'flex min-h-11 items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition',
                temporary
                  ? 'bg-[#4b4ce6]/10 text-[#222226]'
                  : 'bg-[#222226]/[0.02] text-[#222226]/70 hover:bg-[#222226]/[0.05]',
              )}
            >
              <Timer className="h-4 w-4" />
              Временный чат
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={onSend}
          disabled={busy || !value.trim()}
          title="Отправить"
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#222226]/5 text-[#222226] transition hover:bg-[#222226]/10 disabled:opacity-40"
        >
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
        </button>
      </div>
    </div>
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
      <div className="ml-auto max-w-[85%] rounded-2xl bg-[#222226]/[0.04] px-4 py-2">
        <p className="text-[14.3px] font-medium leading-[1.5] tracking-[-0.2145px] text-[#222226]">
          {m.content}
        </p>
      </div>
    )
  return (
    <div className="max-w-[90%]">
      {m.trace.length > 0 && (
        <div className="mb-1.5 space-y-0.5 border-l-2 border-[#e5e5e5] pl-2.5 text-[12px] text-[#222226]/45">
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
        <div className="text-[14.3px] font-medium leading-[1.5] tracking-[-0.2145px] text-[#222226]">
          <Markdown content={m.content || '…'} />
        </div>
      )}
      {m.citations.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {dedupeCitations(m.citations).map((c) => (
            <button
              key={c.n}
              onClick={() => onCite(c)}
              title={c.heading_path}
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition',
                activeCite?.n === c.n && activeCite?.document_id === c.document_id
                  ? 'bg-[#4b4ce6]/10 text-[#222226]'
                  : 'bg-[#222226]/[0.02] text-[#222226]/70 hover:bg-[#222226]/[0.05]',
              )}
            >
              <Eye className="h-3.5 w-3.5" />
              [{c.n}] {c.filename}
              {c.page_start != null ? ` · стр. ${c.page_start + 1}` : ''}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
