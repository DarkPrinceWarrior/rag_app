import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Download,
  Eye,
  FileText,
  Files,
  Folder as FolderIcon,
  Loader2,
  MessagesSquare,
  Minus,
  Plus,
  Search,
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
import { Modal } from '@/components/ui/modal'
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

// Область чата: вся библиотека или комбинация папок и отдельных документов.
type Scope =
  | { kind: 'all' }
  | { kind: 'selection'; folderIds: string[]; docIds: string[] }

const MAX_CHAT_SCOPE_DOCUMENTS = 50
const MAX_CHAT_SCOPE_FOLDERS = 50

function copyScope(scope: Scope): Scope {
  return scope.kind === 'selection'
    ? { kind: 'selection', folderIds: [...scope.folderIds], docIds: [...scope.docIds] }
    : { kind: 'all' }
}

function sameScope(left: Scope, right: Scope): boolean {
  if (left.kind !== right.kind) return false
  if (left.kind === 'all' && right.kind === 'all') return true
  if (left.kind !== 'selection' || right.kind !== 'selection') return false
  if (left.docIds.length !== right.docIds.length || left.folderIds.length !== right.folderIds.length) return false
  const rightDocuments = new Set(right.docIds)
  const rightFolders = new Set(right.folderIds)
  return left.docIds.every((id) => rightDocuments.has(id)) && left.folderIds.every((id) => rightFolders.has(id))
}

function scopeFromSession(session: ChatSession): Scope {
  const docIds = [...(session.document_ids ?? []), ...(session.document_id ? [session.document_id] : [])]
  const folderIds = [...(session.folder_ids ?? []), ...(session.folder_id ? [session.folder_id] : [])]
  if (docIds.length || folderIds.length) return { kind: 'selection', docIds, folderIds }
  return { kind: 'all' }
}

function scopeToBody(scope: Scope): {
  document_ids?: string[]
  folder_ids?: string[]
} {
  if (scope.kind === 'selection') {
    return {
      ...(scope.docIds.length ? { document_ids: scope.docIds } : {}),
      ...(scope.folderIds.length ? { folder_ids: scope.folderIds } : {}),
    }
  }
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

const QUANTITY_WARNING_MARKER = '<!-- docragenslate:quantity-warning -->'
const QUANTITY_WARNING_TEXT =
  'Часть числовых значений ответа не найдена в использованных фрагментах. Сверьте их с первоисточником.'
const QUANTITY_WARNING_SUFFIX =
  `\n\n${QUANTITY_WARNING_MARKER}\n> ⚠️ **Проверьте числовые значения.** ${QUANTITY_WARNING_TEXT}`

export function splitQuantityWarning(content: string): { markdown: string; hasWarning: boolean } {
  if (!content.endsWith(QUANTITY_WARNING_SUFFIX)) return { markdown: content, hasWarning: false }
  return {
    markdown: content.slice(0, -QUANTITY_WARNING_SUFFIX.length).trimEnd(),
    hasWarning: true,
  }
}

function Chat() {
  const { doc, sid: sidParam } = Route.useSearch()
  const navigate = Route.useNavigate()
  const queryClient = useQueryClient()
  const [scope, setScope] = useState<Scope>(
    doc ? { kind: 'selection', folderIds: [], docIds: [doc] } : { kind: 'all' },
  )
  const [sideTab, setSideTab] = useState<'docs' | 'sessions'>('docs')
  const [mobilePanel, setMobilePanel] = useState<'docs' | 'sessions' | null>(null)
  const [mobileScopeDraft, setMobileScopeDraft] = useState<Scope | null>(null)
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

  // При прямом открытии /chat?sid= восстанавливаем не только сообщения, но и
  // точную область источников, включая произвольный набор документов.
  useEffect(() => {
    if (!sidParam || !sessionsQ.data) return
    const session = sessionsQ.data.find((item) => item.id === sidParam)
    if (session) setScope(scopeFromSession(session))
  }, [sidParam, sessionsQ.data])

  function openSession(s: ChatSession) {
    if (busy) return
    setScope(scopeFromSession(s))
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
    if (sameScope(scope, next)) return
    setScope(next)
    if (messages.length > 0 || sessionId.current) newChat()
  }

  function openMobileScopePanel() {
    setMobileScopeDraft(copyScope(scope))
    setMobilePanel('docs')
  }

  function closeMobilePanel() {
    setMobilePanel(null)
    setMobileScopeDraft(null)
  }

  function applyMobileScope() {
    if (mobileScopeDraft) onScopeChange(mobileScopeDraft)
    closeMobilePanel()
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
        {
          message: text,
          session_id: sessionId.current,
          scope_kind: scope.kind,
          ...scopeToBody(scope),
        },
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
      setInput((current) => current.trim() || text)
      patchLast((m) => ({ ...m, trace: [], error: String(e) }))
    }
    setBusy(false)
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }

  const started = messages.length > 0
  const scopeLabel = (() => {
    if (scope.kind === 'all') return 'Вся библиотека'
    if (scope.folderIds.length === 1 && scope.docIds.length === 0) {
      return (foldersQ.data ?? []).find((folder) => folder.id === scope.folderIds[0])?.name ?? '1 папка'
    }
    if (scope.docIds.length === 1 && scope.folderIds.length === 0) {
      return (docsQ.data ?? []).find((document) => document.id === scope.docIds[0])?.filename ?? '1 документ'
    }
    const parts: string[] = []
    if (scope.folderIds.length) parts.push(`${scope.folderIds.length} папки`)
    if (scope.docIds.length) parts.push(`${scope.docIds.length} документа`)
    return parts.join(' · ')
  })()

  return (
    <div className="mx-auto flex min-h-0 w-full max-w-[1136px] flex-1 gap-6 px-4 py-4 lg:py-6">
      {/* Сайдбар: переключатель «Документы» (область чата) / «Мои чаты» (история) */}
      <aside
        data-testid="chat-sidebar"
        className={cn(
          'hidden min-h-0 shrink-0 flex-col gap-5 rounded-[24px] bg-[#222226]/[0.02] p-6 transition-[width] duration-200 md:flex',
          sideTab === 'sessions' ? 'w-[352px]' : 'w-[320px]',
        )}
      >
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

        <div
          data-testid="chat-sidebar-scroll"
          className={cn('min-h-0 flex-1 overflow-y-auto', sideTab === 'sessions' && '-mx-3 px-3')}
        >
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
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="mb-3 grid grid-cols-[minmax(0,1fr)_auto] gap-2 md:hidden">
          <button
            type="button"
            onClick={openMobileScopePanel}
            className="flex min-h-11 min-w-0 items-center gap-2 rounded-xl border border-[#e5e5e5] bg-white px-3 text-left shadow-sm transition active:scale-[0.99]"
            aria-label={`Область чата: ${scopeLabel}`}
          >
            <Files className="h-4 w-4 shrink-0 text-[#4b4ce6]" />
            <span className="min-w-0 flex-1 truncate text-sm font-semibold text-[#222226]">{scopeLabel}</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setMobileScopeDraft(null)
              setMobilePanel('sessions')
            }}
            className="flex min-h-11 min-w-11 items-center justify-center rounded-xl border border-[#e5e5e5] bg-white px-3 text-[#424247] shadow-sm transition active:scale-[0.98]"
            aria-label={`История чатов: ${(sessionsQ.data ?? []).length}`}
          >
            <MessagesSquare className="h-4 w-4" />
          </button>
        </div>
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

      <Modal
        open={mobilePanel !== null}
        onClose={closeMobilePanel}
        labelledBy="mobile-chat-panel-title"
        className="mt-auto -mb-4 flex max-h-[82vh] max-w-none flex-col overflow-hidden rounded-b-none rounded-t-[28px] border-x-0 border-b-0 md:hidden"
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-[#e5e5e5] px-5 py-4">
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-[#222226]/35">Чат с документами</p>
            <h2 id="mobile-chat-panel-title" className="mt-0.5 text-lg font-semibold text-[#222226]">
              {mobilePanel === 'docs' ? 'Область поиска' : 'Мои чаты'}
            </h2>
          </div>
          <button
            type="button"
            onClick={closeMobilePanel}
            className="flex h-11 w-11 items-center justify-center rounded-full bg-[#222226]/5 text-[#424247]"
            aria-label="Закрыть панель"
          >
            <X className="h-4 w-4" />
          </button>
        </header>
        {mobilePanel === 'docs' ? (
          <>
            <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
              <DocPicker
                scope={mobileScopeDraft ?? scope}
                onChange={setMobileScopeDraft}
                docs={docsQ.data ?? []}
                folders={foldersQ.data ?? []}
              />
            </div>
            <div className="shrink-0 border-t border-[#e5e5e5] bg-white px-5 py-4">
              {(mobileScopeDraft ?? scope).kind !== 'all' && (
                <Button
                  variant="outline"
                  className="min-h-11 w-full"
                  onClick={() => setMobileScopeDraft({ kind: 'all' })}
                >
                  Искать по всей библиотеке
                </Button>
              )}
              <Button className="mt-2 min-h-11 w-full" onClick={applyMobileScope} disabled={busy}>
                Готово
              </Button>
            </div>
          </>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
            <SessionList
              sessions={sessionsQ.data ?? []}
              activeSid={sid}
              onOpen={(session) => {
                openSession(session)
                closeMobilePanel()
              }}
              onDelete={deleteSession}
              onNewChat={() => {
                newChat()
                closeMobilePanel()
              }}
              busy={busy}
            />
          </div>
        )}
      </Modal>
    </div>
  )
}

/** Область чата: дерево папок с раскрытием документов. Папки сохраняются как
 * динамическая область, а отдельные документы можно добавить к ней. */
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
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    () => new Set(scope.kind === 'selection' ? scope.folderIds : []),
  )
  const [selectionError, setSelectionError] = useState<string | null>(null)
  const folderIds = new Set(folders.map((folder) => folder.id))
  const looseDocs = docs.filter((document) => !document.folder_id || !folderIds.has(document.folder_id))
  const selectedDocIds = new Set(scope.kind === 'selection' ? scope.docIds : [])
  const selectedFolderIds = new Set(scope.kind === 'selection' ? scope.folderIds : [])
  const checked = (document: Document) =>
    selectedDocIds.has(document.id) || Boolean(document.folder_id && selectedFolderIds.has(document.folder_id))

  function commitSelected(nextDocuments: Set<string>, nextFolders: Set<string>) {
    const orderedDocuments = docs.filter((document) => nextDocuments.has(document.id)).map((document) => document.id)
    const unknownDocuments = [...nextDocuments].filter((id) => !orderedDocuments.includes(id))
    const orderedFolders = folders.filter((folder) => nextFolders.has(folder.id)).map((folder) => folder.id)
    const unknownFolders = [...nextFolders].filter((id) => !orderedFolders.includes(id))
    const documentIds = [...orderedDocuments, ...unknownDocuments]
    const folderIds = [...orderedFolders, ...unknownFolders]
    if (documentIds.length > MAX_CHAT_SCOPE_DOCUMENTS) {
      setSelectionError(`Можно выбрать не более ${MAX_CHAT_SCOPE_DOCUMENTS} документов за один чат.`)
      return
    }
    if (folderIds.length > MAX_CHAT_SCOPE_FOLDERS) {
      setSelectionError(`Можно выбрать не более ${MAX_CHAT_SCOPE_FOLDERS} папок за один чат.`)
      return
    }
    setSelectionError(null)
    onChange(
      documentIds.length || folderIds.length
        ? { kind: 'selection', docIds: documentIds, folderIds }
        : { kind: 'all' },
    )
  }

  function toggleDoc(id: string) {
    const nextDocuments = new Set(selectedDocIds)
    if (nextDocuments.has(id)) nextDocuments.delete(id)
    else nextDocuments.add(id)
    commitSelected(nextDocuments, selectedFolderIds)
  }

  function toggleFolder(folder: Folder, folderDocs: Document[]) {
    const nextDocuments = new Set(selectedDocIds)
    const nextFolders = new Set(selectedFolderIds)
    if (nextFolders.has(folder.id)) nextFolders.delete(folder.id)
    else {
      nextFolders.add(folder.id)
      for (const document of folderDocs) nextDocuments.delete(document.id)
    }
    commitSelected(nextDocuments, nextFolders)
  }

  function toggleExpanded(folderId: string) {
    setExpandedFolders((current) => {
      const next = new Set(current)
      if (next.has(folderId)) next.delete(folderId)
      else next.add(folderId)
      return next
    })
  }

  return (
    <div className="flex flex-col gap-5">
      {folders.length > 0 && (
        <section className="flex flex-col gap-1" aria-labelledby="chat-folders-label">
          <div id="chat-folders-label" className="px-1 pb-1 text-[11px] font-medium uppercase tracking-[0.09em] text-[#a8a8ad]">
            Папки
          </div>
          {folders.map((f) => {
            const folderDocs = docs.filter((document) => document.folder_id === f.id)
            const active = selectedFolderIds.has(f.id)
            const partial = !active && folderDocs.some((document) => selectedDocIds.has(document.id))
            const expanded = expandedFolders.has(f.id)
            return (
              <div key={f.id} className="flex flex-col">
                <div
                  className={cn(
                    'flex min-h-11 items-center rounded-xl transition md:min-h-9',
                    active || partial ? 'bg-[#4b4ce6]/[0.07]' : 'hover:bg-[#222226]/[0.04]',
                  )}
                >
                  <button
                    type="button"
                    onClick={() => toggleFolder(f, folderDocs)}
                    role="checkbox"
                    aria-checked={partial ? 'mixed' : active}
                    aria-label={`Выбрать папку ${f.name}`}
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg md:h-9 md:w-9"
                  >
                    <span
                      className={cn(
                        'flex h-5 w-5 items-center justify-center rounded-md border transition',
                        active || partial
                          ? 'border-[#4b4ce6] bg-[#4b4ce6]'
                          : 'border-[#d8d8dc] bg-white',
                      )}
                    >
                      {active ? <Check className="h-3 w-3 text-white" /> : partial ? <Minus className="h-3 w-3 text-white" /> : null}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => toggleExpanded(f.id)}
                    aria-label={`${expanded ? 'Свернуть' : 'Раскрыть'} папку ${f.name}`}
                    aria-expanded={expanded}
                    className="flex min-h-11 min-w-0 flex-1 items-center gap-2 rounded-lg py-2 pr-2 text-left md:min-h-9"
                  >
                    {expanded ? <ChevronDown className="h-4 w-4 shrink-0 text-[#222226]/45" /> : <ChevronRight className="h-4 w-4 shrink-0 text-[#222226]/45" />}
                    <FolderIcon className={cn('h-4 w-4 shrink-0', active || partial ? 'text-[#4b4ce6]' : 'text-[#222226]/35')} />
                    <span className="min-w-0 flex-1 truncate text-[14px] font-medium text-[#222226]">{f.name}</span>
                    <span className="shrink-0 text-[11px] text-[#a8a8ad]">{folderDocs.length}</span>
                  </button>
                </div>
                {expanded && (
                  <div className="ml-11 flex flex-col border-l border-[#222226]/10 pl-3 md:ml-9">
                    {folderDocs.length === 0 ? (
                      <p className="px-1 py-2 text-[12px] text-[#a8a8ad]">Нет готовых документов</p>
                    ) : (
                      folderDocs.map((document) => (
                        <DocumentScopeChoice
                          key={document.id}
                          document={document}
                          checked={checked(document)}
                          disabled={active}
                          nested
                          onToggle={() => toggleDoc(document.id)}
                        />
                      ))
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </section>
      )}

      <section className="flex flex-col gap-1" aria-labelledby="chat-documents-label">
        <div id="chat-documents-label" className="px-1 pb-1 text-[11px] font-medium uppercase tracking-[0.09em] text-[#a8a8ad]">
          {folders.length > 0 ? 'Документы без папки' : 'Документы'}
        </div>
        {looseDocs.length === 0 ? (
          <p className="px-1 py-1 text-[12px] text-[#a8a8ad]">
            {docs.length === 0 ? 'Нет готовых документов' : 'Все документы распределены по папкам'}
          </p>
        ) : (
          looseDocs.map((document) => (
            <DocumentScopeChoice
              key={document.id}
              document={document}
              checked={checked(document)}
              onToggle={() => toggleDoc(document.id)}
            />
          ))
        )}
      </section>
      {selectionError && (
        <p role="status" className="rounded-lg bg-amber-50 px-3 py-2 text-[12px] leading-[1.4] text-amber-800">
          {selectionError}
        </p>
      )}
    </div>
  )
}

function DocumentScopeChoice({
  document,
  checked,
  disabled = false,
  nested = false,
  onToggle,
}: {
  document: Document
  checked: boolean
  disabled?: boolean
  nested?: boolean
  onToggle: () => void
}) {
  return (
    <label
      className={cn(
        'flex min-h-11 items-center gap-2.5 rounded-lg px-1 md:min-h-9',
        disabled ? 'cursor-default opacity-60' : 'cursor-pointer',
        nested && 'pr-1',
      )}
    >
      <span
        className={cn(
          'flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition',
          checked ? 'border-[#4b4ce6] bg-[#4b4ce6]' : 'border-[#d8d8dc] bg-white',
        )}
      >
        {checked && <Check className="h-3 w-3 text-white" />}
      </span>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={onToggle} className="sr-only" />
      <FileText className="h-3.5 w-3.5 shrink-0 text-[#222226]/30" aria-hidden="true" />
      <span className="min-w-0 flex-1 truncate text-[13.5px] font-medium leading-[1.4] text-[#222226]" title={document.filename}>
        {document.filename}
      </span>
    </label>
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
  const [searchQuery, setSearchQuery] = useState('')
  const dateFormatter = new Intl.DateTimeFormat('ru-RU', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
  const normalizedQuery = searchQuery.trim().toLocaleLowerCase('ru-RU')
  const visibleSessions = normalizedQuery
    ? sessions.filter((session) => session.title.toLocaleLowerCase('ru-RU').includes(normalizedQuery))
    : sessions

  return (
    <div className="flex flex-col gap-4">
      <Button
        variant="outline"
        size="sm"
        className="min-h-11 w-full justify-start rounded-xl border-[#222226]/10 bg-white px-2.5 text-[#424247] shadow-sm hover:border-[#4b4ce6]/20 hover:bg-[#4b4ce6]/[0.04] md:min-h-9"
        onClick={onNewChat}
        disabled={busy}
      >
        <span
          aria-hidden="true"
          data-testid="new-chat-icon-container"
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-[#4b4ce6]/10 text-[#4b4ce6]"
        >
          <Plus className="h-4 w-4" strokeWidth={2} />
        </span>
        Новый чат
      </Button>
      <label className="relative block">
        <span className="sr-only">Поиск по чатам</span>
        <Search
          aria-hidden="true"
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#222226]/30"
        />
        <input
          type="search"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          placeholder="Поиск по чатам"
          autoComplete="off"
          className="min-h-11 w-full rounded-xl border border-[#222226]/10 bg-white py-2 pl-9 pr-3 text-[13px] text-[#222226] shadow-sm outline-none transition placeholder:text-[#222226]/35 focus:border-[#4b4ce6]/35 focus:ring-2 focus:ring-[#4b4ce6]/10 md:min-h-9"
        />
      </label>
      <ul className="flex flex-col gap-1" aria-label="История чатов">
        {sessions.length === 0 && (
          <li className="rounded-xl border border-dashed border-[#222226]/10 px-3 py-4 text-center text-[13px] text-[#a8a8ad]">
            История пуста
          </li>
        )}
        {sessions.length > 0 && visibleSessions.length === 0 && (
          <li className="rounded-xl border border-dashed border-[#222226]/10 px-3 py-4 text-center text-[13px] text-[#a8a8ad]">
            Чаты не найдены
          </li>
        )}
        {visibleSessions.map((session) => {
          const active = session.id === activeSid
          const folderCount = new Set([
            ...(session.folder_ids ?? []),
            ...(session.folder_id ? [session.folder_id] : []),
          ]).size
          const documentCount = new Set([
            ...(session.document_ids ?? []),
            ...(session.document_id ? [session.document_id] : []),
          ]).size
          const scopeMeta = folderCount
            ? `Папок: ${folderCount}${documentCount ? ` · Документов: ${documentCount}` : ''}`
            : documentCount
              ? `Документов: ${documentCount}`
              : 'Вся библиотека'
          const updatedAt = new Date(session.updated_at)
          const updatedLabel = Number.isNaN(updatedAt.getTime()) ? '' : dateFormatter.format(updatedAt)

          return (
            <li
              key={session.id}
              className={cn(
                'group flex min-h-14 items-center rounded-xl transition md:min-h-[52px]',
                active
                  ? 'bg-[#4b4ce6]/[0.09] ring-1 ring-inset ring-[#4b4ce6]/10'
                  : 'hover:bg-[#222226]/[0.04]',
              )}
            >
              <button
                type="button"
                onClick={() => onOpen(session)}
                title={session.title}
                aria-label={`Открыть чат «${session.title}»`}
                aria-current={active ? 'page' : undefined}
                className="flex min-h-14 min-w-0 flex-1 items-center gap-2.5 rounded-xl px-2 py-1.5 text-left md:min-h-[52px]"
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    'flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] transition',
                    active ? 'bg-[#4b4ce6] text-white shadow-sm' : 'bg-[#222226]/[0.05] text-[#222226]/35',
                  )}
                >
                  <MessagesSquare className="h-4 w-4" />
                </span>
                <span className="min-w-0 flex-1">
                  <span
                    className={cn(
                      'block truncate text-[13.5px] font-medium leading-[1.35]',
                      active ? 'text-[#222226]' : 'text-[#222226]/75',
                    )}
                  >
                    {session.title}
                  </span>
                  <span className="mt-0.5 flex min-w-0 items-center gap-1.5 text-[11px] leading-[1.3] text-[#8f8f95]">
                    <span className="truncate">{scopeMeta}</span>
                    {updatedLabel && (
                      <>
                        <span aria-hidden="true" className="shrink-0 text-[#c1c1c5]">
                          ·
                        </span>
                        <time dateTime={session.updated_at} className="shrink-0">
                          {updatedLabel}
                        </time>
                      </>
                    )}
                  </span>
                </span>
              </button>
              <button
                type="button"
                onClick={(event) => onDelete(session, event)}
                aria-label={`Удалить чат «${session.title}»`}
                title="Удалить чат"
                className="mr-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-[#8f8f95] opacity-70 transition hover:bg-white hover:text-destructive focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#4b4ce6]/40 md:h-9 md:w-9 md:opacity-0 md:group-hover:opacity-70 md:focus-visible:opacity-100"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </li>
          )
        })}
      </ul>
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
  const quantityWarning = splitQuantityWarning(m.content)
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
        <>
          <div className="text-[14.3px] font-medium leading-[1.5] tracking-[-0.2145px] text-[#222226]">
            <Markdown content={quantityWarning.markdown || '…'} />
          </div>
          {quantityWarning.hasWarning && <QuantityWarning />}
        </>
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

function QuantityWarning() {
  return (
    <aside
      role="status"
      aria-live="polite"
      aria-label="Проверьте числовые значения"
      className="relative mt-3 overflow-hidden rounded-xl border border-[#b98217]/30 bg-[#f8f4e8] text-[#3d3320] shadow-[0_1px_0_rgba(34,34,38,0.03)]"
      data-testid="quantity-warning"
    >
      <span aria-hidden="true" className="absolute inset-y-0 left-0 w-1 bg-[#b98217]" />
      <div className="grid grid-cols-[32px_minmax(0,1fr)] gap-3 px-4 py-3 pl-5">
        <span
          aria-hidden="true"
          className="flex h-8 w-8 items-center justify-center rounded-md border border-[#b98217]/25 bg-[#b98217]/10 text-[#8a5f0c]"
        >
          <AlertTriangle className="h-4 w-4" strokeWidth={2} />
        </span>
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <p className="text-[12px] font-bold uppercase leading-5 tracking-[0.08em] text-[#70500f]">
              Проверьте числовые значения
            </p>
            <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[#8a5f0c]/65">
              Контроль источников
            </span>
          </div>
          <p className="mt-0.5 text-[13px] font-medium leading-[1.45] text-[#51452d]">
            {QUANTITY_WARNING_TEXT}
          </p>
        </div>
      </div>
    </aside>
  )
}
