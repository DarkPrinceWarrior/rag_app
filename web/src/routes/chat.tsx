import { useEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type ChatSession, type Citation } from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { streamChat } from '@/lib/sse'
import { Button } from '@/components/ui/button'

export const Route = createFileRoute('/chat')({
  validateSearch: (s: Record<string, unknown>): { doc?: string; sid?: string } => ({
    doc: typeof s.doc === 'string' ? s.doc : undefined,
    sid: typeof s.sid === 'string' ? s.sid : undefined,
  }),
  component: Chat,
})

interface Msg {
  role: 'user' | 'assistant'
  content: string
  trace: string[]
  citations: Citation[]
  error?: string
}

function Chat() {
  const { doc, sid: sidParam } = Route.useSearch()
  const navigate = Route.useNavigate()
  const queryClient = useQueryClient()
  const [docId, setDocId] = useState<string>(doc ?? '')
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [temporary, setTemporary] = useState(false) // временный чат — без памяти
  const [sid, setSid] = useState<string | null>(sidParam ?? null) // активная сессия
  const sessionId = useRef<string | null>(sidParam ?? null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const loadedSid = useRef<string | null>(null) // какую сессию уже подняли из истории

  const sessionsQ = useQuery({ queryKey: ['chat-sessions'], queryFn: api.listSessions })

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
    setDocId(s.document_id ?? '')
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

  const docsQ = useQuery({
    queryKey: ['documents'],
    queryFn: api.listDocuments,
    select: (ds) => ds.filter((d) => d.status === 'done'),
  })

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
      await streamChat({ message: text, session_id: sessionId.current, document_id: docId || null }, (ev) => {
        if (ev.type === 'session') {
          sessionId.current = ev.session_id
          loadedSid.current = ev.session_id // эту сессию уже держим в state — не перечитывать
          setSid(ev.session_id)
          if (isNew) {
            navigate({ search: (prev) => ({ ...prev, sid: ev.session_id }) })
            queryClient.invalidateQueries({ queryKey: ['chat-sessions'] })
          }
        }
        else if (ev.type === 'mode' && ev.mode === 'multi_hop')
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
      }, undefined, temporary)
    } catch (e) {
      patchLast((m) => ({ ...m, error: String(e) }))
    }
    setBusy(false)
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }

  return (
    <div className="mx-auto flex h-[calc(100vh-49px)] max-w-5xl gap-3 px-4">
      {/* Сайдбар истории чатов */}
      <aside className="hidden w-56 shrink-0 flex-col py-3 md:flex">
        <Button variant="outline" size="sm" className="mb-2" onClick={newChat} disabled={busy}>
          + Новый чат
        </Button>
        <div className="flex-1 space-y-0.5 overflow-auto">
          {sessionsQ.data?.length === 0 && (
            <p className="px-1 pt-2 text-xs text-muted-foreground">История пуста</p>
          )}
          {sessionsQ.data?.map((s) => (
            <button
              key={s.id}
              onClick={() => openSession(s)}
              title={s.title}
              className={`block w-full truncate rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent ${
                s.id === sid ? 'bg-accent font-medium text-accent-foreground' : 'text-muted-foreground'
              }`}
            >
              {s.title}
            </button>
          ))}
        </div>
      </aside>

      {/* Колонка чата */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 py-3">
          <span className="text-sm text-muted-foreground">Документ:</span>
          <select
            value={docId}
            onChange={(e) => {
              setDocId(e.target.value)
              newChat()
            }}
            className="h-9 rounded-md border bg-card px-2 text-sm"
          >
            <option value="">Вся библиотека</option>
            {docsQ.data?.map((d) => (
              <option key={d.id} value={d.id}>
                {d.filename}
              </option>
            ))}
          </select>
          <label className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground" title="Не сохранять и не использовать долговременную память в этом чате">
            <input type="checkbox" checked={temporary} onChange={(e) => setTemporary(e.target.checked)} />
            Временный
          </label>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-muted-foreground">Выжимка:</span>
            <Button variant="outline" size="sm" disabled={!sid} onClick={() => exportChat('md')}>
              MD
            </Button>
            <Button variant="outline" size="sm" disabled={!sid} onClick={() => exportChat('docx')}>
              DOCX
            </Button>
          </div>
        </div>

        <div className="flex-1 space-y-3 overflow-auto pb-4">
          {messages.length === 0 && (
            <p className="mt-8 text-center text-sm text-muted-foreground">
              Задайте вопрос по переведённым документам — ответ придёт со ссылками на источники.
            </p>
          )}
          {messages.map((m, i) => (
            <Bubble key={i} m={m} />
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="flex gap-2 border-t py-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), send())}
            placeholder="Например: сравни требования к испытаниям и сведи в таблицу"
            className="flex-1 rounded-md border bg-card px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <Button onClick={send} disabled={busy || !input.trim()}>
            {busy ? '…' : 'Спросить'}
          </Button>
        </div>
      </div>
    </div>
  )
}

function Bubble({ m }: { m: Msg }) {
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
      ) : (
        <div className="whitespace-pre-wrap rounded-lg bg-card px-3 py-2 text-sm shadow-sm">{m.content || '…'}</div>
      )}
      {m.citations.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {m.citations.map((c) => (
            <Link
              key={c.n}
              to="/view/$id"
              params={{ id: c.document_id }}
              search={{ seg: c.segment_ids?.[0], page: c.page_start != null ? c.page_start + 1 : undefined }}
              title={c.heading_path}
              className="rounded-md border bg-accent/40 px-2 py-0.5 text-xs text-accent-foreground hover:bg-accent"
            >
              [{c.n}] {c.filename}
              {c.page_start != null ? ` · стр. ${c.page_start + 1}` : ''}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
