import { useRef, useState } from 'react'
import { Link } from '@tanstack/react-router'
import { MessageCircle, X, Loader2 } from 'lucide-react'
import { streamChat } from '@/lib/sse'
import { Button } from '@/components/ui/button'
import { Markdown } from '@/components/Markdown'
import { dedupeCitations } from '@/lib/citations'
import type { Citation } from '@/lib/api'

/** Плавающий ассистент поверх страницы просмотра: чат, привязанный к открытому
 *  документу (document_id). Видит весь документ (RAG по нему), знает текущую
 *  страницу — подсказку о ней подмешиваем в запрос модели, не засоряя реплику.
 *  Переиспользует SSE-машинерию чата (streamChat); сессия общая на время визита. */

interface AMsg {
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  error?: string
}

export function DocAssistant({
  docId,
  page,
  pageText,
  filename,
}: {
  docId: string
  page?: number
  pageText?: string // текст открытой страницы (оригинал/перевод, включая таблицы)
  filename?: string
}) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<AMsg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const sid = useRef<string | null>(null) // общая сессия ассистента на время просмотра
  const bottomRef = useRef<HTMLDivElement>(null)
  const pageRef = useRef(page)
  const pageTextRef = useRef(pageText)
  pageRef.current = page // актуальные страница и её текст на момент отправки
  pageTextRef.current = pageText

  function patchLast(fn: (m: AMsg) => AMsg) {
    setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)))
  }

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    setInput('')
    setBusy(true)
    setMessages((m) => [
      ...m,
      { role: 'user', content: text, citations: [] },
      { role: 'assistant', content: '', citations: [] },
    ])
    // в модель подмешиваем контекст открытой страницы (включая её таблицы),
    // в пузыре показываем чистый текст вопроса. Без текста страницы RAG-ретрив
    // не достаёт нужную страницу по семантике («что за таблица» не якорится).
    // Вопрос идёт ПЕРВЫМ (из него формируется заголовок чата), контекст страницы —
    // следом, как служебный блок для модели.
    const p = pageRef.current
    const pt = (pageTextRef.current ?? '').trim()
    let augmented = text
    if (p && pt) {
      augmented =
        `${text}\n\n` +
        `[Контекст для ответа: пользователь открыл страницу ${p} документа. ` +
        `Отвечай в первую очередь по её содержимому (таблицы — строками через « | »), ` +
        `при необходимости дополняй из остального документа.\n` +
        `Содержимое открытой страницы:\n"""\n${pt.slice(0, 6000)}\n"""]`
    } else if (p) {
      augmented = `${text}\n\n[Контекст: открыта страница ${p} документа.]`
    }
    try {
      await streamChat({ message: augmented, session_id: sid.current, document_id: docId }, (ev) => {
        if (ev.type === 'session') sid.current = ev.session_id
        else if (ev.type === 'delta') patchLast((m) => ({ ...m, content: m.content + ev.text }))
        else if (ev.type === 'done') patchLast((m) => ({ ...m, citations: ev.citations ?? [] }))
        else if (ev.type === 'error') patchLast((m) => ({ ...m, error: ev.detail }))
      })
    } catch (e) {
      patchLast((m) => ({ ...m, error: String(e) }))
    }
    setBusy(false)
    setTimeout(() => bottomRef.current?.scrollIntoView({ block: 'end' }), 0)
  }

  if (!open)
    return (
      <button
        onClick={() => setOpen(true)}
        title="Ассистент по документу"
        className="fixed bottom-5 right-5 z-40 flex h-12 items-center gap-2 rounded-full bg-primary px-4 text-sm font-medium text-primary-foreground shadow-lg transition-opacity hover:opacity-90"
      >
        <MessageCircle className="h-5 w-5" />
        Ассистент
      </button>
    )

  return (
    <div className="fixed bottom-5 right-5 z-40 flex h-[min(72vh,580px)] w-[min(92vw,390px)] flex-col rounded-xl border bg-card shadow-2xl">
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <MessageCircle className="h-4 w-4 text-primary" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">Ассистент по документу</div>
          <div className="truncate text-[11px] text-muted-foreground">
            {page ? `стр. ${page}` : 'весь документ'}
            {filename ? ` · ${filename}` : ''}
          </div>
        </div>
        <button onClick={() => setOpen(false)} title="Свернуть" className="rounded p-1 hover:bg-accent">
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="flex-1 space-y-2.5 overflow-auto px-3 py-3">
        {messages.length === 0 && (
          <p className="mt-6 text-center text-xs leading-relaxed text-muted-foreground">
            Спросите что угодно по этому документу — объяснить таблицу, перевести фрагмент, найти
            раздел. Ассистент видит документ целиком{page ? ` и знает, что открыта стр. ${page}` : ''}.
          </p>
        )}
        {messages.map((m, i) => (
          <ABubble key={i} m={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="flex items-end gap-1.5 border-t p-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void send()
            }
          }}
          rows={1}
          placeholder={page ? `Объясни таблицу на стр. ${page}…` : 'Спросить по документу…'}
          className="max-h-28 flex-1 resize-none rounded-md border bg-background px-2.5 py-1.5 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <Button size="sm" onClick={() => void send()} disabled={busy || !input.trim()}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Спросить'}
        </Button>
      </div>
    </div>
  )
}

function ABubble({ m }: { m: AMsg }) {
  if (m.role === 'user')
    return (
      <div className="ml-auto max-w-[85%] whitespace-pre-wrap rounded-lg bg-primary px-2.5 py-1.5 text-sm text-primary-foreground">
        {m.content}
      </div>
    )
  return (
    <div className="max-w-[92%]">
      {m.error ? (
        <div className="text-sm text-destructive">Ошибка: {m.error}</div>
      ) : (
        <div className="rounded-lg bg-muted px-2.5 py-1.5">
          <Markdown content={m.content || '…'} />
        </div>
      )}
      {m.citations.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {dedupeCitations(m.citations).map((c) => (
            <Link
              key={c.n}
              to="/view/$id"
              params={{ id: c.document_id }}
              search={{ seg: c.segment_ids?.[0], page: c.page_start != null ? c.page_start + 1 : undefined }}
              title={c.heading_path}
              className="rounded border bg-accent/40 px-1.5 py-0.5 text-[11px] hover:bg-accent"
            >
              [{c.n}]{c.page_start != null ? ` стр. ${c.page_start + 1}` : ''}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
