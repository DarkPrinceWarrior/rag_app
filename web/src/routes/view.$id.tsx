import { Fragment, useRef, useState, useEffect } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { api, type Segment } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { PdfPane, type Highlight } from '@/components/PdfPane'

export const Route = createFileRoute('/view/$id')({
  validateSearch: (s: Record<string, unknown>): { seg?: string; page?: number } => ({
    seg: typeof s.seg === 'string' ? s.seg : undefined,
    page: s.page != null ? Number(s.page) : undefined,
  }),
  component: Viewer,
})

const PDF_KINDS = ['pdf_text', 'pdf_scan']

function highlightOf(s: Segment): Highlight | null {
  if (s.bbox && s.bbox.length === 4 && s.page_idx != null && s.page_size?.length === 2)
    return { page: s.page_idx + 1, bbox: s.bbox, pageSize: s.page_size }
  return null
}

function Viewer() {
  const { id } = Route.useParams()
  const { seg, page } = Route.useSearch()
  const [msg, setMsg] = useState('')
  const [cited, setCited] = useState<string | null>(null)
  const [active, setActive] = useState<Highlight | null>(null)

  const docQ = useQuery({ queryKey: ['document', id], queryFn: () => api.getDocument(id) })
  const segsQ = useQuery({ queryKey: ['segments', id], queryFn: () => api.getSegments(id) })
  const isPdf = !!docQ.data && PDF_KINDS.includes(docQ.data.kind)

  // переход от цитаты/поиска: bbox на PDF + скролл и подсветка в тексте
  useEffect(() => {
    if (!segsQ.data) return
    if (seg) {
      const s = segsQ.data.find((x) => x.id === seg)
      if (s) {
        const h = highlightOf(s)
        if (h) setActive(h)
        document.querySelector(`[data-seg="${seg}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
        setCited(seg)
        const t = setTimeout(() => setCited(null), 4000)
        return () => clearTimeout(t)
      }
    } else if (page != null) {
      if (isPdf) setActive({ page, bbox: [], pageSize: [] })
      else document.getElementById(`page-${page}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }
  }, [segsQ.data, seg, page, isPdf])

  async function reexport() {
    setMsg('Экспорт в очереди…')
    await api.reexport(id)
    setMsg('Экспорт пересобирается')
    setTimeout(() => setMsg(''), 4000)
  }

  const header = (
    <div className="sticky top-[49px] z-[5] flex items-center gap-3 border-b bg-card/90 px-5 py-2 backdrop-blur">
      <span className="truncate text-sm font-medium">
        {docQ.data?.filename} · {docQ.data?.status}
      </span>
      <span className="ml-auto text-xs text-primary">{msg}</span>
      <Button size="sm" onClick={reexport}>
        Пересобрать экспорт
      </Button>
    </div>
  )

  if (segsQ.isLoading)
    return (
      <div>
        {header}
        <p className="p-6 text-sm text-muted-foreground">Загрузка…</p>
      </div>
    )

  // PDF: слева канвас оригинала с подсветкой bbox, справа перевод —
  // сплошным документом с сохранением структуры (заголовки/абзацы/таблицы).
  if (isPdf) {
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)]">
          <div className="w-1/2 border-r">
            <PdfPane docId={id} highlight={active} />
          </div>
          <div className="w-1/2 overflow-auto">
            <p className="border-b px-6 py-2 text-xs text-muted-foreground">
              Клик по фрагменту — подсветка в оригинале слева. Правка сохраняется по клику вне поля.
            </p>
            <TranslationDoc
              segs={segsQ.data ?? []}
              cited={cited}
              onSaved={setMsg}
              onPick={(s) => {
                const h = highlightOf(s)
                if (h) setActive(h)
              }}
            />
          </div>
        </div>
      </div>
    )
  }

  // не-PDF (OOXML): оригинал | перевод текстом
  let lastPage: number | null = null
  return (
    <div>
      {header}
      <div className="mx-auto max-w-[1400px] px-4 py-4">
        <div className="space-y-2">
          {segsQ.data?.map((s) => {
            const sep = s.page_idx != null && s.page_idx !== lastPage
            if (sep) lastPage = s.page_idx
            return (
              <div key={s.id}>
                {sep && (
                  <div id={`page-${(s.page_idx ?? 0) + 1}`} className="my-3 text-xs text-muted-foreground">
                    — страница {(s.page_idx ?? 0) + 1} —
                  </div>
                )}
                <SegmentRow s={s} cited={cited === s.id} showSource onSaved={setMsg} />
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// --- Перевод как документ (правая панель PDF-вьювера) ----------------------

function TranslationDoc({
  segs,
  cited,
  onSaved,
  onPick,
}: {
  segs: Segment[]
  cited: string | null
  onSaved: (m: string) => void
  onPick: (s: Segment) => void
}) {
  let lastPage: number | null = null
  return (
    <article className="mx-auto max-w-3xl px-6 py-4 leading-relaxed">
      {segs.map((s) => {
        const sep = s.page_idx != null && s.page_idx !== lastPage
        if (sep) lastPage = s.page_idx
        return (
          <Fragment key={s.id}>
            {sep && (
              <div
                id={`page-${(s.page_idx ?? 0) + 1}`}
                className="my-4 flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground"
              >
                <span className="h-px flex-1 bg-border" />
                страница {(s.page_idx ?? 0) + 1}
                <span className="h-px flex-1 bg-border" />
              </div>
            )}
            <DocBlock s={s} cited={cited === s.id} onSaved={onSaved} onPick={() => onPick(s)} />
          </Fragment>
        )
      })}
    </article>
  )
}

const HEADING_CLASS: Record<number, string> = {
  1: 'mt-5 mb-2 text-2xl font-bold',
  2: 'mt-5 mb-1.5 text-xl font-semibold',
  3: 'mt-4 mb-1 text-lg font-semibold',
}

function DocBlock({
  s,
  cited,
  onSaved,
  onPick,
}: {
  s: Segment
  cited: boolean
  onSaved: (m: string) => void
  onPick: () => void
}) {
  const citeCls = cited ? 'rounded bg-primary/10 ring-1 ring-primary' : ''
  const wrap = '-ml-2 border-l-2 border-transparent pl-2 transition-colors hover:border-border ' + citeCls

  if (s.kind === 'table') {
    return (
      <div data-seg={s.id} onClick={onPick} className={'my-3 ' + wrap}>
        <TableBlock s={s} onSaved={onSaved} />
        {s.needs_review && <ReviewBadge />}
      </div>
    )
  }

  const isHeading = s.kind === 'heading'
  const level = s.heading_level ?? 2
  const typo = isHeading
    ? HEADING_CLASS[level] ?? 'mt-3 mb-1 text-base font-semibold'
    : 'my-2 text-[15px] text-foreground/90'

  return (
    <div data-seg={s.id} onClick={onPick} className={wrap}>
      <Editable
        value={s.translated_text ?? ''}
        segId={s.id}
        className={typo}
        onSaved={onSaved}
      />
      {s.needs_review && <ReviewBadge />}
    </div>
  )
}

function ReviewBadge() {
  return (
    <span className="ml-2 align-middle rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800">
      проверить числа
    </span>
  )
}

/** Инлайн-редактируемый блок текста: правка сохраняется по blur, если изменилась. */
function Editable({
  value,
  segId,
  className,
  onSaved,
}: {
  value: string
  segId: string
  className: string
  onSaved: (m: string) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const orig = useRef(value)

  async function save() {
    const text = ref.current?.textContent ?? ''
    if (text === orig.current) return
    onSaved('Сохранение…')
    try {
      await api.patchSegment(segId, text)
      orig.current = text
      onSaved('Сохранено')
    } catch {
      onSaved('Ошибка сохранения')
    }
    setTimeout(() => onSaved(''), 2000)
  }

  return (
    <div
      ref={ref}
      contentEditable
      suppressContentEditableWarning
      onBlur={save}
      className={
        'inline whitespace-pre-wrap rounded-sm outline-none focus:bg-accent/40 focus:ring-1 focus:ring-primary ' +
        className
      }
    >
      {value}
    </div>
  )
}

/** Таблица перевода как настоящая <table>; правка ячеек реасемблируется в '|'-формат. */
function TableBlock({ s, onSaved }: { s: Segment; onSaved: (m: string) => void }) {
  const ref = useRef<HTMLTableElement>(null)
  const orig = useRef(s.translated_text ?? '')
  const rows = (s.translated_text ?? '')
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => (l.includes(' | ') ? l.split(' | ') : [l]))

  async function save() {
    const tbl = ref.current
    if (!tbl) return
    const text = Array.from(tbl.rows)
      .map((r) => Array.from(r.cells).map((c) => c.textContent ?? '').join(' | '))
      .join('\n')
    if (text === orig.current) return
    onSaved('Сохранение…')
    try {
      await api.patchSegment(s.id, text)
      orig.current = text
      onSaved('Сохранено')
    } catch {
      onSaved('Ошибка сохранения')
    }
    setTimeout(() => onSaved(''), 2000)
  }

  return (
    <table ref={ref} onBlur={save} className="w-full border-collapse text-sm">
      <tbody>
        {rows.map((cells, ri) => (
          <tr key={ri} className={ri === 0 ? 'bg-muted/50 font-medium' : ''}>
            {cells.map((c, ci) => (
              <td
                key={ci}
                contentEditable
                suppressContentEditableWarning
                colSpan={cells.length === 1 ? 99 : 1}
                className="border border-border px-2.5 py-1 align-top outline-none focus:bg-accent/40"
              >
                {c}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// --- OOXML-путь (оригинал | перевод) ---------------------------------------

function SegmentRow({
  s,
  cited,
  showSource,
  onSaved,
  onPick,
}: {
  s: Segment
  cited: boolean
  showSource: boolean
  onSaved: (m: string) => void
  onPick?: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const orig = useRef(s.translated_text ?? '')
  const [review, setReview] = useState(s.needs_review)
  const heading = s.kind === 'heading'

  async function save() {
    const text = ref.current?.textContent ?? ''
    if (text === orig.current) return
    onSaved('Сохранение…')
    try {
      await api.patchSegment(s.id, text)
      orig.current = text
      setReview(false)
      onSaved('Сохранено')
    } catch {
      onSaved('Ошибка сохранения')
    }
    setTimeout(() => onSaved(''), 2000)
  }

  return (
    <div
      data-seg={s.id}
      onClick={onPick}
      className={
        'rounded-lg bg-card px-3 py-2 shadow-sm transition-colors ' +
        (showSource ? 'grid grid-cols-[28px_1fr_1fr] gap-3 ' : 'grid grid-cols-[28px_1fr] gap-3 ') +
        (review ? 'outline outline-2 outline-amber-400 ' : '') +
        (cited ? 'outline outline-2 outline-primary ' : '')
      }
    >
      <div className="pt-1 text-[11px] text-muted-foreground">{s.idx}</div>
      {showSource && (
        <div className={'whitespace-pre-wrap px-2 py-1 text-sm text-foreground/80 ' + (heading ? 'font-bold' : '')}>
          {s.source_text}
          {review && (
            <span className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800">проверить числа</span>
          )}
        </div>
      )}
      <div
        ref={ref}
        contentEditable
        suppressContentEditableWarning
        onBlur={save}
        className={
          'whitespace-pre-wrap rounded-md border border-transparent px-2 py-1 text-sm outline-none hover:border-border focus:border-primary focus:bg-accent/40 ' +
          (heading ? 'font-bold' : '')
        }
      >
        {s.translated_text ?? ''}
      </div>
    </div>
  )
}
