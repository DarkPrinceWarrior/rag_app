import { useEffect, useRef, useState } from 'react'
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

  // PDF: слева канвас оригинала с подсветкой bbox, справа правки перевода
  if (isPdf) {
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)]">
          <div className="w-1/2 border-r">
            <PdfPane docId={id} highlight={active} />
          </div>
          <div className="w-1/2 overflow-auto px-4 py-3">
            <p className="mb-2 text-xs text-muted-foreground">
              Клик по фрагменту — подсветка в оригинале слева. Правка сохраняется по клику вне поля.
            </p>
            <div className="space-y-1.5">
              {segsQ.data?.map((s) => (
                <SegmentRow
                  key={s.id}
                  s={s}
                  cited={cited === s.id}
                  showSource={false}
                  onSaved={setMsg}
                  onPick={() => {
                    const h = highlightOf(s)
                    if (h) setActive(h)
                  }}
                />
              ))}
            </div>
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
