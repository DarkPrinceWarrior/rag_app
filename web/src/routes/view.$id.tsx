import { useRef, useState, useEffect, type ReactNode } from 'react'
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
  if (s.page_idx == null) return null
  if (s.bbox && s.bbox.length === 4 && s.page_size?.length === 2)
    return { page: s.page_idx + 1, bbox: s.bbox, pageSize: s.page_size }
  // нет bbox (старый парс до геометрии) — хотя бы перейти на страницу сегмента
  return { page: s.page_idx + 1, bbox: [], pageSize: [] }
}

function Viewer() {
  const { id } = Route.useParams()
  const { seg, page } = Route.useSearch()
  const [msg, setMsg] = useState('')
  const [cited, setCited] = useState<string | null>(null)
  const [active, setActive] = useState<Highlight | null>(null)
  // какая страница оригинала (слева) показывается — следует за прокруткой перевода
  const [pageHint, setPageHint] = useState(1)
  const rightRef = useRef<HTMLDivElement>(null)

  // scroll-spy: верхняя видимая страница перевода → переключаем оригинал слева
  function syncPage() {
    const cont = rightRef.current
    if (!cont) return
    const threshold = cont.getBoundingClientRect().top + 64
    let cur = 1
    cont.querySelectorAll<HTMLElement>('[data-page]').forEach((el) => {
      if (el.getBoundingClientRect().top <= threshold) cur = Number(el.dataset.page)
    })
    setPageHint(cur)
  }

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

  const segs = segsQ.data ?? []

  // PDF: слева канвас оригинала с подсветкой bbox, справа перевод —
  // сплошным документом с сохранением структуры (заголовки/списки/таблицы).
  if (isPdf) {
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)]">
          <div className="w-1/2 border-r">
            <PdfPane docId={id} highlight={active} pageHint={pageHint} />
          </div>
          <div ref={rightRef} onScroll={syncPage} className="w-1/2 overflow-auto">
            <p className="border-b px-6 py-2 text-xs text-muted-foreground">
              Клик по фрагменту — подсветка в оригинале слева. Правка сохраняется по клику вне поля.
            </p>
            <article className="mx-auto max-w-3xl px-6 py-4">
              <DocFlow
                segs={segs}
                field="translated"
                editable
                citedId={cited}
                onSaved={setMsg}
                onPick={(s) => {
                  const h = highlightOf(s)
                  if (h) setActive(h)
                }}
              />
            </article>
          </div>
        </div>
      </div>
    )
  }

  // не-PDF (OOXML / txt): оригинал | перевод — оба сплошным документом,
  // структура как в оригинале (без разбиения на карточки-секции).
  return (
    <div>
      {header}
      <div className="mx-auto grid max-w-[1600px] grid-cols-2 gap-8 px-6 py-4">
        <section className="border-r pr-8">
          <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Оригинал</div>
          <DocFlow segs={segs} field="source" editable={false} citedId={cited} />
        </section>
        <section>
          <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Перевод</div>
          <DocFlow segs={segs} field="translated" editable citedId={cited} onSaved={setMsg} />
        </section>
      </div>
    </div>
  )
}

// --- Документ-поток: структура как в оригинале -----------------------------

// маркеры списков: явные буллеты (глиф) или ведущее тире — однозначны.
// Нумерацию (1./2.1) НЕ трогаем: даёт ложные срабатывания на измерениях/датах.
const BULLET_RE = /^\s*[•·‣◦▪►○●∙*]\s+|^\s*[–—-]\s+/

function isListItem(kind: string, text: string): boolean {
  return kind === 'paragraph' && BULLET_RE.test(text)
}

const textOf = (s: Segment, field: Field): string =>
  ((field === 'source' ? s.source_text : s.translated_text) ?? '')

type Field = 'source' | 'translated'

function DocFlow({
  segs,
  field,
  editable,
  citedId,
  onSaved,
  onPick,
}: {
  segs: Segment[]
  field: Field
  editable: boolean
  citedId: string | null
  onSaved?: (m: string) => void
  onPick?: (s: Segment) => void
}) {
  const nodes: ReactNode[] = []
  let lastPage: number | null = null
  let i = 0
  while (i < segs.length) {
    const s = segs[i]
    if (s.page_idx != null && s.page_idx !== lastPage) {
      lastPage = s.page_idx
      nodes.push(<PageSep key={`p-${s.id}`} n={s.page_idx + 1} />)
    }
    // группируем подряд идущие списочные пункты в один список
    if (isListItem(s.kind, textOf(s, field))) {
      const items: Segment[] = []
      while (i < segs.length && isListItem(segs[i].kind, textOf(segs[i], field))) {
        items.push(segs[i])
        i++
      }
      nodes.push(
        <ul key={`l-${items[0].id}`} className="my-2 space-y-1">
          {items.map((it) => (
            <ListItem
              key={it.id}
              s={it}
              field={field}
              editable={editable}
              cited={citedId === it.id}
              onSaved={onSaved}
              onPick={onPick}
            />
          ))}
        </ul>,
      )
      continue
    }
    nodes.push(
      <Block
        key={s.id}
        s={s}
        field={field}
        editable={editable}
        cited={citedId === s.id}
        onSaved={onSaved}
        onPick={onPick}
      />,
    )
    i++
  }
  return <>{nodes}</>
}

function PageSep({ n }: { n: number }) {
  return (
    <div
      id={`page-${n}`}
      data-page={n}
      className="my-4 flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground"
    >
      <span className="h-px flex-1 bg-border" />
      страница {n}
      <span className="h-px flex-1 bg-border" />
    </div>
  )
}

// размер + отступ заголовка по уровню (глубже — мельче и с отступом)
const HEADING_CLASS: Record<number, string> = {
  1: 'mt-5 mb-2 text-2xl font-bold',
  2: 'mt-5 mb-1.5 text-xl font-semibold',
  3: 'mt-4 mb-1 text-lg font-semibold ml-3',
  4: 'mt-3 mb-1 text-base font-semibold ml-6',
}
const headingClass = (lvl: number) => HEADING_CLASS[lvl] ?? 'mt-3 mb-1 text-base font-semibold ml-8'

const CITE_CLS = 'rounded bg-primary/10 ring-1 ring-primary'

function Block({
  s,
  field,
  editable,
  cited,
  onSaved,
  onPick,
}: {
  s: Segment
  field: Field
  editable: boolean
  cited: boolean
  onSaved?: (m: string) => void
  onPick?: (s: Segment) => void
}) {
  const wrap =
    '-ml-2 border-l-2 border-transparent pl-2 transition-colors hover:border-border ' + (cited ? CITE_CLS : '')

  if (s.kind === 'table') {
    return (
      <div data-seg={s.id} onClick={() => onPick?.(s)} className={'my-3 ' + wrap}>
        <TableBlock s={s} field={field} editable={editable} onSaved={onSaved} />
        {editable && s.needs_review && <ReviewBadge />}
      </div>
    )
  }

  const isHeading = s.kind === 'heading'
  const typo = isHeading ? headingClass(s.heading_level ?? 2) : 'my-2 text-[15px] leading-relaxed text-foreground/90'

  return (
    <div data-seg={s.id} onClick={() => onPick?.(s)} className={wrap}>
      <Editable value={textOf(s, field)} segId={s.id} className={typo} editable={editable} onSaved={onSaved} />
      {editable && s.needs_review && <ReviewBadge />}
    </div>
  )
}

// списочный пункт: маркер из текста сохраняем (round-trip), даём отступ и висячую строку
function ListItem({
  s,
  field,
  editable,
  cited,
  onSaved,
  onPick,
}: {
  s: Segment
  field: Field
  editable: boolean
  cited: boolean
  onSaved?: (m: string) => void
  onPick?: (s: Segment) => void
}) {
  return (
    <li
      data-seg={s.id}
      onClick={() => onPick?.(s)}
      className={'pl-6 text-[15px] leading-relaxed text-foreground/90 ' + (cited ? CITE_CLS : '')}
      style={{ textIndent: '-1.1rem' }}
    >
      <Editable value={textOf(s, field)} segId={s.id} className="" editable={editable} onSaved={onSaved} />
    </li>
  )
}

function ReviewBadge() {
  return (
    <span className="ml-2 align-middle rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-800">
      проверить числа
    </span>
  )
}

/** Инлайн-редактируемый (или read-only) блок текста; правка по blur, если изменилась. */
function Editable({
  value,
  segId,
  className,
  editable,
  onSaved,
}: {
  value: string
  segId: string
  className: string
  editable: boolean
  onSaved?: (m: string) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const orig = useRef(value)

  async function save() {
    const text = ref.current?.textContent ?? ''
    if (text === orig.current) return
    onSaved?.('Сохранение…')
    try {
      await api.patchSegment(segId, text)
      orig.current = text
      onSaved?.('Сохранено')
    } catch {
      onSaved?.('Ошибка сохранения')
    }
    setTimeout(() => onSaved?.(''), 2000)
  }

  if (!editable)
    return <div className={'whitespace-pre-wrap ' + className}>{value}</div>

  return (
    <div
      ref={ref}
      contentEditable
      suppressContentEditableWarning
      onBlur={save}
      className={
        'whitespace-pre-wrap rounded-sm outline-none focus:bg-accent/40 focus:ring-1 focus:ring-primary ' + className
      }
    >
      {value}
    </div>
  )
}

/** Таблица как настоящая <table>; правка ячеек реасемблируется в '|'-формат. */
function TableBlock({
  s,
  field,
  editable,
  onSaved,
}: {
  s: Segment
  field: Field
  editable: boolean
  onSaved?: (m: string) => void
}) {
  const ref = useRef<HTMLTableElement>(null)
  const orig = useRef(textOf(s, field))
  const rows = textOf(s, field)
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
    onSaved?.('Сохранение…')
    try {
      await api.patchSegment(s.id, text)
      orig.current = text
      onSaved?.('Сохранено')
    } catch {
      onSaved?.('Ошибка сохранения')
    }
    setTimeout(() => onSaved?.(''), 2000)
  }

  return (
    <table ref={ref} onBlur={editable ? save : undefined} className="w-full border-collapse text-sm">
      <tbody>
        {rows.map((cells, ri) => (
          <tr key={ri} className={ri === 0 ? 'bg-muted/50 font-medium' : ''}>
            {cells.map((c, ci) => (
              <td
                key={ci}
                contentEditable={editable}
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
