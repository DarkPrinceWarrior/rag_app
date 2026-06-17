import { useRef, useState, useEffect, type ReactNode } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { api, type Segment } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { PdfPane, type Highlight } from '@/components/PdfPane'
import { DocAssistant } from '@/components/DocAssistant'
import { cleanMath } from '@/lib/cleanMath'

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
  const { seg, page: pageParam } = Route.useSearch()
  const [msg, setMsg] = useState('')
  const [cited, setCited] = useState<string | null>(null)
  const [active, setActive] = useState<Highlight | null>(null)
  // текущая страница (общая для оригинала слева и перевода справа)
  const [page, setPage] = useState(1)
  const [numPages, setNumPages] = useState(0)
  // правая панель PDF: вёрстка (переведённый PDF от BabelDOC) или текст (правки)
  const [rightText, setRightText] = useState(false)

  const docQ = useQuery({ queryKey: ['document', id], queryFn: () => api.getDocument(id) })
  const segsQ = useQuery({ queryKey: ['segments', id], queryFn: () => api.getSegments(id) })
  const isPdf = !!docQ.data && PDF_KINDS.includes(docQ.data.kind)
  // OOXML с PDF-рендером (LibreOffice) — показываем «как в Microsoft»: оригинал и
  // перевод двумя pdf.js-панелями, синхронно, вместо плоского текста.
  const hasOfficeView =
    !!docQ.data && ['docx', 'xlsx', 'pptx'].includes(docQ.data.kind) && !!docQ.data.has_view

  // переход от цитаты/поиска: страница + bbox + подсветка в тексте
  useEffect(() => {
    if (!segsQ.data) return
    if (seg) {
      const s = segsQ.data.find((x) => x.id === seg)
      if (s) {
        if (s.page_idx != null) setPage(s.page_idx + 1)
        const h = highlightOf(s)
        if (h) setActive(h)
        setCited(seg)
        const t = setTimeout(() => {
          document.querySelector(`[data-seg="${seg}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
        }, 50)
        const t2 = setTimeout(() => setCited(null), 4000)
        return () => {
          clearTimeout(t)
          clearTimeout(t2)
        }
      }
    } else if (pageParam != null) {
      if (isPdf) setPage(pageParam)
      else document.getElementById(`page-${pageParam}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }
  }, [segsQ.data, seg, pageParam, isPdf])

  async function reexport() {
    setMsg('Экспорт в очереди…')
    await api.reexport(id)
    setMsg('Экспорт пересобирается')
    setTimeout(() => setMsg(''), 4000)
  }

  async function reparseOcr() {
    if (!confirm('Переразобрать через OCR? Для PDF с нечитаемым текстовым слоем (битый cmap). Текущие сегменты и перевод заменятся.'))
      return
    setMsg('OCR-переразбор в очереди…')
    await api.reparseOcr(id, 'east_slavic')
    setMsg('Переразбор запущен — обновите страницу через ~минуту')
    setTimeout(() => setMsg(''), 8000)
  }

  const isPdfDoc = !!docQ.data && PDF_KINDS.includes(docQ.data.kind)
  const hasTransPdf = !!docQ.data?.exports.includes('pdf')
  const header = (
    <div className="sticky top-[49px] z-[5] flex items-center gap-3 border-b bg-card/90 px-5 py-2 backdrop-blur">
      <span className="truncate text-sm font-medium">
        {docQ.data?.filename} · {docQ.data?.status}
      </span>
      <span className="ml-auto text-xs text-primary">{msg}</span>
      {isPdfDoc && hasTransPdf && (
        <div className="flex items-center overflow-hidden rounded-md border text-xs">
          <button
            onClick={() => setRightText(false)}
            className={'px-2.5 py-1 ' + (!rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            перевод: вёрстка
          </button>
          <button
            onClick={() => setRightText(true)}
            className={'px-2.5 py-1 ' + (rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            текст
          </button>
        </div>
      )}
      {isPdfDoc && (
        <Button variant="outline" size="sm" onClick={reparseOcr} title="Если текст в PDF распознан как латиница-каша">
          OCR-распознавание
        </Button>
      )}
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

  // PDF: слева оригинал постранично, справа перевод ТОЙ ЖЕ страницы; листаются синхронно.
  if (isPdf) {
    const pageSegs = segs.filter((s) => (s.page_idx ?? 0) === page - 1)
    const pageText = pageSegs
      .map(segPlainText)
      .filter((t) => t.trim())
      .join('\n')
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)]">
          <div className="w-1/2 border-r">
            <PdfPane docId={id} page={page} highlight={active} onPageChange={setPage} onNumPages={setNumPages} />
          </div>
          <div className="flex w-1/2 flex-col">
            {hasTransPdf && !rightText ? (
              // перевод с сохранённой вёрсткой (BabelDOC) — отдельной pdf.js-панелью
              <PdfPane docId={id} urlKind="pdf" label="перевод · вёрстка" page={page} highlight={null} onPageChange={setPage} />
            ) : (
              <>
                <div className="flex items-center gap-2 border-b bg-card px-2 py-1.5 text-sm">
                  <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
                    ←
                  </Button>
                  <span className="text-muted-foreground">
                    стр. {page} / {numPages || '…'}
                  </span>
                  <Button variant="ghost" size="sm" disabled={page >= numPages} onClick={() => setPage(page + 1)}>
                    →
                  </Button>
                  <span className="ml-auto text-xs text-muted-foreground">
                    перевод · текст · клик по фрагменту — подсветка слева
                  </span>
                </div>
                <div className="flex-1 overflow-auto">
                  <article className="mx-auto max-w-3xl px-6 py-4">
                    <DocFlow
                      segs={pageSegs}
                      field="translated"
                      editable
                      showPages={false}
                      citedId={cited}
                      onSaved={setMsg}
                      onPick={(s) => {
                        const h = highlightOf(s)
                        if (h) setActive(h)
                      }}
                    />
                  </article>
                </div>
              </>
            )}
          </div>
        </div>
        <DocAssistant docId={id} page={page} pageText={pageText} filename={docQ.data?.filename} />
      </div>
    )
  }

  // OOXML с PDF-рендером (LibreOffice): «как в Microsoft» — оригинал и перевод
  // двумя pdf.js-панелями, листаются синхронно.
  if (hasOfficeView) {
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)]">
          <div className="w-1/2 border-r">
            <PdfPane
              docId={id}
              urlKind="view_orig"
              label="оригинал"
              scale={1.0}
              page={page}
              highlight={null}
              onPageChange={setPage}
              onNumPages={setNumPages}
            />
          </div>
          <div className="w-1/2">
            <PdfPane
              docId={id}
              urlKind="view_ru"
              label="перевод"
              scale={1.0}
              page={page}
              highlight={null}
              onPageChange={setPage}
            />
          </div>
        </div>
        <DocAssistant docId={id} page={page} filename={docQ.data?.filename} />
      </div>
    )
  }

  // не-PDF без рендера (txt, или OOXML до готовности view): оригинал | перевод текстом.
  return (
    <div>
      {header}
      <DocAssistant docId={id} filename={docQ.data?.filename} />
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

// Текст сегмента для контекста ассистента: таблицы — строками через « | »,
// остальное — перевод (или оригинал), с очисткой LaTeX-разметки.
function segPlainText(s: Segment): string {
  const cells = s.table_cells_ru ?? s.table_cells
  if (cells && cells.length)
    return cells.map((row) => row.map((c) => cleanMath(c.text)).join(' | ')).join('\n')
  return cleanMath(s.translated_text || s.source_text || '')
}

function DocFlow({
  segs,
  field,
  editable,
  citedId,
  showPages = true,
  onSaved,
  onPick,
}: {
  segs: Segment[]
  field: Field
  editable: boolean
  citedId: string | null
  showPages?: boolean
  onSaved?: (m: string) => void
  onPick?: (s: Segment) => void
}) {
  const nodes: ReactNode[] = []
  let lastPage: number | null = null
  let i = 0
  while (i < segs.length) {
    const s = segs[i]
    if (showPages && s.page_idx != null && s.page_idx !== lastPage) {
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

  if (!editable) return <div className={'whitespace-pre-wrap ' + className}>{value}</div>

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

/** Таблица с объединёнными ячейками (colSpan/rowSpan). Перевод берётся ПО ПОЗИЦИИ
 *  ячейки из table_cells_ru (а не из ` | `-блоба) — подзаголовки не «уезжают».
 *  Документы без table_cells (старый парс) — старый рендер из текста. */
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
  const cells = field === 'source' ? s.table_cells : (s.table_cells_ru ?? s.table_cells)
  if (!cells || cells.length === 0) return <LegacyTable s={s} field={field} editable={editable} onSaved={onSaved} />
  const caption = field === 'source' ? s.caption : (s.caption_ru ?? s.caption)
  // строки шапки = сколько строк накрывает rowspan первой строки
  const headerRows = Math.max(1, ...cells[0].map((c) => c.rowspan))

  return (
    <div className="my-2 overflow-x-auto">
      {caption && (
        <div className="mb-1 whitespace-pre-line text-xs font-medium text-muted-foreground">{cleanMath(caption)}</div>
      )}
      <table className="border-collapse text-sm">
        <tbody>
          {cells.map((row, ri) => (
            <tr key={ri} className={ri < headerRows ? 'bg-muted/50 font-medium' : ''}>
              {row.map((c, ci) => (
                <td
                  key={ci}
                  colSpan={c.colspan > 1 ? c.colspan : undefined}
                  rowSpan={c.rowspan > 1 ? c.rowspan : undefined}
                  className="border border-border px-2.5 py-1 align-top"
                >
                  {cleanMath(c.text)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Фолбэк для документов без table_cells: ячейки из ` | `-текста, редактируемые. */
function LegacyTable({
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
        {rows.map((rowCells, ri) => (
          <tr key={ri} className={ri === 0 ? 'bg-muted/50 font-medium' : ''}>
            {rowCells.map((c, ci) => (
              <td
                key={ci}
                contentEditable={editable}
                suppressContentEditableWarning
                colSpan={rowCells.length === 1 ? 99 : 1}
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
