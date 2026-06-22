import { useRef, useState, useEffect, createElement, type ReactNode } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { api, downloadUrl, EXPORT_LABELS, SEGMENTS_LIMIT, type Segment } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Menu, MenuItem, MenuLabel } from '@/components/ui/menu'
import { Download, MoreVertical } from 'lucide-react'
import { PdfPane, type Highlight } from '@/components/PdfPane'
import { DocAssistant } from '@/components/DocAssistant'
import { XlsxView } from '@/components/XlsxView'
import { PptxView } from '@/components/PptxView'
import { Markdown } from '@/components/Markdown'
import { authFetch } from '@/lib/auth'
import { cleanMath } from '@/lib/cleanMath'

export const Route = createFileRoute('/view/$id')({
  validateSearch: (s: Record<string, unknown>): { seg?: string; page?: number } => ({
    seg: typeof s.seg === 'string' ? s.seg : undefined,
    page: s.page != null ? Number(s.page) : undefined,
  }),
  component: Viewer,
})

const PDF_KINDS = ['pdf_text', 'pdf_scan']

// Скачивание экспорта: <a download> не шлёт Bearer → тянем через authFetch в blob.
async function downloadExport(url: string, filename: string): Promise<void> {
  const r = await authFetch(url)
  if (!r.ok) return
  const obj = URL.createObjectURL(await r.blob())
  const a = document.createElement('a')
  a.href = obj
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(obj), 1000)
}

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
  // «документ (PDF)» — reflow из перевода, у него СВОЯ пагинация (не совпадает с
  // оригиналом), поэтому отдельный счётчик, не синхронный с левой панелью.
  const [docPage, setDocPage] = useState(1)
  // правая панель PDF: вёрстка (переведённый PDF от BabelDOC) или текст (рендер)
  const [rightText, setRightText] = useState(false)
  // режим «текст»: просмотр (чистый Markdown+формулы) или правка (DocFlow)
  const [edit, setEdit] = useState(false)

  const docQ = useQuery({ queryKey: ['document', id], queryFn: () => api.getDocument(id) })
  const segsQ = useQuery({ queryKey: ['segments', id], queryFn: () => api.getSegments(id) })
  // вьювер грузит первые SEGMENTS_LIMIT сегментов (бэкстоп против дата-дампов
  // на сотни тысяч ячеек). Если их меньше, чем segment_count документа —
  // показан срез, предупреждаем баннером.
  const loadedSegs = segsQ.data?.length ?? 0
  const totalSegs = docQ.data?.segment_count ?? 0
  // xlsx показывается интерактивным гридом со своим капом — баннер сегментов
  // там не нужен (и сбивает с толку).
  const segsTruncated =
    loadedSegs >= SEGMENTS_LIMIT && totalSegs > loadedSegs && docQ.data?.kind !== 'xlsx'
  const isPdf = !!docQ.data && PDF_KINDS.includes(docQ.data.kind)
  // OOXML с PDF-рендером (LibreOffice) — «как в Microsoft». Оригинал рендерится
  // рано (после парсинга) → показываем его, не дожидаясь перевода; перевод
  // (view_ru) — на экспорте. Раздельные флаги: hasViewOrig / hasViewRu.
  const isOffice = !!docQ.data && ['docx', 'xlsx', 'pptx'].includes(docQ.data.kind)
  const hasViewOrig = isOffice && !!docQ.data?.has_view_orig
  const hasViewRu = isOffice && !!docQ.data?.has_view_ru

  // дефолт правой панели — ПО ПРОИСХОЖДЕНИЮ ФОРМАТА (родной формат = истина вёрстки):
  // - pdf_text (родной PDF) → «текст» (чистый MD-рендер): BabelDOC-вёрстка портит
  //   исходную раскладку, а MD читается чисто;
  // - docx (родной Word) → «как в Microsoft» (office-PDF LibreOffice): точная Word-
  //   вёрстка строго лучше MD-реконструкции; «текст» остаётся опцией;
  // - pdf_scan → «вёрстка» (раскладка чертежа и есть содержимое).
  // Ставится раз на смену типа; ручной тумблер не перетирается.
  const defKindRef = useRef<string | null>(null)
  useEffect(() => {
    const k = docQ.data?.kind
    if (k && defKindRef.current !== k) {
      defKindRef.current = k
      setRightText(k === 'pdf_text')
    }
  }, [docQ.data?.kind])

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

  // Выбор движка парсинга pdf_text/docx (mineru / dots.mocr / PaddleOCR-VL 1.6).
  // Меняет parser_backend на документе и переразбирает: сегменты и перевод заменятся.
  async function reparseBackend(backend: string) {
    const cur = docQ.data?.parser_backend || 'mineru'
    if (backend === cur) return
    const names: Record<string, string> = {
      mineru: 'MinerU2.5-Pro + добор',
      dots_mocr: 'dots.mocr',
      paddle_vl: 'PaddleOCR-VL 1.6',
    }
    if (!confirm(`Переразобрать через «${names[backend]}»? Текущие сегменты и перевод заменятся.`)) return
    setMsg(`Переразбор через ${names[backend]} в очереди…`)
    await api.reparse(id, backend)
    setMsg('Переразбор запущен — обновите страницу через ~минуту')
    setTimeout(() => setMsg(''), 8000)
  }

  const isPdfDoc = !!docQ.data && PDF_KINDS.includes(docQ.data.kind)
  const hasTransPdf = !!docQ.data?.exports.includes('pdf')
  const hasDocxExport = !!docQ.data?.exports.includes('docx')
  const dlStem = (docQ.data?.filename ?? 'документ').replace(/\.[^.]+$/, '')
  const header = (
    <>
    {segsTruncated && (
      <div className="border-b bg-amber-50 px-5 py-1.5 text-center text-xs text-amber-800">
        Показаны первые {loadedSegs.toLocaleString('ru')} сегментов из{' '}
        {totalSegs.toLocaleString('ru')} (документ очень большой — остальные не загружены)
      </div>
    )}
    <div className="sticky top-[49px] z-[5] flex items-center gap-3 border-b bg-card/90 px-5 py-2 backdrop-blur">
      <span className="truncate text-sm font-medium">
        {docQ.data?.filename} · {docQ.data?.status}
      </span>
      <span className="ml-auto text-xs text-primary">{msg}</span>
      {isPdfDoc && hasTransPdf && (
        <div className="flex items-center overflow-hidden rounded-md border text-xs">
          <button
            onClick={() => setRightText(false)}
            title="Переведённый документ как PDF: заголовки, абзацы и таблицы с переносом (собран из перевода). Своя пагинация — для постраничного сравнения с оригиналом удобнее «текст»."
            className={'px-2.5 py-1 ' + (!rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            документ (PDF)
          </button>
          <button
            onClick={() => setRightText(true)}
            title="Интерактивный перевод постранично, синхронно с оригиналом: заголовки, абзацы, таблицы, сноски без переполнения. Рекомендуется."
            className={'px-2.5 py-1 ' + (rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            текст
          </button>
        </div>
      )}
      {(isPdfDoc || docQ.data?.kind === 'docx') && (
        <select
          value={docQ.data?.parser_backend || 'mineru'}
          onChange={(e) => reparseBackend(e.target.value)}
          title="Движок парсинга: переразобрать документ выбранным парсером"
          className="rounded-md border bg-background px-2 py-1 text-xs"
        >
          <option value="mineru">парсер: MinerU+добор</option>
          <option value="dots_mocr">парсер: dots.mocr</option>
          <option value="paddle_vl">парсер: PaddleOCR-VL 1.6</option>
        </select>
      )}
      {isPdfDoc && (
        <Button variant="outline" size="sm" onClick={reparseOcr} title="Если текст в PDF распознан как латиница-каша">
          OCR-распознавание
        </Button>
      )}
      {(hasTransPdf || hasDocxExport) && (
        <Menu trigger={<MoreVertical className="h-4 w-4" />} title="Скачать перевод">
          {(close) => (
            <>
              <MenuLabel>Скачать перевод</MenuLabel>
              {(docQ.data?.exports ?? []).map((k) => (
                <MenuItem
                  key={k}
                  icon={<Download className="h-4 w-4" />}
                  onClick={() => {
                    void downloadExport(downloadUrl(id, k), `${dlStem}.ru.${k}`)
                    close()
                  }}
                >
                  {EXPORT_LABELS[k] ?? k}
                </MenuItem>
              ))}
            </>
          )}
        </Menu>
      )}
      <Button size="sm" onClick={reexport}>
        Пересобрать экспорт
      </Button>
    </div>
    </>
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
              // «документ (PDF)» — reflow-PDF перевода (build_docx → LibreOffice):
              // таблицы/абзацы с переносом, без overflow. Своя пагинация (docPage),
              // не синхронная с оригиналом.
              <PdfPane docId={id} urlKind="pdf" label="перевод · документ" page={docPage} highlight={null} onPageChange={setDocPage} />
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
                  {/* Скан-картинка (pdf_scan) — это просмотр VL-описания, править
                      нечего: тумблер «просмотр/править» только у родного pdf_text. */}
                  {docQ.data?.kind === 'pdf_text' && (
                    <div className="ml-auto flex items-center overflow-hidden rounded-md border text-xs">
                      <button
                        onClick={() => setEdit(false)}
                        className={'px-2 py-0.5 ' + (!edit ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
                      >
                        просмотр
                      </button>
                      <button
                        onClick={() => setEdit(true)}
                        className={'px-2 py-0.5 ' + (edit ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
                      >
                        править
                      </button>
                    </div>
                  )}
                </div>
                <div className="flex-1 overflow-auto">
                  <article className="mx-auto max-w-3xl px-6 py-5">
                    {edit && docQ.data?.kind === 'pdf_text' ? (
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
                    ) : (
                      <DocRead
                        segs={pageSegs}
                        citedId={cited}
                        onPick={(s) => {
                          const h = highlightOf(s)
                          if (h) setActive(h)
                        }}
                      />
                    )}
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

  // DOCX с PDF-рендером: слева оригинал (LibreOffice-PDF), справа переключатель
  // «текст» (MD-просмотр: абзацы/таблицы/картинки, как у PDF) ↔ «как в Microsoft»
  // (office-PDF перевода). Сегментам на экспорте проставлен page_idx (физ. страница
  // оригинала) — поэтому правый «текст» листается СИНХРОННО с левым, как у PDF.
  if (hasViewOrig && docQ.data?.kind === 'docx') {
    const pageSegs = segs.filter((s) => (s.page_idx ?? 0) === page - 1)
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
          <div className="flex w-1/2 flex-col">
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
              <div className="ml-2 flex items-center overflow-hidden rounded-md border text-xs">
                <button
                  onClick={() => setRightText(true)}
                  className={'px-2 py-0.5 ' + (rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
                >
                  текст
                </button>
                <button
                  onClick={() => setRightText(false)}
                  className={'px-2 py-0.5 ' + (!rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
                >
                  как в Microsoft
                </button>
              </div>
              {rightText && (
                <div className="ml-auto flex items-center overflow-hidden rounded-md border text-xs">
                  <button
                    onClick={() => setEdit(false)}
                    className={'px-2 py-0.5 ' + (!edit ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
                  >
                    просмотр
                  </button>
                  <button
                    onClick={() => setEdit(true)}
                    className={'px-2 py-0.5 ' + (edit ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
                  >
                    править
                  </button>
                </div>
              )}
            </div>
            {rightText ? (
              <div className="flex-1 overflow-auto">
                <article className="mx-auto max-w-3xl px-6 py-5">
                  {pageSegs.length === 0 ? (
                    <p className="text-sm text-muted-foreground">На этой странице нет текста для перевода.</p>
                  ) : edit ? (
                    <DocFlow segs={pageSegs} field="translated" editable showPages={false} citedId={cited} onSaved={setMsg} />
                  ) : (
                    <DocRead segs={pageSegs} citedId={cited} />
                  )}
                </article>
              </div>
            ) : hasViewRu ? (
              <PdfPane docId={id} urlKind="view_ru" label="перевод" scale={1.0} page={page} highlight={null} onPageChange={setPage} />
            ) : (
              <ViewPending text="Перевод «как в Microsoft» ещё готовится — выберите «текст» или подождите." />
            )}
          </div>
        </div>
        <DocAssistant
          docId={id}
          page={page}
          pageText={pageSegs.map(segPlainText).filter((t) => t.trim()).join('\n')}
          filename={docQ.data?.filename}
        />
      </div>
    )
  }

  // XLSX → ИНТЕРАКТИВНЫЙ грид (а не office-PDF «принт»): настоящая таблица с
  // вкладками листов, линейкой строк/столбцов, выделением ячеек и синхронной
  // прокруткой панелей оригинал|перевод. Данные тянутся из самих xlsx-файлов.
  if (docQ.data?.kind === 'xlsx') {
    return (
      <div>
        {header}
        <XlsxView docId={id} />
        <DocAssistant docId={id} filename={docQ.data?.filename} />
      </div>
    )
  }

  // PPTX → ИНТЕРАКТИВНЫЙ просмотр слайдов (а не office-PDF «принт»): рейка
  // слайдов + оригинал|перевод блоками (текст/таблица/рисунок), выделяемый текст.
  // Тумблер «как в PowerPoint» оставляет office-PDF для точной вёрстки.
  if (docQ.data?.kind === 'pptx') {
    return (
      <div>
        {header}
        <PptxView docId={id} hasViewOrig={hasViewOrig} hasViewRu={hasViewRu} />
        <DocAssistant docId={id} filename={docQ.data?.filename} />
      </div>
    )
  }

  // Прочие office-форматы с PDF-рендером (фолбэк): оригинал и перевод двумя
  // pdf.js-панелями.
  if (hasViewOrig) {
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
            {hasViewRu ? (
              <PdfPane
                docId={id}
                urlKind="view_ru"
                label="перевод"
                scale={1.0}
                page={page}
                highlight={null}
                onPageChange={setPage}
              />
            ) : (
              <ViewPending text="Перевод ещё готовится…" />
            )}
          </div>
        </div>
        <DocAssistant docId={id} page={page} filename={docQ.data?.filename} />
      </div>
    )
  }

  // OOXML, у которого рендер оригинала ещё не готов (идёт обработка) — показываем
  // статус, а не «сплошной текст» как будто это оригинал. Если обработка
  // завершилась без рендера (LibreOffice недоступен) — падаем в текстовый fallback.
  if (isOffice && docQ.data && !['done', 'error'].includes(docQ.data.status)) {
    return (
      <div>
        {header}
        <ViewPending text="Документ обрабатывается — просмотр «как в Microsoft» готовится…" />
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

// --- Чистый просмотр перевода: «документ как на GitHub» ----------------------
// Заголовки/абзацы — через Markdown (inline-формулы $…$, жирный, ссылки),
// блок-формулы (kind=equation) — KaTeX из исходного LaTeX (формулы не
// переводятся), таблицы — TableBlock со спанами, рисунки — подпись. Read-only;
// правки текста — в DocFlow по тумблеру «править». Это решает и «плавающий»
// BabelDOC, и пропажу формул в плоском тексте.

// LaTeX блок-формулы → $$…$$ для remark-math (источник бывает $$…$$ или \[…\])
function eqMarkdown(s: Segment): string {
  const t = (s.source_text || '')
    .trim()
    .replace(/^\\\[/, '')
    .replace(/\\\]$/, '')
    .replace(/^\$\$/, '')
    .replace(/\$\$$/, '')
    .trim()
  return `$$\n${t}\n$$`
}

// Заглушка-статус, пока office-PDF (оригинал/перевод) ещё рендерится.
function ViewPending({ text }: { text: string }) {
  return (
    <div className="flex h-full items-center justify-center p-8 text-center text-sm text-muted-foreground">
      {text}
    </div>
  )
}

// MinerU кладёт все пункты раздела в одну ячейку без переносов → расставляем
// разрывы перед маркерами пунктов/подпунктов, чтобы 1.1 / 1.2 / (a) / (b) шли
// с новой строки (как в оригинале). Только для длинных «прозовых» ячеек.
function splitClauses(t: string): string {
  return (t || '')
    .replace(/\s*(\((?:[a-zа-я]|[ivxl]{1,4})\))\s*/g, '\n$1 ') // (a) (b) (i) (ii)
    .replace(/\s*(\d{1,2}(?:\.\d{1,2}){1,2})\s+(?=[A-ZА-Я“"«(])/g, '\n$1 ') // 1.1 / 1.1.1
    .replace(/\n{2,}/g, '\n')
    .trim()
}

// Картинка из оригинала: тег <img> не шлёт Bearer, поэтому тянем через
// authFetch → object URL (работает и с включённой авторизацией на проде).
function AuthImage({ src, alt }: { src: string; alt?: string }) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    let obj: string | null = null
    let cancelled = false
    authFetch(src)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((b) => {
        if (!cancelled) {
          obj = URL.createObjectURL(b)
          setUrl(obj)
        }
      })
      .catch(() => {})
    return () => {
      cancelled = true
      if (obj) URL.revokeObjectURL(obj)
    }
  }, [src])
  if (!url) return null
  return <img src={url} alt={alt || ''} className="mx-auto max-h-[460px] rounded border bg-white" />
}

// VL иногда пишет внутри ячейки markdown-таблицы литеральный `|` (например
// «Y|BTC» — металл|лиганд). GFM принимает его за разделитель столбцов → строка
// разъезжается и последний столбец отваливается. Экранируем `|`, окружённый
// непробельными символами, ТОЛЬКО в строках-рядах таблицы (начинаются с `|`),
// не трогая структурные ` | ` и инлайн-код в обычных абзацах.
function escapeTablePipes(md: string): string {
  return md
    .split('\n')
    .map((ln) => (/^\s*\|/.test(ln) ? ln.replace(/([^\s|])\|([^\s|])/g, '$1\\|$2') : ln))
    .join('\n')
}

// CommonMark схлопывает одиночный \n внутри абзаца в пробел (soft break) — из-за
// этого многострочные сегменты слипаются: сноски (⁵ … ⁶ …) текут в одну строку, а
// «**Метка:**\n*[плейсхолдер]*» рендерится в одну строку. Превращаем ОДИНОЧНЫЙ \n
// в жёсткий перенос (два пробела + \n = <br>); двойной \n (разрыв абзаца) не трогаем.
function mdHardBreaks(md: string): string {
  return md.replace(/([^\n])\n(?!\n)/g, '$1  \n')
}

function DocRead({
  segs,
  citedId,
  onPick,
  plain = false,
}: {
  segs: Segment[]
  citedId: string | null
  onPick?: (s: Segment) => void
  plain?: boolean // DOCX: абзацы как обычный текст (без Markdown/формул), быстро
}) {
  const nodes: ReactNode[] = []
  let i = 0
  while (i < segs.length) {
    const s = segs[i]
    const cited = citedId === s.id
    const ring = cited ? ' ' + CITE_CLS : ''
    const pick = () => onPick?.(s)

    // DOCX-таблица: ячейки лежат подряд как сегменты с location.t — собираем
    // обратно в таблицу (грид по r/c, несколько абзацев в ячейке склеиваем).
    if (s.location && s.location.t != null) {
      const t = s.location.t
      const cells: Segment[] = []
      while (i < segs.length && segs[i].location?.t === t) {
        cells.push(segs[i])
        i++
      }
      const maxR = Math.max(0, ...cells.map((c) => c.location?.r ?? 0))
      const maxC = Math.max(0, ...cells.map((c) => c.location?.c ?? 0))
      const grid: string[][] = Array.from({ length: maxR + 1 }, () => Array(maxC + 1).fill(''))
      for (const c of cells) {
        const r = c.location?.r ?? 0
        const col = c.location?.c ?? 0
        const txt = textOf(c, 'translated') || textOf(c, 'source')
        grid[r][col] = grid[r][col] ? grid[r][col] + '\n' + txt : txt
      }
      nodes.push(
        <div key={`t-${cells[0].id}`} className="my-3 overflow-x-auto">
          <table className="border-collapse text-sm">
            <tbody>
              {grid.map((row, ri) => (
                <tr key={ri} className={ri === 0 ? 'bg-muted/50 font-medium' : ''}>
                  {row.map((cell, ci) => (
                    <td key={ci} className="whitespace-pre-line border border-border px-2.5 py-1 align-top">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }

    // буллет-списки — группируем подряд идущие пункты
    if (isListItem(s.kind, textOf(s, 'translated'))) {
      const items: Segment[] = []
      while (i < segs.length && isListItem(segs[i].kind, textOf(segs[i], 'translated'))) {
        items.push(segs[i])
        i++
      }
      nodes.push(
        <ul key={`l-${items[0].id}`} className="my-2 list-disc space-y-1 pl-6">
          {items.map((it) => (
            <li
              key={it.id}
              data-seg={it.id}
              onClick={() => onPick?.(it)}
              className={citedId === it.id ? CITE_CLS : ''}
            >
              <Markdown content={textOf(it, 'translated').replace(BULLET_RE, '')} />
            </li>
          ))}
        </ul>,
      )
      continue
    }

    if (s.kind === 'equation') {
      nodes.push(
        <div key={s.id} data-seg={s.id} onClick={pick} className={'my-3 overflow-x-auto' + ring}>
          <Markdown content={eqMarkdown(s)} />
        </div>,
      )
      i++
      continue
    }

    if (s.kind === 'table') {
      // «Прозовая» таблица договора (MinerU кладёт пункты раздела в одну длинную
      // ячейку): сохраняем 2-столбцовую структуру (номер | текст), но внутри col
      // расставляем переносы по маркерам пунктов. Настоящие таблицы (короткие
      // ячейки) идут штатным TableBlock.
      const trows = s.table_cells_ru ?? s.table_cells ?? []
      const clauseTable = trows.flat().some((c) => (c?.text || '').length > 200)
      nodes.push(
        <div key={s.id} data-seg={s.id} onClick={pick} className={'my-3 overflow-x-auto' + ring}>
          {clauseTable ? (
            // table-fixed + w-full: длинная (1500+ симв.) колонка описания иначе
            // распирает таблицу шире страницы и текст красится за рамку. break-words
            // ломает сверхдлинные токены. Узкая колонка-метка (ci=0) — доля ширины.
            <table className="w-full table-fixed border-collapse text-sm">
              <tbody>
                {trows.map((row, ri) => (
                  <tr key={ri}>
                    {row.map((c, ci) => (
                      <td
                        key={ci}
                        colSpan={c.colspan > 1 ? c.colspan : undefined}
                        // rowspan парсера бывает больше числа строк сегмента (таблицу
                        // разбило по границе страницы) — клампим по остатку строк,
                        // иначе браузер перекашивает раскладку
                        rowSpan={c.rowspan > 1 ? Math.min(c.rowspan, trows.length - ri) : undefined}
                        className={
                          'whitespace-pre-line break-words border border-border px-2.5 py-1.5 align-top leading-relaxed ' +
                          (ci === 0 ? 'w-1/5 font-medium' : '')
                        }
                      >
                        {splitClauses(cleanMath(c.text))}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <TableBlock s={s} field="translated" editable={false} />
          )}
        </div>,
      )
      i++
      continue
    }

    if (s.kind === 'image') {
      const cap = textOf(s, 'translated') || textOf(s, 'source')
      // Описание скана/картинки от VL (Qwen-VL) приходит готовым Markdown —
      // заголовки, таблица, списки. Рендерим его как Markdown (таблица = таблица),
      // а не сплошной плоской подписью по центру. Короткие реальные подписи
      // («Рис. 1. …») остаются мелким figcaption по центру.
      const richCap = cap.length > 200 || /(^|\n)\s*#{1,6}\s|\n\s*\||\n-{3,}/.test(cap)
      if (s.image_url || cap.trim())
        nodes.push(
          <figure key={s.id} data-seg={s.id} onClick={pick} className={'my-4' + ring}>
            {s.image_url && <AuthImage src={s.image_url} alt="" />}
            {cap.trim() &&
              (richCap ? (
                <div className="mt-2">
                  <Markdown content={escapeTablePipes(cap)} className="text-[15px] leading-relaxed" />
                </div>
              ) : (
                <figcaption className="mt-1.5 text-center text-sm text-muted-foreground">{cleanMath(cap)}</figcaption>
              ))}
          </figure>,
        )
      i++
      continue
    }

    if (s.kind === 'heading') {
      const lvl = Math.min(Math.max(s.heading_level ?? 2, 1), 4)
      nodes.push(
        createElement(
          `h${lvl}`,
          { key: s.id, 'data-seg': s.id, onClick: pick, className: headingClass(lvl) + ring },
          textOf(s, 'translated') || textOf(s, 'source'),
        ),
      )
      i++
      continue
    }

    // абзац: DOCX — обычный текст (быстро, без формул); PDF — через Markdown
    const body = textOf(s, 'translated') || textOf(s, 'source')
    nodes.push(
      plain ? (
        <p key={s.id} data-seg={s.id} onClick={pick} className={'my-2 whitespace-pre-line text-[15px] leading-relaxed' + ring}>
          {body}
        </p>
      ) : (
        // my-3 даёт отступ МЕЖДУ абзацами: внутренний <p> Markdown обнуляется
        // его же правилом first/last-child (один абзац = и первый, и последний),
        // поэтому пробел держим на обёртке — иначе абзацы слипаются в «стену».
        <div key={s.id} data-seg={s.id} onClick={pick} className={'my-3' + ring}>
          <Markdown content={mdHardBreaks(body)} className="text-[15px] leading-relaxed" />
        </div>
      ),
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
                  rowSpan={c.rowspan > 1 ? Math.min(c.rowspan, cells.length - ri) : undefined}
                  className="border border-border px-2.5 py-1 align-top break-words"
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
