import { useRef, useState, useEffect, createElement, type ReactNode } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { api, downloadUrl, EXPORT_LABELS, SEGMENTS_LIMIT, type Segment, type SegmentVersion } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Menu, MenuItem, MenuLabel } from '@/components/ui/menu'
import { ConfirmDialog } from '@/components/ui/modal'
import { Download, MoreVertical, Maximize2, Pencil, MessageCircle } from 'lucide-react'
import { PdfPane, type Highlight, type Region } from '@/components/PdfPane'
import { DocAssistant } from '@/components/DocAssistant'
import { XlsxView } from '@/components/XlsxView'
import { PptxView } from '@/components/PptxView'
import { Markdown } from '@/components/Markdown'
import { authFetch } from '@/lib/auth'
import { cleanMath } from '@/lib/cleanMath'
import { cn } from '@/lib/utils'

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
  const navigate = Route.useNavigate()
  const [msg, setMsg] = useState('')
  const [cited, setCited] = useState<string | null>(null)
  const [active, setActive] = useState<Highlight | null>(null)
  // текущая страница (общая для оригинала слева и перевода справа)
  const [page, setPage] = useState(1)
  // «документ (PDF)» / docx «как в Microsoft» — переведённый PDF с другой
  // пагинацией: НЕ синхронизируем с оригиналом, а ходим кросс-навигацией по клику.
  const [docPage, setDocPage] = useState(1) // правая панель pdf_text (reflow)
  const [ruPage, setRuPage] = useState(1) // правая панель docx (view_ru)
  // подсветка цели при переходе по клику между панелями (кросс-навигация)
  const [leftHi, setLeftHi] = useState<Highlight | null>(null)
  const [rightHi, setRightHi] = useState<Highlight | null>(null)
  const [crossSelectedSegId, setCrossSelectedSegId] = useState<string | null>(null)
  // правая панель PDF: вёрстка (переведённый PDF от BabelDOC) или текст (рендер)
  const [rightText, setRightText] = useState(false)
  // «текст»-режим (Figma 41:1317): выделение сегмента гасит остальные (для
  // сравнения/навигации); сама правка перевода — на отдельном экране сегмента
  // (/view/$id/segment/$segId), сюда её больше не открываем инлайн.
  const [selectedSegId, setSelectedSegId] = useState<string | null>(null)
  const [assistantOpen, setAssistantOpen] = useState(false)
  const sourceColRef = useRef<HTMLDivElement>(null)
  const translatedColRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function clearSelectedSegment(event: KeyboardEvent) {
      if (event.key !== 'Escape') return
      setSelectedSegId(null)
      setCrossSelectedSegId(null)
      setLeftHi(null)
      setRightHi(null)
      setActive(null)
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    }
    window.addEventListener('keydown', clearSelectedSegment)
    return () => window.removeEventListener('keydown', clearSelectedSegment)
  }, [])

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
    const documentKindKey = k ? `${id}:${k}` : null
    if (k && defKindRef.current !== documentKindKey) {
      defKindRef.current = documentKindKey
      setRightText(k === 'pdf_text')
      setSelectedSegId(null)
      setCrossSelectedSegId(null)
      setLeftHi(null)
      setRightHi(null)
      setActive(null)
    }
  }, [docQ.data?.kind, id])

  // переход от цитаты/поиска: страница + bbox + подсветка в тексте
  useEffect(() => {
    if (!segsQ.data) return
    if (seg) {
      const s = segsQ.data.find((x) => x.id === seg)
      if (s) {
        // URL/search state intentionally drives the imperative PDF pane position.
        // eslint-disable-next-line react-hooks/set-state-in-effect
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
    }
    // Внутренние RAG-сегменты намеренно отсутствуют в публичном API. Их цитата
    // всё равно должна открыть исходную страницу по page_start.
    if (pageParam != null) {
      if (isPdf) setPage(pageParam)
      else document.getElementById(`page-${pageParam}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
    }
  }, [segsQ.data, seg, pageParam, isPdf])

  // Подтверждение деструктивных/долгих действий вьювера — строгая модалка
  // (вместо нативного confirm): иконка, заголовок, описание «что произойдёт».
  const [confirmAction, setConfirmAction] = useState<{
    title: string
    description?: ReactNode
    points?: ReactNode[]
    warning?: ReactNode
    note?: ReactNode
    confirmLabel: string
    tone: 'default' | 'danger'
    run: () => Promise<void>
  } | null>(null)
  const [confirmBusy, setConfirmBusy] = useState(false)

  async function runConfirm() {
    if (!confirmAction) return
    setConfirmBusy(true)
    try {
      await confirmAction.run()
    } finally {
      setConfirmBusy(false)
      setConfirmAction(null)
    }
  }

  const PARSER_NAMES: Record<string, string> = {
    mineru: 'MinerU2.5-Pro + добор',
    paddle_vl: 'PaddleOCR-VL 1.6',
  }

  function parserName(backend: string): string {
    return PARSER_NAMES[backend] ?? 'недоступный парсер'
  }

  function askReexport() {
    setConfirmAction({
      title: 'Пересобрать перевод и экспорт?',
      tone: 'default',
      confirmLabel: 'Пересобрать',
      description: 'Перевод соберётся заново из текущих сегментов. Сам разбор документа не трогаем.',
      points: [
        'Сегменты и распознавание остаются как есть',
        'Все сегменты переводятся заново',
        <>Пересобираются файлы экспорта: <b className="font-medium text-foreground">PDF</b> и <b className="font-medium text-foreground">DOCX</b> (side-by-side)</>,
      ],
      note: '≈ 1–2 минуты. Обновите страницу после завершения.',
      run: async () => {
        setMsg('Экспорт в очереди…')
        await api.reexport(id)
        setMsg('Экспорт пересобирается')
        setTimeout(() => setMsg(''), 4000)
      },
    })
  }

  function askReparseOcr() {
    setConfirmAction({
      title: 'Переразобрать через OCR-распознавание?',
      tone: 'danger',
      confirmLabel: 'Переразобрать (OCR)',
      description: 'Для PDF с нечитаемым текстовым слоем — битый cmap, когда текст выглядит как латиница-каша.',
      points: [
        'Документ распознаётся заново постранично (OCR)',
        'Затем — повторный перевод и пересборка экспорта',
      ],
      warning: 'Текущие сегменты, перевод и экспорт будут заменены.',
      note: 'Займёт от минуты. Обновите страницу после завершения.',
      run: async () => {
        setMsg('OCR-переразбор в очереди…')
        await api.reparseOcr(id, 'east_slavic')
        setMsg('Переразбор запущен — обновите страницу через ~минуту')
        setTimeout(() => setMsg(''), 8000)
      },
    })
  }

  // Выбор движка парсинга PDF (MinerU / PaddleOCR-VL 1.6).
  function askReparseBackend(backend: string) {
    const cur = docQ.data?.parser_backend || 'mineru'
    if (backend === cur) return
    setConfirmAction({
      title: `Сменить парсер на «${parserName(backend)}»?`,
      tone: 'danger',
      confirmLabel: 'Переразобрать',
      description: (
        <>
          Сейчас документ разобран движком{' '}
          <span className="font-medium text-foreground">{parserName(cur)}</span>. Он будет
          полностью переразобран другим парсером.
        </>
      ),
      points: [
        <>Полный переразбор движком <span className="font-medium text-foreground">{parserName(backend)}</span></>,
        'Повторный перевод всех сегментов',
        'Пересборка экспорта (PDF/DOCX)',
      ],
      warning: 'Текущие сегменты, перевод и экспорт будут заменены.',
      note: 'Несколько минут. Обновите страницу после завершения.',
      run: async () => {
        setMsg(`Переразбор через ${parserName(backend)} в очереди…`)
        await api.reparse(id, backend)
        setMsg('Переразбор запущен — обновите страницу через ~минуту')
        setTimeout(() => setMsg(''), 8000)
      },
    })
  }

  const segs = segsQ.data ?? []

  // Переведённая страница, на которой лежит контент страницы оригинала.
  function rightPageForLeft(leftPage: number): number | null {
    const mappedPages = segs
      .flatMap((segment) =>
        segment.loc_left?.page === leftPage - 1 && segment.loc_right ? [segment.loc_right.page] : [],
      )
      .sort((a, b) => a - b)
    if (mappedPages.length) return mappedPages[Math.floor(mappedPages.length / 2)] + 1

    let nearestPage: number | null = null
    let nearestDistance = Infinity
    for (const segment of segs) {
      if (segment.loc_left && segment.loc_right) {
        const distance = Math.abs(segment.loc_left.page - (leftPage - 1))
        if (distance < nearestDistance) {
          nearestDistance = distance
          nearestPage = segment.loc_right.page
        }
      }
    }
    return nearestPage != null ? nearestPage + 1 : null
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
    <div className="sticky top-[49px] z-[5] flex flex-wrap items-center gap-3 border-b bg-card/90 px-5 py-2 backdrop-blur max-md:static max-md:gap-2 max-md:px-3">
      <span className="min-w-0 truncate text-sm font-medium max-md:basis-full">
        {docQ.data?.filename} · {docQ.data?.status}
      </span>
      <span className="ml-auto text-xs text-primary">{msg}</span>
      {isPdfDoc && hasTransPdf && (
        <div className="flex items-center overflow-hidden rounded-md border text-xs">
          <button
            onClick={() => {
              // вход в layout-режим: встать на переведённую страницу с тем же
              // контентом (RU объёмнее — номер не совпадает), а не сбрасывать на 1
              const rp = rightPageForLeft(page)
              if (rp != null) setDocPage(rp)
              changeViewerMode(false)
            }}
            title="Переведённый документ как PDF: заголовки, абзацы и таблицы с переносом (собран из перевода). Своя пагинация — для постраничного сравнения с оригиналом удобнее «текст»."
            className={'px-2.5 py-1 max-md:min-h-11 ' + (!rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            документ (PDF)
          </button>
          <button
            onClick={() => changeViewerMode(true)}
            title="Интерактивный перевод постранично, синхронно с оригиналом: заголовки, абзацы, таблицы, сноски без переполнения. Рекомендуется."
            className={'px-2.5 py-1 max-md:min-h-11 ' + (rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            текст
          </button>
        </div>
      )}
      {hasViewOrig && docQ.data?.kind === 'docx' && (
        <div className="flex items-center overflow-hidden rounded-md border text-xs">
          <button
            onClick={() => {
              const rp = rightPageForLeft(page)
              if (rp != null) setRuPage(rp)
              changeViewerMode(false)
            }}
            title="Переведённый документ с сохранённой вёрсткой Word (LibreOffice-рендер). Точная раскладка оригинала."
              className={'px-2.5 py-1 max-md:min-h-11 ' + (!rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            как в Microsoft
          </button>
          <button
            onClick={() => changeViewerMode(true)}
            title="Интерактивный перевод постранично, синхронно с оригиналом: абзацы, таблицы, картинки."
              className={'px-2.5 py-1 max-md:min-h-11 ' + (rightText ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
          >
            текст
          </button>
        </div>
      )}
      {isPdfDoc && (
        <select
          value={
            PARSER_NAMES[docQ.data?.parser_backend || 'mineru']
              ? docQ.data?.parser_backend || 'mineru'
              : ''
          }
          onChange={(e) => askReparseBackend(e.target.value)}
          title="Движок парсинга PDF: переразобрать документ выбранным парсером"
          className="min-h-8 max-w-full rounded-md border bg-background px-2 py-1 text-xs max-md:min-h-11 max-md:flex-1"
        >
          {docQ.data?.parser_backend && !PARSER_NAMES[docQ.data.parser_backend] && (
            <option value="" disabled>парсер: недоступен</option>
          )}
          <option value="mineru">парсер: MinerU+добор</option>
          <option value="paddle_vl">парсер: PaddleOCR-VL 1.6</option>
        </select>
      )}
      {isPdfDoc && (
        <Button className="max-md:flex-1" variant="outline" size="sm" onClick={askReparseOcr} title="Если текст в PDF распознан как латиница-каша">
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
      <Button className="max-md:flex-1" size="sm" onClick={askReexport}>
        Пересобрать экспорт
      </Button>
    </div>
    <ConfirmDialog
      open={!!confirmAction}
      onClose={() => setConfirmAction(null)}
      onConfirm={runConfirm}
      title={confirmAction?.title ?? ''}
      description={confirmAction?.description}
      points={confirmAction?.points}
      warning={confirmAction?.warning}
      note={confirmAction?.note}
      confirmLabel={confirmAction?.confirmLabel}
      tone={confirmAction?.tone}
      busy={confirmBusy}
    />
    </>
  )

  if (segsQ.isLoading)
    return (
      <div>
        {header}
        <p className="p-6 text-sm text-muted-foreground">Загрузка…</p>
      </div>
    )

  // --- Кросс-навигация PDF↔PDF (pdf_text/docx): клик по фрагменту на одной панели
  // подсвечивает его на другой и листает туда (страницы НЕ синхронны — после
  // перевода объём другой). regionsFor — кликабельные сегменты текущей страницы.
  const regionsFor = (side: 'left' | 'right', curPage: number): Region[] =>
    segs.flatMap((s) => {
      const loc = side === 'left' ? s.loc_left : s.loc_right
      return loc && loc.page === curPage - 1 && loc.bbox?.length === 4
        ? [{ segId: s.id, bbox: loc.bbox, pageSize: loc.pagesize }]
        : []
    })
  const crossToRight = (segId: string, setRight: (p: number) => void) => {
    if (crossSelectedSegId === segId) {
      clearCrossSelection()
      return
    }
    const s = segs.find((x) => x.id === segId)
    if (s?.loc_right && s.loc_right.bbox?.length === 4) {
      setCrossSelectedSegId(segId)
      setLeftHi(null)
      setRight(s.loc_right.page + 1)
      setRightHi({ page: s.loc_right.page + 1, bbox: s.loc_right.bbox, pageSize: s.loc_right.pagesize })
    }
  }
  const crossToLeft = (segId: string) => {
    if (crossSelectedSegId === segId) {
      clearCrossSelection()
      return
    }
    const s = segs.find((x) => x.id === segId)
    if (s?.loc_left && s.loc_left.bbox?.length === 4) {
      setCrossSelectedSegId(segId)
      setRightHi(null)
      setPage(s.loc_left.page + 1)
      setLeftHi({ page: s.loc_left.page + 1, bbox: s.loc_left.bbox, pageSize: s.loc_left.pagesize })
    }
  }
  function clearCrossSelection() {
    setCrossSelectedSegId(null)
    setLeftHi(null)
    setRightHi(null)
    setActive(null)
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
  }
  function clearAllSelection() {
    setSelectedSegId(null)
    clearCrossSelection()
  }
  function changeTextPage(nextPage: number) {
    clearAllSelection()
    setPage(nextPage)
  }
  function changeViewerMode(textMode: boolean) {
    clearAllSelection()
    setRightText(textMode)
  }
  // --- «текст»-режим (Figma 41:1317): выделение сегмента с обеих сторон сразу
  // (для сравнения/навигации); правка перевода — на отдельном экране сегмента. ---
  function scrollSegIntoView(segId: string) {
    for (const ref of [sourceColRef, translatedColRef]) {
      const el = ref.current?.querySelector(`[data-seg="${segId}"]`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }
  function selectSeg(segId: string) {
    setSelectedSegId((cur) => {
      const next = cur === segId ? null : segId
      if (next) setTimeout(() => scrollSegIntoView(next), 0)
      return next
    })
  }
  function goToEditSegment(s: Segment) {
    navigate({ to: '/view/$id/segment/$segId', params: { id, segId: s.id } })
  }

  // PDF: слева оригинал, справа перевод; кросс-навигация по клику (страницы не синхронны).
  if (isPdf) {
    const pageSegs = segs.filter((s) => (s.page_idx ?? 0) === page - 1)
    const pageText = pageSegs
      .map(segPlainText)
      .filter((t) => t.trim())
      .join('\n')
    const maxPage = segs.length ? Math.max(...segs.map((s) => s.page_idx ?? 0)) + 1 : 1
    const textMode = !hasTransPdf || rightText
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)] flex-col max-md:h-auto">
          {textMode && (
            <div className="flex items-center gap-2 border-b bg-card px-4 py-1.5 text-sm">
              <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => changeTextPage(page - 1)}>
                ←
              </Button>
              <span className="text-muted-foreground">
                стр. {page} / {maxPage}
              </span>
              <Button variant="ghost" size="sm" disabled={page >= maxPage} onClick={() => changeTextPage(page + 1)}>
                →
              </Button>
            </div>
          )}
          <div className="flex flex-1 overflow-hidden max-md:flex-col max-md:overflow-visible">
            <div
              className={cn(
                'w-1/2 border-r max-md:w-full max-md:border-b max-md:border-r-0',
                !textMode && 'max-md:min-h-[70vh]',
              )}
            >
              {textMode ? (
                <div ref={sourceColRef} className="h-full overflow-y-auto max-md:h-auto max-md:overflow-visible">
                  <PaneHeader label="Оригинал" lang={docQ.data?.source_lang} />
                  <article className="mx-auto max-w-3xl px-6 py-4">
                    <DocRead segs={pageSegs} field="source" citedId={cited} selectedId={selectedSegId} onSelectSeg={selectSeg} />
                  </article>
                </div>
              ) : (
                <PdfPane
                  docId={id}
                  page={page}
                  fitWidth
                  highlight={leftHi || active}
                  onPageChange={(nextPage) => {
                    clearCrossSelection()
                    setPage(nextPage)
                  }}
                  regions={hasTransPdf ? regionsFor('left', page) : undefined}
                  onRegionClick={(sid) => crossToRight(sid, setDocPage)}
                  onBackgroundClick={clearCrossSelection}
                />
              )}
            </div>
            <div
              className={cn(
                'flex w-1/2 flex-col max-md:w-full',
                !textMode && 'max-md:min-h-[70vh]',
              )}
            >
              {textMode ? (
                <div ref={translatedColRef} className="h-full overflow-y-auto max-md:h-auto max-md:overflow-visible">
                  <PaneHeader label="Перевод" lang="ru" />
                  <article className="mx-auto max-w-3xl px-6 py-4">
                    <DocRead
                      segs={pageSegs}
                      field="translated"
                      citedId={cited}
                      editable
                      selectedId={selectedSegId}
                      onSelectSeg={selectSeg}
                      onStartEdit={goToEditSegment}
                    />
                  </article>
                </div>
              ) : (
                // «документ (PDF)» — reflow-PDF перевода. Своя пагинация (не синхронна с
                // оригиналом); связь — кросс-навигацией по клику (regions + highlight).
                <PdfPane
                  docId={id}
                  urlKind="pdf"
                  label="перевод · документ — кликните фрагмент, чтобы найти его в оригинале"
                  fitWidth
                  page={docPage}
                  highlight={rightHi}
                  onPageChange={(nextPage) => {
                    clearCrossSelection()
                    setDocPage(nextPage)
                  }}
                  regions={regionsFor('right', docPage)}
                  onRegionClick={crossToLeft}
                  onBackgroundClick={clearCrossSelection}
                />
              )}
            </div>
          </div>
        </div>
        {textMode && (
          <EditBar
            editing={false}
            saving={false}
            onCancel={() => {}}
            onSave={() => {}}
            showSaveCancel={false}
            assistantOpen={assistantOpen}
            onToggleAssistant={() => setAssistantOpen((o) => !o)}
          />
        )}
        <DocAssistant
          docId={id}
          page={page}
          pageText={pageText}
          filename={docQ.data?.filename}
          // Вне «текст»-режима триггер открытия — свой плавающий лаунчер
          // DocAssistant (нижнего бара с «Открыть» здесь нет).
          open={textMode ? assistantOpen : undefined}
          onOpenChange={textMode ? setAssistantOpen : undefined}
        />
      </div>
    )
  }

  // DOCX с PDF-рендером: слева оригинал (LibreOffice-PDF), справа переключатель
  // «текст» (MD-просмотр: абзацы/таблицы/картинки, как у PDF) ↔ «как в Microsoft»
  // (office-PDF перевода). Сегментам на экспорте проставлен page_idx (физ. страница
  // оригинала) — поэтому правый «текст» листается СИНХРОННО с левым, как у PDF.
  if (hasViewOrig && docQ.data?.kind === 'docx') {
    const pageSegs = segs.filter((s) => (s.page_idx ?? 0) === page - 1)
    const maxPage = segs.length ? Math.max(...segs.map((s) => s.page_idx ?? 0)) + 1 : 1
    const textMode = rightText
    return (
      <div>
        {header}
        <div className="flex h-[calc(100vh-97px)] flex-col max-md:h-auto">
          {textMode && (
            <div className="flex items-center gap-2 border-b bg-card px-4 py-1.5 text-sm">
              <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => changeTextPage(page - 1)}>
                ←
              </Button>
              <span className="text-muted-foreground">
                стр. {page} / {maxPage}
              </span>
              <Button variant="ghost" size="sm" disabled={page >= maxPage} onClick={() => changeTextPage(page + 1)}>
                →
              </Button>
            </div>
          )}
          <div className="flex flex-1 overflow-hidden max-md:flex-col max-md:overflow-visible">
            <div
              aria-label="Оригинал документа"
              className={cn(
                'w-1/2 border-r max-md:w-full max-md:border-b max-md:border-r-0',
                !textMode && 'max-md:min-h-[70vh]',
              )}
            >
              {textMode ? (
                <div ref={sourceColRef} className="h-full overflow-y-auto max-md:h-auto max-md:overflow-visible">
                  <PaneHeader label="Оригинал" lang={docQ.data?.source_lang} />
                  <article className="mx-auto max-w-3xl px-6 py-4">
                    {pageSegs.length === 0 ? (
                      <p className="text-sm text-muted-foreground">На этой странице нет текста.</p>
                    ) : (
                      <DocRead segs={pageSegs} field="source" citedId={cited} selectedId={selectedSegId} onSelectSeg={selectSeg} />
                    )}
                  </article>
                </div>
              ) : (
                <PdfPane
                  docId={id}
                  urlKind="view_orig"
                  label="оригинал"
                  fitWidth
                  page={page}
                  highlight={leftHi}
                  onPageChange={(nextPage) => {
                    clearCrossSelection()
                    setPage(nextPage)
                  }}
                  regions={regionsFor('left', page)}
                  onRegionClick={(sid) => crossToRight(sid, setRuPage)}
                  onBackgroundClick={clearCrossSelection}
                />
              )}
            </div>
            <div
              aria-label="Перевод документа"
              className={cn(
                'flex w-1/2 flex-col max-md:w-full',
                !textMode && 'max-md:min-h-[70vh]',
              )}
            >
              {/* Тумблер «текст | как в Microsoft» — в шапке. «Как в Microsoft» = view_ru
                  со СВОЕЙ пагинацией (ruPage, не синхронна с оригиналом — объём после
                  перевода другой); связь — кросс-навигацией по клику. */}
              {!textMode ? (
                hasViewRu ? (
                  <PdfPane
                    docId={id}
                    urlKind="view_ru"
                    label="перевод — кликните фрагмент, чтобы найти его в оригинале"
                    fitWidth
                    page={ruPage}
                    highlight={rightHi}
                    onPageChange={(nextPage) => {
                      clearCrossSelection()
                      setRuPage(nextPage)
                    }}
                    regions={regionsFor('right', ruPage)}
                    onRegionClick={crossToLeft}
                    onBackgroundClick={clearCrossSelection}
                  />
                ) : (
                  <ViewPending text="Перевод «как в Microsoft» ещё готовится — выберите «текст» или подождите." />
                )
              ) : (
                <div ref={translatedColRef} className="h-full overflow-y-auto max-md:h-auto max-md:overflow-visible">
                  <PaneHeader label="Перевод" lang="ru" />
                  <article className="mx-auto max-w-3xl px-6 py-4">
                    {pageSegs.length === 0 ? (
                      <p className="text-sm text-muted-foreground">На этой странице нет текста для перевода.</p>
                    ) : (
                      <DocRead
                        segs={pageSegs}
                        field="translated"
                        citedId={cited}
                        editable
                        selectedId={selectedSegId}
                        onSelectSeg={selectSeg}
                        onStartEdit={goToEditSegment}
                      />
                    )}
                  </article>
                </div>
              )}
            </div>
          </div>
        </div>
        {textMode && (
          <EditBar
            editing={false}
            saving={false}
            onCancel={() => {}}
            onSave={() => {}}
            showSaveCancel={false}
            assistantOpen={assistantOpen}
            onToggleAssistant={() => setAssistantOpen((o) => !o)}
          />
        )}
        <DocAssistant
          docId={id}
          page={page}
          pageText={pageSegs.map(segPlainText).filter((t) => t.trim()).join('\n')}
          filename={docQ.data?.filename}
          open={textMode ? assistantOpen : undefined}
          onOpenChange={textMode ? setAssistantOpen : undefined}
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
        <div className="flex h-[calc(100vh-97px)] max-md:h-auto max-md:flex-col">
          <div aria-label="Оригинал документа" className="w-1/2 border-r max-md:min-h-[70vh] max-md:w-full max-md:border-b max-md:border-r-0">
            <PdfPane
              docId={id}
              urlKind="view_orig"
              label="оригинал"
              fitWidth
              page={page}
              highlight={null}
              onPageChange={setPage}
            />
          </div>
          <div aria-label="Перевод документа" className="w-1/2 max-md:min-h-[70vh] max-md:w-full">
            {hasViewRu ? (
              <PdfPane
                docId={id}
                urlKind="view_ru"
                label="перевод"
                fitWidth
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
      <div className="mx-auto grid max-w-[1600px] grid-cols-2 gap-8 px-6 py-4 max-md:grid-cols-1 max-md:gap-6 max-md:px-4">
        <section className="border-r pr-8 max-md:border-b max-md:border-r-0 max-md:pb-6 max-md:pr-0">
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

export type Field = 'source' | 'translated'

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
    .map((ln) => {
      if (!/^\s*\|/.test(ln)) return ln
      // строку-разделитель GFM (|---|:--:|…) НЕ трогаем — иначе экранированные пайпы
      // ломают распознавание таблицы и она рендерится сырым текстом
      if (/^\s*\|?[\s:|-]+$/.test(ln)) return ln
      return ln.replace(/([^\s|])\|([^\s|])/g, '$1\\|$2')
    })
    .join('\n')
}

// CommonMark схлопывает одиночный \n внутри абзаца в пробел (soft break) — из-за
// этого многострочные сегменты слипаются: сноски (⁵ … ⁶ …) текут в одну строку, а
// «**Метка:**\n*[плейсхолдер]*» рендерится в одну строку. Превращаем ОДИНОЧНЫЙ \n
// в жёсткий перенос (два пробела + \n = <br>); двойной \n (разрыв абзаца) не трогаем.
function mdHardBreaks(md: string): string {
  return md.replace(/([^\n])\n(?!\n)/g, '$1  \n')
}

// Рамка выделения/затемнения сегмента (Figma 41:1317): пока что-то выделено,
// остальные сегменты гаснут (opacity-30); выделенный подсвечивается —
// нейтрально для оригинала, синим для перевода (там же появляется «Редактировать»).
function segBoxCls(field: Field, selectedId: string | null | undefined, segId: string, cited: boolean): string {
  const isSelected = selectedId === segId
  const isDimmed = selectedId != null && !isSelected
  return cn(
    'rounded-lg transition-opacity cursor-pointer',
    isDimmed && 'opacity-30',
    cited && CITE_CLS,
    isSelected && (field === 'source' ? 'bg-[#222226]/[0.02] border border-[#222226]/[0.22] p-3' : 'bg-[#392dc1]/[0.06] border border-[#4b4ce6] p-3'),
  )
}

export function DocRead({
  segs,
  citedId,
  plain = false,
  editable = false,
  field = 'translated',
  selectedId = null,
  onSelectSeg,
  editingId = null,
  pendingText = '',
  onPendingTextChange,
  onStartEdit,
}: {
  segs: Segment[]
  citedId: string | null
  plain?: boolean // DOCX: абзацы как обычный текст (без Markdown/формул), быстро
  editable?: boolean // правка перевода: кнопка «Редактировать» + история (§4.7.2)
  field?: Field // 'source' — колонка оригинала (без правки), 'translated' — перевод
  selectedId?: string | null
  onSelectSeg?: (segId: string) => void
  editingId?: string | null
  pendingText?: string
  onPendingTextChange?: (t: string) => void
  onStartEdit?: (s: Segment) => void
}) {
  const nodes: ReactNode[] = []
  let i = 0
  while (i < segs.length) {
    const s = segs[i]
    const pick = () => onSelectSeg?.(s.id)
    const boxCls = (segId: string) => segBoxCls(field, selectedId, segId, citedId === segId)

    // DOCX-таблица: ячейки лежат подряд как сегменты с location.t — собираем
    // обратно в таблицу (грид по r/c, несколько абзацев в ячейке склеиваем).
    if (s.location && s.location.t != null) {
      const t = s.location.t
      const cells: Segment[] = []
      while (i < segs.length && segs[i].location?.t === t) {
        cells.push(segs[i])
        i++
      }
      const rowIndexes = cells.flatMap((c) => (c.location?.r != null ? [c.location.r] : []))
      // DOCX-таблица может продолжаться на другой физической странице. Полное
      // table_size нужно для пустых колонок, но не для строк текущего pageSegs:
      // иначе на каждой странице рисовалась вся таблица с пустыми строками до/после.
      const minR = rowIndexes.length ? Math.min(...rowIndexes) : 0
      const maxR = rowIndexes.length ? Math.max(...rowIndexes) : minR
      const maxC = Math.max(
        0,
        ...cells.flatMap((c) => (c.location?.c != null ? [c.location.c] : [])),
        ...cells.map((c) => (c.table_size?.[1] ?? 0) - 1),
      )
      const grid: string[][] = Array.from(
        { length: maxR - minR + 1 },
        () => Array(maxC + 1).fill(''),
      )
      for (const c of cells) {
        const r = (c.location?.r ?? minR) - minR
        const col = c.location?.c ?? 0
        const txt = textOf(c, field) || textOf(c, field === 'source' ? 'translated' : 'source')
        grid[r][col] = grid[r][col] ? grid[r][col] + '\n' + txt : txt
      }
      nodes.push(
        <div key={`t-${cells[0].id}`} data-seg={cells[0].id} onClick={() => pickAt(cells[0])} className={cn('my-3 overflow-x-auto', boxCls(cells[0].id))}>
          <table className="border-collapse text-sm">
            <tbody>
              {grid.map((row, ri) => (
                <tr
                  key={ri}
                  className={minR === 0 && ri === 0 ? 'bg-muted/50 font-medium' : ''}
                >
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
    if (isListItem(s.kind, textOf(s, field))) {
      const items: Segment[] = []
      while (i < segs.length && isListItem(segs[i].kind, textOf(segs[i], field))) {
        items.push(segs[i])
        i++
      }
      nodes.push(
        <ul key={`l-${items[0].id}`} className="my-2 list-disc space-y-1 pl-6">
          {items.map((it) => (
            <li key={it.id} data-seg={it.id} onClick={() => onSelectSeg?.(it.id)} className={boxCls(it.id)}>
              <Markdown content={textOf(it, field).replace(BULLET_RE, '')} />
            </li>
          ))}
        </ul>,
      )
      continue
    }

    if (s.kind === 'equation') {
      nodes.push(
        <div key={s.id} data-seg={s.id} onClick={pick} className={cn('my-3 overflow-x-auto', boxCls(s.id))}>
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
      const trows = (field === 'source' ? s.table_cells : s.table_cells_ru ?? s.table_cells) ?? []
      const clauseTable = trows.flat().some((c) => (c?.text || '').length > 200)
      nodes.push(
        <div key={s.id} data-seg={s.id} onClick={pick} className={cn('my-3 overflow-x-auto', boxCls(s.id))}>
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
            <TableBlock s={s} field={field} editable={false} />
          )}
        </div>,
      )
      i++
      continue
    }

    if (s.kind === 'image') {
      const cap = textOf(s, field) || textOf(s, field === 'source' ? 'translated' : 'source')
      // Описание скана/картинки от VL (Qwen-VL) приходит готовым Markdown —
      // заголовки, таблица, списки. Рендерим его как Markdown (таблица = таблица),
      // а не сплошной плоской подписью по центру. Короткие реальные подписи
      // («Рис. 1. …») остаются мелким figcaption по центру.
      const richCap = cap.length > 200 || /(^|\n)\s*#{1,6}\s|\n\s*\||\n-{3,}/.test(cap)
      if (s.image_url || cap.trim())
        nodes.push(
          <figure key={s.id} data-seg={s.id} onClick={pick} className={cn('my-4', boxCls(s.id))}>
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

    const canEditHere = editable && field === 'translated'

    if (s.kind === 'heading') {
      const lvl = Math.min(Math.max(s.heading_level ?? 2, 1), 4)
      const htext = textOf(s, field) || textOf(s, field === 'source' ? 'translated' : 'source')
      nodes.push(
        canEditHere ? (
          <TranslatedBlock
            key={s.id}
            s={s}
            body={htext}
            typo={headingClass(lvl)}
            markdown={false}
            selected={selectedId === s.id}
            dimmed={selectedId != null && selectedId !== s.id}
            editing={editingId === s.id}
            pendingText={pendingText}
            onSelect={() => onSelectSeg?.(s.id)}
            onStartEdit={() => onStartEdit?.(s)}
            onPendingTextChange={(t) => onPendingTextChange?.(t)}
          />
        ) : (
          createElement(
            `h${lvl}`,
            { key: s.id, 'data-seg': s.id, onClick: pick, className: cn(headingClass(lvl), boxCls(s.id)) },
            htext,
          )
        ),
      )
      i++
      continue
    }

    // абзац: при canEditHere — Редактировать + история (§4.7.2); иначе DOCX plain / PDF Markdown
    const body = textOf(s, field) || textOf(s, field === 'source' ? 'translated' : 'source')
    nodes.push(
      canEditHere ? (
        <TranslatedBlock
          key={s.id}
          s={s}
          body={body}
          typo="text-[15px] leading-relaxed"
          selected={selectedId === s.id}
          dimmed={selectedId != null && selectedId !== s.id}
          editing={editingId === s.id}
          pendingText={pendingText}
          onSelect={() => onSelectSeg?.(s.id)}
          onStartEdit={() => onStartEdit?.(s)}
          onPendingTextChange={(t) => onPendingTextChange?.(t)}
        />
      ) : plain ? (
        <p key={s.id} data-seg={s.id} onClick={pick} className={cn('my-2 whitespace-pre-line text-[15px] leading-relaxed', boxCls(s.id))}>
          {body}
        </p>
      ) : (
        // my-3 даёт отступ МЕЖДУ абзацами: внутренний <p> Markdown обнуляется
        // его же правилом first/last-child (один абзац = и первый, и последний),
        // поэтому пробел держим на обёртке — иначе абзацы слипаются в «стену».
        <div key={s.id} data-seg={s.id} onClick={pick} className={cn('my-3', boxCls(s.id))}>
          <Markdown content={mdHardBreaks(body)} className="text-[15px] leading-relaxed" />
        </div>
      ),
    )
    i++
  }
  return <>{nodes}</>

  function pickAt(seg: Segment) {
    onSelectSeg?.(seg.id)
  }
}

// Абзац/заголовок перевода в «текст»-режиме (Figma 41:1317): статичный текст,
// при наведении (не по клику — правка дизайнера) — кнопка «Редактировать» и
// история; клик по абзацу выделяет его и гасит остальные (для сравнения);
// правка — textarea, сохранение/отмена — кнопками нижнего бара (не по blur).
function TranslatedBlock({
  s,
  body,
  typo,
  markdown = true,
  selected,
  dimmed = false,
  editing,
  pendingText,
  onSelect,
  onStartEdit,
  onPendingTextChange,
}: {
  s: Segment
  body: string
  typo: string
  markdown?: boolean
  selected: boolean
  dimmed?: boolean
  editing: boolean
  pendingText: string
  onSelect: () => void
  onStartEdit: () => void
  onPendingTextChange: (t: string) => void
}) {
  const [history, setHistory] = useState<SegmentVersion[] | null>(null)
  const [hovered, setHovered] = useState(false)
  const showControls = (selected || hovered) && !editing
  async function openHistory() {
    try {
      setHistory(await api.listSegmentVersions(s.id))
    } catch {
      // история — не критично, просто не открылась
    }
  }
  return (
    <div
      data-seg={s.id}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onSelect}
      className={cn(
        'relative my-2 rounded-lg transition-opacity',
        dimmed && 'opacity-30',
        selected ? 'bg-[#392dc1]/[0.06] border border-[#4b4ce6] p-3' : 'border border-transparent p-3',
      )}
    >
      {s.needs_review && <ReviewBadge />}
      {editing ? (
        <textarea
          autoFocus
          value={pendingText}
          onChange={(e) => onPendingTextChange(e.target.value)}
          rows={Math.max(2, Math.ceil(pendingText.length / 70))}
          className={cn('w-full resize-none whitespace-pre-wrap bg-transparent outline-none', typo)}
        />
      ) : markdown ? (
        <Markdown content={mdHardBreaks(body)} className={typo} />
      ) : (
        <div className={typo}>{body}</div>
      )}
      {showControls && (
        <button
          type="button"
          title="Редактировать перевод"
          onMouseDown={(e) => e.preventDefault()}
          onClick={(e) => {
            e.stopPropagation()
            onStartEdit()
          }}
          className="absolute right-2 top-2 flex h-9 w-9 items-center justify-center rounded-full bg-[#4b4ce6] text-white shadow transition hover:opacity-90 max-md:h-11 max-md:w-11"
        >
          <Pencil className="h-4 w-4" />
        </button>
      )}
      {showControls && (
        <button
          type="button"
          title="История правок перевода"
          onMouseDown={(e) => e.preventDefault()}
          onClick={(e) => {
            e.stopPropagation()
            void openHistory()
          }}
          className="absolute -bottom-2.5 right-11 z-10 flex items-center gap-0.5 rounded border bg-card px-1 py-0.5 text-[10px] leading-none text-muted-foreground shadow-sm hover:bg-accent hover:text-foreground max-md:-bottom-5 max-md:right-14 max-md:min-h-11 max-md:min-w-11 max-md:px-2"
        >
          <Maximize2 className="h-2.5 w-2.5" />
          история
        </button>
      )}
      {history && (
        <div onClick={(e) => e.stopPropagation()} className="absolute bottom-5 right-0 z-30 max-h-72 w-80 overflow-auto rounded-md border bg-card p-2 text-xs shadow-lg">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-medium">История правок ({history.length})</span>
            <button
              type="button"
              aria-label="Закрыть историю правок"
              onClick={() => setHistory(null)}
              className="flex items-center justify-center text-muted-foreground hover:text-foreground max-md:h-11 max-md:w-11"
            >
              ✕
            </button>
          </div>
          {history.length === 0 && <div className="text-muted-foreground">Правок ещё не было.</div>}
          {history.map((h) => (
            <div key={h.id} className="border-t py-1">
              <div className="text-muted-foreground">
                {h.editor} · {new Date(h.created_at).toLocaleString('ru')}
              </div>
              <div className="line-through opacity-60">{(h.old_text ?? '').slice(0, 140)}</div>
              <div className="text-emerald-700">{(h.new_text ?? '').slice(0, 140)}</div>
              <button
                onClick={() => {
                  onStartEdit()
                  onPendingTextChange(h.old_text ?? '')
                  setHistory(null)
                }}
                className="mt-0.5 text-primary hover:underline"
              >
                восстановить прежний текст
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Заголовок колонки в «текст»-режиме: «Оригинал RU» / «Перевод EN» (Figma 41:1317).
export function PaneHeader({ label, lang }: { label: string; lang?: string | null }) {
  return (
    <div className="sticky top-0 z-[1] flex items-baseline gap-2 bg-card/95 px-6 pb-2 pt-4 text-base font-semibold backdrop-blur">
      <span className="text-foreground">{label}</span>
      {lang && lang !== 'auto' && <span className="uppercase text-muted-foreground/70">{lang}</span>}
    </div>
  )
}

// Нижний плавающий бар «текст»-режима (Figma 41:1317): Вернуть/Сохранить активны
// только во время правки сегмента; ИИ-консультант открывает DocAssistant сбоку.
// showSaveCancel=false — только тумблер ассистента (обычная страница документа
// больше не редактирует инлайн: правка теперь на отдельном экране сегмента).
export function EditBar({
  editing,
  saving,
  onCancel,
  onSave,
  assistantOpen,
  onToggleAssistant,
  showSaveCancel = true,
}: {
  editing: boolean
  saving: boolean
  onCancel: () => void
  onSave: () => void
  assistantOpen: boolean
  onToggleAssistant: () => void
  showSaveCancel?: boolean
}) {
  return (
    <div className="fixed bottom-6 left-1/2 z-30 flex -translate-x-1/2 items-center gap-4 rounded-3xl border bg-card p-2 shadow-[0_7px_14px_rgba(0,0,0,0.07)]">
      {showSaveCancel && (
        <>
          <button
            type="button"
            onClick={onCancel}
            disabled={!editing || saving}
            className="w-[120px] rounded-2xl bg-[#392dc1]/[0.06] px-4 py-2 text-sm font-semibold text-[#4138cd] transition hover:bg-[#392dc1]/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Вернуть
          </button>
          <span className="h-5 w-px bg-border" />
        </>
      )}
      <div className="flex items-center gap-2 px-1">
        <MessageCircle className="h-5 w-5 text-primary" />
        <span className="text-sm font-semibold">ИИ-консультант</span>
        <button
          type="button"
          onClick={onToggleAssistant}
          className="rounded-2xl bg-[#222226]/5 px-4 py-2 text-sm font-semibold text-[#424247] transition hover:bg-[#222226]/10"
        >
          {assistantOpen ? 'Свернуть' : 'Открыть'}
        </button>
      </div>
      {showSaveCancel && (
        <>
          <span className="h-5 w-px bg-border" />
          <button
            type="button"
            onClick={onSave}
            disabled={!editing || saving}
            className="w-[120px] rounded-2xl bg-[#4b4ce6] px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {saving ? 'Сохраняю…' : 'Сохранить'}
          </button>
        </>
      )}
    </div>
  )
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

// Метка «Требует проверки» (Figma 44:1072): сегмент не сошёлся при числовой
// валидации после перевода — ручная правка её снимает (segments.py).
function ReviewBadge() {
  return (
    <span className="mb-2 inline-flex items-center rounded-full bg-[#952d2d]/10 px-2 py-1 text-[11px] font-medium text-[#c43232]">
      Требует проверки
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
  const [history, setHistory] = useState<SegmentVersion[] | null>(null)
  const [focused, setFocused] = useState(false)

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

  async function openHistory() {
    try {
      setHistory(await api.listSegmentVersions(segId)) // ТЗ §4.7.2
    } catch {
      onSaved?.('Не удалось загрузить историю')
    }
  }

  function restore(text: string | null) {
    if (ref.current && text != null) {
      ref.current.textContent = text
      void save()
    }
    setHistory(null)
  }

  if (!editable) return <div className={'whitespace-pre-wrap ' + className}>{value}</div>

  // Значок «развернуть историю» — в правом нижнем углу рамки выделенного сегмента.
  // Скрыт по умолчанию (чисто), проявляется при выделении (фокусе), наведении или
  // когда история открыта.
  const vis = focused || history ? 'opacity-100' : 'opacity-0 group-hover:opacity-70'
  return (
    <div className="group relative">
      <div
        ref={ref}
        contentEditable
        suppressContentEditableWarning
        onFocus={() => setFocused(true)}
        onBlur={() => {
          setFocused(false)
          void save()
        }}
        className={
          'whitespace-pre-wrap rounded px-1.5 py-0.5 -mx-1.5 outline-none focus:bg-accent/20 focus:ring-1 focus:ring-primary/50 ' +
          className
        }
      >
        {value}
      </div>
      <button
        type="button"
        title="История правок перевода"
        onMouseDown={(e) => e.preventDefault()}
        onClick={openHistory}
        className={
          'absolute -bottom-2.5 right-0 z-10 flex items-center gap-0.5 rounded border bg-card px-1 py-0.5 text-[10px] leading-none text-muted-foreground shadow-sm transition-opacity hover:bg-accent hover:text-foreground ' +
          vis
        }
      >
        <Maximize2 className="h-2.5 w-2.5" />
        история
      </button>
      {history && (
        <div className="absolute bottom-5 right-0 z-30 max-h-72 w-80 overflow-auto rounded-md border bg-card p-2 text-xs shadow-lg">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-medium">История правок ({history.length})</span>
            <button onClick={() => setHistory(null)} className="text-muted-foreground hover:text-foreground">
              ✕
            </button>
          </div>
          {history.length === 0 && <div className="text-muted-foreground">Правок ещё не было.</div>}
          {history.map((h) => (
            <div key={h.id} className="border-t py-1">
              <div className="text-muted-foreground">
                {h.editor} · {new Date(h.created_at).toLocaleString('ru')}
              </div>
              <div className="line-through opacity-60">{(h.old_text ?? '').slice(0, 140)}</div>
              <div className="text-emerald-700">{(h.new_text ?? '').slice(0, 140)}</div>
              <button
                onClick={() => restore(h.old_text)}
                className="mt-0.5 text-primary hover:underline"
              >
                восстановить прежний текст
              </button>
            </div>
          ))}
        </div>
      )}
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
