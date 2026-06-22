import { useEffect, useRef, useState } from 'react'
import * as pdfjs from 'pdfjs-dist'
import type { PDFDocumentProxy, RenderTask } from 'pdfjs-dist'
// worker бандлится локально (?worker) — без CDN (roadmap § 9)
import PdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?worker'
import { bearer } from '@/lib/auth'
import { downloadUrl } from '@/lib/api'
import { Button } from '@/components/ui/button'

pdfjs.GlobalWorkerOptions.workerPort = new PdfjsWorker()

export interface Highlight {
  page: number // 1-based
  bbox: number[] // [x0,y0,x1,y1] в пунктах, origin top-left
  pageSize: number[] // [w,h] в пунктах
}

const SCALE = 1.4

/** Контролируемая страница: `page` приходит сверху, стрелки зовут `onPageChange`
 *  (правая панель перевода листается синхронно). numPages сообщается наверх. */
export function PdfPane({
  docId,
  page,
  highlight,
  onPageChange,
  onNumPages,
  urlKind = 'original',
  label = 'оригинал',
  scale = SCALE,
  hideToolbar = false,
  fitWidth = false,
}: {
  docId: string
  page: number
  highlight: Highlight | null
  onPageChange: (p: number) => void
  onNumPages?: (n: number) => void
  urlKind?: string // источник PDF: original | view_orig | view_ru
  label?: string
  scale?: number
  hideToolbar?: boolean // спрятать собственный тулбар (когда счётчик уже снаружи)
  fitWidth?: boolean // вписывать страницу по ширине панели (для широких слайдов)
}) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  // текущая операция рендера: ДВА render() на одном canvas одновременно
  // (первый рендер + перерисовка от ResizeObserver) портят кадр — pdf.js
  // выдаёт перевёрнутую/битую страницу. Держим задачу, чтобы отменить
  // предыдущую перед новой и при размонтировании эффекта.
  const renderTaskRef = useRef<RenderTask | null>(null)
  const [numPages, setNumPages] = useState(0)
  const [boxW, setBoxW] = useState(0)
  const [err, setErr] = useState('')

  // ширина области просмотра (для fitWidth) — пересчитываем при ресайзе панели
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setBoxW(entries[0].contentRect.width))
    ro.observe(el)
    setBoxW(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const token = await bearer()
        // загрузка по URL: pdf.js тянет страницы лениво порейндж-запросами
        // (сервер отдаёт Accept-Ranges/206) — большой PDF открывается сразу,
        // а не качается целиком.
        const pdf = await pdfjs.getDocument({
          url: downloadUrl(docId, urlKind),
          httpHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
          withCredentials: false,
        }).promise
        if (cancelled) return
        pdfRef.current = pdf
        setNumPages(pdf.numPages)
        onNumPages?.(pdf.numPages)
      } catch (e) {
        if (!cancelled) setErr(String(e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [docId, urlKind, onNumPages])

  // рендер текущей страницы + bbox-оверлей
  useEffect(() => {
    const pdf = pdfRef.current
    const canvas = canvasRef.current
    if (!pdf || !canvas || page < 1 || page > numPages) return
    let cancelled = false
    ;(async () => {
      const pg = await pdf.getPage(page)
      if (cancelled) return
      // рендерим в физических пикселях (×devicePixelRatio) и ужимаем CSS-размером —
      // иначе на retina/масштабе слайды и шрифты выглядят пиксельными/«рваными».
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      // fitWidth: вписать страницу по ширине панели (широкие слайды иначе обрезаются)
      const natW = pg.getViewport({ scale: 1 }).width
      const base = fitWidth && boxW > 0 ? Math.max(0.2, (boxW - 24) / natW) : scale
      const vpCss = pg.getViewport({ scale: base })
      const vpDev = pg.getViewport({ scale: base * dpr })
      canvas.width = Math.floor(vpDev.width)
      canvas.height = Math.floor(vpDev.height)
      canvas.style.width = `${Math.floor(vpCss.width)}px`
      canvas.style.height = `${Math.floor(vpCss.height)}px`
      const ctx = canvas.getContext('2d')!
      // отменяем предыдущий незавершённый рендер на этом canvas (иначе гонка)
      renderTaskRef.current?.cancel()
      const task = pg.render({ canvasContext: ctx, viewport: vpDev, canvas })
      renderTaskRef.current = task
      try {
        await task.promise
      } catch (e) {
        // RenderingCancelledException при отмене — это норма, молча выходим
        if ((e as { name?: string })?.name === 'RenderingCancelledException') return
        throw e
      }
      if (cancelled) return
      const box = boxRef.current!
      if (highlight && highlight.page === page && highlight.bbox.length === 4 && highlight.pageSize.length === 2) {
        const [x0, y0, x1, y1] = highlight.bbox
        const sx = vpCss.width / highlight.pageSize[0]
        const sy = vpCss.height / highlight.pageSize[1]
        box.style.display = 'block'
        box.style.left = `${x0 * sx}px`
        box.style.top = `${y0 * sy}px`
        box.style.width = `${(x1 - x0) * sx}px`
        box.style.height = `${(y1 - y0) * sy}px`
      } else {
        box.style.display = 'none'
      }
    })()
    return () => {
      cancelled = true
      renderTaskRef.current?.cancel()
    }
  }, [page, numPages, highlight, scale, fitWidth, boxW])

  if (err) return <div className="p-4 text-sm text-destructive">Не удалось открыть PDF: {err}</div>

  return (
    <div className="flex h-full flex-col">
      {!hideToolbar && (
        <div className="flex items-center gap-2 border-b bg-card px-2 py-1.5 text-sm">
          <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
            ←
          </Button>
          <span className="text-muted-foreground">
            стр. {page} / {numPages || '…'}
          </span>
          <Button variant="ghost" size="sm" disabled={page >= numPages} onClick={() => onPageChange(page + 1)}>
            →
          </Button>
          <span className="ml-auto text-xs text-muted-foreground">{label}</span>
        </div>
      )}
      {hideToolbar && (
        <div className="border-b bg-muted/40 px-3 py-1 text-xs font-medium text-muted-foreground">{label}</div>
      )}
      <div ref={containerRef} className="flex-1 overflow-auto bg-muted/40 p-3">
        <div className="relative mx-auto w-fit shadow">
          <canvas ref={canvasRef} className="block" />
          <div
            ref={boxRef}
            className="pointer-events-none absolute rounded-sm border-2 border-primary bg-primary/15"
            style={{ display: 'none' }}
          />
        </div>
      </div>
    </div>
  )
}
