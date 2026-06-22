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

// Кликабельный регион сегмента на текущей странице (кросс-навигация панелей).
export interface Region {
  segId: string
  bbox: number[] // [x0,y0,x1,y1] top-left, pt
  pageSize: number[] // [w,h] pt
}

const SCALE = 1.4

/** Контролируемая страница: `page` приходит сверху, стрелки зовут `onPageChange`.
 *  regions — кликабельные сегменты текущей страницы; клик зовёт onRegionClick. */
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
  regions,
  onRegionClick,
}: {
  docId: string
  page: number
  highlight: Highlight | null
  onPageChange: (p: number) => void
  onNumPages?: (n: number) => void
  urlKind?: string // источник PDF: original | view_orig | view_ru | pdf
  label?: string
  scale?: number
  hideToolbar?: boolean // спрятать собственный тулбар (когда счётчик уже снаружи)
  fitWidth?: boolean // вписывать страницу по ширине панели (для широких слайдов)
  regions?: Region[] // кликабельные сегменты на текущей странице (кросс-навигация)
  onRegionClick?: (segId: string) => void
}) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  // текущая операция рендера: ДВА render() на одном canvas одновременно
  // (первый рендер + перерисовка от ResizeObserver) портят кадр — pdf.js
  // выдаёт перевёрнутую/битую страницу. Держим задачу, чтобы отменить
  // предыдущую перед новой и при размонтировании эффекта.
  const renderTaskRef = useRef<RenderTask | null>(null)
  const [numPages, setNumPages] = useState(0)
  const [boxW, setBoxW] = useState(0)
  // CSS-размер отрисованной страницы — для позиционирования оверлеев (bbox).
  const [vp, setVp] = useState<{ w: number; h: number }>({ w: 0, h: 0 })
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

  // рендер текущей страницы (оверлеи — отдельно в JSX по vp)
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
      setVp({ w: Math.floor(vpCss.width), h: Math.floor(vpCss.height) })
    })()
    return () => {
      cancelled = true
      renderTaskRef.current?.cancel()
    }
  }, [page, numPages, scale, fitWidth, boxW])

  if (err) return <div className="p-4 text-sm text-destructive">Не удалось открыть PDF: {err}</div>

  const hi =
    highlight && highlight.page === page && highlight.bbox.length === 4 && highlight.pageSize.length === 2
      ? highlight
      : null

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
          {/* кликабельные регионы сегментов — кросс-навигация на другую панель */}
          {vp.w > 0 &&
            regions?.map((r) =>
              r.bbox?.length === 4 && r.pageSize?.length === 2 ? (
                <button
                  key={r.segId}
                  title="Найти этот фрагмент на другой стороне"
                  onClick={() => onRegionClick?.(r.segId)}
                  className="absolute cursor-pointer rounded-sm transition-colors hover:bg-primary/20 hover:ring-1 hover:ring-primary"
                  style={{
                    left: (r.bbox[0] * vp.w) / r.pageSize[0],
                    top: (r.bbox[1] * vp.h) / r.pageSize[1],
                    width: ((r.bbox[2] - r.bbox[0]) * vp.w) / r.pageSize[0],
                    height: ((r.bbox[3] - r.bbox[1]) * vp.h) / r.pageSize[1],
                  }}
                />
              ) : null,
            )}
          {/* подсветка цели (куда перешли по клику с другой стороны) */}
          {vp.w > 0 && hi && (
            <div
              className="pointer-events-none absolute rounded-sm border-2 border-primary bg-primary/25"
              style={{
                left: (hi.bbox[0] * vp.w) / hi.pageSize[0],
                top: (hi.bbox[1] * vp.h) / hi.pageSize[1],
                width: ((hi.bbox[2] - hi.bbox[0]) * vp.w) / hi.pageSize[0],
                height: ((hi.bbox[3] - hi.bbox[1]) * vp.h) / hi.pageSize[1],
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}
