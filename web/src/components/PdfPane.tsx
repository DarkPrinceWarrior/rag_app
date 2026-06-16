import { useEffect, useRef, useState } from 'react'
import * as pdfjs from 'pdfjs-dist'
import type { PDFDocumentProxy } from 'pdfjs-dist'
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
}: {
  docId: string
  page: number
  highlight: Highlight | null
  onPageChange: (p: number) => void
  onNumPages?: (n: number) => void
  urlKind?: string // источник PDF: original | view_orig | view_ru
  label?: string
  scale?: number
}) {
  const pdfRef = useRef<PDFDocumentProxy | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const [numPages, setNumPages] = useState(0)
  const [err, setErr] = useState('')

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const token = await bearer()
        const resp = await fetch(downloadUrl(docId, urlKind), {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        if (!resp.ok) throw new Error(`PDF: ${resp.status}`)
        const data = await resp.arrayBuffer()
        const pdf = await pdfjs.getDocument({ data }).promise
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
      const viewport = pg.getViewport({ scale })
      canvas.width = viewport.width
      canvas.height = viewport.height
      const ctx = canvas.getContext('2d')!
      await pg.render({ canvasContext: ctx, viewport, canvas }).promise
      const box = boxRef.current!
      if (highlight && highlight.page === page && highlight.bbox.length === 4 && highlight.pageSize.length === 2) {
        const [x0, y0, x1, y1] = highlight.bbox
        const sx = viewport.width / highlight.pageSize[0]
        const sy = viewport.height / highlight.pageSize[1]
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
    }
  }, [page, numPages, highlight, scale])

  if (err) return <div className="p-4 text-sm text-destructive">Не удалось открыть PDF: {err}</div>

  return (
    <div className="flex h-full flex-col">
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
      <div className="flex-1 overflow-auto bg-muted/40 p-3">
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
