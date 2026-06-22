import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, slideImageUrl, type Slide, type SlideBlock } from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import { PdfPane } from '@/components/PdfPane'

// <img> не шлёт Bearer → тянем картинку слайда через authFetch в object URL.
function AuthImage({ src }: { src: string }) {
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
  return <img src={url} alt="" className="my-2 max-h-[420px] rounded border bg-white" />
}

function Blocks({
  docId,
  slide,
  side,
}: {
  docId: string
  slide: Slide
  side: 'orig' | 'ru'
}) {
  const title = side === 'orig' ? slide.title : slide.title_ru
  let titleSkipped = false
  return (
    <article className="space-y-2 px-5 py-4 text-sm">
      {title && <h2 className="text-lg font-semibold leading-snug">{title}</h2>}
      {slide.blocks.map((b: SlideBlock, i) => {
        if (b.type === 'image' && b.shape != null) {
          return <AuthImage key={i} src={slideImageUrl(docId, slide.index, b.shape)} />
        }
        if (b.type === 'table' && b.rows) {
          return (
            <div key={i} className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <tbody>
                  {b.rows.map((row, r) => (
                    <tr key={r}>
                      {row.map((cell, c) => (
                        <td
                          key={c}
                          className={
                            'border border-border px-2 py-1 align-top ' +
                            (r === 0 ? 'bg-muted font-medium' : '')
                          }
                        >
                          {side === 'orig' ? cell.orig : cell.ru}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
        if (b.type === 'text' && b.lines) {
          return (
            <div key={i} className="space-y-1">
              {b.lines.map((ln, j) => {
                const txt = side === 'orig' ? ln.orig : ln.ru
                // заголовок слайда уже показан как <h2> — не дублируем
                if (!titleSkipped && title && ln.orig === slide.title) {
                  titleSkipped = true
                  return null
                }
                return (
                  <p
                    key={j}
                    style={{ paddingLeft: ln.level ? ln.level * 16 : 0 }}
                    className={ln.level ? 'list-item list-inside list-disc text-muted-foreground' : ''}
                  >
                    {txt}
                  </p>
                )
              })}
            </div>
          )
        }
        return null
      })}
    </article>
  )
}

export function PptxView({
  docId,
  hasViewOrig,
  hasViewRu,
}: {
  docId: string
  hasViewOrig: boolean
  hasViewRu: boolean
}) {
  const q = useQuery({ queryKey: ['slides', docId], queryFn: () => api.getSlides(docId) })
  const [active, setActive] = useState(0)
  const [mode, setMode] = useState<'text' | 'pdf'>('text')

  if (q.isLoading) return <p className="p-6 text-sm text-muted-foreground">Загрузка слайдов…</p>
  if (q.isError || !q.data) return <p className="p-6 text-sm text-destructive">Не удалось загрузить презентацию.</p>
  const slides = q.data.slides
  if (!slides.length) return <p className="p-6 text-sm text-muted-foreground">В презентации нет слайдов.</p>
  const idx = Math.min(active, slides.length - 1)
  const slide = slides[idx]

  return (
    <div className="flex h-[calc(100vh-97px)]">
      {/* рейка слайдов */}
      <div className="w-48 shrink-0 overflow-y-auto border-r bg-muted/20">
        {slides.map((s, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            className={
              'block w-full border-b px-3 py-2 text-left text-xs ' +
              (i === idx ? 'bg-card font-medium text-foreground' : 'text-muted-foreground hover:bg-muted/50')
            }
          >
            <span className="mr-1 tabular-nums text-muted-foreground">{i + 1}.</span>
            {s.title_ru || s.title || `Слайд ${i + 1}`}
          </button>
        ))}
      </div>

      {/* основная область */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b bg-card px-3 py-1.5 text-sm">
          <Button variant="ghost" size="sm" disabled={idx <= 0} onClick={() => setActive(idx - 1)}>
            ←
          </Button>
          <span className="text-muted-foreground">
            слайд {idx + 1} / {slides.length}
          </span>
          <Button variant="ghost" size="sm" disabled={idx >= slides.length - 1} onClick={() => setActive(idx + 1)}>
            →
          </Button>
          {hasViewOrig && (
            <div className="ml-auto flex items-center overflow-hidden rounded-md border text-xs">
              <button
                onClick={() => setMode('text')}
                className={'px-2.5 py-1 ' + (mode === 'text' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
              >
                интерактивно
              </button>
              <button
                onClick={() => setMode('pdf')}
                className={'px-2.5 py-1 ' + (mode === 'pdf' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent')}
              >
                как в PowerPoint
              </button>
            </div>
          )}
        </div>

        {mode === 'text' ? (
          <div className="flex min-h-0 flex-1">
            <div className="w-1/2 min-w-0 overflow-auto border-r">
              <div className="border-b bg-muted/40 px-3 py-1 text-xs font-medium text-muted-foreground">оригинал</div>
              <Blocks docId={docId} slide={slide} side="orig" />
            </div>
            <div className="w-1/2 min-w-0 overflow-auto">
              <div className="border-b bg-muted/40 px-3 py-1 text-xs font-medium text-muted-foreground">
                {q.data.translated_ready ? 'перевод' : 'перевод · готовится…'}
              </div>
              <Blocks docId={docId} slide={slide} side="ru" />
            </div>
          </div>
        ) : (
          <div className="flex min-h-0 flex-1">
            <div className="w-1/2 border-r">
              <PdfPane docId={docId} urlKind="view_orig" label="оригинал" fitWidth hideToolbar page={idx + 1} highlight={null} onPageChange={(p) => setActive(p - 1)} />
            </div>
            <div className="w-1/2">
              {hasViewRu ? (
                <PdfPane docId={docId} urlKind="view_ru" label="перевод" fitWidth hideToolbar page={idx + 1} highlight={null} onPageChange={(p) => setActive(p - 1)} />
              ) : (
                <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
                  Перевод «как в PowerPoint» ещё готовится…
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
