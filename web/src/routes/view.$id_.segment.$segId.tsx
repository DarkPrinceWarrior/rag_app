import { useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { Check } from 'lucide-react'
import { api, type Segment } from '@/lib/api'
import { Markdown } from '@/components/Markdown'
import { cn } from '@/lib/utils'
import { PaneHeader } from './view.$id'

// Отдельный экран правки сегмента (Figma 44:775/44:1072, взамен правки на
// месте внутри выделенной области): только два блока — оригинал и перевод,
// без остального документа, без прокрутки и без ИИ-консультанта. Перевод
// редактируется сразу («свободное редактирование» — своей текстовой зоны
// «выделить, чтобы редактировать» не нужно). «Требует проверки» — ручной
// флаг (не только автоматический от числовой валидации).
export const Route = createFileRoute('/view/$id_/segment/$segId')({
  component: SegmentEditor,
})

function SegmentEditor() {
  const { id, segId } = Route.useParams()
  const docQ = useQuery({ queryKey: ['document', id], queryFn: () => api.getDocument(id) })
  const segsQ = useQuery({ queryKey: ['segments', id], queryFn: () => api.getSegments(id) })
  const segs = segsQ.data ?? []
  const target = segs.find((s) => s.id === segId)

  if (segsQ.isLoading || docQ.isLoading)
    return <p className="p-6 text-sm text-muted-foreground">Загрузка…</p>

  if (!target)
    return (
      <div className="p-6">
        <p className="text-sm text-muted-foreground">Сегмент не найден.</p>
        <Link to="/view/$id" params={{ id }} className="mt-2 inline-block text-sm text-primary hover:underline">
          ← Вернуться к документу
        </Link>
      </div>
    )

  return <LoadedSegmentEditor key={target.id} id={id} target={target} sourceLang={docQ.data?.source_lang} />
}

function LoadedSegmentEditor({
  id,
  target,
  sourceLang,
}: {
  id: string
  target: Segment
  sourceLang?: string | null
}) {
  const navigate = Route.useNavigate()
  const [text, setText] = useState(target.translated_text ?? '')
  const [needsReview, setNeedsReview] = useState(target.needs_review)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  function goBack() {
    navigate({ to: '/view/$id', params: { id } })
  }

  async function save() {
    setSaving(true)
    setError('')
    try {
      await api.patchSegment(target.id, text, needsReview)
      goBack()
    } catch {
      setError('Ошибка сохранения')
      setSaving(false)
    }
  }

  return (
    <div>
      <div className="flex flex-wrap items-start justify-center gap-6 px-6 py-10 md:px-[168px]">
        <div className="w-full max-w-[548px]">
          <PaneHeader label="Оригинал" lang={sourceLang} />
          <div className="mt-2 rounded-lg bg-[#222226]/[0.02] p-3">
            <Markdown content={target.source_text} className="text-[14.3px] leading-relaxed" />
          </div>
        </div>

        <div className="w-full max-w-[548px]">
          <PaneHeader label="Перевод" lang="ru" />
          <div className="mt-2 flex flex-col gap-4 rounded-lg border border-[#4b4ce6] bg-[#392dc1]/[0.06] p-3">
            {needsReview && (
              <span className="inline-flex w-fit items-center rounded-full bg-[#952d2d]/10 px-2 py-1 text-[11px] font-medium text-[#c43232]">
                Требует проверки
              </span>
            )}
            <textarea
              autoFocus
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={Math.max(3, Math.ceil(text.length / 60))}
              className="w-full resize-none whitespace-pre-wrap bg-transparent text-[14.3px] leading-relaxed outline-none"
            />
          </div>

          <div className="mt-4 flex flex-col gap-4">
            <button
              type="button"
              onClick={() => setNeedsReview((v) => !v)}
              className={cn(
                'flex w-fit items-center gap-2 rounded-lg px-4 py-3 text-[16px] font-medium transition',
                needsReview
                  ? 'bg-[#952d2d]/10 text-[#c43232]'
                  : 'bg-[#222226]/[0.02] text-[#424247] hover:bg-[#222226]/[0.05]',
              )}
            >
              <span
                className={cn(
                  'flex h-5 w-5 shrink-0 items-center justify-center rounded border',
                  needsReview ? 'border-[#c43232] bg-[#c43232]' : 'border-[#e5e5e5] bg-white',
                )}
              >
                {needsReview && <Check className="h-3 w-3 text-white" />}
              </span>
              Требует проверки
            </button>

            <div className="h-px bg-[#e5e5e5]" />

            {error && <p className="text-sm text-destructive">{error}</p>}

            <div className="flex gap-2">
              <button
                type="button"
                onClick={goBack}
                disabled={saving}
                className="flex-1 rounded-2xl bg-[#222226]/5 px-6 py-3 text-base font-semibold text-[#424247] transition hover:bg-[#222226]/10 disabled:opacity-50"
              >
                Отмена
              </button>
              <button
                type="button"
                onClick={() => void save()}
                disabled={saving}
                className="flex-1 rounded-2xl bg-[#4b4ce6] px-6 py-3 text-base font-semibold text-[#ebf1ff] transition hover:opacity-90 disabled:opacity-50"
              >
                {saving ? 'Сохраняю…' : 'Сохранить и назад'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
