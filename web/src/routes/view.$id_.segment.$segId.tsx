import { useEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
import { api } from '@/lib/api'
import { DocAssistant } from '@/components/DocAssistant'
import { DocRead, PaneHeader, EditBar } from './view.$id'

// Отдельный экран правки сегмента (Figma 44:1037, взамен правки на месте
// внутри выделенной области): тот же двухколоночный «текст»-режим, но целевой
// сегмент всегда выделен/редактируется, остальные — контекст (размыт).
// Глоссарий и метка на скролле из макета не переносим — этих фич нет.
export const Route = createFileRoute('/view/$id_/segment/$segId')({
  component: SegmentEditor,
})

function SegmentEditor() {
  const { id, segId } = Route.useParams()
  const navigate = Route.useNavigate()
  const [pendingText, setPendingText] = useState('')
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const [assistantOpen, setAssistantOpen] = useState(false)
  const sourceColRef = useRef<HTMLDivElement>(null)
  const translatedColRef = useRef<HTMLDivElement>(null)

  const docQ = useQuery({ queryKey: ['document', id], queryFn: () => api.getDocument(id) })
  const segsQ = useQuery({ queryKey: ['segments', id], queryFn: () => api.getSegments(id) })
  const segs = segsQ.data ?? []
  const target = segs.find((s) => s.id === segId)

  useEffect(() => {
    for (const ref of [sourceColRef, translatedColRef]) {
      ref.current?.querySelector(`[data-seg="${segId}"]`)?.scrollIntoView({ block: 'center' })
    }
  }, [segId, segs.length])

  function startEdit() {
    if (!target) return
    setPendingText(target.translated_text ?? '')
    setEditing(true)
  }
  function cancelEdit() {
    setEditing(false)
    setPendingText('')
  }
  async function saveEdit() {
    if (!target) return
    if (pendingText === (target.translated_text ?? '')) {
      setEditing(false)
      return
    }
    setSaving(true)
    setMsg('Сохранение…')
    try {
      await api.patchSegment(segId, pendingText)
      setMsg('Сохранено')
      navigate({ to: '/view/$id', params: { id } })
      return
    } catch {
      setMsg('Ошибка сохранения')
    }
    setSaving(false)
    setTimeout(() => setMsg(''), 2000)
  }

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

  return (
    <div>
      <div className="sticky top-[49px] z-[5] flex items-center gap-3 border-b bg-card/90 px-5 py-2 backdrop-blur">
        <Link
          to="/view/$id"
          params={{ id }}
          className="flex shrink-0 items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          {docQ.data?.filename}
        </Link>
        <span className="ml-auto text-xs text-primary">{msg}</span>
      </div>
      <div className="flex h-[calc(100vh-137px)]">
        <div className="w-1/2 border-r">
          <div ref={sourceColRef} className="h-full overflow-y-auto">
            <PaneHeader label="Оригинал" lang={docQ.data?.source_lang} />
            <article className="mx-auto max-w-3xl px-6 py-4">
              <DocRead segs={segs} field="source" citedId={null} selectedId={segId} onSelectSeg={() => {}} />
            </article>
          </div>
        </div>
        <div className="w-1/2">
          <div ref={translatedColRef} className="h-full overflow-y-auto">
            <PaneHeader label="Перевод" lang="ru" />
            <article className="mx-auto max-w-3xl px-6 py-4">
              <DocRead
                segs={segs}
                field="translated"
                citedId={null}
                editable
                selectedId={segId}
                onSelectSeg={() => {}}
                editingId={editing ? segId : null}
                pendingText={pendingText}
                onPendingTextChange={setPendingText}
                onStartEdit={startEdit}
              />
            </article>
          </div>
        </div>
      </div>
      <EditBar
        editing={editing}
        saving={saving}
        onCancel={cancelEdit}
        onSave={() => void saveEdit()}
        assistantOpen={assistantOpen}
        onToggleAssistant={() => setAssistantOpen((o) => !o)}
      />
      <DocAssistant
        docId={id}
        filename={docQ.data?.filename}
        open={assistantOpen}
        onOpenChange={setAssistantOpen}
      />
    </div>
  )
}
