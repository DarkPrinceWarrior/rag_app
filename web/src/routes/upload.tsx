import { useCallback, useRef, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, CloudUpload, FileText, Loader2, Package, Trash2, X } from 'lucide-react'
import { api, type Document } from '@/lib/api'
import { cn } from '@/lib/utils'
import { FORMAT_TONE, formatBytes, formatFileCount, formatFromFilename, inProgress } from '@/lib/format'
import { Select } from '@/components/ui/select'

export const Route = createFileRoute('/upload')({ component: UploadPage })

// ТЗ §4.2 (совпадает с ALLOWED_EXTENSIONS на бэкенде, documents.py)
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.jpg', '.jpeg', '.png', '.txt']
const ACCEPT = ALLOWED_EXTENSIONS.join(',')

// Парсер выбирается сразу на загрузке (Figma 41:854/41:1029) — на одном
// документе или на всех документах пачки сразу, вместо переразбора постфактум
// из вьювера (тот же набор бэкендов, см. view.$id.tsx PARSER_NAMES).
const PARSER_OPTIONS = [
  { value: '', label: 'Автоматически (по умолчанию)' },
  { value: 'mineru', label: 'MinerU + добор' },
  { value: 'dots_mocr', label: 'dots.mocr' },
  { value: 'paddle_vl', label: 'PaddleOCR-VL 1.6' },
]

const STAGE_TITLE: Record<string, string> = {
  uploaded: 'Готовлю документы',
  parsing: 'Разбираю документ',
  parsed: 'Разбираю документ',
  translating: 'Делаю перевод',
  translated: 'Собираю файл',
  exporting: 'Собираю файл',
}
const STAGE_ORDER = ['uploaded', 'parsing', 'parsed', 'translating', 'translated', 'exporting']

function fileExt(name: string) {
  return (/\.[a-z0-9]+$/i.exec(name)?.[0] ?? '').toLowerCase()
}

function UploadPage() {
  const qc = useQueryClient()
  const navigate = Route.useNavigate()
  const fileInput = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [parserBackend, setParserBackend] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const [rejected, setRejected] = useState<string[]>([])
  const [batchIds, setBatchIds] = useState<string[]>([])

  const addFiles = useCallback((list: FileList | File[]) => {
    const incoming = Array.from(list)
    const ok = incoming.filter((f) => ALLOWED_EXTENSIONS.includes(fileExt(f.name)))
    const bad = incoming.filter((f) => !ALLOWED_EXTENSIONS.includes(fileExt(f.name))).map((f) => f.name)
    if (ok.length) setFiles((cur) => [...cur, ...ok])
    setRejected(bad)
  }, [])

  const removeFile = (idx: number) => setFiles((cur) => cur.filter((_, i) => i !== idx))
  const replaceFile = (idx: number, file: File) => setFiles((cur) => cur.map((f, i) => (i === idx ? file : f)))

  const upload = useMutation({
    mutationFn: async () => {
      const created = await Promise.all(files.map((f) => api.uploadDocument(f, parserBackend || undefined)))
      return created.map((d) => d.id)
    },
    onSuccess: (ids) => {
      setBatchIds(ids)
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const batchQ = useQuery({
    queryKey: ['upload-batch', batchIds],
    queryFn: () => Promise.all(batchIds.map((id) => api.getDocument(id))),
    enabled: batchIds.length > 0,
    refetchInterval: (q) => (q.state.data?.some(inProgress) ? 1500 : false),
  })

  const cancel = useMutation({
    mutationFn: () => Promise.all(batchIds.map((id) => api.deleteDocument(id))),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] })
      navigate({ to: '/' })
    },
  })

  if (batchIds.length > 0) {
    return (
      <ProcessingView
        docs={batchQ.data ?? []}
        loading={batchQ.isLoading}
        cancelPending={cancel.isPending}
        onCancel={() => cancel.mutate()}
        onDone={() => navigate({ to: '/' })}
      />
    )
  }

  return (
    <div className="mx-auto max-w-[1136px] px-4 py-12">
      <input
        ref={fileInput}
        type="file"
        hidden
        multiple
        accept={ACCEPT}
        onChange={(e) => {
          if (e.target.files?.length) addFiles(e.target.files)
          e.currentTarget.value = ''
        }}
      />
      <div className="flex flex-wrap items-start gap-6">
        <div className="flex min-w-0 max-w-[695px] flex-1 flex-col gap-3">
          {files.length === 0 ? (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
              }}
              className={cn(
                'flex flex-col items-center gap-8 rounded-2xl border border-dashed px-6 py-10 text-center transition',
                dragOver ? 'border-[#4b4ce6] bg-[#4b4ce6]/5' : 'border-[#c1c1c1]',
              )}
            >
              <CloudUpload className="h-6 w-6 text-[#222226]/60" />
              <div className="flex flex-col items-center gap-2">
                <p className="text-base font-semibold text-[#222226]">Выберите или перетащите файлы</p>
                <p className="text-sm font-medium text-[#222226]/80">
                  PDF, DOCX, XLSX, PPTX, JPG, PNG, TXT — до 200 МБ
                </p>
              </div>
              <button
                type="button"
                onClick={() => fileInput.current?.click()}
                className="rounded-2xl bg-[#222226]/5 px-4 py-2 text-base font-semibold text-[#424247] transition hover:bg-[#222226]/10"
              >
                Выбрать файлы
              </button>
            </div>
          ) : (
            files.map((file, idx) => {
              const format = formatFromFilename(file.name) ?? 'FILE'
              const tone = FORMAT_TONE[format] ?? FORMAT_TONE.TXT
              return (
                <div
                  key={`${file.name}-${idx}`}
                  className="flex items-center justify-between gap-4 rounded-2xl bg-white px-6 py-4 ring-1 ring-[#e5e5e5]"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className={cn('flex h-[53px] w-[53px] shrink-0 items-center justify-center rounded-lg', tone.badge)}>
                      <FileText className="h-5 w-5" />
                    </span>
                    <div className="min-w-0">
                      <p className="text-[11.11px] font-medium text-[#222226]/80">{format}</p>
                      <p className="truncate text-base font-semibold text-[#222226]">{file.name}</p>
                      <p className="text-sm font-medium text-[#222226]/80">{formatBytes(file.size)}</p>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <label className="cursor-pointer rounded-2xl bg-[#222226]/5 px-4 py-2 text-base font-semibold text-[#424247] transition hover:bg-[#222226]/10">
                      Заменить
                      <input
                        type="file"
                        hidden
                        accept={ACCEPT}
                        onChange={(e) => {
                          const f = e.target.files?.[0]
                          if (f && ALLOWED_EXTENSIONS.includes(fileExt(f.name))) replaceFile(idx, f)
                          e.currentTarget.value = ''
                        }}
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => removeFile(idx)}
                      aria-label="Удалить файл"
                      className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#952d2d]/10 text-[#c43232] transition hover:bg-[#952d2d]/15"
                    >
                      <Trash2 className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              )
            })
          )}
          {files.length > 0 && (
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="self-start rounded-2xl bg-[#222226]/5 px-4 py-2 text-sm font-semibold text-[#424247] transition hover:bg-[#222226]/10"
            >
              + Добавить ещё файлы
            </button>
          )}
          {rejected.length > 0 && (
            <p className="text-xs text-destructive">Не поддерживается: {rejected.join(', ')}</p>
          )}
          {upload.isError && <p className="text-xs text-destructive">Ошибка загрузки: {String(upload.error)}</p>}
        </div>

        <div className="flex w-[385px] shrink-0 flex-col gap-6 pt-2">
          <div className="flex flex-col gap-4">
            <h2 className="text-[23px] font-semibold leading-[1.3] text-[#222226]">Настройки</h2>
            <div>
              <p className="mb-1 text-[11.11px] font-medium text-[#c1c1c1]">
                Парсер {files.length > 1 ? 'для всех документов' : 'документа'}
              </p>
              <Select
                value={parserBackend}
                onChange={setParserBackend}
                options={PARSER_OPTIONS}
                className="w-full justify-between rounded-lg border-0 bg-[#f3f3f3] px-4 py-3 text-base font-medium text-[#424247] hover:bg-[#eeeeee]"
              />
            </div>
          </div>
          <div className="flex flex-col items-center gap-2">
            <button
              type="button"
              disabled={files.length === 0 || upload.isPending}
              onClick={() => upload.mutate()}
              className={cn(
                'flex w-full items-center justify-center gap-2 rounded-2xl px-6 py-3 text-base font-semibold transition',
                files.length === 0
                  ? 'cursor-not-allowed bg-[#222226]/5 text-[#222226]/22'
                  : 'bg-[#222226] text-white hover:opacity-90',
              )}
            >
              {upload.isPending ? <Loader2 className="h-5 w-5 animate-spin" /> : <Package className="h-5 w-5" />}
              {upload.isPending ? 'Загружаю…' : 'Обработать'}
            </button>
            <p className="text-sm font-medium text-[#c1c1c1]">
              {files.length === 0
                ? 'Загрузите файл и выберите настройки'
                : `${formatFileCount(files.length)} готово к обработке`}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Обработка (Figma 41:1122) и завершение (41:1380) объединены: «к переводу»
// намеренно нет — при нескольких документах непонятно, какой один открывать.
// «Отмена» остаётся в обеих фазах: во время обработки прерывает (документы
// только что созданы — удаляем), после завершения — просто уводит в библиотеку.
function ProcessingView({
  docs,
  loading,
  cancelPending,
  onCancel,
  onDone,
}: {
  docs: Document[]
  loading: boolean
  cancelPending: boolean
  onCancel: () => void
  onDone: () => void
}) {
  const allSettled = !loading && docs.length > 0 && docs.every((d) => !inProgress(d))
  const single = docs.length === 1 ? docs[0] : null
  const doneCount = docs.filter((d) => d.status === 'done').length
  const errorCount = docs.filter((d) => d.status === 'error').length
  const failedOnly = allSettled && errorCount > 0 && doneCount === 0

  const totalSegments = docs.reduce((s, d) => s + (d.segment_count || 0), 0)
  const translatedSegments = docs.reduce((s, d) => s + (d.translated_count || 0), 0)
  const segmentPercent = totalSegments > 0 ? Math.round((translatedSegments / totalSegments) * 100) : 0
  const stage = STAGE_ORDER.find((s) => docs.some((d) => d.status === s))

  const title = allSettled
    ? failedOnly
      ? 'Ошибка обработки'
      : 'Успешно'
    : docs.length > 1
      ? 'Обрабатываю документы'
      : (stage && STAGE_TITLE[stage]) || 'Обрабатываю…'

  const subtitle = allSettled
    ? `${doneCount} из ${docs.length} готово${errorCount ? `, ошибок: ${errorCount}` : ''}`
    : single
      ? single.segment_count
        ? `${single.translated_count} / ${single.segment_count} сегментов`
        : ''
      : `${doneCount} из ${docs.length} готово`

  const progressPercent = allSettled ? 100 : docs.length > 1 ? Math.round((doneCount / docs.length) * 100) : segmentPercent

  return (
    <div className="mx-auto flex max-w-[1136px] flex-col items-center gap-11 px-4 py-24">
      <div className="flex flex-col items-center gap-6">
        <div className="relative flex h-[280px] w-[281px] items-center justify-center overflow-hidden rounded-lg">
          <img src="/upload-illustration.png" alt="" className="h-full w-full scale-[1.3] object-cover" />
          {allSettled && (
            <span
              className={cn(
                'absolute bottom-[29px] flex h-14 w-14 items-center justify-center rounded-full backdrop-blur-[21px]',
                failedOnly ? 'bg-[#952d2d]/10' : 'bg-[#392dc1]/[0.06]',
              )}
            >
              {failedOnly ? (
                <X className="h-6 w-6 text-[#c43232]" />
              ) : (
                <CheckCircle2 className="h-6 w-6 text-[#4b4ce6]" />
              )}
            </span>
          )}
        </div>
        <div className="flex flex-col items-center gap-2 text-center">
          <p className="text-sm font-medium text-[#c1c1c1]">
            {allSettled ? 'Обработка окончена' : 'Обрабатываю…'}
          </p>
          <div className="flex flex-col items-center gap-0.5">
            <p className="text-[23px] font-semibold leading-[1.3] text-[#222226]">{title}</p>
            {subtitle && <p className="text-sm font-medium text-[#666]">{subtitle}</p>}
          </div>
        </div>
      </div>

      <div className="flex w-full max-w-[549px] flex-col items-center gap-9">
        <div className="h-1 w-full overflow-hidden rounded-full bg-[#d9d9d9]/[0.23]">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#2f0fa7] to-[#4b4ce6] transition-all"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        <div className="flex w-full max-w-[384px] items-center gap-1 rounded-[20px] border border-black/[0.06] bg-white p-1">
          <button
            type="button"
            onClick={allSettled ? onDone : onCancel}
            disabled={!allSettled && cancelPending}
            className="flex-1 rounded-2xl bg-[#222226]/5 px-4 py-2 text-base font-semibold text-[#424247] transition hover:bg-[#222226]/10 disabled:opacity-50"
          >
            {allSettled ? 'На главную' : cancelPending ? 'Отменяю…' : 'Отмена'}
          </button>
        </div>
      </div>
    </div>
  )
}
