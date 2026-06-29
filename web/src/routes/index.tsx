import { useCallback, useEffect, useRef, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Check,
  CloudUpload,
  Download,
  FolderInput,
  Languages,
  MoreVertical,
  PlusCircle,
  Trash2,
  X,
} from 'lucide-react'
import {
  api,
  EXPORT_LABELS,
  downloadUrl,
  translationDownloadUrl,
  type Document,
  type Folder,
  type DocFilters,
} from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { cn } from '@/lib/utils'
import { useLibrarySearch } from '@/lib/librarySearch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Menu, MenuItem, MenuLabel, MenuSeparator } from '@/components/ui/menu'
import { ConfirmDialog } from '@/components/ui/modal'
import { StatusBadge } from '@/components/StatusBadge'

export const Route = createFileRoute('/')({ component: Library })

const inProgress = (d: Document) => !['done', 'error'].includes(d.status)

// Бейдж направления перевода: источник определён автоматически, цель всегда RU.
// Русский документ не переводится; "auto" — язык ещё не определён (до перевода).
const DIRECTION: Record<string, { label: string; cls: string }> = {
  en: { label: 'EN → RU', cls: 'bg-blue-50 text-blue-700' },
  zh: { label: 'ZH → RU', cls: 'bg-rose-50 text-rose-700' },
  ru: { label: 'RU · без перевода', cls: 'bg-muted text-muted-foreground' },
}

const FORMAT_TONE: Record<string, { badge: string; surface: string }> = {
  DOCX: { badge: 'bg-blue-50 text-[#0a78ff]', surface: 'group-hover:bg-blue-50/60' },
  PDF: { badge: 'bg-red-50 text-[#ff160a]', surface: 'group-hover:bg-red-50/50' },
  PPTX: { badge: 'bg-amber-50 text-[#ff9d0a]', surface: 'group-hover:bg-amber-50/70' },
  XLSX: { badge: 'bg-emerald-50 text-[#008562]', surface: 'group-hover:bg-emerald-50/60' },
  TXT: { badge: 'bg-slate-100 text-slate-700', surface: 'group-hover:bg-slate-100' },
  IMAGE: { badge: 'bg-violet-50 text-violet-700', surface: 'group-hover:bg-violet-50/70' },
}

function documentFormat(d: Document) {
  const ext = /\.([a-z0-9]+)$/i.exec(d.filename)?.[1]?.toUpperCase()
  if (ext === 'JPG' || ext === 'JPEG' || ext === 'PNG') return 'IMAGE'
  if (ext) return ext
  if (d.kind.startsWith('pdf')) return 'PDF'
  return d.kind.toUpperCase()
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 Б'
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toLocaleString('ru-RU', {
    maximumFractionDigits: value >= 10 || unit === 0 ? 0 : 1,
  })} ${units[unit]}`
}

function formatDate(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'дата не указана'
  return new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' }).format(date)
}

function formatFileCount(count: number) {
  const mod10 = count % 10
  const mod100 = count % 100
  const word = mod10 === 1 && mod100 !== 11 ? 'файл' : mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14) ? 'файла' : 'файлов'
  return `${count} ${word}`
}

function Library() {
  const qc = useQueryClient()
  const { submitted, filters, clearSearch } = useLibrarySearch()
  const [folder, setFolder] = useState<string>('') // '' = все
  const [folderToDelete, setFolderToDelete] = useState<Folder | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const docsQ = useQuery({
    queryKey: ['documents', filters],
    queryFn: () => api.listDocuments(filters),
    refetchInterval: (q) => (q.state.data?.some(inProgress) ? 2500 : false),
  })
  const foldersQ = useQuery({ queryKey: ['folders'], queryFn: api.listFolders })

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  })
  const deleteFolder = useMutation({
    mutationFn: (target: Folder) => api.deleteFolder(target.id),
    onSuccess: (_, target) => {
      if (folder === target.id) setFolder('')
      setFolderToDelete(null)
      qc.invalidateQueries({ queryKey: ['folders'] })
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const hasFilters = Object.values(filters).some(Boolean)
  const statsDocsQ = useQuery({
    queryKey: ['documents', 'folder-stats'],
    queryFn: () => api.listDocuments({}),
    enabled: hasFilters,
    refetchInterval: (q) => (q.state.data?.some(inProgress) ? 2500 : false),
  })
  const allDocs = docsQ.data ?? []
  const docs = allDocs.filter((d) => !folder || d.folder_id === folder)
  const searchTerm = submitted.trim().toLocaleLowerCase('ru-RU')
  const searchActive = searchTerm.length >= 2
  const folders = foldersQ.data ?? []
  const visibleFolders = searchActive
    ? folders.filter((f) => f.name.toLocaleLowerCase('ru-RU').includes(searchTerm))
    : folders
  const visibleDocs = searchActive
    ? docs.filter((d) => d.filename.toLocaleLowerCase('ru-RU').includes(searchTerm))
    : docs
  const selectedFolder = foldersQ.data?.find((f) => f.id === folder)
  const statsDocs = hasFilters ? (statsDocsQ.data ?? allDocs) : allDocs
  const folderStats = new Map<string, { count: number; size: number }>()
  for (const d of statsDocs) {
    if (!d.folder_id) continue
    const current = folderStats.get(d.folder_id) ?? { count: 0, size: 0 }
    current.count += 1
    current.size += d.size_bytes
    folderStats.set(d.folder_id, current)
  }

  return (
    <div className="mx-auto max-w-[1136px] px-4 pb-14 pt-8">
      <input
        ref={fileInput}
        type="file"
        hidden
        accept=".pdf,.docx,.xlsx,.pptx,.jpg,.jpeg,.png,.txt"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) upload.mutate(file)
          e.currentTarget.value = ''
        }}
      />

      {upload.isError && <p className="mt-3 text-sm text-destructive">Ошибка загрузки: {String(upload.error)}</p>}

      {!searchActive && (
      <section className="mt-8">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <h2 className="text-[23px] font-semibold leading-[1.3] text-[#222226]">Папки</h2>
            {folder && (
              <button
                type="button"
                onClick={() => setFolder('')}
                className="rounded-full bg-[#222226]/5 px-3 py-1 text-xs font-medium text-muted-foreground transition hover:bg-[#222226]/10 hover:text-foreground"
              >
                Все документы
              </button>
            )}
          </div>
          <NewFolder onCreated={() => qc.invalidateQueries({ queryKey: ['folders'] })} />
        </div>
        <div className="mt-6 flex gap-2 overflow-x-auto pb-2">
          {foldersQ.isLoading && <p className="py-10 text-sm text-muted-foreground">Загрузка папок…</p>}
          {!foldersQ.isLoading && foldersQ.data?.length === 0 && (
            <div className="rounded-lg border border-dashed bg-card px-5 py-8 text-sm text-muted-foreground">
              Папок пока нет. Создайте первую папку для группировки документов.
            </div>
          )}
          {folders.map((f) => {
            const stats = folderStats.get(f.id)
            return (
              <FolderCard
                key={f.id}
                folder={f}
                active={folder === f.id}
                count={stats?.count ?? f.documents}
                size={stats?.size ?? 0}
                onClick={() => setFolder((current) => (current === f.id ? '' : f.id))}
                onDelete={() => setFolderToDelete(f)}
              />
            )
          })}
        </div>
      </section>
      )}

      {searchActive && visibleFolders.length > 0 && (
        <section className="mt-8">
          <div>
            <h2 className="text-[23px] font-semibold leading-[1.3] text-[#222226]">
              Папки: {submitted}
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">Карточки папок с совпадением в названии</p>
          </div>
          <div className="mt-6 flex gap-2 overflow-x-auto pb-2">
            {visibleFolders.map((f) => {
              const stats = folderStats.get(f.id)
              return (
                <FolderCard
                  key={f.id}
                  folder={f}
                  active={folder === f.id}
                  count={stats?.count ?? f.documents}
                  size={stats?.size ?? 0}
                  onClick={() => {
                    setFolder(f.id)
                    clearSearch()
                  }}
                  onDelete={() => setFolderToDelete(f)}
                />
              )
            })}
          </div>
        </section>
      )}

      <section className={searchActive ? 'mt-8' : 'mt-12'}>
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-[23px] font-semibold leading-[1.3] text-[#222226]">
              {searchActive ? `Документы: ${submitted}` : 'Документы'}
            </h2>
            {searchActive ? (
              <p className="mt-1 text-xs text-muted-foreground">
                Карточки документов с совпадением в названии
              </p>
            ) : selectedFolder ? (
              <p className="mt-1 text-xs text-muted-foreground">Папка: {selectedFolder.name}</p>
            ) : null}
          </div>
          {searchActive ? (
            <Button variant="ghost" className="h-10 rounded-2xl px-4" onClick={clearSearch}>
              Сбросить поиск
            </Button>
          ) : (
            <Button
              variant="ghost"
              className="h-10 rounded-2xl bg-[#222226]/5 px-4 text-[#424247] hover:bg-[#222226]/10"
              disabled={upload.isPending}
              onClick={() => fileInput.current?.click()}
            >
              <CloudUpload className="h-4 w-4" />
              {upload.isPending ? 'Загружаю…' : 'Загрузить ещё'}
            </Button>
          )}
        </div>

        {docsQ.isLoading ? (
          <p className="mt-6 text-sm text-muted-foreground">Загрузка…</p>
        ) : visibleDocs.length === 0 ? (
          <p className="mt-6 rounded-lg border border-dashed bg-card px-5 py-8 text-sm text-muted-foreground">
            {searchActive
              ? 'По названию документа ничего не найдено. Совпадения в содержимом показаны ниже.'
              : 'Пока нет документов. Загрузите PDF/DOCX/XLSX/PPTX/JPG/PNG/TXT.'}
          </p>
        ) : (
          <DocList docs={visibleDocs} folders={folders} />
        )}
      </section>

      <SearchResults folder={folder} filters={filters} />
      <ConfirmDialog
        open={!!folderToDelete}
        onClose={() => !deleteFolder.isPending && setFolderToDelete(null)}
        onConfirm={() => folderToDelete && deleteFolder.mutate(folderToDelete)}
        title={folderToDelete ? `Удалить папку «${folderToDelete.name}»?` : 'Удалить папку?'}
        description="Папка исчезнет из библиотеки, но документы из неё не удалятся."
        points={[
          'Документы останутся в общей библиотеке без папки.',
          'Переводы, превью, индекс поиска и чаты по документам сохранятся.',
        ]}
        warning="Саму папку восстановить нельзя."
        confirmLabel="Удалить папку"
        tone="danger"
        busy={deleteFolder.isPending}
      />
    </div>
  )
}

function FolderCard({
  folder,
  active,
  onClick,
  count,
  size,
  onDelete,
}: {
  folder: Folder
  active: boolean
  onClick: () => void
  count: number
  size: number
  onDelete?: () => void
}) {
  return (
    <article
      className={cn(
        'group relative flex w-[237px] shrink-0 flex-col gap-[11px] rounded-lg border bg-card p-1 pb-4 transition',
        active ? 'border-[#ef9a11]/60 shadow-[0_7px_14px_rgba(0,0,0,0.07)]' : 'border-[#e5e5e5] hover:shadow-[0_7px_14px_rgba(0,0,0,0.07)]',
      )}
    >
      {onDelete && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          title="Удалить папку"
          className="absolute right-2 top-2 z-[1] flex h-7 w-7 items-center justify-center rounded-full bg-white/[0.85] text-muted-foreground opacity-0 shadow-sm transition hover:text-destructive group-hover:opacity-100"
        >
          <X className="h-4 w-4" />
        </button>
      )}
      <button type="button" onClick={onClick} className="flex flex-col gap-[11px] text-left">
        <div
          className={cn(
            'flex h-[137px] w-full items-center justify-center rounded-md transition-colors',
            active ? 'bg-amber-50/80' : 'bg-[#222226]/[0.02] group-hover:bg-[#ef9a11]/10',
          )}
        >
          <FolderIllustration active={active} />
        </div>
        <div className="w-full px-4 text-center">
          <div className="truncate text-[14.3px] font-medium leading-[1.5] text-[#222226]">
            {folder.name}
          </div>
          <div className="mt-1 flex items-center justify-center gap-2 whitespace-nowrap text-[11.11px] font-medium leading-[1.5] text-[#c1c1c1]">
            <span>{formatFileCount(count)}</span>
            <span className="text-[#d9d9d9]">•</span>
            <span>{formatBytes(size)}</span>
          </div>
        </div>
      </button>
    </article>
  )
}

function FolderIllustration({ active }: { active: boolean }) {
  return (
    <div className="relative h-[86px] w-[96px]">
      <div
        className={cn(
          'absolute left-[8px] top-[9px] h-[18px] w-[38px] rounded-t-[10px]',
          active ? 'bg-[#f6fbff]' : 'bg-[#d8eafa]',
        )}
      />
      <div
        className={cn(
          'absolute inset-x-0 bottom-0 h-[68px] rounded-[10px] border shadow-[inset_0_1px_2px_rgba(255,255,255,0.85),0_5px_12px_rgba(74,103,139,0.2)]',
          active
            ? 'border-[#d4e3f6] bg-gradient-to-b from-[#f6fbff] to-[#c6d8f1]'
            : 'border-[#c5d7ee] bg-gradient-to-b from-[#eef7ff] via-[#d9e7f8] to-[#bccce3]',
        )}
      />
      <div className="absolute inset-x-[10px] top-[28px] h-px bg-white/70" />
    </div>
  )
}

function NewFolder({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const create = useMutation({
    mutationFn: () => api.createFolder(name.trim()),
    onSuccess: () => {
      setName('')
      setOpen(false)
      onCreated()
    },
  })
  if (!open)
    return (
      <Button
        variant="ghost"
        className="h-10 rounded-2xl bg-[#222226]/5 px-4 text-[#424247] hover:bg-[#222226]/10"
        onClick={() => setOpen(true)}
      >
        <PlusCircle className="h-4 w-4" />
        Создать папку
      </Button>
    )
  return (
    <span className="flex items-center gap-1 rounded-2xl bg-[#222226]/5 p-1">
      <Input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && name.trim()) create.mutate()
          if (e.key === 'Escape') setOpen(false)
        }}
        // увели курсор, ничего не введя — поле сворачивается (задержка, чтобы успел клик по «ок»)
        onBlur={() => setTimeout(() => !name.trim() && setOpen(false), 120)}
        placeholder="название"
        className="h-8 w-40 rounded-xl border-0 bg-white"
      />
      <Button size="sm" className="rounded-xl" disabled={!name.trim()} onClick={() => create.mutate()}>
        ок
      </Button>
    </span>
  )
}

function DocList({ docs, folders }: { docs: Document[]; folders: Folder[] }) {
  return (
    <div className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
      {docs.map((d) => (
        <DocCard key={d.id} d={d} folders={folders} />
      ))}
    </div>
  )
}

function DocCard({ d, folders }: { d: Document; folders: Folder[] }) {
  const qc = useQueryClient()
  const [deleteOpen, setDeleteOpen] = useState(false)
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['documents'] })
    qc.invalidateQueries({ queryKey: ['folders'] }) // счётчики папок
  }
  const del = useMutation({
    mutationFn: () => api.deleteDocument(d.id),
    onSuccess: () => {
      setDeleteOpen(false)
      refresh()
    },
  })
  const move = useMutation({
    mutationFn: (folderId: string | null) => api.moveDocument(d.id, folderId),
    onSuccess: refresh,
  })
  // доп. переводы документа (ТЗ §4.3): RU→EN/RU→ZH
  const translations = useQuery({
    queryKey: ['translations', d.id],
    queryFn: () => api.listTranslations(d.id),
    enabled: d.status === 'done',
    refetchInterval: (q) =>
      q.state.data?.some((t) => t.status === 'translating' || t.status === 'exporting') ? 2500 : false,
  })
  const translate = useMutation({
    mutationFn: (lang: string) => api.createTranslation(d.id, lang),
    onSuccess: () => translations.refetch(),
  })
  const TR_LANGS = [
    { code: 'en', label: 'English' },
    { code: 'zh', label: '中文 (упрощённый)' },
    { code: 'ru', label: 'Русский' },
  ]
  const srcLang = d.source_lang || 'ru'
  const trList = translations.data ?? []
  // не-ru документ уже переведён на ru основным потоком → ru повторно не предлагаем
  const offerLangs = TR_LANGS.filter((l) => l.code !== srcLang && !(srcLang !== 'ru' && l.code === 'ru'))
  const progress =
    d.status === 'translating' && d.segment_count
      ? `${Math.round((d.translated_count / d.segment_count) * 100)}%`
      : null
  const format = documentFormat(d)
  const tone = FORMAT_TONE[format] ?? FORMAT_TONE.TXT
  const canOpen = Boolean(d.status === 'done' || d.has_view || d.has_view_orig || d.has_view_ru)
  return (
    <>
      <article className="group flex min-h-[333px] w-full min-w-0 flex-col rounded-lg border border-[#e5e5e5] bg-card p-1 pb-4 shadow-sm transition hover:border-[#ef9a11]/60 hover:shadow-[0_7px_14px_rgba(0,0,0,0.07)]">
        <DocumentPreview d={d} tone={tone} canOpen={canOpen} />

        <div className="flex min-w-0 flex-1 flex-col px-3 pt-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <span className={cn('inline-flex rounded px-1.5 py-0.5 text-[11px] font-semibold leading-none', tone.badge)}>
              {format}
            </span>
            <div className="mt-2 line-clamp-2 min-h-[38px] text-[13px] font-medium leading-[1.45] text-[#222226]">
              {d.filename}
            </div>
          </div>
          <Menu
            trigger={<MoreVertical className="h-4 w-4" />}
            triggerClassName="mt-0 h-8 w-8 shrink-0 rounded-full text-muted-foreground hover:bg-[#222226]/5"
            title="Действия"
          >
            {(close) => (
              <>
                <MenuLabel>Скачать перевод</MenuLabel>
                {d.exports.map((k) => (
                  <MenuItem
                    key={k}
                    icon={<Download className="h-4 w-4" />}
                    onClick={() => {
                      void downloadFile(downloadUrl(d.id, k))
                      close()
                    }}
                  >
                    {EXPORT_LABELS[k] ?? k}
                  </MenuItem>
                ))}
                {d.exports.length === 0 && (
                  <div className="px-2 py-1 text-xs text-muted-foreground">перевод ещё не готов</div>
                )}
                <MenuSeparator />
                <MenuItem
                  icon={<Download className="h-4 w-4" />}
                  onClick={() => {
                    void downloadFile(downloadUrl(d.id, 'original'))
                    close()
                  }}
                >
                  Оригинал (как загружен)
                </MenuItem>

                {d.status === 'done' && offerLangs.length > 0 && (
                  <>
                    <MenuSeparator />
                    <MenuLabel>Перевести на язык</MenuLabel>
                    {offerLangs.map((l) => {
                      const t = trList.find((x) => x.target_lang === l.code)
                      const busy = t?.status === 'translating' || t?.status === 'exporting'
                      return (
                        <MenuItem
                          key={l.code}
                          icon={<Languages className="h-4 w-4" />}
                          disabled={busy || translate.isPending}
                          onClick={() => {
                            translate.mutate(l.code)
                            close()
                          }}
                        >
                          {l.label}
                          {t
                            ? t.status === 'done'
                              ? ' — готово ✓'
                              : t.status === 'error'
                                ? ' — ошибка'
                                : ' — перевод…'
                            : ''}
                        </MenuItem>
                      )
                    })}
                    {trList
                      .filter((t) => t.status === 'done' && t.has_export)
                      .map((t) => (
                        <MenuItem
                          key={`dl-${t.target_lang}`}
                          icon={<Download className="h-4 w-4" />}
                          onClick={() => {
                            void downloadFile(translationDownloadUrl(d.id, t.target_lang))
                            close()
                          }}
                        >
                          Скачать перевод — {t.target_lang.toUpperCase()}
                        </MenuItem>
                      ))}
                  </>
                )}

                <MenuSeparator />
                <MenuLabel>Переместить в папку</MenuLabel>
                <MenuItem
                  icon={d.folder_id == null ? <Check className="h-4 w-4" /> : <FolderInput className="h-4 w-4" />}
                  disabled={move.isPending}
                  onClick={() => {
                    if (d.folder_id != null) move.mutate(null)
                    close()
                  }}
                >
                  Без папки
                </MenuItem>
                {folders.map((f) => (
                  <MenuItem
                    key={f.id}
                    icon={d.folder_id === f.id ? <Check className="h-4 w-4" /> : <FolderInput className="h-4 w-4" />}
                    disabled={move.isPending}
                    onClick={() => {
                      if (d.folder_id !== f.id) move.mutate(f.id)
                      close()
                    }}
                  >
                    {f.name}
                  </MenuItem>
                ))}

                <MenuSeparator />
                <MenuItem
                  destructive
                  disabled={del.isPending}
                  icon={<Trash2 className="h-4 w-4" />}
                  onClick={() => {
                    setDeleteOpen(true)
                    close()
                  }}
                >
                  Удалить
                </MenuItem>
              </>
            )}
          </Menu>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
          <span>{formatBytes(d.size_bytes)}</span>
          <span className="text-[#d9d9d9]">•</span>
          <span>{formatDate(d.created_at)}</span>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[11px]">
          <StatusBadge status={d.status} />
          {d.source_lang && DIRECTION[d.source_lang] && (
            <span className={cn('rounded px-1.5 py-0.5 font-medium', DIRECTION[d.source_lang].cls)}>
              {DIRECTION[d.source_lang].label}
            </span>
          )}
          {trList.map((t) => (
            <span
              key={t.target_lang}
              className={cn(
                'rounded px-1.5 py-0.5 font-medium',
                t.status === 'done'
                  ? 'bg-emerald-50 text-emerald-700'
                  : t.status === 'error'
                    ? 'bg-destructive/10 text-destructive'
                    : 'bg-amber-50 text-amber-700',
              )}
            >
              → {t.target_lang.toUpperCase()}
              {t.status === 'done' ? ' ✓' : t.status === 'error' ? ' ✗' : '…'}
            </span>
          ))}
          {progress && <span className="rounded bg-amber-50 px-1.5 py-0.5 font-medium text-amber-700">{progress}</span>}
          {d.page_count != null && (
            <span className="rounded bg-[#222226]/5 px-1.5 py-0.5 font-medium text-muted-foreground">
              {d.page_count} стр.
            </span>
          )}
          {d.review_count > 0 && (
            <span className="rounded bg-amber-50 px-1.5 py-0.5 font-medium text-amber-700">
              проверить числа: {d.review_count}
            </span>
          )}
          {del.isError && <span className="text-destructive">Ошибка удаления</span>}
          {d.error && <span className="text-destructive">{d.error.slice(0, 80)}</span>}
        </div>

        <div className="mt-auto flex items-center justify-between gap-2 pt-4">
          {d.status === 'error' ? (
            <Button variant="ghost" size="sm" className="h-8 rounded-xl px-2.5" onClick={() => void api.retry(d.id)}>
              Повторить
            </Button>
          ) : d.status !== 'done' ? (
            <span className="text-xs text-muted-foreground">Обработка…</span>
          ) : (
            <span />
          )}
          {!canOpen && d.status === 'done' && (
            <span className="text-[11px] text-muted-foreground">превью готовится</span>
          )}
        </div>
        </div>
      </article>
      <ConfirmDialog
        open={deleteOpen}
        onClose={() => !del.isPending && setDeleteOpen(false)}
        onConfirm={() => del.mutate()}
        title={`Удалить «${d.filename}»?`}
        description="Документ будет удалён из библиотеки вместе со связанными данными."
        points={[
          'Исходный файл, перевод и экспортированные артефакты будут удалены.',
          'Поисковый индекс, сегменты и связанные чаты по документу будут очищены.',
        ]}
        warning="Действие необратимо."
        confirmLabel="Удалить документ"
        tone="danger"
        busy={del.isPending}
      />
    </>
  )
}

function DocumentPreview({
  d,
  tone,
  canOpen,
}: {
  d: Document
  tone: { badge: string; surface: string }
  canOpen: boolean
}) {
  const [previewUnavailable, setPreviewUnavailable] = useState(false)
  const previewUrl = d.preview_url && !previewUnavailable ? d.preview_url : null
  const handlePreviewUnavailable = useCallback(() => setPreviewUnavailable(true), [])
  const preview = (
    <div
      className={cn(
        'relative flex h-[234px] w-full items-center justify-center overflow-hidden rounded-md bg-[#222226]/[0.02] transition-colors',
        tone.surface,
      )}
    >
      <div className="relative h-[219px] w-[213px] rounded-[6px] border border-[#e3e5ea] bg-white shadow-[0_10px_24px_rgba(30,42,62,0.08)]">
        {previewUrl ? (
          <AuthenticatedPreviewImage
            src={previewUrl}
            alt=""
            onUnavailable={handlePreviewUnavailable}
          />
        ) : (
          <DocumentPreviewPlaceholder />
        )}
      </div>
      {!canOpen && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/45 text-xs font-medium text-muted-foreground">
          {d.status === 'error' ? 'ошибка обработки' : 'превью готовится'}
        </div>
      )}
    </div>
  )

  if (!canOpen) return preview

  return (
    <Link to="/view/$id" params={{ id: d.id }} aria-label={`Открыть ${d.filename}`} className="block">
      {preview}
    </Link>
  )
}

function AuthenticatedPreviewImage({
  src,
  alt,
  onUnavailable,
}: {
  src: string
  alt: string
  onUnavailable: () => void
}) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    let obj: string | null = null
    let cancelled = false
    setUrl(null)
    authFetch(src)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((blob) => {
        if (cancelled) return
        obj = URL.createObjectURL(blob)
        setUrl(obj)
      })
      .catch(() => {
        if (!cancelled) onUnavailable()
      })
    return () => {
      cancelled = true
      if (obj) URL.revokeObjectURL(obj)
    }
  }, [src, onUnavailable])

  if (!url) {
    return <DocumentPreviewPlaceholder pulse />
  }
  return <img src={url} alt={alt} className="absolute inset-0 h-full w-full object-contain" />
}

function DocumentPreviewPlaceholder({ pulse = false }: { pulse?: boolean }) {
  return (
    <>
      <div className={cn('absolute left-5 right-5 top-6 h-3 rounded bg-[#222226]/10', pulse && 'animate-pulse')} />
      <div className={cn('absolute left-5 right-8 top-12 h-2 rounded bg-[#222226]/[0.07]', pulse && 'animate-pulse')} />
      <div className={cn('absolute left-5 right-16 top-[72px] h-2 rounded bg-[#222226]/[0.07]', pulse && 'animate-pulse')} />
      <div className="absolute left-5 top-24 h-[54px] w-[74px] rounded border border-[#e3e5ea] bg-gradient-to-br from-[#f4f8ff] to-[#dfe8f5]" />
      <div className="absolute left-[110px] right-5 top-24 space-y-2">
        <div className={cn('h-2 rounded bg-[#222226]/[0.08]', pulse && 'animate-pulse')} />
        <div className={cn('h-2 rounded bg-[#222226]/[0.08]', pulse && 'animate-pulse')} />
        <div className={cn('h-2 w-2/3 rounded bg-[#222226]/[0.08]', pulse && 'animate-pulse')} />
      </div>
      <div className="absolute bottom-8 left-5 right-5 grid grid-cols-3 gap-2">
        <div className="h-12 rounded border border-[#e3e5ea] bg-[#222226]/[0.025]" />
        <div className="h-12 rounded border border-[#e3e5ea] bg-[#222226]/[0.025]" />
        <div className="h-12 rounded border border-[#e3e5ea] bg-[#222226]/[0.025]" />
      </div>
    </>
  )
}

// Скачивание через authFetch (download-роут за require_user) → blob → клик
async function downloadFile(url: string) {
  const r = await authFetch(url)
  if (!r.ok) return
  const blob = await r.blob()
  const cd = r.headers.get('Content-Disposition') || ''
  const name = /filename="?([^"]+)"?/.exec(cd)?.[1] || 'document'
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

function SearchResults({ folder, filters }: { folder: string; filters: DocFilters }) {
  const { submitted } = useLibrarySearch()
  const searchQ = useQuery({
    queryKey: ['search', submitted, folder, filters],
    queryFn: () => api.search(submitted, { ...(folder ? { folder_id: folder } : {}), ...filters }),
    enabled: submitted.length >= 2,
  })
  if (submitted.length < 2) return null
  const contentHits = (searchQ.data ?? []).filter((h) => h.match !== 'filename')

  return (
    <section className="mt-10">
      <div className="max-w-[832px]">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <h2 className="truncate text-[23px] font-semibold leading-[1.3] text-[#222226]">
              Совпадения в содержимом
            </h2>
            <div className="mt-0.5 text-xs text-muted-foreground">Фрагменты документов по запросу: {submitted}</div>
          </div>
        </div>
        {searchQ.isLoading ? (
          <p className="mt-6 text-sm text-muted-foreground">Ищу…</p>
        ) : (
          searchQ.data && (
            <div className="mt-6 grid gap-2">
              {contentHits.length === 0 && (
                <p className="rounded-lg border border-dashed bg-card px-5 py-8 text-sm text-muted-foreground">
                  Совпадений внутри документов не найдено.
                </p>
              )}
              {contentHits.map((h, i) => (
                <Link
                  key={`${h.document_id}-${h.chunk_id || `f${i}`}`}
                  to="/view/$id"
                  params={{ id: h.document_id }}
                  search={{ page: h.page_start != null ? h.page_start + 1 : undefined }}
                  className="group block rounded-lg border border-[#e5e5e5] bg-card px-4 py-3 text-sm shadow-sm transition hover:border-[#6269f3]/35 hover:bg-[#222226]/[0.02] hover:shadow-[0_7px_14px_rgba(0,0,0,0.05)]"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-medium leading-[1.45] text-[#222226]">{h.filename}</div>
                      {h.heading_path && (
                        <div className="mt-0.5 truncate text-xs text-muted-foreground">{h.heading_path}</div>
                      )}
                    </div>
                    <span className="shrink-0 rounded-full bg-[#222226]/5 px-2 py-1 text-[11px] font-medium leading-none text-muted-foreground">
                      {h.page_start != null ? `стр. ${h.page_start + 1}` : 'фрагмент'}
                    </span>
                  </div>
                  {h.snippet && (
                    <div className="mt-2 line-clamp-2 border-l-2 border-[#6269f3]/25 pl-3 leading-relaxed text-muted-foreground">
                      {h.snippet}
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )
        )}
      </div>
    </section>
  )
}
