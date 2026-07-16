import { useCallback, useEffect, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Check,
  CloudUpload,
  Download,
  FileText,
  FolderInput,
  Languages,
  MoreVertical,
  Pencil,
  PlusCircle,
  Search,
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
} from '@/lib/api'
import { authFetch } from '@/lib/auth'
import { cn } from '@/lib/utils'
import { useLibrarySearch } from '@/lib/librarySearch'
import {
  DIRECTION,
  FORMAT_TONE,
  documentFormat,
  formatBytes,
  formatDate,
  formatDocCount,
  formatFileCount,
  inProgress,
} from '@/lib/format'
import { Button } from '@/components/ui/button'
import { Menu, MenuItem, MenuLabel, MenuSeparator } from '@/components/ui/menu'
import { ConfirmDialog, Modal } from '@/components/ui/modal'
import { StatusBadge } from '@/components/StatusBadge'

export const Route = createFileRoute('/')({ component: Library })

function Library() {
  const qc = useQueryClient()
  const navigate = Route.useNavigate()
  const { submitted, filters, clearSearch } = useLibrarySearch()
  const [folder, setFolder] = useState<string>('') // '' = все
  const [folderToDelete, setFolderToDelete] = useState<Folder | null>(null)
  const [editFolderOpen, setEditFolderOpen] = useState(false)

  const docsQ = useQuery({
    queryKey: ['documents', filters],
    queryFn: () => api.listDocuments(filters),
    refetchInterval: (q) => (q.state.data?.some(inProgress) ? 2500 : false),
  })
  const foldersQ = useQuery({ queryKey: ['folders'], queryFn: api.listFolders })

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
    <div className="px-4 pb-14 pt-8 md:px-[168px]">
      {!searchActive && !folder && (
      <section className="mt-8">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-[23px] font-semibold leading-[1.3] text-[#222226]">Папки</h2>
          <NewFolder
            docs={allDocs}
            onCreated={() => {
              qc.invalidateQueries({ queryKey: ['folders'] })
              qc.invalidateQueries({ queryKey: ['documents'] })
            }}
          />
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
              />
            )
          })}
        </div>
      </section>
      )}

      {/* Внутри папки (Figma 54:1618): хлебная крошка вместо строки папок */}
      {!searchActive && folder && selectedFolder && (
        <section className="mt-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2 whitespace-nowrap text-[23px] font-semibold leading-[1.3]">
              <button
                type="button"
                onClick={() => setFolder('')}
                className="text-[#222226]/[0.22] transition hover:text-[#222226]/50"
              >
                Папки
              </button>
              <span className="text-[#222226]/[0.22]">/</span>
              <span className="text-[#222226]">{selectedFolder.name}</span>
            </div>
            <Button
              variant="ghost"
              className="h-10 rounded-2xl bg-[#222226]/5 px-4 text-[#424247] hover:bg-[#222226]/10"
              onClick={() => setEditFolderOpen(true)}
            >
              <Pencil className="h-4 w-4" />
              Редактировать папку
            </Button>
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
                />
              )
            })}
          </div>
        </section>
      )}

      <section className={searchActive || folder ? 'mt-8' : 'mt-12'}>
        {(searchActive || !folder) && (
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-[23px] font-semibold leading-[1.3] text-[#222226]">
                {searchActive ? `Документы: ${submitted}` : 'Документы'}
              </h2>
              {searchActive && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Карточки документов с совпадением в названии
                </p>
              )}
            </div>
            {searchActive ? (
              <Button variant="ghost" className="h-10 rounded-2xl px-4" onClick={clearSearch}>
                Сбросить поиск
              </Button>
            ) : (
              <Button
                variant="ghost"
                className="h-10 rounded-2xl bg-[#222226]/5 px-4 text-[#424247] hover:bg-[#222226]/10"
                onClick={() => navigate({ to: '/upload' })}
              >
                <CloudUpload className="h-4 w-4" />
                Загрузить ещё
              </Button>
            )}
          </div>
        )}

        {docsQ.isLoading ? (
          <p className="mt-6 text-sm text-muted-foreground">Загрузка…</p>
        ) : visibleDocs.length === 0 ? (
          <p className="mt-6 rounded-lg border border-dashed bg-card px-5 py-8 text-sm text-muted-foreground">
            {searchActive
              ? 'По названию документа ничего не найдено.'
              : 'Пока нет документов. Загрузите PDF/DOCX/XLSX/PPTX/JPG/PNG/TXT.'}
          </p>
        ) : (
          <DocList docs={visibleDocs} folders={folders} />
        )}
      </section>

      <FolderModal
        key={editFolderOpen ? (selectedFolder?.id ?? 'create') : 'closed'}
        open={editFolderOpen}
        onClose={() => setEditFolderOpen(false)}
        docs={allDocs}
        folder={selectedFolder ?? null}
        onSaved={() => {
          qc.invalidateQueries({ queryKey: ['folders'] })
          qc.invalidateQueries({ queryKey: ['documents'] })
        }}
        onRequestDelete={(f) => {
          setEditFolderOpen(false)
          setFolderToDelete(f)
        }}
      />
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
        'group relative flex w-[237px] shrink-0 flex-col gap-[11px] rounded-lg border border-[#e5e5e5] bg-card p-1 pb-4 transition hover:shadow-[0_7px_14px_rgba(0,0,0,0.07)]',
        active && 'shadow-[0_7px_14px_rgba(0,0,0,0.07)]',
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
            active ? 'bg-[#392dc1]/[0.06]' : 'bg-[#222226]/[0.02] group-hover:bg-[#392dc1]/[0.06]',
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

function NewFolder({ docs, onCreated }: { docs: Document[]; onCreated: () => void }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <Button
        variant="ghost"
        className="h-10 rounded-2xl bg-[#222226]/5 px-4 text-[#424247] hover:bg-[#222226]/10"
        onClick={() => setOpen(true)}
      >
        <PlusCircle className="h-4 w-4" />
        Создать папку
      </Button>
      <FolderModal
        key={open ? 'create' : 'closed'}
        open={open}
        onClose={() => setOpen(false)}
        docs={docs}
        folder={null}
        onSaved={onCreated}
      />
    </>
  )
}

// Модалка создания/редактирования папки (Figma 54:3374 create / 54:3750 edit):
// имя + выбор документов. Бэкенд не принимает документы при создании папки —
// на клиенте создаём/переименовываем папку, затем перемещаем документы
// (moveDocument) по разнице с исходным набором.
function FolderModal({
  open,
  onClose,
  docs,
  folder,
  onSaved,
  onRequestDelete,
}: {
  open: boolean
  onClose: () => void
  docs: Document[]
  folder: Folder | null
  onSaved: () => void
  onRequestDelete?: (folder: Folder) => void
}) {
  const isEdit = folder != null
  const initialDocumentIds = () =>
    folder ? new Set(docs.filter((document) => document.folder_id === folder.id).map((document) => document.id)) : new Set<string>()
  const [name, setName] = useState(folder?.name ?? '')
  const [docSearch, setDocSearch] = useState('')
  const [selected, setSelected] = useState<Set<string>>(initialDocumentIds)
  const [initialSelected] = useState<Set<string>>(initialDocumentIds)

  const save = useMutation({
    mutationFn: async () => {
      if (folder) {
        if (name.trim() !== folder.name) await api.renameFolder(folder.id, name.trim())
        const added = [...selected].filter((id) => !initialSelected.has(id))
        const removed = [...initialSelected].filter((id) => !selected.has(id))
        await Promise.all([
          ...added.map((id) => api.moveDocument(id, folder.id)),
          ...removed.map((id) => api.moveDocument(id, null)),
        ])
        return
      }
      const created = await api.createFolder(name.trim())
      await Promise.all([...selected].map((id) => api.moveDocument(id, created.id)))
    },
    onSuccess: () => {
      onSaved()
      onClose()
    },
  })

  const handleClose = () => {
    if (save.isPending) return
    onClose()
  }

  const toggle = (id: string) =>
    setSelected((cur) => {
      const next = new Set(cur)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const term = docSearch.trim().toLocaleLowerCase('ru-RU')
  const visibleDocs = term ? docs.filter((d) => d.filename.toLocaleLowerCase('ru-RU').includes(term)) : docs
  const addedCount = [...selected].filter((id) => !initialSelected.has(id)).length
  const removedCount = [...initialSelected].filter((id) => !selected.has(id)).length

  return (
    <Modal
      open={open}
      onClose={handleClose}
      labelledBy="folder-modal-title"
      className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-[24px] border-[#e5e5e5] p-0"
    >
      <div className="flex shrink-0 items-center justify-between border-b border-[#e5e5e5] px-8 py-6">
        <h2 id="folder-modal-title" className="text-base font-semibold leading-[1.5] text-[#222226]">
          {isEdit ? 'Редактирование папки' : 'Создать папку'}
        </h2>
        <div className="flex shrink-0 items-center gap-4">
          {isEdit && folder && (
            <button
              type="button"
              onClick={() => {
                onRequestDelete?.(folder)
                handleClose()
              }}
              className="flex items-center gap-2 rounded-2xl bg-[#952d2d]/10 px-4 py-2 text-base font-semibold text-[#c43232] transition hover:bg-[#952d2d]/15"
            >
              <Trash2 className="h-5 w-5" />
              Удалить папку
            </button>
          )}
          <button
            type="button"
            onClick={handleClose}
            aria-label="Закрыть"
            className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#222226]/5 text-[#424247] transition hover:bg-[#222226]/10"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-8">
        <div>
          <label className="text-[11.11px] font-medium leading-[1.5] tracking-[-0.0889px] text-[#c1c1c1]">
            Название папки
          </label>
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && name.trim() && !save.isPending) save.mutate()
            }}
            placeholder="Название папки..."
            className="mt-1 w-full border-0 border-b border-[#e5e5e5] pb-4 text-[28px] font-medium leading-[1.3] tracking-[-0.4645px] text-[#222226] outline-none placeholder:text-[#222226]/22"
          />
        </div>

        <div className="flex min-h-0 flex-1 flex-col gap-3">
          <h3 className="text-base font-semibold leading-[1.5] text-[#222226]">
            Добавить документы в папку
          </h3>
          <div className="flex shrink-0 items-center gap-2.5 rounded-[32px] bg-[#f3f3f3] px-4 py-3">
            <Search className="h-5 w-5 shrink-0 text-[#222226]/40" />
            <input
              value={docSearch}
              onChange={(e) => setDocSearch(e.target.value)}
              placeholder="Поиск документов"
              className="w-full border-0 bg-transparent text-sm font-medium text-[#222226] outline-none placeholder:text-[#222226]/22"
            />
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-[#e5e5e5]">
            {visibleDocs.length === 0 && (
              <div className="p-4 text-center text-sm text-muted-foreground">Документы не найдены</div>
            )}
            {visibleDocs.map((d) => {
              const checked = selected.has(d.id)
              return (
                <label
                  key={d.id}
                  className={cn(
                    'flex cursor-pointer items-center gap-3 border-b border-[#e5e5e5] p-4 last:border-b-0',
                    checked ? 'bg-[#392dc1]/[0.06]' : 'bg-white hover:bg-[#222226]/[0.02]',
                  )}
                >
                  <span
                    className={cn(
                      'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg',
                      checked ? 'bg-[#4b4ce6]' : 'bg-[#f3f3f3]',
                    )}
                  >
                    <FileText className={cn('h-4 w-4', checked ? 'text-white' : 'text-[#c1c1c1]')} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[14.3px] font-medium leading-[1.5] text-[#222226]">
                      {d.filename}
                    </span>
                    <span className="block text-[11.11px] font-medium leading-[1.5] text-[#c1c1c1]">
                      {formatDate(d.created_at)}
                    </span>
                  </span>
                  <input type="checkbox" checked={checked} onChange={() => toggle(d.id)} className="sr-only" />
                  <span
                    className={cn(
                      'flex h-5 w-5 shrink-0 items-center justify-center rounded border',
                      checked ? 'border-[#4b4ce6] bg-[#4b4ce6]' : 'border-[#e5e5e5] bg-white',
                    )}
                  >
                    {checked && <Check className="h-3 w-3 text-white" />}
                  </span>
                </label>
              )
            })}
          </div>
        </div>

        {save.isError && (
          <p className="text-xs text-destructive">Ошибка сохранения папки: {String(save.error)}</p>
        )}
      </div>

      <div className="flex shrink-0 items-center justify-between gap-4 border-t border-[#e5e5e5] px-8 py-6">
        {isEdit ? (
          <div className="text-[14.3px] font-medium leading-[1.5] text-[#222226]">
            <p>
              <span className="text-[#222226]/50">Добавлено: </span>
              {formatDocCount(addedCount)}
            </p>
            <p>
              <span className="text-[#222226]/50">Удалено: </span>
              {formatDocCount(removedCount)}
            </p>
          </div>
        ) : (
          <p className="text-[14.3px] font-medium leading-[1.5] text-[#222226]">
            {selected.size > 0 ? `Выбрано: ${formatDocCount(selected.size)}` : ''}
          </p>
        )}
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={handleClose}
            disabled={save.isPending}
            className="rounded-2xl bg-[#222226]/5 px-6 py-3 text-base font-semibold text-[#424247] transition hover:bg-[#222226]/10 disabled:opacity-50"
          >
            Отмена
          </button>
          <button
            type="button"
            onClick={() => save.mutate()}
            disabled={!name.trim() || save.isPending}
            className="rounded-2xl bg-[#4b4ce6] px-6 py-3 text-base font-semibold text-[#ebf1ff] transition hover:opacity-90 disabled:opacity-50"
          >
            {save.isPending ? 'Сохраняю…' : isEdit ? 'Сохранить' : 'Создать папку'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

function DocList({ docs, folders }: { docs: Document[]; folders: Folder[] }) {
  return (
    <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-[repeat(auto-fill,minmax(237px,280px))] sm:justify-center">
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
  const retry = useMutation({
    mutationFn: () => api.retry(d.id),
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
      <article data-testid="document-card" className="group flex min-h-[333px] w-full min-w-0 flex-col rounded-lg border border-[#e5e5e5] bg-card p-1 pb-4 shadow-sm transition hover:border-[#ef9a11]/60 hover:shadow-[0_7px_14px_rgba(0,0,0,0.07)]">
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
          {retry.isError && <span className="text-destructive">Не удалось повторить обработку</span>}
          {d.error && <span className="text-destructive">{d.error.slice(0, 80)}</span>}
        </div>

        <div className="mt-auto flex items-center justify-between gap-2 pt-4">
          {d.status === 'error' ? (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 rounded-xl px-2.5"
              disabled={retry.isPending}
              onClick={() => retry.mutate()}
            >
              {retry.isPending ? 'Повторяю…' : 'Повторить'}
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
        'relative flex h-[218px] w-full items-center justify-center overflow-hidden rounded-md bg-[#222226]/[0.02] transition-colors',
        tone.surface,
      )}
    >
      <div className="relative h-[202px] w-[calc(100%-20px)]">
        {previewUrl ? (
          <AuthenticatedPreviewImage
            src={previewUrl}
            alt=""
            onUnavailable={handlePreviewUnavailable}
          />
        ) : (
          <PreviewPaper />
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
  const [loaded, setLoaded] = useState<{ src: string; url: string } | null>(null)
  const url = loaded?.src === src ? loaded.url : null
  useEffect(() => {
    let obj: string | null = null
    let cancelled = false
    authFetch(src)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((blob) => {
        if (cancelled) return
        obj = URL.createObjectURL(blob)
        setLoaded({ src, url: obj })
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
    return <PreviewPaper pulse />
  }
  return (
    <img
      src={url}
      alt={alt}
      className="absolute inset-0 h-full w-full rounded-[5px] object-contain [filter:drop-shadow(0_8px_14px_rgba(30,42,62,0.12))]"
    />
  )
}

function PreviewPaper({ pulse = false }: { pulse?: boolean }) {
  return (
    <div className="absolute left-1/2 top-1/2 h-[202px] w-[148px] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-[5px] border border-[#e3e5ea]/80 bg-white/80 shadow-[0_8px_16px_rgba(30,42,62,0.08)]">
      <DocumentPreviewPlaceholder pulse={pulse} />
    </div>
  )
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
