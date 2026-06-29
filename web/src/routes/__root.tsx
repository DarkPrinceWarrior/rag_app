import { useEffect, useRef, useState, type FormEvent } from 'react'
import { createRootRoute, Link, Outlet, useRouterState } from '@tanstack/react-router'
import { Check, ChevronDown, Search, SlidersHorizontal, X } from 'lucide-react'
import { currentUser } from '@/lib/auth'
import { LibrarySearchContext, useLibrarySearch, type LibrarySearchContextValue } from '@/lib/librarySearch'
import type { DocFilters } from '@/lib/api'
import { cn } from '@/lib/utils'

function HeaderSearch() {
  const { query, filters, setQuery, setFilters, submitSearch, clearSearch } = useLibrarySearch()
  const [filtersOpen, setFiltersOpen] = useState(false)
  const filtersRef = useRef<HTMLDivElement>(null)
  const path = useRouterState({ select: (s) => s.location.pathname })
  const showFilters = path === '/'
  const activeFilterCount = Object.values(filters).filter(Boolean).length
  const setFilter = (key: keyof DocFilters, value: string) =>
    setFilters({ ...filters, [key]: value || undefined })

  useEffect(() => {
    if (!filtersOpen) return
    const onDown = (e: MouseEvent) => {
      if (!filtersRef.current?.contains(e.target as Node)) setFiltersOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setFiltersOpen(false)
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [filtersOpen])

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    submitSearch()
    setFiltersOpen(false)
  }

  return (
    <div ref={filtersRef} className="relative min-w-0 flex-1 md:max-w-[600px]">
      <form
        onSubmit={onSubmit}
        className="flex h-10 w-full items-center gap-2 rounded-[32px] bg-[#f3f3f3] py-1.5 pl-4 pr-1.5 text-[#222226]"
      >
        <Search className="h-5 w-5 shrink-0 text-[#222226]/35" />
        <input
          value={query}
          onChange={(e) => {
            const value = e.target.value
            setQuery(value)
            if (!value.trim()) clearSearch()
          }}
          placeholder="Найти среди документов"
          className="min-w-0 flex-1 bg-transparent text-[16px] font-medium leading-[1.5] text-[#222226] outline-none placeholder:text-[#222226]/25"
        />
        {query && (
          <button
            type="button"
            aria-label="Очистить поиск"
            onClick={clearSearch}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[#222226]/35 transition hover:bg-white hover:text-[#222226]/70"
          >
            <X className="h-4 w-4" />
          </button>
        )}
        {showFilters && (
          <button
            type="button"
            aria-label="Фильтры"
            aria-expanded={filtersOpen}
            onClick={() => setFiltersOpen((v) => !v)}
            className={cn(
              'relative flex h-8 min-w-8 shrink-0 items-center justify-center rounded-full px-2 text-[#222226]/45 transition',
              filtersOpen || activeFilterCount
                ? 'bg-white text-[#222226] shadow-[0_1px_4px_rgba(34,34,38,0.08)]'
                : 'hover:bg-white hover:text-[#222226]/70',
            )}
          >
            <SlidersHorizontal className="h-4 w-4" />
            {activeFilterCount > 0 && (
              <span className="ml-1 text-[11px] font-semibold leading-none">{activeFilterCount}</span>
            )}
          </button>
        )}
      </form>

      {showFilters && filtersOpen && (
        <div className="absolute right-0 top-12 z-40 w-[min(92vw,420px)] rounded-2xl border border-[#222226]/[0.08] bg-white p-3 shadow-[0_18px_42px_rgba(34,34,38,0.14)]">
          <div className="flex items-center justify-between gap-3">
            <div className="text-sm font-medium text-[#222226]">Фильтры поиска</div>
            {activeFilterCount > 0 && (
              <button
                type="button"
                onClick={() => setFilters({})}
                className="rounded-full px-2 py-1 text-xs font-medium text-muted-foreground transition hover:bg-[#222226]/5 hover:text-[#222226]"
              >
                Сбросить
              </button>
            )}
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            <FilterKindPicker value={filters.kind ?? ''} onChange={(value) => setFilter('kind', value)} />
            <label className="grid gap-1 text-[11px] font-medium text-muted-foreground">
              От
              <input
                type="date"
                className="h-9 min-w-0 rounded-xl border border-[#222226]/[0.08] bg-[#f7f7f7] px-3 text-sm font-medium text-[#222226] outline-none transition focus:border-[#6269f3]"
                value={filters.date_from ?? ''}
                onChange={(e) => setFilter('date_from', e.target.value)}
              />
            </label>
            <label className="grid gap-1 text-[11px] font-medium text-muted-foreground">
              До
              <input
                type="date"
                className="h-9 min-w-0 rounded-xl border border-[#222226]/[0.08] bg-[#f7f7f7] px-3 text-sm font-medium text-[#222226] outline-none transition focus:border-[#6269f3]"
                value={filters.date_to ?? ''}
                onChange={(e) => setFilter('date_to', e.target.value)}
              />
            </label>
          </div>
        </div>
      )}
    </div>
  )
}

const FILTER_KIND_OPTIONS = [
  { value: '', label: 'Любой' },
  { value: 'pdf_text', label: 'PDF текст' },
  { value: 'pdf_scan', label: 'PDF скан' },
  { value: 'docx', label: 'DOCX' },
  { value: 'xlsx', label: 'XLSX' },
  { value: 'pptx', label: 'PPTX' },
  { value: 'text', label: 'TXT' },
]

function FilterKindPicker({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLLabelElement>(null)
  const current = FILTER_KIND_OPTIONS.find((option) => option.value === value) ?? FILTER_KIND_OPTIONS[0]

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <label ref={ref} className="relative grid gap-1 text-[11px] font-medium text-muted-foreground">
      Тип
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex h-9 min-w-0 items-center gap-2 rounded-xl border border-[#222226]/[0.08] bg-[#f7f7f7] pl-3 pr-2 text-left text-sm font-medium text-[#222226] outline-none transition hover:bg-[#f1f1f1] focus:border-[#6269f3]"
      >
        <span className="min-w-0 flex-1 truncate">{current.label}</span>
        <ChevronDown className={cn('h-4 w-4 shrink-0 text-[#222226]/40 transition', open && 'rotate-180')} />
      </button>
      {open && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 overflow-hidden rounded-xl border border-[#222226]/[0.08] bg-white p-1 shadow-[0_12px_28px_rgba(34,34,38,0.14)]">
          {FILTER_KIND_OPTIONS.map((option) => (
            <button
              key={option.value || 'all'}
              type="button"
              onClick={() => {
                onChange(option.value)
                setOpen(false)
              }}
              className={cn(
                'flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-[#222226] transition hover:bg-[#222226]/[0.05]',
                value === option.value && 'bg-[#222226]/[0.05] font-semibold',
              )}
            >
              <span className="min-w-0 flex-1 truncate">{option.label}</span>
              {value === option.value && <Check className="h-4 w-4 shrink-0 text-[#6269f3]" />}
            </button>
          ))}
        </div>
      )}
    </label>
  )
}

function BarsIcon() {
  return (
    <span className="flex h-6 w-6 flex-col items-center justify-center gap-[3px]" aria-hidden="true">
      <span className="h-[1.7px] w-4 rounded-full bg-[#424247]" />
      <span className="h-[1.7px] w-4 rounded-full bg-[#424247]" />
      <span className="h-[1.7px] w-4 rounded-full bg-[#424247]" />
    </span>
  )
}

function ProfileButton() {
  const user = currentUser()
  const path = useRouterState({ select: (s) => s.location.pathname })
  const active = path.startsWith('/account')
  return (
    <Link
      to="/account"
      title={`Профиль: ${user.username}`}
      aria-label="Профиль"
      className={cn(
        'relative block h-10 w-10 shrink-0 overflow-hidden rounded-full transition',
        active && 'ring-2 ring-[#6269f3] ring-offset-2',
      )}
    >
      <span className="absolute inset-0 rounded-full bg-[radial-gradient(circle_at_55%_33%,#fff4a8_0_11%,#ffd537_12%_29%,#f6a61d_30%_48%,#b06a35_49%_66%,#e9e2d5_67%_100%)]" />
      <span className="absolute inset-[3px] rounded-full bg-[radial-gradient(circle_at_47%_34%,rgba(255,255,255,0.65),rgba(255,255,255,0)_32%),linear-gradient(145deg,rgba(255,224,69,0.95),rgba(203,112,25,0.95)_55%,rgba(91,70,63,0.9))]" />
      <span className="absolute inset-x-[8px] bottom-[5px] h-[10px] rounded-full bg-black/10 blur-[2px]" />
    </Link>
  )
}

function TabLink({ to, label }: { to: string; label: string }) {
  const path = useRouterState({ select: (s) => s.location.pathname })
  const active = to === '/' ? path === '/' : path.startsWith(to)
  return (
    <Link
      to={to}
      className={cn(
        'flex items-start px-4 py-2 text-[16px] font-medium leading-[1.5] text-[#222226] transition',
        active
          ? 'border-b-2 border-[#6269f3] bg-gradient-to-b from-white/0 to-[rgba(75,76,230,0.08)]'
          : 'opacity-50 hover:opacity-80',
      )}
    >
      {label}
    </Link>
  )
}

function RootLayout() {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [filters, setFilters] = useState<DocFilters>({})

  const searchContext: LibrarySearchContextValue = {
    query,
    submitted,
    filters,
    setQuery,
    setFilters,
    submitSearch: (value = query) => setSubmitted(value.trim()),
    clearSearch: () => {
      setQuery('')
      setSubmitted('')
    },
  }

  return (
    <LibrarySearchContext.Provider value={searchContext}>
      <div className="min-h-screen bg-white">
        <header className="sticky top-0 z-20 border-b border-[#222226]/[0.12] bg-white">
          <div className="flex min-h-[57px] items-center justify-between gap-2 px-3 py-3 md:gap-4 md:px-8">
            <button
              type="button"
              aria-label="Меню"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#222226]/5 text-[#424247] transition hover:bg-[#222226]/10"
            >
              <BarsIcon />
            </button>
            <HeaderSearch />
            <ProfileButton />
          </div>
          <nav className="flex h-[72px] items-end overflow-x-auto border-t border-[#222226]/[0.04] px-4 pt-8 md:px-[168px]">
            <div className="flex items-center gap-1.5">
              <TabLink to="/" label="Документы" />
              <TabLink to="/chat" label="Чат" />
            </div>
          </nav>
        </header>
        <main>
          <Outlet />
        </main>
      </div>
    </LibrarySearchContext.Provider>
  )
}

export const Route = createRootRoute({ component: RootLayout })
