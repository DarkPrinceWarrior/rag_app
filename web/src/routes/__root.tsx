import { useState, type FormEvent } from 'react'
import { createRootRoute, Link, Outlet, useRouterState } from '@tanstack/react-router'
import { Menu as MenuIcon, Search } from 'lucide-react'
import { currentUser } from '@/lib/auth'
import { LibrarySearchContext, useLibrarySearch, type LibrarySearchContextValue } from '@/lib/librarySearch'
import { cn } from '@/lib/utils'

function HeaderSearch() {
  const { query, setQuery, submitSearch } = useLibrarySearch()

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    submitSearch()
  }

  return (
    <form
      onSubmit={onSubmit}
      className="flex h-10 w-full max-w-[600px] items-center gap-2.5 rounded-[32px] bg-[#f3f3f3] px-4 text-[#222226]"
    >
      <Search className="h-5 w-5 shrink-0 text-[#222226]/35" />
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Найти среди документов"
        className="min-w-0 flex-1 bg-transparent text-[16px] font-medium leading-[1.5] text-[#222226] outline-none placeholder:text-[#222226]/25"
      />
    </form>
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

  const searchContext: LibrarySearchContextValue = {
    query,
    submitted,
    setQuery,
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
          <div className="flex min-h-[57px] items-center justify-between gap-4 px-4 py-3 md:px-8">
            <button
              type="button"
              aria-label="Меню"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#222226]/5 text-[#424247] transition hover:bg-[#222226]/10"
            >
              <MenuIcon className="h-5 w-5" />
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
