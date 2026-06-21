import { createRootRoute, Link, Outlet, useRouterState } from '@tanstack/react-router'
import { currentUser, logout } from '@/lib/auth'
import { Button } from '@/components/ui/button'

function NavLink({ to, label }: { to: string; label: string }) {
  const path = useRouterState({ select: (s) => s.location.pathname })
  const active = to === '/' ? path === '/' : path.startsWith(to)
  return (
    <Link
      to={to}
      className={
        'rounded-md px-3 py-1.5 text-sm transition-colors ' +
        (active ? 'bg-accent text-accent-foreground font-medium' : 'text-muted-foreground hover:text-foreground')
      }
    >
      {label}
    </Link>
  )
}

function AccountChip() {
  const user = currentUser()
  const path = useRouterState({ select: (s) => s.location.pathname })
  const active = path.startsWith('/account')
  return (
    <Link
      to="/account"
      title="Личный кабинет"
      className={
        'flex items-center gap-2 rounded-full py-1 pl-1 pr-3 text-sm transition-colors ' +
        (active ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground')
      }
    >
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
        {user.username.slice(0, 1).toUpperCase()}
      </span>
      <span className="max-w-[10rem] truncate">{user.username}</span>
    </Link>
  )
}

function RootLayout() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 flex items-center gap-2 border-b bg-card/90 px-5 py-2.5 backdrop-blur">
        <span className="mr-3 font-semibold">rag_app</span>
        <nav className="flex items-center gap-1">
          <NavLink to="/" label="Библиотека" />
          <NavLink to="/chat" label="Чат" />
          <NavLink to="/extract" label="Таблицы" />
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <AccountChip />
          <Button variant="ghost" size="sm" onClick={logout}>
            Выйти
          </Button>
        </div>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  )
}

export const Route = createRootRoute({ component: RootLayout })
