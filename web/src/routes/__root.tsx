import { createRootRoute, Link, Outlet, useRouterState } from '@tanstack/react-router'
import { logout } from '@/lib/auth'
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

function RootLayout() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 flex items-center gap-2 border-b bg-card/90 px-5 py-2.5 backdrop-blur">
        <span className="mr-3 font-semibold">rag_app</span>
        <nav className="flex items-center gap-1">
          <NavLink to="/" label="Библиотека" />
          <NavLink to="/chat" label="Чат" />
          <NavLink to="/memory" label="Память" />
          <NavLink to="/extract" label="Таблицы" />
          <NavLink to="/translate" label="Фрагмент" />
        </nav>
        <Button variant="ghost" size="sm" className="ml-auto" onClick={logout}>
          Выйти
        </Button>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  )
}

export const Route = createRootRoute({ component: RootLayout })
