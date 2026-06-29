import { createContext, useContext } from 'react'
import type { DocFilters } from '@/lib/api'

export type LibrarySearchContextValue = {
  query: string
  submitted: string
  filters: DocFilters
  setQuery: (value: string) => void
  setFilters: (value: DocFilters) => void
  submitSearch: (value?: string) => void
  clearSearch: () => void
}

export const LibrarySearchContext = createContext<LibrarySearchContextValue | null>(null)

export function useLibrarySearch() {
  const ctx = useContext(LibrarySearchContext)
  if (!ctx) throw new Error('useLibrarySearch must be used inside RootLayout')
  return ctx
}
