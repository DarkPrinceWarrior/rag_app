import { createContext, useContext } from 'react'

export type LibrarySearchContextValue = {
  query: string
  submitted: string
  setQuery: (value: string) => void
  submitSearch: (value?: string) => void
  clearSearch: () => void
}

export const LibrarySearchContext = createContext<LibrarySearchContextValue | null>(null)

export function useLibrarySearch() {
  const ctx = useContext(LibrarySearchContext)
  if (!ctx) throw new Error('useLibrarySearch must be used inside RootLayout')
  return ctx
}
