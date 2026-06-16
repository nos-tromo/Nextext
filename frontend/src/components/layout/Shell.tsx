import type { ReactNode } from 'react'

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-full">
      <header className="border-b border-border px-6 py-4">
        <h1 className="text-lg font-semibold">Nextext</h1>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
    </div>
  )
}
