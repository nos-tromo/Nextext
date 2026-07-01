import type { ReactNode } from 'react'
import { Shell as UIShell } from '@infra/ui'
import { StatusBar } from './StatusBar'

/**
 * Nextext app shell: adapts the shared, sticky `@infra/ui` Shell by supplying
 * the app title and the global job StatusBar as its right-aligned actions slot.
 * The shared shell keeps the header pinned to the top while the page scrolls.
 */
export function Shell({ children }: { children: ReactNode }) {
  return (
    <UIShell title="Nextext" actions={<StatusBar />}>
      {children}
    </UIShell>
  )
}
