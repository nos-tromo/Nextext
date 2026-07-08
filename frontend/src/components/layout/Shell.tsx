import type { ReactNode } from 'react'
import { Shell as UIShell } from '@infra/ui'
import { StatusBar } from './StatusBar'
import { VersionBadge } from '../VersionBadge'
import { useOwnerJobStream } from '../../hooks/useOwnerJobStream'

/**
 * Nextext app shell: adapts the shared, sticky `@infra/ui` Shell by supplying
 * the app title and the global job StatusBar plus the version badge as its
 * right-aligned actions slot. The shared shell keeps the header pinned to the
 * top while the page scrolls.
 *
 * Mounts {@link useOwnerJobStream} once here so the whole session shares a
 * single owner-multiplexed SSE connection feeding every job's live progress —
 * job cards read that shared store instead of each opening their own stream.
 */
export function Shell({ children }: { children: ReactNode }) {
  useOwnerJobStream()
  return (
    <UIShell
      title="Nextext"
      actions={
        <div className="flex items-center gap-2">
          <StatusBar />
          <VersionBadge />
        </div>
      }
    >
      {children}
    </UIShell>
  )
}
