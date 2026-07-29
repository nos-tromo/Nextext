import type { ReactNode } from 'react'
import { AppHeader, Shell as UIShell } from '@infra/ui'
import { StatusBar } from './StatusBar'
import { VersionBadge } from '../VersionBadge'
import { useOwnerJobStream } from '../../hooks/useOwnerJobStream'
import { useT } from '../../i18n/LanguageContext'

/**
 * Nextext app shell: the shared `@infra/ui` AppHeader (portal link, theme
 * toggle) sits above the shared, sticky `@infra/ui` Shell, which supplies the
 * app title and the global job StatusBar plus the version badge as its
 * right-aligned actions slot. Nextext has no server-exposed signed-in
 * identity to show in AppHeader's `user` slot (the backend's trusted-header
 * principal is never echoed back to the browser), so it's left undefined.
 *
 * Mounts {@link useOwnerJobStream} once here so the whole session shares a
 * single owner-multiplexed SSE connection feeding every job's live progress —
 * job cards read that shared store instead of each opening their own stream.
 */
export function Shell({ children }: { children: ReactNode }) {
  useOwnerJobStream()
  const t = useT()
  return (
    <>
      <AppHeader
        title="Nextext"
        homeLabel={t('header.home')}
        themeLabels={{
          system: t('header.theme_system'),
          light: t('header.theme_light'),
          dark: t('header.theme_dark'),
        }}
      />
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
    </>
  )
}
