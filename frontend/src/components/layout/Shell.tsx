import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AppHeader } from '@infra/ui'
import { StatusBar } from './StatusBar'
import { getVersion } from '../../api/meta'
import { useOwnerJobStream } from '../../hooks/useOwnerJobStream'
import { useT } from '../../i18n/LanguageContext'

/**
 * Nextext app shell: a single header row — the shared `@infra/ui` AppHeader
 * (back link, title, version, theme toggle) — above the page body. Nextext
 * has no server-exposed signed-in identity to show in AppHeader's `user`
 * slot (the backend's trusted-header principal is never echoed back to the
 * browser), so it's left undefined. The global job {@link StatusBar} (empty
 * when there are no jobs) renders as a slim, non-header strip directly below
 * AppHeader rather than inside it, so it never competes with the single
 * header row.
 *
 * Mounts {@link useOwnerJobStream} once here so the whole session shares a
 * single owner-multiplexed SSE connection feeding every job's live progress —
 * job cards read that shared store instead of each opening their own stream.
 */
export function Shell({ children }: { children: ReactNode }) {
  useOwnerJobStream()
  const t = useT()
  const { data } = useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity,
  })
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <AppHeader
        title="Nextext"
        version={data?.version ? `v${data.version}` : undefined}
        className="sticky top-0 z-20"
        homeLabel={t('header.home')}
        themeLabels={{
          system: t('header.theme_system'),
          light: t('header.theme_light'),
          dark: t('header.theme_dark'),
        }}
      />
      <div className="flex justify-end px-6 py-2">
        <StatusBar />
      </div>
      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">{children}</main>
    </div>
  )
}
