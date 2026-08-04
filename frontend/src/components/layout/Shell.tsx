import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AppShell } from '@infra/ui'
import { getVersion } from '../../api/meta'
import { useOwnerJobStream } from '../../hooks/useOwnerJobStream'
import { useWhoami } from '../../hooks/useWhoami'
import { useT } from '../../i18n/LanguageContext'

/**
 * Nextext app shell: the shared `@infra/ui` AppShell chrome (back link,
 * title, version, identity menu, theme toggle) wrapping a scrolling canvas.
 * It carries the signed-in identity (`user`, from the authenticated
 * `GET /whoami`, preferring the gateway's decorative display name over the
 * raw username; undefined while loading/on error) and the release version.
 * The global job status bar now lives in the jobs column on Home rather
 * than in the shell chrome.
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
  const { data: whoami } = useWhoami()
  return (
    <AppShell
      title="Nextext"
      version={data?.version ? `v${data.version}` : undefined}
      user={whoami?.display_name ?? whoami?.username}
      homeLabel={t('header.home')}
      themeLabels={{
        system: t('header.theme_system'),
        light: t('header.theme_light'),
        dark: t('header.theme_dark'),
      }}
      signOutLabel={t('header.sign_out')}
    >
      <div className="flex h-full min-h-0 flex-col p-8">{children}</div>
    </AppShell>
  )
}
