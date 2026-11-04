import type { ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AppShell, PurposeBar, type PurposeBarLabels } from '@infra/ui'
import { getVersion } from '../../api/meta'
import { PURPOSE_SCOPE } from '../../api/client'
import { purposeRequired, useAppConfig } from '../../hooks/useAppConfig'
import { useOwnerJobStream } from '../../hooks/useOwnerJobStream'
import { useRetryOnPurpose } from '../../hooks/useRetryOnPurpose'
import { MediaPanel } from '../player/MediaPanel'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
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
 * The header's `PurposeBar` is where the user states the purpose every
 * request then carries (`X-Audit-Purpose`, sent by the API client); it warns
 * while none is stated when `GET /config` says the backend requires one, and
 * the client opens it when a request is refused for lacking one. A query
 * refused that way is refetched once a purpose is stated
 * ({@link useRetryOnPurpose}).
 *
 * Mounts {@link useOwnerJobStream} once here so the whole session shares a
 * single owner-multiplexed SSE connection feeding every job's live progress —
 * job cards read that shared store instead of each opening their own stream.
 *
 * Mounts {@link MediaPanel} the same way: one player for the app, so any
 * transcript row can start playback without owning player state. While it is
 * open the canvas reserves room for it on wide screens, so the page reflows
 * beside the panel rather than disappearing underneath it.
 */
export function Shell({ children }: { children: ReactNode }) {
  useOwnerJobStream()
  useRetryOnPurpose()
  const t = useT()
  const playerOpen = useMediaPlayerStore((s) => s.session !== null)
  const { data } = useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity,
  })
  const { data: whoami } = useWhoami()
  const { data: config } = useAppConfig()
  const purposeLabels: PurposeBarLabels = {
    purpose: t('purpose.purpose'),
    empty: t('purpose.empty'),
    required: t('purpose.required'),
    title: t('purpose.title'),
    description: t('purpose.description'),
    requested: t('purpose.requested'),
    caseRef: t('purpose.case_ref'),
    reason: t('purpose.reason'),
    save: t('purpose.save'),
    cancel: t('purpose.cancel'),
    clear: t('purpose.clear'),
    errorLength: t('purpose.error_length'),
    errorControl: t('purpose.error_control'),
    errorHeader: t('purpose.error_header'),
    errorEmpty: t('purpose.error_empty'),
  }
  return (
    <>
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
        headerActions={
          <PurposeBar scope={PURPOSE_SCOPE} required={purposeRequired(config)} labels={purposeLabels} />
        }
      >
        <div
          data-testid="shell-canvas"
          className={`flex min-h-full flex-col p-8 ${playerOpen ? 'md:pr-[29rem]' : ''}`}
        >
          {children}
        </div>
      </AppShell>
      {/* One player for the whole app: any result tab can start playback
          through the media store without owning panel state. */}
      <MediaPanel />
    </>
  )
}
