import { Banner } from '@infra/ui'
import { DownloadButtons } from './DownloadButtons'
import { TimeButton } from './TimeCell'
import { useT } from '../../i18n/LanguageContext'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { formatStamp } from '../../lib/formatStamp'
import { cn } from '../../lib/cn'
import type { FaceEntry, FacesResult } from '../../api/types'

interface FacesTabProps {
  jobId: string
  faces: FacesResult
  stem: string
  fileName: string
  mediaUrl: string | null
  /** Whether this deployment groups faces per person (`NEXTEXT_FACES=group`). */
  groupCapable: boolean
}

/**
 * Whether the playhead of this job's recording is inside one of an entry's appearances.
 *
 * @param entry - The face entry.
 * @param seconds - The playhead position.
 * @returns `true` while the entry is on screen.
 */
function onScreen(entry: FaceEntry, seconds: number): boolean {
  return entry.appearances.some((a) => a.start_seconds <= seconds && seconds < a.end_seconds)
}

/**
 * Shows who is on screen when: one card per person (or, ungrouped, per face
 * track) with its thumbnail, how often and when it was seen, its appearances
 * as seek controls, and its best image to download.
 *
 * The thumbnails arrive as `data:` URIs inside the result, so looking at them
 * is the one snapshot query rather than a download per face. The tab-level
 * `faces.zip` / `faces.csv` exist only when there are entries — the backend
 * answers 404 for an empty or unavailable result — so they are offered only
 * then.
 *
 * @param jobId - The job identifier, forwarded to the downloads and the player.
 * @param faces - The job's face result.
 * @param stem - Upload filename without extension; prefixes download names.
 * @param fileName - Original upload name, shown as the player's title.
 * @param mediaUrl - Capability URL for playback; `null` disables seeking.
 * @param groupCapable - Whether "not grouped" is worth saying.
 */
export function FacesTab({ jobId, faces, stem, fileName, mediaUrl, groupCapable }: FacesTabProps) {
  const t = useT()
  const playingHere = useMediaPlayerStore((s) => s.session?.jobId === jobId)
  const currentTime = useMediaPlayerStore((s) => s.currentTime)
  const unavailable = faces.outcome === 'unavailable'
  const hasEntries = faces.entries.length > 0

  return (
    <div className="space-y-4">
      {faces.outcome === 'partial' && <Banner variant="info">{t('results.faces_partial')}</Banner>}
      {unavailable && <Banner variant="info">{t('results.faces_unavailable')}</Banner>}
      {faces.truncated && <Banner variant="info">{t('results.faces_truncated')}</Banner>}
      {groupCapable && !faces.grouped && hasEntries && (
        <Banner variant="info">{t('results.faces_not_grouped')}</Banner>
      )}

      {!hasEntries && !unavailable && <p className="text-sm text-muted-foreground">{t('results.faces_none')}</p>}

      {hasEntries && (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {faces.entries.map((entry) => {
            const label = t(entry.assigned ? 'results.face_person' : 'results.face_single', { n: entry.number })
            const current = playingHere && onScreen(entry, currentTime)
            return (
              <li
                key={entry.number}
                aria-current={current ? 'true' : undefined}
                className={cn(
                  'flex gap-3 rounded-md border border-border p-3',
                  current && 'bg-accent/60 ring-2 ring-primary',
                )}
              >
                <img
                  src={entry.thumbnail}
                  alt={label}
                  className="h-16 w-16 shrink-0 rounded object-cover"
                />
                <div className="min-w-0 flex-1 space-y-1 text-sm">
                  <p className="font-medium text-foreground">{label}</p>
                  <p className="text-muted-foreground">
                    {t(entry.detections === 1 ? 'results.face_sightings_one' : 'results.face_sightings_other', {
                      count: entry.detections,
                    })}
                  </p>
                  <p className="text-muted-foreground">
                    {t('results.face_seen', {
                      first: formatStamp(entry.first_seen_seconds),
                      last: formatStamp(entry.last_seen_seconds),
                    })}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {entry.appearances.map((appearance) => (
                      <TimeButton
                        key={appearance.start_seconds}
                        label={formatStamp(appearance.start_seconds)}
                        seconds={appearance.start_seconds}
                        jobId={jobId}
                        fileName={fileName}
                        mediaUrl={mediaUrl}
                      />
                    ))}
                  </div>
                  <DownloadButtons
                    jobId={jobId}
                    items={[
                      {
                        name: entry.best_artifact,
                        label: 'JPG',
                        title: t('artifacts.download_face_best'),
                        fileName: `${stem}_${entry.best_artifact}`,
                      },
                    ]}
                  />
                </div>
              </li>
            )
          })}
        </ul>
      )}

      {hasEntries && !unavailable && (
        <DownloadButtons
          jobId={jobId}
          items={[
            { name: 'faces.zip', label: 'ZIP', title: t('artifacts.download_faces'), fileName: `${stem}_faces.zip` },
            {
              name: 'faces.csv',
              label: 'CSV',
              title: t('artifacts.download_faces_csv'),
              fileName: `${stem}_faces.csv`,
            },
          ]}
        />
      )}
    </div>
  )
}
