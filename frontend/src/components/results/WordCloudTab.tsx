import { useArtifactImage } from '../../hooks/useArtifactImage'
import { Spinner } from '../common/Spinner'
import { Banner } from '@infra/ui'
import { useT } from '../../i18n/LanguageContext'
import { DownloadButtons } from './DownloadButtons'

interface WordCloudTabProps {
  jobId: string
  stem: string
}

/**
 * Displays the word-cloud image artifact for a completed job.
 *
 * Fetches the image as a blob object URL via {@link useArtifactImage}, shows a
 * {@link Spinner} while loading, a {@link Banner} on failure, and the
 * `<img>` once the URL is available. Provides a stem-prefixed PNG download
 * button.
 *
 * @param jobId - The job identifier used to fetch the wordcloud artifact.
 * @param stem - Upload filename without extension; used to prefix the download name.
 */
export function WordCloudTab({ jobId, stem }: WordCloudTabProps) {
  const t = useT()
  const { url, loading, error } = useArtifactImage(jobId, 'wordcloud.png', true)

  return (
    <div className="space-y-4">
      {loading && <Spinner label={t('results.wordcloud_loading')} />}
      {error && <Banner variant="danger">{error}</Banner>}
      {url && (
        <img
          src={url}
          alt={t('results.wordcloud_alt')}
          className="max-w-full rounded-md border border-border"
        />
      )}
      <DownloadButtons
        jobId={jobId}
        items={[{ name: 'wordcloud.png', label: 'PNG', fileName: `${stem}_wordcloud.png` }]}
      />
    </div>
  )
}
