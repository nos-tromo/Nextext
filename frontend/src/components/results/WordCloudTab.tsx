import { useArtifactImage } from '../../hooks/useArtifactImage'
import { Spinner } from '@infra/ui'
import { Banner } from '@infra/ui'
import { useT } from '../../i18n/LanguageContext'

interface WordCloudTabProps {
  jobId: string
}

/**
 * Displays the word-cloud image artifact for a completed job.
 *
 * Fetches the image as a blob object URL via {@link useArtifactImage}, shows a
 * {@link Spinner} while loading, a {@link Banner} on failure, and the
 * `<img>` once the URL is available. Its PNG download sits in the result
 * panel's header.
 *
 * @param jobId - The job identifier used to fetch the wordcloud artifact.
 */
export function WordCloudTab({ jobId }: WordCloudTabProps) {
  const t = useT()
  const { url, loading, error } = useArtifactImage(jobId, 'wordcloud.png', true)

  return (
    <div className="space-y-4">
      {loading && <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner label={t('results.wordcloud_loading')} /><span aria-hidden>{t('results.wordcloud_loading')}</span></div>}
      {error && <Banner variant="danger">{t(error.key, error.vars)}</Banner>}
      {url && (
        <img
          src={url}
          alt={t('results.wordcloud_alt')}
          className="max-w-full rounded-md border border-border"
        />
      )}
    </div>
  )
}
