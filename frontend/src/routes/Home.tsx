import { useSubmitBatch } from '../hooks/useJobs'
import { UploadForm } from '../components/upload/UploadForm'
import { BatchProgress } from '../components/jobs/BatchProgress'
import { Banner } from '@infra/ui'
import { useT } from '../i18n/LanguageContext'
import { describeError } from '../api/errorMessage'

export function Home() {
  const t = useT()
  const submit = useSubmitBatch()

  // Collect per-file submission errors from the last batch (files that failed
  // to reach the backend at all — distinct from a job that runs and fails).
  // The file name is client-side and safe to render; the error itself is
  // never rendered verbatim, only its localized catalog text.
  const fileErrors: string[] =
    submit.data
      ?.filter((s) => s.error)
      .map((s) => `${s.file_name}: ${t(s.error!.key, s.error!.vars)}`) ?? []

  const submitErrorDescriptor = submit.error ? describeError(submit.error) : null

  return (
    <div className="space-y-8">
      {submitErrorDescriptor && (
        <Banner variant="danger">
          {t('errors.upload_failed')} {t(submitErrorDescriptor.key, submitErrorDescriptor.vars)}
        </Banner>
      )}
      {fileErrors.length > 0 && (
        <Banner variant="danger">{fileErrors.join('\n')}</Banner>
      )}
      <section>
        <h2 className="mb-3 text-base font-semibold">{t('home.new_job')}</h2>
        <UploadForm pending={submit.isPending} onRun={(files, options) => submit.mutate({ files, options })} />
      </section>
      <section>
        <h2 className="mb-3 text-base font-semibold">{t('home.jobs_heading')}</h2>
        <BatchProgress />
      </section>
    </div>
  )
}
