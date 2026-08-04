import { useSubmitBatch } from '../hooks/useJobs'
import { UploadForm } from '../components/upload/UploadForm'
import { BatchProgress } from '../components/jobs/BatchProgress'
import { StatusBar } from '../components/layout/StatusBar'
import { Banner, Card, PageHeader } from '@infra/ui'
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
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title={t('page.title')} caption={t('page.caption')} />
      <div className="grid min-h-0 flex-1 items-start gap-6 lg:grid-cols-[minmax(20rem,26rem)_1fr]">
        <section className="space-y-4">
          <Card title={t('home.new_job')}>
            <UploadForm
              pending={submit.isPending}
              onRun={(files, options) => submit.mutate({ files, options })}
            />
          </Card>
          {submitErrorDescriptor && (
            <Banner variant="danger">
              {t('errors.upload_failed')} {t(submitErrorDescriptor.key, submitErrorDescriptor.vars)}
            </Banner>
          )}
          {fileErrors.length > 0 && <Banner variant="danger">{fileErrors.join('\n')}</Banner>}
        </section>
        <section className="min-w-0 space-y-3">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-base font-semibold">{t('home.jobs_heading')}</h2>
            <StatusBar />
          </div>
          <BatchProgress />
        </section>
      </div>
    </div>
  )
}
