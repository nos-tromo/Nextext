import { useSubmitBatch } from '../hooks/useJobs'
import { UploadForm } from '../components/upload/UploadForm'
import { BatchProgress } from '../components/jobs/BatchProgress'
import { Banner } from '@infra/ui'

export function Home() {
  const submit = useSubmitBatch()

  // Collect per-file submission errors from the last batch (files that failed
  // to reach the backend at all — distinct from a job that runs and fails).
  const fileErrors: string[] =
    submit.data
      ?.filter((s) => s.error)
      .map((s) => `${s.file_name}: ${s.error}`) ?? []

  return (
    <div className="space-y-8">
      {submit.error && (
        <Banner variant="danger">{`Submission failed: ${submit.error.message}`}</Banner>
      )}
      {fileErrors.length > 0 && (
        <Banner variant="danger">{fileErrors.join('\n')}</Banner>
      )}
      <section>
        <h2 className="mb-3 text-base font-semibold">New job</h2>
        <UploadForm pending={submit.isPending} onRun={(files, options) => submit.mutate({ files, options })} />
      </section>
      <section>
        <h2 className="mb-3 text-base font-semibold">Jobs</h2>
        <BatchProgress />
      </section>
    </div>
  )
}
