import { useSubmitBatch } from '../hooks/useJobs'
import { UploadForm } from '../components/upload/UploadForm'
import { BatchProgress } from '../components/jobs/BatchProgress'
import { ErrorBanner } from '../components/common/ErrorBanner'

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
        <ErrorBanner message={`Submission failed: ${submit.error.message}`} />
      )}
      {fileErrors.length > 0 && (
        <ErrorBanner message={fileErrors.join('\n')} />
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
