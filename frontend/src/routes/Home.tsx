import { useSubmitBatch } from '../hooks/useJobs'
import { UploadForm } from '../components/upload/UploadForm'
import { BatchProgress } from '../components/jobs/BatchProgress'

export function Home() {
  const submit = useSubmitBatch()
  return (
    <div className="space-y-8">
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
