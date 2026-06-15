import { useHealth } from '../hooks/useHealth'
import { useLanguages } from '../hooks/useLanguages'
import { Spinner } from '../components/common/Spinner'
import { ErrorBanner } from '../components/common/ErrorBanner'

export function Home() {
  const health = useHealth()
  const languages = useLanguages()

  if (health.isLoading || languages.isLoading) return <Spinner label="Contacting backend…" />
  if (health.error) return <ErrorBanner message={`Backend unreachable: ${String(health.error)}`} />

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Backend OK — version {health.data?.version}, inference{' '}
        {health.data?.inference ? 'reachable' : 'unreachable'}.
      </p>
      <p className="text-sm text-muted-foreground">
        {languages.data?.whisper.length ?? 0} source languages,{' '}
        {languages.data?.target.length ?? 0} target languages loaded.
      </p>
      <p className="text-foreground">Job UI arrives in Plan 02.</p>
    </div>
  )
}
